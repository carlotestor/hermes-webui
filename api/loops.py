"""WebUI /loop: hermes_cli.loops owns the state; one server thread fires and judges ticks."""
import logging
import threading
from contextlib import contextmanager

_WAKE = threading.Event()


@contextmanager
def _home(profile):
    from api.profiles import get_hermes_home_for_profile
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(str(get_hermes_home_for_profile(profile)))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _wakeup_reply(messages):
    """Assistant text after the newest user message; "" if the wakeup turn produced none (never a stale reply)."""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return ""
        if m.get("role") == "assistant":
            return str(m.get("content") or "")
    return ""


def run_loop_command(session_id, args):
    from api.models import get_session
    from hermes_cli.loops import LoopManager, dispatch_loop_command, parse_loop_args
    p = parse_loop_args(args)
    if not p["error"] and p["prompt"].startswith("/"):  # WebUI slash commands run in the browser, not the agent
        return "/loop: looping slash commands isn't supported in the WebUI."
    try:
        profile = get_session(session_id, metadata_only=True).profile
    except KeyError:  # fail closed: never fall back to another profile's state.db
        return "/loop: open a saved chat first."
    with _home(profile):
        out = dispatch_loop_command(LoopManager(session_id=session_id), args,
                                    route={"platform": "webui", "chat_id": session_id})
    _WAKE.set()
    return out["output"]


def run_due_loops():
    from api.background_process import _session_has_active_turn
    from api.models import get_session
    from api.profiles import _profiles_root
    from api.routes import start_session_turn
    from api.streaming import _session_has_cancel_marker
    from hermes_cli.loops import LoopManager, goal_blocks_loop_tick, list_active_loops
    root = _profiles_root()
    for profile in [None] + (sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []):
        with _home(profile):
            for sid, state in list_active_loops():
                if (state.route or {}).get("platform") != "webui" or _session_has_active_turn(sid):
                    continue
                mgr = LoopManager(session_id=sid)
                try:
                    if state.awaiting_response:  # wakeup turn ended: same verdicts as the CLI post-turn hook
                        s = get_session(sid)
                        if _session_has_cancel_marker(s):
                            mgr.pause(reason="user-interrupted (Stop)")
                        else:
                            mgr.complete_tick(_wakeup_reply(s.messages))
                    elif not goal_blocks_loop_tick(sid) and (msg := mgr.fire_tick()):
                        try:
                            status = start_session_turn(sid, msg, source="loop_wakeup").get("_status", 200)
                        except Exception:  # nothing ran: roll the tick back so it is never judged
                            mgr.abandon_tick()
                            raise
                        if status >= 400:
                            mgr.clear() if status == 404 else mgr.abandon_tick()
                except KeyError:  # session deleted
                    mgr.clear()
                except Exception:
                    logging.getLogger(__name__).warning("/loop tick failed for %s", sid, exc_info=True)


def start_loop_scheduler():
    def _run():
        while True:
            _WAKE.wait(15)
            _WAKE.clear()
            try:
                run_due_loops()
            except Exception:
                logging.getLogger(__name__).debug("/loop scheduler pass failed", exc_info=True)
    threading.Thread(target=_run, name="webui-loop-scheduler", daemon=True).start()
