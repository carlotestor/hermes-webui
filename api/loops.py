"""WebUI /loop: hermes_cli.loops owns the state; one server thread fires and judges ticks."""
import json
import logging
import threading
import weakref
from contextlib import contextmanager

_WAKE = threading.Event()
_LOCKS = weakref.WeakValueDictionary()
_LOCKS_GUARD = threading.Lock()
_TURN_PREFIX = "webui_loop_turn:"


def _session_lock(session_id):
    # One lock per session: the /loop command, the scheduler and profile retagging all take it.
    with _LOCKS_GUARD:
        lock = _LOCKS.get(session_id)
        if lock is None:
            lock = _LOCKS[session_id] = threading.Lock()
        return lock


@contextmanager
def _home(profile):
    from api.profiles import get_hermes_home_for_profile
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(str(get_hermes_home_for_profile(profile)))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _cas(db, key, expected, value):
    """Write state_meta[key]=value only if it still holds `expected` (another writer wins otherwise)."""
    def _do(conn):
        row = conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()
        if (row[0] if row else None) != expected:
            return False
        conn.execute("INSERT INTO state_meta (key, value) VALUES (?, ?) "
                     "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
        return True
    return db._execute_write(_do)


def _wakeup_reply(messages, turn):
    """Assistant text answering the recorded wakeup turn; "" if that turn produced none or is not found."""
    msgs = list(messages or [])
    token, at = (turn or {}).get("token"), (turn or {}).get("started_at")
    for i, m in enumerate(msgs):
        if m.get("role") == "user" and ((token and m.get("_active_turn_token") == token)
                                         or (at and m.get("timestamp") == at)):
            reply = ""
            for n in msgs[i + 1:]:
                if n.get("role") == "user":
                    break
                if n.get("role") == "assistant":
                    reply = str(n.get("content") or "")
            return reply
    return ""


def retag_session_profile(session, profile):
    """Move an empty session to `profile`, clearing the loop it left in its old profile."""
    from hermes_cli.loops import LoopManager
    with _session_lock(session.session_id):
        old = getattr(session, "profile", None)
        session.profile = profile
        with _home(old):
            LoopManager(session_id=session.session_id).clear()


def run_loop_command(session_id, args):
    from api.models import get_session
    from hermes_cli.loops import LoopManager, dispatch_loop_command, parse_loop_args
    p = parse_loop_args(args)
    if not p["error"] and p["prompt"].startswith("/"):  # WebUI slash commands run in the browser, not the agent
        return "/loop: looping slash commands isn't supported in the WebUI."
    with _session_lock(session_id):
        try:
            profile = get_session(session_id, metadata_only=True).profile
        except KeyError:  # fail closed: never fall back to another profile's state.db
            return "/loop: open a saved chat first."
        with _home(profile):
            out = dispatch_loop_command(LoopManager(session_id=session_id), args,
                                        route={"platform": "webui", "chat_id": session_id})
    _WAKE.set()
    return out["output"]


def _conditional_manager():
    from hermes_cli import loops as cli

    class _CasLoop(cli.LoopManager):
        """LoopManager whose saves only land if the row is still the one this pass read."""
        def __init__(self, session_id, db):
            self.session_id, self._db, self._key = session_id, db, cli._meta_key(session_id)
            self._raw = db.get_meta(self._key)
            self._state = cli._parse_state(self._raw, session_id) if self._raw else None
            self.lost = False

        def _save(self):
            new = self._state.to_json()
            if not self.lost and _cas(self._db, self._key, self._raw, new):
                self._raw = new
            else:
                self.lost = True  # a /loop command or another process changed it first
            return self._state
    return _CasLoop


def _run_one(profile, sid):
    from api.background_process import _session_has_active_turn
    from api.models import get_session
    from api.process_event_utils import build_active_turn_token
    from api.profiles import _profiles_match
    from api.routes import start_session_turn
    from api.streaming import _session_has_cancel_marker
    from hermes_cli import loops as cli
    db = cli._get_session_db()
    if db is None:
        return
    mgr = _conditional_manager()(sid, db)
    state = mgr.state
    if (state is None or state.status != "active" or (state.route or {}).get("platform") != "webui"
            or _session_has_active_turn(sid)):
        return
    try:
        if state.awaiting_response:  # wakeup turn ended: same verdicts as the CLI post-turn hook
            s = get_session(sid)
            if _session_has_cancel_marker(s):
                mgr.pause(reason="user-interrupted (Stop)")
            else:
                mgr.complete_tick(_wakeup_reply(s.messages, json.loads(db.get_meta(_TURN_PREFIX + sid) or "{}")))
            return
        if cli.goal_blocks_loop_tick(sid):
            return
        if not _profiles_match(get_session(sid, metadata_only=True).profile, profile):
            mgr.clear()  # the session moved to another profile: never launch this loop there
            return
        msg = mgr.fire_tick()
        if not msg or mgr.lost:
            return
        try:
            resp = start_session_turn(sid, msg, source="loop_wakeup")
        except Exception:  # nothing ran: roll the tick back so it is never judged
            mgr.abandon_tick()
            raise
        status = resp.get("_status", 200)
        if status >= 400:
            mgr.clear() if status == 404 else mgr.abandon_tick()
            return
        at = resp.get("pending_started_at")
        db.set_meta(_TURN_PREFIX + sid, json.dumps(
            {"token": build_active_turn_token(resp.get("stream_id"), at), "started_at": at}))
    except KeyError:  # session deleted
        mgr.clear()


def run_due_loops():
    from api.profiles import _profiles_root
    from hermes_cli.loops import list_active_loops
    root = _profiles_root()
    for profile in [None] + (sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []):
        with _home(profile):
            for sid, state in list_active_loops():
                if (state.route or {}).get("platform") != "webui":
                    continue
                try:
                    with _session_lock(sid):
                        _run_one(profile, sid)
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
