"""WebUI bridge for Hermes ``/loop`` recurring in-session wakeups.

Loop state is owned by ``hermes_cli.loops.LoopManager`` (the ``loop:<sid>`` row in the
session profile's ``state.db``), the same store the CLI, TUI and gateway drive. WebUI-created
loops carry ``route={"platform": "webui", ...}``: the gateway wakeup scanner finds no adapter for
that platform and the TUI poller treats any routed loop as someone else's, so only this module's
scheduler fires them. Ticks start server-side through ``routes.start_session_turn`` (the same
path process wakeups use), so a loop keeps running with the browser tab closed.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:  # Module attributes so tests can monkeypatch them.
    from hermes_cli import loops as _agent_loops  # type: ignore
except Exception:  # pragma: no cover - depends on installed hermes-agent
    _agent_loops = None  # type: ignore

LOOP_WAKEUP_SOURCE = "loop_wakeup"
WEBUI_LOOP_PLATFORM = "webui"
SCAN_INTERVAL_SECONDS = 15.0
# A fired tick whose turn ended without reaching the post-turn hook (provider error, cancel,
# server restart) would stay ``awaiting_response`` forever; complete it once the session has
# been idle this long after the fire.
STALE_TICK_GRACE_SECONDS = 60.0
_CONTROL_WORDS = {"", "status", "pause", "resume", "stop", "clear", "cancel"}
_HELP_WORDS = {"help", "--help", "-h"}
LOOP_HELP = (
    "Usage: /loop [interval] <prompt>\n"
    "  /loop 5m check the deploy status      — first run now, then every 5m\n"
    "  /loop keep fixing tests until green   — self-paced (backs off while output is unchanged)\n"
    "Controls: /loop status · /loop pause · /loop resume · /loop stop\n"
    "The loop stops itself when the agent reports the task is done (LOOP_COMPLETE), "
    "or after loops.max_ticks runs (default 100)."
)

_SCHEDULER_LOCK = threading.Lock()
_SCHEDULER_THREAD: Optional[threading.Thread] = None
_SCHEDULER_WAKE = threading.Event()
_SCHEDULER_STOP = threading.Event()


def loops_available() -> bool:
    return _agent_loops is not None and _home_override_api() is not None


def _home_override_api():
    try:
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    except Exception:  # pragma: no cover - depends on installed hermes-agent
        return None
    return set_hermes_home_override, reset_hermes_home_override


@contextlib.contextmanager
def _profile_scope(profile_home: str | Path | None) -> Iterator[None]:
    """Pin the agent's SessionDB/config resolution to *profile_home* for this thread only."""
    api = _home_override_api() if profile_home else None
    if api is None:
        yield
        return
    set_home, reset_home = api
    token = set_home(str(Path(profile_home).expanduser()))
    try:
        yield
    finally:
        reset_home(token)


def _state_payload(state: Any) -> Optional[Dict[str, Any]]:
    if state is None or getattr(state, "status", "") == "cleared":
        return None
    return {
        "prompt": getattr(state, "prompt", "") or "",
        "status": getattr(state, "status", "") or "",
        "mode": getattr(state, "mode", "") or "",
        "interval_seconds": float(getattr(state, "interval_seconds", 0) or 0),
        "current_delay": float(getattr(state, "current_delay", 0) or 0),
        "times": int(getattr(state, "times", 0) or 0),
        "until": getattr(state, "until", "") or "",
        "ticks_fired": int(getattr(state, "ticks_fired", 0) or 0),
        "max_ticks": int(getattr(state, "max_ticks", 0) or 0),
        "next_due_at": float(getattr(state, "next_due_at", 0) or 0),
        "awaiting_response": bool(getattr(state, "awaiting_response", False)),
        "paused_reason": getattr(state, "paused_reason", None),
        "last_stop_reason": getattr(state, "last_stop_reason", None),
    }


def _is_webui_route(state: Any) -> bool:
    route = getattr(state, "route", None) or {}
    return isinstance(route, dict) and route.get("platform") == WEBUI_LOOP_PLATFORM


def loop_command_payload(
    session_id: str,
    args: str = "",
    *,
    profile_home: str | Path | None = None,
    profile: str | None = None,
) -> Dict[str, Any]:
    """Run ``/loop <args>`` for a WebUI session through the agent's own command handler."""
    sid = str(session_id or "").strip()
    if not sid:
        return {"ok": False, "error": "session_required", "message": "No active session."}
    if not loops_available():
        return {
            "ok": False,
            "error": "loops_unavailable",
            "message": "/loop needs a hermes-agent version that ships hermes_cli.loops.",
        }
    arg = str(args or "").strip()
    if arg.lower() in _HELP_WORDS:
        return {"ok": True, "action": "help", "created": False, "message": LOOP_HELP, "loop": None}
    if arg.lower() in _CONTROL_WORDS:
        with _profile_scope(profile_home):
            mgr = _agent_loops.LoopManager(session_id=sid)
            result = _agent_loops.dispatch_loop_command(mgr, arg)
            state = mgr.state
        return {
            "ok": True,
            "action": arg.lower() or "status",
            "created": False,
            "message": str(result.get("output") or ""),
            "loop": _state_payload(state),
        }

    parsed = _agent_loops.parse_loop_args(arg)
    if parsed.get("error"):
        usage = "Usage: /loop [interval] <prompt> — see /loop help."
        text = usage if parsed["error"] == "empty" else f"/loop: {parsed['error']}"
        return {"ok": False, "error": "invalid_args", "message": text}
    if parsed.get("times") or parsed.get("until"):
        return {
            "ok": False,
            "error": "unsupported_flag",
            "message": "/loop: --times and --until are not supported in the WebUI. The loop stops "
            "itself when the task is done, or after the default run limit.",
        }
    prompt = str(parsed.get("prompt") or "")
    if prompt.lstrip().startswith("/"):
        return {
            "ok": False,
            "error": "slash_prompt_unsupported",
            "message": "/loop: looping a slash command is not supported in the WebUI yet — "
            "loop a plain prompt instead.",
        }

    route = {"platform": WEBUI_LOOP_PLATFORM, "chat_id": sid}
    if profile:
        route["profile"] = str(profile)
    with _profile_scope(profile_home):
        mgr = _agent_loops.LoopManager(session_id=sid)
        replacing = mgr.has_loop()
        # The run limit is a hard stop (``times``), not the agent's resumable max_ticks pause.
        limit = _agent_loops.max_ticks_default()
        try:
            state = mgr.set(prompt, interval_seconds=parsed.get("interval_seconds"),
                            times=limit, route=route)
        except ValueError as exc:
            return {"ok": False, "error": "invalid_args", "message": f"/loop: {exc}"}
        goal_active = False
        with contextlib.suppress(Exception):
            goal_active = bool(_agent_loops.goal_blocks_loop_tick(sid))
        floor = _agent_loops.format_interval(state.interval_seconds)
        ceiling = _agent_loops.format_interval(_agent_loops.self_paced_ceiling_seconds())

    lines = [f"↻ Loop set ({state.cadence_label()}): {state.prompt}"]
    if replacing:
        lines.append("(replaced the previous loop for this session)")
    requested = parsed.get("interval_seconds")
    if requested is not None and requested < state.interval_seconds:
        lines.append(f"(interval raised to the {floor} minimum — loops.min_interval_seconds)")
    if state.mode == "self_paced":
        lines.append(f"Self-paced: backs off up to {ceiling} while nothing changes.")
    stop = "Stops when the task is done"
    lines.append(f"{stop}, or after {limit} runs (loops.max_ticks)." if limit else f"{stop} (no run limit).")
    lines.append("First wakeup fires now. Controls: /loop status · pause · resume · stop.")
    if goal_active:
        lines.append("Note: an active /goal is driving this session — loop wakeups "
                     "defer until the goal finishes, pauses, or parks.")
    wake_scheduler()
    return {
        "ok": True,
        "action": "set",
        "created": True,
        "message": "\n".join(lines),
        "loop": _state_payload(state),
    }


def last_assistant_text(messages: Iterable[Any]) -> str:
    """Plain text of the newest assistant message (string or content-part list)."""
    for msg in reversed(list(messages or [])):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if text:
                        parts.append(str(text))
            return "\n".join(parts)
        return str(content or "")
    return ""


def evaluate_loop_after_turn(
    session_id: str,
    last_response: str,
    *,
    profile_home: str | Path | None = None,
) -> Dict[str, Any]:
    """Complete the in-flight tick for a finished ``loop_wakeup`` turn; ``{}`` when none."""
    if not loops_available() or not session_id:
        return {}
    with _profile_scope(profile_home):
        mgr = _agent_loops.LoopManager(session_id=str(session_id))
        state = mgr.state
        if state is None or not state.awaiting_response:
            return {}
        decision = mgr.complete_tick(str(last_response or ""))
        payload = _state_payload(mgr.state)
    return {**(decision or {}), "loop": payload}


# ── Scheduler ────────────────────────────────────────────────────────────────


def _profile_homes() -> List[Tuple[Optional[str], Path]]:
    """``[(profile_name, home)]`` for every profile whose state.db could hold a WebUI loop."""
    from api import profiles as _profiles

    if _profiles._is_isolated_profile_mode():
        name = _profiles._isolated_profile_name()
        return [(name, _profiles.get_hermes_home_for_profile(name))]
    homes: List[Tuple[Optional[str], Path]] = [(None, _profiles.get_hermes_home_for_profile(None))]
    root = _profiles._profiles_root()
    try:
        children = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        children = []
    for child in children:
        if (child / "state.db").exists():
            homes.append((child.name, child))
    return homes


def _session_has_active_turn(session_id: str) -> bool:
    from api.background_process import _session_has_active_turn as _active

    return _active(session_id)


def _webui_session_exists(session_id: str) -> bool:
    from api.models import get_session

    try:
        get_session(session_id, metadata_only=True)
        return True
    except KeyError:
        return False


def _start_turn(session_id: str, message: str) -> Dict[str, Any]:
    from api.routes import start_session_turn

    return start_session_turn(session_id, message, source=LOOP_WAKEUP_SOURCE) or {}


def _fire_one(session_id: str, profile_home: Path, now: float) -> str:
    """Advance one WebUI loop; returns what happened (for logs and tests)."""
    with _profile_scope(profile_home):
        mgr = _agent_loops.LoopManager(session_id=session_id)
        state = mgr.state
        if state is None or state.status != "active" or not _is_webui_route(state):
            return "skip"
        if state.awaiting_response:
            fired_at = float(state.last_fired_at or 0)
            if now - fired_at < STALE_TICK_GRACE_SECONDS or _session_has_active_turn(session_id):
                return "in_flight"
            mgr.complete_tick("")
            return "recovered_stale_tick"
        if not mgr.is_due(now):
            return "not_due"
        if not _webui_session_exists(session_id):
            mgr.clear()
            return "cleared_missing_session"
        if _session_has_active_turn(session_id) or _agent_loops.goal_blocks_loop_tick(session_id):
            return "busy"
        wakeup = mgr.fire_tick()
        if not wakeup:
            return "not_due"
    try:
        resp = _start_turn(session_id, wakeup)
        status = int(resp.get("_status", 200) or 200)
    except Exception:
        logger.warning("/loop wakeup turn raised for session %s", session_id, exc_info=True)
        status = 500
        resp = {}
    if status >= 400:
        with _profile_scope(profile_home):
            _agent_loops.LoopManager(session_id=session_id).abandon_tick()
        if status != 409:
            logger.warning("/loop wakeup failed for session %s: status=%s err=%r",
                           session_id, status, resp.get("error"))
        return "start_failed"
    return "fired"


def fire_due_loops(now: Optional[float] = None) -> Dict[str, str]:
    """One scheduler pass over every profile; ``{session_id: outcome}``."""
    if not loops_available():
        return {}
    now = time.time() if now is None else now
    outcomes: Dict[str, str] = {}
    for _name, home in _profile_homes():
        try:
            with _profile_scope(home):
                active = _agent_loops.list_active_loops()
        except Exception:
            logger.debug("/loop scan failed for %s", home, exc_info=True)
            continue
        for session_id, state in active:
            if not _is_webui_route(state):
                continue
            try:
                outcomes[session_id] = _fire_one(session_id, home, now)
            except Exception:
                logger.warning("/loop tick failed for session %s", session_id, exc_info=True)
                outcomes[session_id] = "error"
    return outcomes


def wake_scheduler() -> None:
    """Run the next scan now (a new loop's first wakeup fires immediately)."""
    _SCHEDULER_WAKE.set()


def _scheduler_loop() -> None:
    while not _SCHEDULER_STOP.is_set():
        try:
            fire_due_loops()
        except Exception:
            logger.warning("/loop scheduler pass failed", exc_info=True)
        _SCHEDULER_WAKE.wait(SCAN_INTERVAL_SECONDS)
        _SCHEDULER_WAKE.clear()


def start_loop_scheduler() -> bool:
    """Start the daemon scheduler once per process; False when loops are unavailable."""
    global _SCHEDULER_THREAD
    if not loops_available():
        return False
    with _SCHEDULER_LOCK:
        if _SCHEDULER_THREAD is not None and _SCHEDULER_THREAD.is_alive():
            return True
        _SCHEDULER_STOP.clear()
        _SCHEDULER_THREAD = threading.Thread(
            target=_scheduler_loop, name="hermes-webui-loop-scheduler", daemon=True)
        _SCHEDULER_THREAD.start()
    return True


def stop_loop_scheduler(timeout: float = 2.0) -> None:
    _SCHEDULER_STOP.set()
    _SCHEDULER_WAKE.set()
    thread = _SCHEDULER_THREAD
    if thread is not None:
        thread.join(timeout)
