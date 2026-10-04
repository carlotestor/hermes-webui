"""Wake WebUI sessions for async-delegation completions delivered by the Gateway.

With the Gateway chat backend the agent runs in the Gateway process, so its
``async_delegation`` completions never reach this process's
``process_registry.completion_queue`` (``api.background_process`` drain).
For api_server sessions the Gateway instead persists each completion as a
delivery row in the profile's state.db (``display_kind`` ``async_delegation_complete``,
or ``hidden`` for presentation-suppressed notices) and deliberately starts no turn: the client owns the next turn
(``gateway.wake.persist_delegation_delivery``). Without a consumer here the
parent agent only sees the result when the user next types.

This poller leases those rows with the Agent's reservation API
(``SessionDB.reserve_caller_history_deliveries``), copies each result once into
the session's model-facing ``context_messages`` (keyed by row id, so a retry
never duplicates it), and starts a wakeup turn whose prompt is a FIXED nudge:
the stored copy is the only one the model sees (the legacy Gateway path reads
the state.db row instead; the runs-API fold skips content the caller already
carries). Rows are committed only once the wake turn is persisted, and released
otherwise; a WebUI crash in between just lets the lease expire and the next
poll retries. Without reserve+commit the poller stays inert: the old final
claim could lose a result, so there is no fallback to it.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 5.0
RESERVATION_OWNER = "hermes-webui"
RESERVATION_TTL_S = 120.0
WAKE_PROMPT = ("[IMPORTANT: A delegated subagent finished. Its result is already in your "
               "conversation history above; review it and continue the task.]")
_REQUIRED_API = ("reserve_caller_history_deliveries", "commit_caller_history_deliveries",
                 "release_caller_history_deliveries")
# Rows older than this at first sight are left for the next user turn to fold.
MAX_ROW_AGE_S = 6 * 3600

_THREAD: threading.Thread | None = None
_STOP = threading.Event()
_LOCK = threading.Lock()

_PENDING_SQL = (
    "SELECT DISTINCT session_id FROM messages WHERE role = 'user'"
    " AND display_kind IN ('async_delegation_complete', 'hidden')"
    " AND coalesce(json_extract(display_metadata, '$.delegation_id'), '') != ''"
    " AND json_extract(display_metadata, '$.caller_history_consumed') IS NULL"
    " AND timestamp >= ?"
)


def _profile_state_dbs() -> list[tuple[str, Path]]:
    from api.profiles import _DEFAULT_HERMES_HOME

    base = Path(_DEFAULT_HERMES_HOME)
    out = [("default", base / "state.db")]
    profiles_dir = base / "profiles"
    if profiles_dir.is_dir():
        out += [(p.name, p / "state.db") for p in sorted(profiles_dir.iterdir()) if p.is_dir()]
    return [(name, path) for name, path in out if path.is_file()]


def _pending_session_ids(db_path: Path, since: float) -> list[str]:
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    try:
        return [row[0] for row in conn.execute(_PENDING_SQL, (since,))]
    finally:
        conn.close()


def _session_db_cls():
    from hermes_state import SessionDB

    return SessionDB


def _claim_api_available() -> bool:
    try:
        cls = _session_db_cls()
    except Exception:
        return False
    return all(callable(getattr(cls, name, None)) for name in _REQUIRED_API)


def _sidecar_profile_matches(sidecar: Path, profile: str) -> bool:
    import json
    from api.profiles import _profiles_match

    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and _profiles_match(data.get("profile"), profile)


def _webui_session_id(db_path: Path, session_id: str, profile: str) -> str | None:
    """The WebUI sidecar of *profile* that owns *session_id*, following compression lineage upward."""
    import sqlite3
    from api.config import SESSION_DIR

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    try:
        sid, seen = session_id, set()
        while sid and sid not in seen:
            sidecar = Path(SESSION_DIR) / f"{sid}.json"
            if sidecar.is_file():
                return sid if _sidecar_profile_matches(sidecar, profile) else None
            seen.add(sid)
            row = conn.execute(
                "SELECT p.id FROM sessions s JOIN sessions p ON p.id = s.parent_session_id"
                " WHERE s.id = ? AND p.end_reason = 'compression'", (sid,)).fetchone()
            sid = row[0] if row else None
        return None
    finally:
        conn.close()


def _start_wakeup(webui_sid: str) -> bool:
    """True once the wake turn is persisted as the session's pending turn."""
    from api.routes import start_session_turn

    try:
        resp = start_session_turn(webui_sid, WAKE_PROMPT, source="process_wakeup") or {}
    except Exception:
        logger.warning("gateway delegation wakeup raised for %s", webui_sid, exc_info=True)
        return False
    status = int(resp.get("_status", 200) or 200)
    if status >= 400 and status != 409:
        logger.warning("gateway delegation wakeup failed for %s: %s %r", webui_sid, status, resp.get("error"))
    return status < 400 and bool(resp.get("stream_id"))


def _store_results_in_context(webui_sid: str, rows: list[dict]) -> None:
    """Append each reserved result once to the sidecar's model-facing context; raises on failure."""
    from api.config import _get_session_agent_lock
    from api.models import get_session
    from api.routes import _ensure_full_session_before_mutation

    with _get_session_agent_lock(webui_sid):
        s = _ensure_full_session_before_mutation(webui_sid, get_session(webui_sid))
        context = list(getattr(s, "context_messages", None) or [])
        if not context:
            context = [m for m in (getattr(s, "messages", None) or []) if isinstance(m, dict)
                       and m.get("role") in ("user", "assistant") and not m.get("_error")]
        have = {m.get("_delegation_delivery_id") for m in context if isinstance(m, dict)}
        added = [{"role": "user", "content": r["content"], "_delegation_delivery_id": r["id"],
                  "timestamp": time.time()} for r in rows if r["id"] not in have]
        if added:
            s.context_messages = context + added
            s.save(touch_updated_at=False)


def _claim_and_wake(db_path: Path, state_sid: str, webui_sid: str) -> int:
    """Reserve the session's pending rows, store them once, wake with a fixed prompt, then commit."""
    from api.background_process import _session_has_active_turn

    if _session_has_active_turn(webui_sid):
        return 0  # rows stay pending: the running turn's successor folds them, or the next poll
    db = _session_db_cls()(db_path)
    try:
        rows = db.reserve_caller_history_deliveries(state_sid, RESERVATION_OWNER, RESERVATION_TTL_S)
        rows = [r for r in rows if isinstance(r.get("content"), str) and r["content"].strip()]
        if not rows:
            return 0
        token = rows[0]["reservation_token"]
        try:
            _store_results_in_context(webui_sid, rows)
            woke = _start_wakeup(webui_sid)
        except Exception:
            logger.warning("gateway delegation wakeup failed for %s", webui_sid, exc_info=True)
            woke = False
        if woke:
            db.commit_caller_history_deliveries(token, RESERVATION_OWNER)
            return 1
        db.release_caller_history_deliveries(reservation_token=token, owner=RESERVATION_OWNER)
        return 0
    finally:
        db.close()


def poll_once(since: float) -> int:
    woken = 0
    for profile, db_path in _profile_state_dbs():
        try:
            session_ids = _pending_session_ids(db_path, since)
        except Exception:
            logger.debug("gateway delegation poll failed for profile %s", profile, exc_info=True)
            continue
        for sid in session_ids:
            try:
                webui_sid = _webui_session_id(db_path, sid, profile)
                if webui_sid:
                    woken += _claim_and_wake(db_path, sid, webui_sid)
            except Exception:
                logger.warning("gateway delegation wakeup raised for %s", sid, exc_info=True)
    return woken


def _loop() -> None:
    since = time.time() - MAX_ROW_AGE_S
    while not _STOP.wait(POLL_INTERVAL_S):
        poll_once(since)


def start_gateway_delegation_poller() -> bool:
    """Start the poller when chat runs on the Gateway backend. Never raises."""
    global _THREAD
    try:
        from api.config import get_config
        from api.gateway_chat import webui_gateway_chat_enabled

        if not webui_gateway_chat_enabled(get_config()):
            return False
        if not _claim_api_available():
            logger.info("gateway delegation poller inactive: installed Hermes Agent lacks the delivery reserve/commit API")
            return False
        with _LOCK:
            if _THREAD is not None and _THREAD.is_alive():
                return False
            _STOP.clear()
            _THREAD = threading.Thread(target=_loop, name="hermes-webui-gateway-deleg-wakeup", daemon=True)
            _THREAD.start()
        return True
    except Exception:
        logger.warning("gateway delegation poller failed to start", exc_info=True)
        return False


def stop_gateway_delegation_poller(timeout: float = 2.0) -> None:
    _STOP.set()
    th = _THREAD
    if th is not None and th.is_alive():
        th.join(timeout=timeout)
