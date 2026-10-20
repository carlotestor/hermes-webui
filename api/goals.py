"""WebUI bridge for Hermes persistent session goals."""

from __future__ import annotations

import copy
import inspect
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:  # Exposed as a module attribute so tests can monkeypatch it directly.
    from hermes_cli.goals import (  # type: ignore
        CONTINUATION_PROMPT_TEMPLATE,
        DEFAULT_MAX_TURNS,
        GoalManager as _NativeGoalManager,
        GoalState,
        judge_goal,
    )
except Exception:  # pragma: no cover - depends on installed hermes-agent
    CONTINUATION_PROMPT_TEMPLATE = ""  # type: ignore
    DEFAULT_MAX_TURNS = 20  # type: ignore
    _NativeGoalManager = None  # type: ignore
    GoalState = None  # type: ignore
    judge_goal = None  # type: ignore

GoalManager = _NativeGoalManager  # type: ignore

_DB_CACHE: dict[str, Any] = {}


def _default_max_turns() -> int:
    """Return the configured /goal turn budget, defaulting to Hermes' 20 turns."""
    try:
        from api import config as _config

        cfg = getattr(_config, "cfg", {}) or {}
        goals_cfg = cfg.get("goals", {}) if isinstance(cfg, dict) else {}
        if not isinstance(goals_cfg, dict):
            return int(DEFAULT_MAX_TURNS or 20)
        return max(1, int(goals_cfg.get("max_turns", DEFAULT_MAX_TURNS or 20) or 20))
    except Exception:
        return int(DEFAULT_MAX_TURNS or 20)


def _meta_key(session_id: str) -> str:
    return f"goal:{session_id}"


def _profile_db(profile_home: str | Path):
    """Return a SessionDB pinned to *profile_home*, without reading HERMES_HOME.

    The upstream Hermes GoalManager persists through hermes_cli.goals.load_goal(),
    which resolves SessionDB from process-global HERMES_HOME. WebUI sessions are
    profile-scoped and can run concurrently, so the WebUI bridge uses an explicit
    state.db path whenever the caller provides the session's profile home.
    """
    home = Path(profile_home).expanduser().resolve()
    key = str(home)
    cached = _DB_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        from hermes_state import SessionDB  # type: ignore

        db = SessionDB(db_path=home / "state.db")
    except Exception as exc:  # pragma: no cover - import/env dependent
        logger.debug("GoalManager profile DB unavailable for %s: %s", home, exc)
        return None
    _DB_CACHE[key] = db
    return db


def _profile_home_context_api():
    """Return the native context-local Hermes home setters when available."""
    try:
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    except Exception:  # pragma: no cover - depends on installed hermes-agent
        return None
    return set_hermes_home_override, reset_hermes_home_override


def _native_profile_context_api(profile_home: str | Path):
    """Return context setters only when SessionDB resolves that context at call time."""
    context_api = _profile_home_context_api()
    if context_api is None:
        return None
    try:
        from hermes_state import _default_db_path  # type: ignore
    except Exception:  # pragma: no cover - depends on installed hermes-agent
        return None
    if not callable(_default_db_path):
        return None

    home = Path(profile_home).expanduser().resolve()
    set_home, reset_home = context_api
    try:
        token = set_home(home)
    except Exception:  # pragma: no cover - depends on installed hermes-agent
        return None
    try:
        resolved_db_path = Path(_default_db_path()).expanduser().resolve()
    except Exception:  # pragma: no cover - depends on installed hermes-agent
        return None
    finally:
        reset_home(token)
    if resolved_db_path != home / "state.db":
        return None
    return context_api


class _ProfileGoalManager:
    """Run the native GoalManager under a context-local profile home."""

    def __init__(
        self,
        session_id: str,
        *,
        profile_home: str | Path,
        default_max_turns: int = 20,
        context_api=None,
    ):
        if _NativeGoalManager is None:
            raise RuntimeError("Hermes goal manager unavailable")
        context_api = context_api or _profile_home_context_api()
        if context_api is None:
            raise RuntimeError("Hermes profile context unavailable")
        self.session_id = session_id
        self.profile_home = Path(profile_home).expanduser().resolve()
        self._set_home, self._reset_home = context_api
        self._manager = self._scoped(
            _NativeGoalManager,
            session_id=session_id,
            default_max_turns=default_max_turns,
        )

    def _scoped(self, func, *args, **kwargs):
        token = self._set_home(self.profile_home)
        try:
            return func(*args, **kwargs)
        finally:
            self._reset_home(token)

    @property
    def state(self):
        return self._manager.state

    def __getattr__(self, name):
        value = getattr(self._manager, name)
        if not callable(value):
            return value

        def scoped_call(*args, **kwargs):
            return self._scoped(value, *args, **kwargs)

        return scoped_call

    def _restore_state(self, snapshot) -> None:
        from hermes_cli.goals import save_goal  # type: ignore

        self._manager._state = snapshot
        self._scoped(save_goal, self.session_id, snapshot)


class _LegacyProfileGoalManager:
    """Explicit-DB fallback for Hermes versions without profile context."""

    def __init__(self, session_id: str, *, profile_home: str | Path, default_max_turns: int = 20):
        if GoalState is None:
            raise RuntimeError("Hermes goal state unavailable")
        self.session_id = session_id
        self.profile_home = Path(profile_home).expanduser().resolve()
        self.default_max_turns = int(default_max_turns or DEFAULT_MAX_TURNS or 20)
        self._state = self._load()

    @property
    def state(self):
        return self._state

    def _load(self):
        db = _profile_db(self.profile_home)
        if db is None or not self.session_id:
            return None
        try:
            raw = db.get_meta(_meta_key(self.session_id))
        except Exception as exc:
            logger.debug("GoalManager profile get_meta failed: %s", exc)
            return None
        if not raw:
            return None
        try:
            return GoalState.from_json(raw)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("GoalManager profile state parse failed for %s: %s", self.session_id, exc)
            return None

    def _save(self, state) -> None:
        db = _profile_db(self.profile_home)
        if db is None or not self.session_id or state is None:
            return
        try:
            db.set_meta(_meta_key(self.session_id), state.to_json())
        except Exception as exc:
            logger.debug("GoalManager profile set_meta failed: %s", exc)

    def is_active(self) -> bool:
        return self._state is not None and self._state.status == "active"

    def has_goal(self) -> bool:
        return self._state is not None and self._state.status in ("active", "paused")

    def status_line(self) -> str:
        s = self._state
        if s is None or s.status in ("cleared",):
            return "No active goal. Set one with /goal <text>."
        turns = f"{s.turns_used}/{s.max_turns} turns"
        if s.status == "active":
            return f"⊙ Goal (active, {turns}): {s.goal}"
        if s.status == "paused":
            extra = f" — {s.paused_reason}" if s.paused_reason else ""
            return f"⏸ Goal (paused, {turns}{extra}): {s.goal}"
        if s.status == "done":
            return f"✓ Goal done ({turns}): {s.goal}"
        return f"Goal ({s.status}, {turns}): {s.goal}"

    def set(self, goal: str, *, max_turns: Optional[int] = None):
        goal = (goal or "").strip()
        if not goal:
            raise ValueError("goal text is empty")
        state = GoalState(  # type: ignore[operator]
            goal=goal,
            status="active",
            turns_used=0,
            max_turns=int(max_turns) if max_turns else self.default_max_turns,
            created_at=time.time(),
            last_turn_at=0.0,
        )
        self._state = state
        self._save(state)
        return state

    def pause(self, reason: str = "user-paused"):
        if not self._state:
            return None
        self._state.status = "paused"
        self._state.paused_reason = reason
        self._save(self._state)
        return self._state

    def resume(self, *, reset_budget: bool = True):
        if not self._state:
            return None
        self._state.status = "active"
        self._state.paused_reason = None
        if reset_budget:
            self._state.turns_used = 0
        self._save(self._state)
        return self._state

    def clear(self) -> None:
        if self._state is None:
            return
        self._state.status = "cleared"
        self._save(self._state)
        self._state = None

    def evaluate_after_turn(self, last_response: str, *, user_initiated: bool = True) -> Dict[str, Any]:
        state = self._state
        if state is None or state.status != "active":
            return {
                "status": state.status if state else None,
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "inactive",
                "reason": "no active goal",
                "message": "",
            }

        state.turns_used += 1
        state.last_turn_at = time.time()

        if judge_goal is None:
            verdict, reason = "continue", "goal judge unavailable"
        else:
            verdict, reason, *_ = judge_goal(state.goal, str(last_response or ""))
        state.last_verdict = verdict
        state.last_reason = reason

        if verdict == "done":
            state.status = "done"
            self._save(state)
            return {
                "status": "done",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "done",
                "reason": reason,
                "message": f"✓ Goal achieved: {reason}",
            }

        if state.turns_used >= state.max_turns:
            state.status = "paused"
            state.paused_reason = f"turn budget exhausted ({state.turns_used}/{state.max_turns})"
            self._save(state)
            return {
                "status": "paused",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "continue",
                "reason": reason,
                "message": (
                    f"⏸ Goal paused — {state.turns_used}/{state.max_turns} turns used. "
                    "Use /goal resume to keep going, or /goal clear to stop."
                ),
            }

        self._save(state)
        return {
            "status": "active",
            "should_continue": True,
            "continuation_prompt": self.next_continuation_prompt(),
            "verdict": "continue",
            "reason": reason,
            "message": f"↻ Continuing toward goal ({state.turns_used}/{state.max_turns}): {reason}",
        }

    def next_continuation_prompt(self) -> Optional[str]:
        if not self._state or self._state.status != "active":
            return None
        return CONTINUATION_PROMPT_TEMPLATE.format(goal=self._state.goal)


def _manager(session_id: str, *, profile_home: str | Path | None = None):
    if GoalManager is None:
        return None
    if profile_home and GoalManager is _NativeGoalManager and GoalState is not None:
        try:
            context_api = _native_profile_context_api(profile_home)
            if context_api is not None:
                return _ProfileGoalManager(
                    session_id=session_id,
                    profile_home=profile_home,
                    default_max_turns=_default_max_turns(),
                    context_api=context_api,
                )
            return _LegacyProfileGoalManager(
                session_id=session_id,
                profile_home=profile_home,
                default_max_turns=_default_max_turns(),
            )
        except Exception as exc:
            logger.debug("Profile-scoped GoalManager unavailable: %s", exc)
            return None
    return GoalManager(session_id=session_id, default_max_turns=_default_max_turns())


def _state_payload(state: Any) -> Optional[Dict[str, Any]]:
    if state is None:
        return None
    return {
        "goal": getattr(state, "goal", "") or "",
        "status": getattr(state, "status", "") or "",
        "turns_used": int(getattr(state, "turns_used", 0) or 0),
        "max_turns": int(getattr(state, "max_turns", 0) or 0),
        "last_verdict": getattr(state, "last_verdict", None),
        "last_reason": getattr(state, "last_reason", None),
        "paused_reason": getattr(state, "paused_reason", None),
    }


def _payload(
    *,
    ok: bool = True,
    action: str,
    message: str,
    state: Any = None,
    error: str | None = None,
    kickoff_prompt: str | None = None,
    decision: Dict[str, Any] | None = None,
    message_key: str | None = None,
    message_args: list[Any] | None = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "ok": bool(ok),
        "action": action,
        "message": message,
        "goal": _state_payload(state),
    }
    if error:
        body["error"] = error
    if kickoff_prompt:
        body["kickoff_prompt"] = kickoff_prompt
    if decision is not None:
        body["decision"] = decision
    if message_key:
        body["message_key"] = message_key
    if message_args is not None:
        body["message_args"] = [a for a in message_args if a is not None]
    return body


def _goal_status_payload(state: Any, *, default_message: str | None = None) -> Dict[str, Any]:
    """Build localized-status style payload fields from a goal state."""
    if default_message is None:
        default_message = "No active goal. Set one with /goal <text>."
    if state is None:
        return {"message": default_message, "message_key": "goal_status_none"}
    status = str(getattr(state, "status", "") or "").strip()
    if status in ("cleared",):
        return {"message": default_message, "message_key": "goal_status_none"}
    turns_used = int(getattr(state, "turns_used", 0) or 0)
    max_turns = int(getattr(state, "max_turns", 0) or 0)
    goal = str(getattr(state, "goal", "") or "")
    if status == "active":
        return {
            "message": f"⊙ Goal (active, {turns_used}/{max_turns} turns): {goal}",
            "message_key": "goal_status_active",
            "message_args": [turns_used, max_turns, goal],
        }
    if status == "paused":
        reason = str(getattr(state, "paused_reason", "") or "")
        return {
            "message": f"⏸ Goal (paused, {turns_used}/{max_turns}{' — ' + reason if reason else ''}): {goal}",
            "message_key": "goal_status_paused",
            "message_args": [turns_used, max_turns, reason, goal],
        }
    if status == "done":
        return {
            "message": f"✓ Goal done ({turns_used}/{max_turns}): {goal}",
            "message_key": "goal_status_done",
            "message_args": [turns_used, max_turns, goal],
        }
    return {
        "message": f"Goal ({status}, {turns_used}/{max_turns}): {goal}",
        "message_args": [status, turns_used, max_turns, goal],
    }


def _extract_goal_turns_from_message(message: str) -> tuple[int, int]:
    """Best-effort extraction for continuation messages like '(1/20)'."""
    if not message:
        return 0, 0
    match = re.search(r"\((\d+)\s*/\s*(\d+)\)", message)
    if not match:
        return 0, 0
    try:
        return int(match.group(1)), int(match.group(2))
    except Exception:
        return 0, 0


def _goal_decision_payload(
    decision: Dict[str, Any],
    state: Any,
) -> Dict[str, Any]:
    """Attach goal message i18n key/args to an evaluation decision."""
    if not isinstance(decision, dict):
        return decision
    status = str(decision.get("status") or "").strip()
    reason = str(decision.get("reason") or "").strip()
    turns_used = int(getattr(state, "turns_used", 0) or 0)
    max_turns = int(getattr(state, "max_turns", 0) or 0)
    if (turns_used, max_turns) == (0, 0):
        turns_used, max_turns = _extract_goal_turns_from_message(str(decision.get("message") or ""))

    if status == "done":
        return {
            **decision,
            "message_key": "goal_achieved",
            "message_args": [reason],
        }
    if status == "paused":
        return {
            **decision,
            "message_key": "goal_paused_budget_exhausted",
            "message_args": [turns_used, max_turns],
        }
    if decision.get("should_continue"):
        return {
            **decision,
            "message_key": "goal_continuing",
            "message_args": [turns_used, max_turns, reason],
        }
    return decision


def goal_state_snapshot(session_id: str, *, profile_home: str | Path | None = None) -> Any:
    """Return a deep copy of current goal state for rollback before kickoff."""
    mgr = _manager(str(session_id or ""), profile_home=profile_home)
    if mgr is None:
        return None
    return copy.deepcopy(getattr(mgr, "state", None))


def restore_goal_state(session_id: str, snapshot: Any, *, profile_home: str | Path | None = None) -> None:
    """Restore a prior goal state after kickoff stream creation fails."""
    mgr = _manager(str(session_id or ""), profile_home=profile_home)
    if mgr is None:
        return
    if snapshot is None:
        try:
            mgr.clear()
        except Exception:
            pass
        return
    if isinstance(mgr, _ProfileGoalManager):
        mgr._restore_state(snapshot)
        return
    if isinstance(mgr, _LegacyProfileGoalManager):
        mgr._state = snapshot
        mgr._save(snapshot)
        return
    try:
        from hermes_cli.goals import save_goal  # type: ignore

        save_goal(str(session_id or ""), snapshot)
    except Exception as exc:  # pragma: no cover - native fallback only
        logger.debug("Goal state restore failed for %s: %s", session_id, exc)


def goal_command_payload(
    session_id: str,
    args: str = "",
    *,
    stream_running: bool = False,
    profile_home: str | Path | None = None,
) -> Dict[str, Any]:
    """Return the WebUI response payload for a /goal command.

    Mirrors the gateway command semantics:
    - /goal or /goal status shows status
    - /goal pause pauses
    - /goal resume resumes without auto-starting a turn
    - /goal clear|stop|done clears
    - /goal <text> sets a new active goal and returns kickoff_prompt so the
      caller can start the first normal user-role turn immediately.
    """
    sid = str(session_id or "").strip()
    if not sid:
        return _payload(ok=False, action="error", error="missing_session", message="session_id required")

    mgr = _manager(sid, profile_home=profile_home)
    if mgr is None:
        return _payload(ok=False, action="error", error="unavailable", message="Goals unavailable on this session.")

    text = str(args or "").strip()
    lower = text.lower()

    if not text or lower == "status":
        state = getattr(mgr, "state", None)
        status_payload = _goal_status_payload(state)
        return _payload(action="status", state=state, **status_payload)

    if lower == "pause":
        state = mgr.pause(reason="user-paused")
        if state is None:
            return _payload(
                ok=False,
                action="pause",
                error="no_goal",
                message="No goal set.",
                message_key="goal_no_goal",
            )
        return _payload(
            action="pause",
            message=f"⏸ Goal paused: {state.goal}",
            message_key="goal_paused",
            message_args=[str(state.goal)],
            state=state,
        )

    if lower == "resume":
        state = mgr.resume()
        if state is None:
            return _payload(
                ok=False,
                action="resume",
                error="no_goal",
                message="No goal to resume.",
                message_key="goal_no_goal",
            )
        return _payload(
            action="resume",
            message=(
                f"▶ Goal resumed: {state.goal}\n"
                "Send a new message, or type continue, to kick it off."
            ),
            message_key="goal_resumed",
            message_args=[str(state.goal)],
            state=state,
        )

    if lower in ("clear", "stop", "done"):
        had = bool(mgr.has_goal())
        mgr.clear()
        return _payload(
            action="clear",
            message="Goal cleared." if had else "No active goal.",
            message_key="goal_cleared" if had else "goal_no_goal",
            state=getattr(mgr, "state", None),
        )

    if stream_running:
        return _payload(
            ok=False,
            action="set",
            error="agent_running",
            message=(
                "Agent is running — use /goal status / pause / clear mid-run, "
                "or /stop before setting a new goal."
            ),
        )

    try:
        state = mgr.set(text)
    except ValueError as exc:
        return _payload(ok=False, action="set", error="invalid_goal", message=f"Invalid goal: {exc}")

    return _payload(
        action="set",
        message=(
            f"⊙ Goal set ({state.max_turns}-turn budget): {state.goal}\n"
            "I'll keep working until the goal is done, you pause/clear it, or the budget is exhausted.\n"
            "Controls: /goal status · /goal pause · /goal resume · /goal clear"
        ),
        message_key="goal_set",
        message_args=[state.max_turns, state.goal],
        state=state,
        kickoff_prompt=state.goal,
    )


def has_active_goal(
    session_id: str,
    *,
    profile_home: str | Path | None = None,
) -> bool:
    """Return True when the session has an active standing goal to evaluate."""
    sid = str(session_id or "").strip()
    if not sid:
        return False
    mgr = _manager(sid, profile_home=profile_home)
    if mgr is None:
        return False
    try:
        return bool(mgr.is_active())
    except Exception as exc:
        logger.debug("goal active-state check failed for session=%s: %s", sid, exc)
        return False


def _gather_goal_turn_background_state(
    session_id: str,
    *,
    profile_home: str | Path | None = None,
) -> tuple[Any, int]:
    """Return ``(background_processes, active_delegations)`` for *session_id*."""
    sid = str(session_id or "").strip()
    if not sid:
        return None, 0
    try:
        from hermes_cli.goals import (  # type: ignore
            count_active_delegations,
            gather_background_processes,
        )
    except Exception:
        return None, 0
    context_api = _native_profile_context_api(profile_home) if profile_home else None
    token = None
    if context_api is not None:
        try:
            token = context_api[0](Path(profile_home).expanduser().resolve())
        except Exception:
            token = None
    try:
        bg_procs = None
        active_deleg = 0
        try:
            bg_procs = gather_background_processes(owner_task_id=sid)
        except Exception:
            bg_procs = None
        try:
            active_deleg = int(count_active_delegations(sid) or 0)
        except Exception:
            active_deleg = 0
        return bg_procs, active_deleg
    finally:
        if token is not None and context_api is not None:
            try:
                context_api[1](token)
            except Exception:
                pass


def gateway_evaluated_goal_in_turn(
    session_id: str,
    *,
    min_evaluated_at: float | None,
    profile_home: str | Path | None = None,
) -> bool:
    """Return True when the Gateway already evaluated *session_id*'s goal in this turn."""
    if min_evaluated_at is None or GoalManager is None or GoalManager is not _NativeGoalManager:
        return False
    sid = str(session_id or "").strip()
    if not sid:
        return False
    if profile_home is not None:
        try:
            db_file = Path(profile_home).expanduser().resolve() / "state.db"
            if not db_file.is_file():
                return False
        except Exception:
            return False
    mgr = _manager(sid, profile_home=profile_home)
    if mgr is None:
        return False
    state = getattr(mgr, "state", None)
    if state is None:
        return False
    try:
        last_turn_at = float(getattr(state, "last_turn_at", 0.0) or 0.0)
        return last_turn_at > 0.0 and last_turn_at >= float(min_evaluated_at)
    except Exception:
        return False


def _decision_from_evaluated_state(mgr: Any, state: Any) -> Dict[str, Any]:
    """Reconstruct an evaluation decision dict from a GoalState already updated in-turn."""
    status = str(getattr(state, "status", "") or "").strip()
    verdict = str(getattr(state, "last_verdict", "") or "continue").strip() or "continue"
    reason = str(getattr(state, "last_reason", "") or "").strip()
    paused_reason = str(getattr(state, "paused_reason", "") or "").strip()
    turns_used = int(getattr(state, "turns_used", 0) or 0)
    max_turns = int(getattr(state, "max_turns", 0) or 0)

    if status == "done":
        return {
            "status": "done",
            "should_continue": False,
            "continuation_prompt": None,
            "verdict": "done",
            "reason": reason,
            "message": f"✓ Goal achieved: {reason}" if reason else "✓ Goal achieved",
        }

    if status == "paused":
        if verdict == "blocked" or paused_reason.startswith("judged unachievable"):
            msg = (
                f"🚫 Goal judged unachievable — paused: {reason or paused_reason} "
                "Re-scope with /goal set, or override with /goal resume."
            )
        elif paused_reason.startswith("turn budget exhausted"):
            msg = (
                f"⏸ Goal paused — {turns_used}/{max_turns} turns used. "
                "Use /goal resume to keep going, or /goal clear to stop."
            )
        else:
            detail = paused_reason or reason
            msg = f"⏸ Goal paused — {detail}" if detail else "⏸ Goal paused."
        return {
            "status": "paused",
            "should_continue": False,
            "continuation_prompt": None,
            "verdict": verdict,
            "reason": reason or paused_reason,
            "message": msg,
        }

    is_waiting = verdict in ("wait", "waiting")
    if not is_waiting and hasattr(mgr, "is_waiting"):
        try:
            is_waiting = bool(mgr.is_waiting())
        except Exception:
            is_waiting = False
    if is_waiting:
        if hasattr(mgr, "_waiting_decision"):
            try:
                wait_dec = dict(mgr._waiting_decision(state) or {})
                wait_dec["verdict"] = verdict if verdict in ("wait", "waiting") else "waiting"
                return wait_dec
            except Exception:
                pass
        return {
            "status": "active",
            "should_continue": False,
            "continuation_prompt": None,
            "verdict": verdict if verdict in ("wait", "waiting") else "waiting",
            "reason": reason,
            "message": f"⏳ Goal parked — waiting: {reason}" if reason else "⏳ Goal parked",
        }

    prompt = None
    if hasattr(mgr, "next_continuation_prompt"):
        try:
            prompt = mgr.next_continuation_prompt()
        except Exception:
            prompt = None
    if not prompt and getattr(state, "goal", ""):
        prompt = CONTINUATION_PROMPT_TEMPLATE.format(goal=state.goal)
    if verdict == "gate_failed":
        msg = f"✗ Quality gate failed ({turns_used}/{max_turns} turns): {reason}"
    else:
        msg = f"↻ Continuing toward goal ({turns_used}/{max_turns}): {reason}"
    return {
        "status": "active",
        "should_continue": bool(prompt),
        "continuation_prompt": prompt,
        "verdict": verdict,
        "reason": reason,
        "message": msg,
    }


def stream_channel_has_subscribers(channel: Any) -> bool:
    """Return True when at least one live SSE subscriber is attached to *channel*."""
    if channel is None:
        return False
    snap_fn = getattr(channel, "diagnostic_snapshot", None)
    if not callable(snap_fn):
        return True
    try:
        snap = snap_fn() or {}
        return int(snap.get("subscriber_count", 0) or 0) > 0
    except Exception:
        return True


def detached_goal_continuation_skip_reason(session_id: str) -> str:
    """Return why a detached goal continuation must NOT start, or '' if safe.

    Invariants (re-checked at fire time, not only at teardown):
      1. never after a user cancel / pause / clear (pending marker gone, goal
         inactive, or a cancel marker persisted on the transcript);
      2. never races a queued user message or an active/pending turn;
      3. stays inside the owning profile (goal state resolved from the
         session's own profile home; isolated mode refuses foreign sessions).
    """
    sid = str(session_id or "").strip()
    if not sid:
        return "missing_session"
    from api.config import PENDING_GOAL_CONTINUATION

    if sid not in PENDING_GOAL_CONTINUATION:
        return "not_pending"
    try:
        from api.models import get_session

        s = get_session(sid)
    except Exception:
        return "session_unavailable"
    owner = getattr(s, "profile", None)
    try:
        from api.profiles import (
            _is_isolated_profile_mode,
            _profiles_match,
            get_active_profile_name,
            get_hermes_home_for_profile,
        )

        if _is_isolated_profile_mode() and not _profiles_match(
            owner, get_active_profile_name()
        ):
            return "foreign_profile"
        profile_home = get_hermes_home_for_profile(owner or "default")
    except Exception:
        return "profile_unresolved"
    try:
        from api.streaming import _session_has_cancel_marker

        if _session_has_cancel_marker(s):
            return "cancelled"
    except Exception:
        return "cancel_state_unknown"
    if getattr(s, "pending_user_message", None):
        return "turn_pending"
    try:
        from api.background_process import _session_has_active_turn

        if _session_has_active_turn(sid):
            return "turn_active"
    except Exception:
        return "turn_state_unknown"
    if not has_active_goal(sid, profile_home=profile_home):
        return "goal_inactive"
    return ""


def start_detached_goal_continuation(
    session_id: str,
    continuation_prompt: str,
) -> None:
    """Start a goal continuation turn server-side when no browser SSE subscriber is attached."""
    sid = str(session_id or "").strip()
    prompt = str(continuation_prompt or "").strip()
    if not sid or not prompt:
        return

    def _runner() -> None:
        try:
            from api.routes import start_session_turn

            reason = detached_goal_continuation_skip_reason(sid)
            if reason:
                logger.debug(
                    "Detached goal continuation skipped for session %s: %s",
                    sid,
                    reason,
                )
                return
            start_session_turn(sid, prompt, source="goal_wakeup")
        except Exception:
            logger.debug(
                "Detached goal continuation start failed for session %s",
                sid,
                exc_info=True,
            )

    threading.Thread(
        target=_runner,
        name=f"goal-continue-{sid[:12]}",
        daemon=True,
    ).start()


def evaluate_goal_after_turn(
    session_id: str,
    last_response: str,
    *,
    user_initiated: bool = True,
    profile_home: str | Path | None = None,
    min_evaluated_at: float | None = None,
) -> Dict[str, Any]:
    """Evaluate a completed turn against the standing goal, if any."""
    sid = str(session_id or "").strip()
    if not sid:
        return {
            "status": None,
            "should_continue": False,
            "continuation_prompt": None,
            "verdict": "inactive",
            "reason": "missing session_id",
            "message": "",
        }
    mgr = _manager(sid, profile_home=profile_home)
    if mgr is None:
        return {
            "status": None,
            "should_continue": False,
            "continuation_prompt": None,
            "verdict": "inactive",
            "reason": "goals unavailable",
            "message": "",
        }
    try:
        state = getattr(mgr, "state", None)
        if (
            min_evaluated_at is not None
            and state is not None
            and float(getattr(state, "last_turn_at", 0.0) or 0.0) >= float(min_evaluated_at)
        ):
            decision = _decision_from_evaluated_state(mgr, state)
            return _goal_decision_payload(decision, state)
        if not mgr.is_active():
            return {
                "status": getattr(state, "status", None),
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "inactive",
                "reason": "no active goal",
                "message": "",
            }
        eval_kwargs: Dict[str, Any] = {"user_initiated": user_initiated}
        target_eval = getattr(
            getattr(mgr, "_manager", mgr),
            "evaluate_after_turn",
            mgr.evaluate_after_turn,
        )
        try:
            params = inspect.signature(target_eval).parameters
            accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            if (
                accepts_var_kw
                or "background_processes" in params
                or "active_delegations" in params
            ):
                bg_procs, active_deleg = _gather_goal_turn_background_state(
                    sid, profile_home=profile_home
                )
                if accepts_var_kw or "background_processes" in params:
                    eval_kwargs["background_processes"] = bg_procs
                if accepts_var_kw or "active_delegations" in params:
                    eval_kwargs["active_delegations"] = active_deleg
        except Exception:
            pass
        decision = mgr.evaluate_after_turn(str(last_response or ""), **eval_kwargs)
    except Exception as exc:
        logger.debug("goal evaluation failed for session=%s: %s", sid, exc)
        return {
            "status": None,
            "should_continue": False,
            "continuation_prompt": None,
            "verdict": "error",
            "reason": f"goal evaluation failed: {type(exc).__name__}",
            "message": "",
        }
    if not isinstance(decision, dict):
        decision = {}
    decision.setdefault("should_continue", False)
    decision.setdefault("continuation_prompt", None)
    decision.setdefault("message", "")
    decision = dict(decision)
    decision = _goal_decision_payload(decision, getattr(mgr, "state", None))
    return decision
