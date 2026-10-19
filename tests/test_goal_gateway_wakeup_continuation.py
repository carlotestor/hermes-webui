"""Regression tests for /goal continuation across Gateway turns, subagent wakeups, and resume."""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import pytest

from api import gateway_chat, goals as webui_goals, models, routes
from api.config import PENDING_GOAL_CONTINUATION, STREAMS, create_stream_channel
from api.models import new_session


@pytest.mark.parametrize("wakeup_source", ["process_wakeup", "delegation_wakeup", "goal_wakeup"])
def test_start_session_turn_marks_wakeup_goal_related_when_goal_active(
    monkeypatch, tmp_path, wakeup_source
):
    """A subagent/process wakeup on a session with an active goal must run with goal_related=True."""
    session = SimpleNamespace(
        session_id="sid-wakeup-goal",
        profile="deepseek",
        workspace=str(tmp_path),
        model="gpt-5.5",
        model_provider="openai-codex",
        process_wakeup_pause=None,
        save=lambda **_kw: None,
    )
    PENDING_GOAL_CONTINUATION.discard(session.session_id)

    monkeypatch.setattr(routes, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_resolve_chat_workspace_with_recovery", lambda _s, _w: str(tmp_path))
    monkeypatch.setattr(routes, "_read_profile_model_config", lambda _s, _p: ("openai-codex", "gpt-5.5", {}))
    monkeypatch.setattr(
        routes,
        "_resolve_compatible_session_model_state",
        lambda model, provider, **_kw: (model, provider, False),
    )
    monkeypatch.setattr(routes, "clear_process_wakeup_pause_if_model_changed", lambda *_a, **_kw: False)
    monkeypatch.setattr(routes, "process_wakeup_pause_matches", lambda *_a, **_kw: False)
    monkeypatch.setattr(routes, "suppress_process_wakeup_for_provider_pause", lambda *_a, **_kw: None)
    monkeypatch.setattr(webui_goals, "has_active_goal", lambda sid, **_kw: sid == "sid-wakeup-goal")

    captured = {}

    def fake_start_run(s, **kwargs):
        captured.update(kwargs)
        return {"stream_id": "wakeup-stream", "session_id": s.session_id, "_status": 200}

    monkeypatch.setattr(routes, "_start_run", fake_start_run)

    resp = routes.start_session_turn(
        session.session_id,
        "[IMPORTANT: A delegated subagent finished.]",
        source=wakeup_source,
    )

    assert resp["_status"] == 200
    assert captured["source"] == wakeup_source
    assert captured["goal_related"] is True


def test_goal_resume_arms_pending_goal_continuation_and_pause_clear_discard(monkeypatch, tmp_path):
    """/goal resume arms PENDING_GOAL_CONTINUATION so the user's next kickoff turn is goal_related."""
    class FakeState:
        goal = "replace helpers with native code"
        status = "active"
        turns_used = 0
        max_turns = 100
        last_verdict = None
        last_reason = None
        paused_reason = None

    class FakeGoalManager:
        def __init__(self, session_id, default_max_turns=20):
            self.session_id = session_id
            self.state = FakeState()

        def resume(self, reset_budget=True):
            self.state.status = "active"
            return self.state

        def pause(self, reason="user-paused"):
            self.state.status = "paused"
            self.state.paused_reason = reason
            return self.state

        def has_goal(self):
            return True

        def clear(self):
            self.state = None

    session = SimpleNamespace(
        session_id="sid-goal-resume",
        profile="deepseek",
        workspace=str(tmp_path),
        model="gpt-5.5",
        model_provider="openai-codex",
        active_stream_id=None,
    )
    PENDING_GOAL_CONTINUATION.discard(session.session_id)
    monkeypatch.setattr(webui_goals, "GoalManager", FakeGoalManager)
    monkeypatch.setattr(routes, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "j", lambda _h, payload, status=200, **_kw: {"status": status, "payload": payload})

    try:
        res_resume = routes._handle_goal_command(
            object(), {"session_id": session.session_id, "args": "resume"}
        )
        assert res_resume["status"] == 200
        assert session.session_id in PENDING_GOAL_CONTINUATION

        res_pause = routes._handle_goal_command(
            object(), {"session_id": session.session_id, "args": "pause"}
        )
        assert res_pause["status"] == 200
        assert session.session_id not in PENDING_GOAL_CONTINUATION

        PENDING_GOAL_CONTINUATION.add(session.session_id)
        res_clear = routes._handle_goal_command(
            object(), {"session_id": session.session_id, "args": "clear"}
        )
        assert res_clear["status"] == 200
        assert session.session_id not in PENDING_GOAL_CONTINUATION
    finally:
        PENDING_GOAL_CONTINUATION.discard(session.session_id)


def test_evaluate_goal_after_turn_forwards_background_processes_and_active_delegations(
    monkeypatch, tmp_path
):
    """evaluate_goal_after_turn passes live background_processes and active_delegations to the manager."""
    captured = {}

    class FakeState:
        goal = "ship all waves"
        status = "active"
        turns_used = 1
        max_turns = 20
        last_turn_at = 0.0

    class FakeGoalManager:
        def __init__(self, session_id, default_max_turns=20):
            self.session_id = session_id
            self.state = FakeState()

        def is_active(self):
            return True

        def evaluate_after_turn(
            self,
            last_response,
            *,
            user_initiated=True,
            background_processes=None,
            active_delegations=0,
        ):
            captured["last_response"] = last_response
            captured["user_initiated"] = user_initiated
            captured["background_processes"] = background_processes
            captured["active_delegations"] = active_delegations
            return {
                "status": "active",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "wait",
                "reason": "waiting on 16 subagents",
                "message": "⏳ Goal parked (judge) — waiting on 300s: waiting on 16 subagents",
            }

    monkeypatch.setattr(webui_goals, "_manager", lambda _sid, **_kw: FakeGoalManager(_sid))
    monkeypatch.setattr(
        webui_goals,
        "_gather_goal_turn_background_state",
        lambda _sid, **_kw: ([{"session_id": "proc_1", "status": "running"}], 16),
    )

    decision = webui_goals.evaluate_goal_after_turn(
        "sid-deleg",
        "Spawned 16 subagents; ending turn.",
        user_initiated=True,
        profile_home=tmp_path,
    )

    assert captured == {
        "last_response": "Spawned 16 subagents; ending turn.",
        "user_initiated": True,
        "background_processes": [{"session_id": "proc_1", "status": "running"}],
        "active_delegations": 16,
    }
    assert decision["verdict"] == "wait"
    assert decision["should_continue"] is False


def test_gateway_worker_adopts_in_turn_gateway_goal_evaluation_without_double_judging(
    tmp_path, monkeypatch
):
    """When the Gateway already evaluated the goal in state.db during the turn, WebUI adopts it
    without calling evaluate_after_turn a second time (and even if goal_related was lost on restart).
    """
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", session_dir / "_index.json")
    monkeypatch.setattr(models, "SESSIONS", OrderedDict())

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"wave 3 merged"}}]}\n\n'
            yield b"data: [DONE]\n\n"

    monkeypatch.setenv("HERMES_WEBUI_GATEWAY_BASE_URL", "http://gateway.local")
    monkeypatch.setattr(gateway_chat.urllib.request, "urlopen", lambda req, timeout=0: FakeResponse())

    import time as _time

    eval_calls = []

    class FakeState:
        goal = "remove all replaceable helpers"
        status = "active"
        turns_used = 1
        max_turns = 100
        last_verdict = "continue"
        last_reason = "Go fell only 2.4%; another wave is needed."
        paused_reason = None
        last_turn_at = _time.time() + 60.0

    class FakeMgr:
        def __init__(self):
            self.state = FakeState()

        def is_active(self):
            return True

        def is_waiting(self):
            return False

        def next_continuation_prompt(self):
            return f"[Continuing toward your standing goal]\nGoal: {self.state.goal}"

        def evaluate_after_turn(self, *_a, **_kw):
            eval_calls.append(True)
            raise AssertionError("mgr.evaluate_after_turn must not be called a second time")

    fake_mgr = FakeMgr()
    (tmp_path / "state.db").write_bytes(b"sqlite")
    monkeypatch.setattr(webui_goals, "_NativeGoalManager", FakeMgr)
    monkeypatch.setattr(webui_goals, "GoalManager", FakeMgr)
    monkeypatch.setattr(webui_goals, "_manager", lambda _sid, **_kw: fake_mgr)
    monkeypatch.setattr("api.profiles.get_hermes_home_for_profile", lambda _p: tmp_path)

    s = new_session()
    stream_id = "stream-gateway-in-turn-goal"
    s.active_stream_id = stream_id
    s.pending_user_message = "continue - subagents have finished"
    s.pending_attachments = []
    s.pending_started_at = 123
    s.save()
    channel = create_stream_channel()
    subscriber = channel.subscribe()
    STREAMS[stream_id] = channel
    PENDING_GOAL_CONTINUATION.discard(s.session_id)

    try:
        gateway_chat._run_gateway_chat_streaming(
            s.session_id,
            "continue - subagents have finished",
            "test-model",
            str(tmp_path),
            stream_id,
            [],
            goal_related=False,
        )

        events = []
        while not subscriber.empty():
            events.append(subscriber.get_nowait())
        event_names = [item[0] for item in events]

        assert eval_calls == []
        assert fake_mgr.state.turns_used == 1
        assert "goal" in event_names
        assert "goal_continue" in event_names
        assert s.session_id in PENDING_GOAL_CONTINUATION
        goal_continue_ev = next(item[1] for item in events if item[0] == "goal_continue")
        assert "remove all replaceable helpers" in goal_continue_ev["continuation_prompt"]
        assert "1/100" in goal_continue_ev["message"]
    finally:
        PENDING_GOAL_CONTINUATION.discard(s.session_id)


def test_gateway_worker_starts_detached_goal_continuation_when_no_sse_subscriber(
    tmp_path, monkeypatch
):
    """When goal_continue fires on a stream with 0 SSE subscribers, teardown starts the next turn server-side."""
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", session_dir / "_index.json")
    monkeypatch.setattr(models, "SESSIONS", OrderedDict())

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"wakeup handled"}}]}\n\n'
            yield b"data: [DONE]\n\n"

    monkeypatch.setenv("HERMES_WEBUI_GATEWAY_BASE_URL", "http://gateway.local")
    monkeypatch.setattr(gateway_chat.urllib.request, "urlopen", lambda req, timeout=0: FakeResponse())
    monkeypatch.setattr(webui_goals, "has_active_goal", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        webui_goals,
        "evaluate_goal_after_turn",
        lambda *_a, **_kw: {
            "status": "active",
            "should_continue": True,
            "continuation_prompt": "[Continuing toward your standing goal]\nGoal: finish wave 4",
            "message": "↻ Continuing toward goal (2/100): spawn wave 4",
            "message_key": "goal_continuing",
            "message_args": [2, 100, "spawn wave 4"],
        },
    )

    import threading as _threading

    detached_done = _threading.Event()
    detached_calls = []

    def fake_start_session_turn(sid, prompt, *, source="process_wakeup", **_kw):
        detached_calls.append((sid, prompt, source))
        detached_done.set()
        return {"_status": 200, "stream_id": "next-goal-stream", "session_id": sid}

    monkeypatch.setattr(routes, "start_session_turn", fake_start_session_turn)

    s = new_session()
    stream_id = "stream-gateway-detached-goal"
    s.active_stream_id = stream_id
    s.pending_user_message = "[IMPORTANT: A delegated subagent finished.]"
    s.pending_user_source = "process_wakeup"
    s.pending_attachments = []
    s.pending_started_at = 123
    s.save()
    # Channel has NO subscriber attached (closed tab / background session wakeup).
    channel = create_stream_channel()
    STREAMS[stream_id] = channel
    PENDING_GOAL_CONTINUATION.discard(s.session_id)

    try:
        gateway_chat._run_gateway_chat_streaming(
            s.session_id,
            "[IMPORTANT: A delegated subagent finished.]",
            "test-model",
            str(tmp_path),
            stream_id,
            [],
            goal_related=True,
        )

        assert detached_done.wait(timeout=3.0), "detached goal continuation did not fire start_session_turn"
        assert detached_calls == [
            (s.session_id, "[Continuing toward your standing goal]\nGoal: finish wave 4", "goal_wakeup")
        ]
    finally:
        PENDING_GOAL_CONTINUATION.discard(s.session_id)
