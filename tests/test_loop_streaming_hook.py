"""/loop post-turn hook runs inside the real _run_agent_streaming worker."""

from __future__ import annotations

import queue
import sys
import types
from unittest import mock

import pytest

import api.config as config
import api.loops as webui_loops
import api.models as models
import api.streaming as streaming
from api.models import Session


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", session_dir / "_index.json")
    models.SESSIONS.clear()
    for registry in (config.STREAMS, config.CANCEL_FLAGS, config.AGENT_INSTANCES,
                     config.STREAM_PARTIAL_TEXT, config.STREAM_REASONING_TEXT,
                     config.STREAM_LIVE_TOOL_CALLS, config.STREAM_GOAL_RELATED,
                     config.SESSION_AGENT_LOCKS):
        registry.clear()
    fake_runtime = types.ModuleType("hermes_cli.runtime_provider")
    fake_runtime.resolve_runtime_provider = lambda requested=None, **_kw: {
        "provider": requested or "test-provider", "api_key": "synthetic-key", "base_url": None,
    }
    fake_cli = types.ModuleType("hermes_cli")
    fake_cli.runtime_provider = fake_runtime
    fake_state = types.ModuleType("hermes_state")
    fake_state.SessionDB = mock.Mock(return_value=None)
    monkeypatch.setitem(sys.modules, "hermes_cli", fake_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.runtime_provider", fake_runtime)
    monkeypatch.setitem(sys.modules, "hermes_state", fake_state)
    yield
    models.SESSIONS.clear()
    config.STREAMS.clear()
    config.SESSION_AGENT_LOCKS.clear()


class _Agent:
    def __init__(self, **kwargs):
        self.session_id = kwargs.get("session_id")
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.reasoning_callback = kwargs.get("reasoning_callback")
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.context_compressor = None
        self._last_error = None
        self.ephemeral_system_prompt = None

    def interrupt(self, _message):
        pass

    def run_conversation(self, **kwargs):
        history = list(kwargs.get("conversation_history") or [])
        return {"messages": history + [
            {"role": "user", "content": kwargs.get("persist_user_message", "")},
            {"role": "assistant", "content": "job exited 0\nLOOP_COMPLETE"},
        ]}


def _run_turn(tmp_path, monkeypatch, source):
    stream_id = f"stream-{source}"
    session = Session(
        session_id=f"sid-{source}", workspace=str(tmp_path), model="test-model",
        model_provider="test-provider", messages=[], context_messages=[],
        active_stream_id=stream_id, pending_user_message="[/loop wakeup #1, every 5m]\nRecurring task: x",
        pending_user_source=source,
    )
    session.save()
    models.SESSIONS[session.session_id] = session
    q = queue.Queue()
    config.STREAMS[stream_id] = q
    config.STREAM_PARTIAL_TEXT[stream_id] = ""
    calls = []

    def fake_evaluate(session_id, last_response, *, profile_home=None):
        calls.append((session_id, last_response))
        return {"status": "done", "stopped": True, "message": "✓ Loop finished after 1 tick — task complete."}

    monkeypatch.setattr(webui_loops, "evaluate_loop_after_turn", fake_evaluate)
    with mock.patch.object(streaming, "_get_ai_agent", return_value=_Agent), \
         mock.patch.object(streaming, "resolve_model_provider", return_value=("test-model", "test-provider", None)), \
         mock.patch("api.config._resolve_cli_toolsets", return_value=[]):
        streaming._run_agent_streaming(
            session_id=session.session_id, msg_text=session.pending_user_message,
            model="test-model", model_provider="test-provider",
            workspace=str(tmp_path), stream_id=stream_id,
        )
    events = []
    while not q.empty():
        item = q.get_nowait()
        events.append((item[0], item[1]))
    return session.session_id, calls, events


def test_loop_wakeup_turn_completes_tick_and_emits_loop_event(tmp_path, monkeypatch):
    sid, calls, events = _run_turn(tmp_path, monkeypatch, "loop_wakeup")

    assert calls == [(sid, "job exited 0\nLOOP_COMPLETE")]
    loop_events = [data for name, data in events if name == "loop"]
    assert loop_events == [{
        "session_id": sid, "message": "✓ Loop finished after 1 tick — task complete.",
        "status": "done", "loop": None,
    }]
    names = [name for name, _ in events]
    assert names.index("loop") < names.index("done")


def test_ordinary_turn_never_touches_loop_state(tmp_path, monkeypatch):
    _sid, calls, events = _run_turn(tmp_path, monkeypatch, "webui")

    assert calls == []
    assert not any(name == "loop" for name, _ in events)
