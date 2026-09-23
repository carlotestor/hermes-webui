"""WebUI /loop parity: hermes_cli.loops.LoopManager state + server-side tick scheduler."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.conftest import requires_agent_modules

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMANDS_JS = (REPO_ROOT / "static" / "commands.js").read_text(encoding="utf-8")
MESSAGES_JS = (REPO_ROOT / "static" / "messages.js").read_text(encoding="utf-8")
ROUTES_PY = (REPO_ROOT / "api" / "routes.py").read_text(encoding="utf-8")
STREAMING_PY = (REPO_ROOT / "api" / "streaming.py").read_text(encoding="utf-8")
GATEWAY_CHAT_PY = (REPO_ROOT / "api" / "gateway_chat.py").read_text(encoding="utf-8")
SERVER_PY = (REPO_ROOT / "server.py").read_text(encoding="utf-8")

pytestmark = requires_agent_modules


@pytest.fixture
def loop_env(tmp_path, monkeypatch):
    """A scratch profile home, an isolated goals DB cache, and stubbed session/turn hooks."""
    pytest.importorskip("hermes_cli.loops")
    from hermes_cli import goals as agent_goals

    from api import loops as webui_loops

    if not webui_loops.loops_available():
        pytest.skip("installed hermes-agent lacks hermes_cli.loops / home override")

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(agent_goals, "_DB_CACHE", {})
    monkeypatch.setattr(webui_loops, "_profile_homes", lambda: [(None, home)])
    monkeypatch.setattr(webui_loops, "_webui_session_exists", lambda sid: True)
    monkeypatch.setattr(webui_loops, "_session_has_active_turn", lambda sid: False)
    monkeypatch.setattr(webui_loops, "wake_scheduler", lambda: None)
    started = []

    def fake_start(sid, message):
        started.append((sid, message))
        return {"_status": 200, "stream_id": f"s{len(started)}"}

    monkeypatch.setattr(webui_loops, "_start_turn", fake_start)
    return webui_loops, home, started


def _load(webui_loops, home, sid):
    with webui_loops._profile_scope(home):
        return webui_loops._agent_loops.load_loop(sid)


def _make_due(webui_loops, home, sid):
    """Move the persisted next_due_at into the past (fire_tick checks the wall clock)."""
    with webui_loops._profile_scope(home):
        state = webui_loops._agent_loops.load_loop(sid)
        state.next_due_at = time.time() - 1
        webui_loops._agent_loops.save_loop(sid, state)


def test_set_persists_loop_in_agent_store_with_webui_route(loop_env):
    webui_loops, home, _started = loop_env

    payload = webui_loops.loop_command_payload("sid1", "5m check the deploy", profile_home=home, profile="p1")

    assert payload["ok"] and payload["created"]
    assert "Loop set (every 5m): check the deploy" in payload["message"]
    assert "Stops when the task is done, or after 100 runs (loops.max_ticks)." in payload["message"]
    state = _load(webui_loops, home, "sid1")
    assert state is not None and state.status == "active"
    assert state.interval_seconds == 300
    assert state.times == 100 and state.until == ""
    assert state.route == {"platform": "webui", "chat_id": "sid1", "profile": "p1"}
    assert (home / "state.db").exists()


@pytest.mark.parametrize("args", ["5m poll CI --times 3", "5m watch the queue --until the queue is empty"])
def test_times_and_until_flags_rejected(loop_env, args):
    webui_loops, home, started = loop_env

    payload = webui_loops.loop_command_payload("sid1", args, profile_home=home)

    assert not payload["ok"] and payload["error"] == "unsupported_flag"
    assert "--times and --until are not supported" in payload["message"]
    assert _load(webui_loops, home, "sid1") is None
    assert started == []


def test_help_documents_default_limit_without_flags(loop_env):
    webui_loops, home, _started = loop_env

    message = webui_loops.loop_command_payload("sid1", "help", profile_home=home)["message"]

    assert "Usage: /loop [interval] <prompt>" in message
    assert "LOOP_COMPLETE" in message and "default 100" in message
    assert "--times" not in message and "--until" not in message


def test_controls_round_trip_through_loop_manager(loop_env):
    webui_loops, home, _started = loop_env
    webui_loops.loop_command_payload("sid1", "5m poll CI", profile_home=home)

    assert "Loop (active, every 5m, 0/100 runs" in webui_loops.loop_command_payload("sid1", "status", profile_home=home)["message"]
    assert webui_loops.loop_command_payload("sid1", "pause", profile_home=home)["loop"]["status"] == "paused"
    assert webui_loops.loop_command_payload("sid1", "resume", profile_home=home)["loop"]["status"] == "active"
    stopped = webui_loops.loop_command_payload("sid1", "stop", profile_home=home)
    assert stopped["message"] == "✓ Loop stopped."
    assert stopped["loop"] is None
    assert _load(webui_loops, home, "sid1").status == "cleared"


def test_slash_command_prompt_rejected(loop_env):
    webui_loops, home, _started = loop_env

    payload = webui_loops.loop_command_payload("sid1", "10m /recap", profile_home=home)

    assert not payload["ok"] and payload["error"] == "slash_prompt_unsupported"
    assert _load(webui_loops, home, "sid1") is None


def test_scheduler_fires_first_tick_immediately_then_waits_for_interval(loop_env):
    webui_loops, home, started = loop_env
    webui_loops.loop_command_payload("sid1", "5m check the deploy", profile_home=home)

    assert webui_loops.fire_due_loops() == {"sid1": "fired"}
    assert len(started) == 1
    sid, message = started[0]
    assert sid == "sid1"
    assert message.startswith("[/loop wakeup #1, every 5m]")
    assert "Recurring task: check the deploy" in message

    # Tick in flight: a second pass must not double-fire.
    assert webui_loops.fire_due_loops() == {"sid1": "in_flight"}

    decision = webui_loops.evaluate_loop_after_turn("sid1", "deploy still rolling", profile_home=home)
    assert decision["status"] == "active" and decision["stopped"] is False
    assert webui_loops.fire_due_loops() == {"sid1": "not_due"}
    next_due = _load(webui_loops, home, "sid1").next_due_at
    assert 299 <= next_due - time.time() <= 301
    _make_due(webui_loops, home, "sid1")
    assert webui_loops.fire_due_loops() == {"sid1": "fired"}
    assert started[-1][1].startswith("[/loop wakeup #2, every 5m]")


def test_loop_complete_marker_finishes_loop(loop_env):
    webui_loops, home, _started = loop_env
    webui_loops.loop_command_payload("sid1", "5m watch the job", profile_home=home)
    webui_loops.fire_due_loops()

    decision = webui_loops.evaluate_loop_after_turn("sid1", "job exited 0\nLOOP_COMPLETE", profile_home=home)

    assert decision["status"] == "done"
    assert decision["message"] == "✓ Loop finished after 1 tick — task complete."
    assert _load(webui_loops, home, "sid1").status == "done"
    _make_due(webui_loops, home, "sid1")
    assert webui_loops.fire_due_loops() == {}


def test_default_run_limit_finishes_loop(loop_env, monkeypatch):
    """With no LOOP_COMPLETE, the loop ends for good (done, not paused) after the default limit."""
    webui_loops, home, started = loop_env
    monkeypatch.setattr(webui_loops._agent_loops, "max_ticks_default", lambda: 2)
    webui_loops.loop_command_payload("sid1", "30s ping", profile_home=home)

    assert webui_loops.fire_due_loops() == {"sid1": "fired"}
    assert webui_loops.evaluate_loop_after_turn("sid1", "pong", profile_home=home)["status"] == "active"
    _make_due(webui_loops, home, "sid1")
    assert webui_loops.fire_due_loops() == {"sid1": "fired"}
    decision = webui_loops.evaluate_loop_after_turn("sid1", "pong", profile_home=home)

    assert decision["status"] == "done" and decision["stopped"] is True
    assert "ran 2/2 times" in decision["message"]
    assert _load(webui_loops, home, "sid1").status == "done"
    _make_due(webui_loops, home, "sid1")
    assert webui_loops.fire_due_loops() == {}
    assert len(started) == 2


def test_unlimited_config_has_no_run_limit(loop_env, monkeypatch):
    webui_loops, home, _started = loop_env
    monkeypatch.setattr(webui_loops._agent_loops, "max_ticks_default", lambda: 0)

    payload = webui_loops.loop_command_payload("sid1", "5m ping", profile_home=home)

    assert "Stops when the task is done (no run limit)." in payload["message"]
    state = _load(webui_loops, home, "sid1")
    assert state.times == 0 and state.max_ticks == 0


def test_busy_session_and_failed_start_do_not_consume_tick(loop_env, monkeypatch):
    webui_loops, home, started = loop_env
    webui_loops.loop_command_payload("sid1", "5m check", profile_home=home)

    monkeypatch.setattr(webui_loops, "_session_has_active_turn", lambda sid: True)
    assert webui_loops.fire_due_loops() == {"sid1": "busy"}
    assert _load(webui_loops, home, "sid1").ticks_fired == 0

    monkeypatch.setattr(webui_loops, "_session_has_active_turn", lambda sid: False)
    monkeypatch.setattr(webui_loops, "_start_turn", lambda sid, msg: {"_status": 409, "error": "active stream"})
    assert webui_loops.fire_due_loops() == {"sid1": "start_failed"}
    state = _load(webui_loops, home, "sid1")
    assert state.ticks_fired == 0 and state.awaiting_response is False
    assert started == []


def test_stale_in_flight_tick_is_recovered(loop_env):
    """A wakeup turn that never reached the post-turn hook must not wedge the loop."""
    webui_loops, home, _started = loop_env
    webui_loops.loop_command_payload("sid1", "5m check", profile_home=home)
    webui_loops.fire_due_loops()

    later = time.time() + webui_loops.STALE_TICK_GRACE_SECONDS + 1
    assert webui_loops.fire_due_loops(now=later) == {"sid1": "recovered_stale_tick"}
    state = _load(webui_loops, home, "sid1")
    assert state.awaiting_response is False and state.status == "active"


def test_deleted_session_clears_loop(loop_env, monkeypatch):
    webui_loops, home, started = loop_env
    webui_loops.loop_command_payload("sid1", "5m check", profile_home=home)
    monkeypatch.setattr(webui_loops, "_webui_session_exists", lambda sid: False)

    assert webui_loops.fire_due_loops() == {"sid1": "cleared_missing_session"}
    assert _load(webui_loops, home, "sid1").status == "cleared"
    assert started == []


def test_scheduler_ignores_loops_owned_by_other_surfaces(loop_env):
    """CLI/TUI loops (no route) and gateway chats (telegram route) belong to their own drivers."""
    webui_loops, home, started = loop_env
    with webui_loops._profile_scope(home):
        webui_loops._agent_loops.LoopManager(session_id="cli-sid").set("check", interval_seconds=60)
        webui_loops._agent_loops.LoopManager(session_id="tg-sid").set(
            "check", interval_seconds=60, route={"platform": "telegram", "chat_id": "42"})

    assert webui_loops.fire_due_loops() == {}
    assert started == []


def test_evaluate_without_tick_in_flight_is_noop(loop_env):
    webui_loops, home, _started = loop_env
    webui_loops.loop_command_payload("sid1", "5m check", profile_home=home)

    assert webui_loops.evaluate_loop_after_turn("sid1", "LOOP_COMPLETE", profile_home=home) == {}
    assert _load(webui_loops, home, "sid1").status == "active"


def test_last_assistant_text_handles_content_parts():
    from api.loops import last_assistant_text

    messages = [
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "done"}, {"type": "text", "text": "LOOP_COMPLETE"}]},
    ]
    assert last_assistant_text(messages) == "done\nLOOP_COMPLETE"


def test_wiring_routes_hooks_and_frontend():
    assert 'parsed.path == "/api/loop"' in ROUTES_PY
    assert "def _handle_loop_command(handler, body)" in ROUTES_PY
    assert "_turn_pending_source == 'loop_wakeup'" in STREAMING_PY
    assert "evaluate_loop_after_turn" in STREAMING_PY
    assert 'pending_source == "loop_wakeup"' in GATEWAY_CHAT_PY
    assert "start_loop_scheduler" in SERVER_PY
    assert "{name:'loop'" in COMMANDS_JS and "async function cmdLoop(args)" in COMMANDS_JS
    assert "api('/api/loop'" in COMMANDS_JS
    assert "source.addEventListener('loop'" in MESSAGES_JS
