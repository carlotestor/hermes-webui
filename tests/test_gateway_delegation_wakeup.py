"""Gateway-backend async-delegation completions must wake the WebUI session.

The Gateway persists them as ``async_delegation_complete`` delivery rows in the
profile state.db and starts no turn; the WebUI poller claims and wakes.
"""
import json
import time

import pytest

hermes_state = pytest.importorskip("hermes_state")

import api.background_process as bp
import api.gateway_delegation_wakeup as gdw


def _deliver(db, sid, deleg_id):
    db.append_delegation_delivery(sid, f"[ASYNC DELEGATION COMPLETE — {deleg_id}]\nresult", {"delegation_id": deleg_id})


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    sessions = tmp_path / "sessions"
    home.mkdir(); sessions.mkdir()
    db_path = home / "state.db"
    db = hermes_state.SessionDB(db_path)
    db.create_session("sid1", source="api_server")
    (sessions / "sid1.json").write_text(json.dumps({"session_id": "sid1"}))
    monkeypatch.setattr(gdw, "_profile_state_dbs", lambda: [("default", db_path)])
    monkeypatch.setattr("api.config.SESSION_DIR", sessions)
    monkeypatch.setattr(bp, "_session_has_active_turn", lambda _sid: False)
    calls = []
    status = {"v": 200}

    def fake_start(sid, prompt, source="process_wakeup"):
        calls.append((sid, prompt, source))
        return {"_status": status["v"], "stream_id": "s"}

    monkeypatch.setattr("api.routes.start_session_turn", fake_start)
    gdw._RETRY.clear()
    yield db, calls, status
    db.close()


def test_delivery_row_starts_wakeup_turn_once(env):
    db, calls, _ = env
    _deliver(db, "sid1", "deleg_a")
    assert gdw.poll_once(time.time() - 60) == 1
    assert calls and calls[0][0] == "sid1" and "ASYNC DELEGATION COMPLETE — deleg_a" in calls[0][1]
    assert calls[0][2] == "process_wakeup"
    # Claimed exactly once: the next poll (and the Gateway's next-run fold) see nothing.
    assert gdw.poll_once(time.time() - 60) == 0
    assert db.claim_caller_history_deliveries("sid1") == []


def test_busy_session_leaves_row_unclaimed(env, monkeypatch):
    db, calls, _ = env
    _deliver(db, "sid1", "deleg_b")
    monkeypatch.setattr(bp, "_session_has_active_turn", lambda _sid: True)
    assert gdw.poll_once(time.time() - 60) == 0
    assert calls == []
    monkeypatch.setattr(bp, "_session_has_active_turn", lambda _sid: False)
    assert gdw.poll_once(time.time() - 60) == 1


def test_start_race_409_is_retried(env):
    db, calls, status = env
    _deliver(db, "sid1", "deleg_c")
    status["v"] = 409
    assert gdw.poll_once(time.time() - 60) == 0
    status["v"] = 200
    assert gdw.poll_once(time.time() - 60) == 1
    assert "deleg_c" in calls[-1][1]


def test_non_webui_session_is_ignored(env):
    db, calls, _ = env
    db.create_session("other", source="api_server")
    _deliver(db, "other", "deleg_d")
    assert gdw.poll_once(time.time() - 60) == 0
    assert calls == []
    # Left for its own client's next run to fold.
    assert len(db.claim_caller_history_deliveries("other")) == 1
