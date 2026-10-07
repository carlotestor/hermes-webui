"""Gateway-backend async-delegation completions must wake the WebUI session.

The Gateway persists them as delivery rows in the profile state.db and starts
no turn; the WebUI poller reserves them and wakes with a fixed prompt; the
Gateway worker commits the lease only once the Gateway accepts the run. A fake SessionDB implements the Agent's
reserve/commit/release contract (hermes-agent#125451).
"""
import json

import pytest

import api.background_process as bp
import api.gateway_delegation_wakeup as gdw
from api.models import Session

_REAL = {"pending": gdw._pending_session_ids, "webui_sid": gdw._webui_session_id}


class FakeDB:
    """In-memory model of the Agent's caller-history reservation contract."""

    rows: dict = {}
    clock = [1000.0]

    def __init__(self, _path=None):
        pass

    def close(self):
        pass

    @classmethod
    def add(cls, sid, deleg_id):
        rid = len(cls.rows) + 1
        cls.rows[rid] = {"id": rid, "session_id": sid, "content": f"[ASYNC DELEGATION COMPLETE — {deleg_id}]\nresult",
                         "consumed": False, "lease": None}
        return rid

    def _pending(self, sid):
        now = self.clock[0]
        return [r for r in self.rows.values() if r["session_id"] == sid and not r["consumed"]
                and (r["lease"] is None or r["lease"]["until"] <= now)]

    def reserve_caller_history_deliveries(self, session_id, owner, ttl_seconds, limit=None):
        token = f"tok{self.clock[0]}-{sum(1 for r in self.rows.values() if r['lease'])}"
        out = []
        for r in self._pending(session_id)[:limit]:
            r["lease"] = {"token": token, "owner": owner, "until": self.clock[0] + ttl_seconds}
            out.append({"id": r["id"], "content": r["content"], "reservation_token": token})
        return out

    def _held(self, token_or_ids, owner):
        # A token settles its own lease even after expiry (a re-reserve replaces the token); ids need a live lease.
        if isinstance(token_or_ids, str):
            return [r for r in self.rows.values() if not r["consumed"] and r["lease"]
                    and r["lease"]["token"] == token_or_ids and r["lease"]["owner"] == owner]
        return [r for r in self.rows.values() if not r["consumed"] and r["lease"] and r["id"] in token_or_ids
                and r["lease"]["owner"] == owner and r["lease"]["until"] > self.clock[0]]

    def commit_caller_history_deliveries(self, token, owner):
        held = self._held(token, owner)
        for r in held:
            r["consumed"], r["lease"] = True, None
        return len(held)

    def release_caller_history_deliveries(self, session_id=None, row_ids=None, *, reservation_token=None, owner=None):
        held = self._held(reservation_token, owner)
        for r in held:
            r["lease"] = None
        return len(held)

    def claim_caller_history_deliveries(self, session_id):
        """The Agent's next-run fold: final consume of pending (unreserved) rows."""
        got = self._pending(session_id)
        for r in got:
            r["consumed"], r["lease"] = True, None
        return [{"id": r["id"], "content": r["content"]} for r in got]


@pytest.fixture
def env(tmp_path, monkeypatch):
    import api.config as config
    import api.models as models

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    db_path = tmp_path / "state.db"
    db_path.write_text("")
    FakeDB.rows = {}
    FakeDB.clock = [1000.0]
    monkeypatch.setattr(models, "SESSION_DIR", sessions)
    monkeypatch.setattr(config, "SESSION_DIR", sessions)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", sessions / "_index.json")
    models.SESSIONS.clear()
    Session(session_id="sid1", messages=[{"role": "user", "content": "go"},
                                         {"role": "assistant", "content": "spawned"}]).save()
    monkeypatch.setattr(gdw, "_session_db_cls", lambda: FakeDB)
    monkeypatch.setattr(gdw, "_pending_session_ids",
                        lambda _p, _s: sorted({r["session_id"] for r in FakeDB.rows.values()}))
    monkeypatch.setattr(gdw, "_webui_session_id",
                        lambda _p, sid, prof: sid if (sessions / f"{sid}.json").is_file()
                        and gdw._sidecar_profile_matches(sessions / f"{sid}.json", prof) else None)
    monkeypatch.setattr(gdw, "_profile_state_dbs", lambda: [("default", db_path)])
    monkeypatch.setattr(gdw, "_profile_uses_gateway", lambda _p: True)
    gdw._BACKOFF.clear()
    monkeypatch.setattr(bp, "_session_has_active_turn", lambda _sid: False)
    calls = []
    status = {"v": 200}

    def fake_start(sid, prompt, source="process_wakeup"):
        # Records whether any row was committed before the wake turn was persisted.
        calls.append({"sid": sid, "prompt": prompt, "source": source,
                      "committed_before": any(r["consumed"] for r in FakeDB.rows.values())})
        if status["v"] < 400 and status.get("accept", True):
            gdw.settle_reservation(sid, accepted=True)  # what the Gateway worker does on acceptance
        return {"_status": status["v"], "stream_id": "s" if status["v"] < 400 else None}

    monkeypatch.setattr("api.routes.start_session_turn", fake_start)
    yield calls, status
    models.SESSIONS.clear()


def _context(sid="sid1"):
    from api.models import SESSIONS
    SESSIONS.clear()
    return Session.load(sid).context_messages


def test_wake_uses_fixed_prompt_and_result_reaches_parent_once(env):
    calls, _ = env
    FakeDB.add("sid1", "deleg_a")
    assert gdw.poll_once(0) == 1
    assert calls[0]["prompt"] == gdw.WAKE_PROMPT and "deleg_a" not in calls[0]["prompt"]
    assert calls[0]["source"] == "process_wakeup"
    assert sum("deleg_a" in str(m.get("content")) for m in _context()) == 1
    # Committed: neither the next poll nor the Agent's next-run fold sees it again.
    assert gdw.poll_once(0) == 0 and len(calls) == 1
    assert FakeDB().claim_caller_history_deliveries("sid1") == []


def test_commit_only_after_wake_turn_persisted(env):
    calls, _ = env
    FakeDB.add("sid1", "deleg_b")
    assert gdw.poll_once(0) == 1
    assert calls[0]["committed_before"] is False
    assert FakeDB.rows[1]["consumed"] is True
    assert Session.load("sid1").delegation_reservation is None


def test_launched_but_unaccepted_wake_keeps_lease_until_worker_settles(env):
    """start_session_turn returning a stream only means the local worker launched."""
    calls, status = env
    status["accept"] = False
    FakeDB.add("sid1", "deleg_h")
    assert gdw.poll_once(0) == 1
    assert FakeDB.rows[1]["consumed"] is False and FakeDB.rows[1]["lease"] is not None
    assert Session.load("sid1").delegation_reservation["owner"] == gdw.RESERVATION_OWNER
    gdw.settle_reservation("sid1", accepted=False)  # the worker's teardown: Gateway never accepted
    assert FakeDB.rows[1]["consumed"] is False and FakeDB.rows[1]["lease"] is None
    status["accept"] = True
    gdw._BACKOFF.clear()
    assert gdw.poll_once(0) == 1 and FakeDB.rows[1]["consumed"] is True


def test_stored_result_is_tagged_as_a_wakeup_notice(env):
    FakeDB.add("sid1", "deleg_t")
    gdw.poll_once(0)
    stored = [m for m in _context() if m.get("_delegation_delivery_id")]
    assert stored and stored[0]["_source"] == "process_wakeup"


def test_refused_wake_backs_off(env, monkeypatch):
    calls, status = env
    status["v"] = 409
    FakeDB.add("sid1", "deleg_k")
    clock = [5000.0]
    monkeypatch.setattr(gdw.time, "time", lambda: clock[0])
    assert gdw.poll_once(0) == 0 and len(calls) == 1
    assert gdw.poll_once(0) == 0 and len(calls) == 1  # backing off: no re-reserve, no re-wake
    clock[0] += gdw.POLL_INTERVAL_S * 2 + 1
    assert gdw.poll_once(0) == 0 and len(calls) == 2
    status["v"] = 200
    clock[0] += gdw.BACKOFF_MAX_S + 1
    assert gdw.poll_once(0) == 1 and FakeDB.rows[1]["consumed"] is True


def test_blank_delivery_is_not_left_reserved(env):
    calls, _ = env
    rid = FakeDB.add("sid1", "deleg_blank")
    FakeDB.rows[rid]["content"] = "   "
    assert gdw.poll_once(0) == 0 and calls == []
    assert FakeDB.rows[rid]["lease"] is None and FakeDB.rows[rid]["consumed"] is True


def test_profile_without_gateway_backend_is_skipped(env, monkeypatch):
    calls, _ = env
    monkeypatch.setattr(gdw, "_profile_uses_gateway", lambda p: p == "work")
    FakeDB.add("sid1", "deleg_p")
    assert gdw.poll_once(0) == 0 and calls == [] and FakeDB.rows[1]["lease"] is None


def test_poller_starts_when_only_a_named_profile_uses_the_gateway(monkeypatch):
    monkeypatch.setattr("api.gateway_chat.webui_gateway_chat_enabled", lambda *_a, **_k: False)
    monkeypatch.setattr(gdw, "_session_db_cls", lambda: FakeDB)
    for name in gdw._REQUIRED_API:
        assert callable(getattr(FakeDB, name))
    monkeypatch.setattr(gdw, "_loop", lambda: gdw._STOP.wait(5))
    try:
        assert gdw.start_gateway_delegation_poller() is True
    finally:
        gdw.stop_gateway_delegation_poller()


@pytest.mark.parametrize("failure", ["409", "500", "raise"])
def test_failed_wake_releases_reservation(env, monkeypatch, failure):
    _calls, status = env
    FakeDB.add("sid1", "deleg_c")
    if failure == "raise":
        def boom(*_a, **_k):
            raise RuntimeError("gateway unreachable")
        monkeypatch.setattr("api.routes.start_session_turn", boom)
    else:
        status["v"] = int(failure)
    assert gdw.poll_once(0) == 0
    assert FakeDB.rows[1]["consumed"] is False and FakeDB.rows[1]["lease"] is None
    assert Session.load("sid1").delegation_reservation is None
    # Released rows are retried (after the back-off), and the stored copy is not duplicated.
    monkeypatch.setattr("api.routes.start_session_turn",
                        lambda sid, *_a, **_k: gdw.settle_reservation(sid, True) or {"_status": 200, "stream_id": "s"})
    gdw._BACKOFF.clear()
    assert gdw.poll_once(0) == 1
    assert FakeDB.rows[1]["consumed"] is True
    assert sum("deleg_c" in str(m.get("content")) for m in _context()) == 1


def test_expired_reservation_is_retried(env):
    calls, _ = env
    FakeDB.add("sid1", "deleg_d")
    # A WebUI crash after reserving: the lease is held but never committed or released.
    FakeDB().reserve_caller_history_deliveries("sid1", gdw.RESERVATION_OWNER, gdw.RESERVATION_TTL_S)
    assert gdw.poll_once(0) == 0 and calls == []
    FakeDB.clock[0] += gdw.RESERVATION_TTL_S + 1
    assert gdw.poll_once(0) == 1
    assert FakeDB.rows[1]["consumed"] is True


def test_busy_session_leaves_row_pending(env, monkeypatch):
    calls, _ = env
    FakeDB.add("sid1", "deleg_e")
    monkeypatch.setattr(bp, "_session_has_active_turn", lambda _sid: True)
    assert gdw.poll_once(0) == 0 and calls == []
    assert FakeDB.rows[1]["lease"] is None
    monkeypatch.setattr(bp, "_session_has_active_turn", lambda _sid: False)
    assert gdw.poll_once(0) == 1


def test_sidecar_of_another_profile_is_not_woken(env, tmp_path):
    calls, _ = env
    path = tmp_path / "sessions" / "sid1.json"
    data = json.loads(path.read_text())
    data["profile"] = "work"
    path.write_text(json.dumps(data))
    FakeDB.add("sid1", "deleg_f")
    assert gdw.poll_once(0) == 0 and calls == []
    assert FakeDB.rows[1]["lease"] is None


def test_non_webui_session_is_ignored(env):
    calls, _ = env
    FakeDB.add("other", "deleg_g")
    assert gdw.poll_once(0) == 0 and calls == []
    assert FakeDB.rows[1]["lease"] is None


class _OldAgentDB:
    def claim_caller_history_deliveries(self, sid):
        raise AssertionError("must not fall back to the final claim")

    def release_caller_history_deliveries(self, sid, ids):
        pass


def test_poller_inert_on_agent_without_reserve_commit(monkeypatch):
    monkeypatch.setattr("api.gateway_chat.webui_gateway_chat_enabled", lambda _cfg: True)
    monkeypatch.setattr(gdw, "_session_db_cls", lambda: _OldAgentDB)
    assert gdw._claim_api_available() is False
    assert gdw.start_gateway_delegation_poller() is False


def test_poller_inert_without_hermes_state(monkeypatch):
    monkeypatch.setattr("api.gateway_chat.webui_gateway_chat_enabled", lambda _cfg: True)

    def missing():
        raise ImportError("hermes_state")

    monkeypatch.setattr(gdw, "_session_db_cls", missing)
    assert gdw.start_gateway_delegation_poller() is False


def test_real_agent_session_db_round_trip(env, tmp_path, monkeypatch):
    hermes_state = pytest.importorskip("hermes_state")
    if not callable(getattr(hermes_state.SessionDB, "reserve_caller_history_deliveries", None)):
        pytest.skip("installed Hermes Agent predates the reservation API")
    calls, _ = env
    monkeypatch.setattr(gdw, "_session_db_cls", lambda: hermes_state.SessionDB)
    monkeypatch.setattr(gdw, "_pending_session_ids", _REAL["pending"])
    monkeypatch.setattr(gdw, "_webui_session_id", _REAL["webui_sid"])
    db_path = tmp_path / "real-state.db"
    monkeypatch.setattr(gdw, "_profile_state_dbs", lambda: [("default", db_path)])
    db = hermes_state.SessionDB(db_path)
    try:
        db.create_session("sid1", source="api_server")
        db.append_delegation_delivery("sid1", "[ASYNC DELEGATION COMPLETE — deleg_r]\nresult", {"delegation_id": "deleg_r"})
        assert gdw.poll_once(0) == 1
        assert calls[0]["prompt"] == gdw.WAKE_PROMPT
        assert sum("deleg_r" in str(m.get("content")) for m in _context()) == 1
        assert db.claim_caller_history_deliveries("sid1") == []
        assert gdw.poll_once(0) == 0
    finally:
        db.close()



@pytest.mark.parametrize("gateway_up", [True, False])
def test_gateway_worker_settles_lease_on_profile_gateway(env, tmp_path, monkeypatch, gateway_up):
    """The real worker commits only once the session profile's Gateway answers, else releases."""
    import io
    import urllib.error

    import api.gateway_chat as gateway_chat
    import api.streaming as streaming
    from api import profiles
    from api.config import STREAMS, create_stream_channel

    root = tmp_path / "hermes"
    for name, home in (("default", root), ("work", root / "profiles" / "work")):
        home.mkdir(parents=True, exist_ok=True)
        (home / ".env").write_text(f"HERMES_WEBUI_GATEWAY_BASE_URL=http://{name}-gateway:8642\n"
                                   f"HERMES_WEBUI_GATEWAY_API_KEY={name}-key\n")
    monkeypatch.setattr(profiles, "_DEFAULT_HERMES_HOME", root)
    monkeypatch.setattr(profiles, "_loaded_profile_env_keys", set())
    monkeypatch.setenv("HERMES_WEBUI_GATEWAY_BASE_URL", "http://default-gateway:8642")
    monkeypatch.setenv("HERMES_WEBUI_GATEWAY_API_KEY", "default-key")
    monkeypatch.delenv("HERMES_WEBUI_GATEWAY_USE_RUNS_API", raising=False)
    monkeypatch.setattr(gateway_chat, "_gateway_reasoning_effort_for_request", lambda *a, **k: None)
    monkeypatch.setattr(streaming, "_load_webui_prefill_context", lambda cfg: {
        "status": "not_configured", "source": "none", "label": "", "message_count": 0, "messages": []})
    monkeypatch.setattr(streaming, "_prefill_messages_with_webui_context", lambda ctx, cfg: [])
    seen = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def fake_urlopen(req, timeout=None):
        if req.full_url.endswith("/v1/capabilities"):
            raise urllib.error.URLError("no runs api")
        seen.append((req.full_url, req.get_header("Authorization"), FakeDB.rows[1]["consumed"]))
        if not gateway_up:
            raise urllib.error.URLError("connection refused")
        return Resp(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n')

    monkeypatch.setattr(gateway_chat.urllib.request, "urlopen", fake_urlopen)
    s = Session.load("sid1")
    s.profile = "work"
    s.active_stream_id, s.pending_user_message, s.pending_user_source = "st1", gdw.WAKE_PROMPT, "process_wakeup"
    s.pending_attachments, s.pending_started_at = [], 1.0
    s.save()
    FakeDB.add("sid1", "deleg_w")
    token = FakeDB().reserve_caller_history_deliveries("sid1", gdw.RESERVATION_OWNER, 60)[0]["reservation_token"]
    gdw._store_results_in_context("sid1", [{"id": 1, "content": "r"}],
                                  {"db": "x", "token": token, "owner": gdw.RESERVATION_OWNER})
    STREAMS["st1"] = create_stream_channel()
    gateway_chat._run_gateway_chat_streaming("sid1", gdw.WAKE_PROMPT, "m", str(tmp_path), "st1", [])

    assert [(u, a) for u, a, _ in seen] == [("http://work-gateway:8642/v1/chat/completions", "Bearer work-key")]
    assert seen[0][2] is False  # not committed before the Gateway answered
    assert FakeDB.rows[1]["consumed"] is gateway_up
    assert (FakeDB.rows[1]["lease"] is None) and Session.load("sid1").delegation_reservation is None
