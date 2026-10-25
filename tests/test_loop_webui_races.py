import pytest
import threading
from pathlib import Path
from types import SimpleNamespace as NS

from tests.conftest import requires_agent_modules
from tests import test_issue7193_eager_start_rollback as _r7193

_start7193 = _r7193._start
issue7193_env = _r7193.issue7193_env  # reused fixture: the eager chat-start sandbox


def _src(rel):
    return (Path(__file__).resolve().parents[1] / rel).read_text(encoding="utf-8")


def _setup(tmp_path, monkeypatch, session, due=True):
    """Sandboxed loop store on one session; `due` makes its loop due without firing it."""
    from hermes_cli import goals, loops as agent
    from api import background_process, loops, models, profiles, routes
    from api.process_event_utils import build_active_turn_token
    started = []
    monkeypatch.setattr(goals, "_DB_CACHE", {})
    monkeypatch.setattr(profiles, "get_hermes_home_for_profile", lambda p: tmp_path / (p or "default"))
    monkeypatch.setattr(profiles, "_profiles_root", lambda: tmp_path / "none")
    monkeypatch.setattr(models, "get_session", lambda sid, **kw: session)
    monkeypatch.setattr(background_process, "_session_has_active_turn", lambda sid: False)

    def start(sid, msg, source):  # records the wakeup user row the way _start_run stamps it
        started.append(msg)
        sid_, at = f"stream{len(started)}", 1000.0 + len(started)
        session.messages.append({"role": "user", "content": msg, "_source": source, "timestamp": at,
                                 "_active_turn_token": build_active_turn_token(sid_, at)})
        return {"stream_id": sid_, "pending_started_at": at}

    monkeypatch.setattr(routes, "start_session_turn", start)
    loops.run_loop_command("s1", "5m check")
    if due:
        with loops._home(None):
            _due(agent)
    return agent, loops, started


def _due(agent, sid="s1"):
    s = agent.load_loop(sid)
    s.next_due_at = 0
    agent.save_loop(sid, s)


@requires_agent_modules
def test_retag_between_load_and_launch_never_runs_other_profile(tmp_path, monkeypatch):
    session = NS(session_id="s1", profile=None, messages=[])
    agent, loops, started = _setup(tmp_path, monkeypatch, session)

    def retag(sid):  # chat start retags the empty session to profile b after the scheduler listed it
        session.profile = "b"
        return False

    monkeypatch.setattr(agent, "goal_blocks_loop_tick", retag)
    loops.run_due_loops()
    assert started == []
    with loops._home(None):
        assert agent.load_loop("s1").status == "cleared"  # the loop left in profile a is dropped


@requires_agent_modules
@pytest.mark.parametrize("pinned,expected_profile,result,loop_status", [
    (False, "b", "retagged", "cleared"),
    (True, None, "pinned_empty", "active"),
])
def test_retag_session_profile_clears_old_loop_only_on_a_move(
        tmp_path, monkeypatch, pinned, expected_profile, result, loop_status):
    session = NS(session_id="s1", profile=None, messages=[], pinned=pinned)
    agent, loops, _ = _setup(tmp_path, monkeypatch, session, due=False)
    from api.routes import _retag_empty_session_profile
    assert loops.retag_session_profile(session, "b", _retag_empty_session_profile) == (expected_profile, result)
    assert session.profile == expected_profile
    with loops._home(None):
        assert agent.load_loop("s1").status == loop_status
    if result == "retagged":  # both retag sites in routes.py go through the helper, carrying the request profile
        src = _src("api/routes.py")
        assert src.count("retag_session_profile(\n") == 2
        assert src.count("s, requested_profile, _retag_empty_session_profile") == 2
        assert "request_profile=_get_active_profile_name())" in src


@requires_agent_modules
def test_stop_from_another_writer_is_not_overwritten(tmp_path, monkeypatch):
    session = NS(session_id="s1", profile=None, messages=[])
    agent, loops, started = _setup(tmp_path, monkeypatch, session)

    def cli_stop(sid):  # another process (CLI) stops the loop after the scheduler loaded it
        agent.clear_loop(sid)
        return False

    monkeypatch.setattr(agent, "goal_blocks_loop_tick", cli_stop)
    loops.run_due_loops()
    assert started == []
    with loops._home(None):
        assert agent.load_loop("s1").status == "cleared"


@requires_agent_modules
def test_stop_command_waits_for_scheduler_pass(tmp_path, monkeypatch):
    session = NS(session_id="s1", profile=None, messages=[])
    agent, loops, started = _setup(tmp_path, monkeypatch, session)
    out, t = [], []

    def race(sid):  # /loop stop arrives from a request thread mid-pass
        t.append(threading.Thread(target=lambda: out.append(loops.run_loop_command("s1", "stop"))))
        t[0].start()
        t[0].join(0.3)
        assert t[0].is_alive(), "stop must wait for the scheduler's transition"
        return False

    monkeypatch.setattr(agent, "goal_blocks_loop_tick", race)
    loops.run_due_loops()
    t[0].join(10)
    assert out == ["✓ Loop stopped."]
    with loops._home(None):
        assert agent.load_loop("s1").status == "cleared"
    loops.run_due_loops()
    assert len(started) == 1  # never brought back after stop reported success


@requires_agent_modules
def test_user_turn_before_judgment_does_not_decide_tick(tmp_path, monkeypatch):
    session = NS(session_id="s1", profile=None, messages=[])
    agent, loops, started = _setup(tmp_path, monkeypatch, session)
    loops.run_due_loops()
    session.messages += [{"role": "assistant", "content": "still deploying"},
                         {"role": "user", "content": "unrelated", "timestamp": 2000.0},
                         {"role": "assistant", "content": "sure\nLOOP_COMPLETE"}]
    loops.run_due_loops()
    with loops._home(None):
        s = agent.load_loop("s1")
    assert s.status == "active" and not s.awaiting_response and s.ticks_fired == 1


def test_retag_without_agent_modules_still_retags(monkeypatch):
    import sys
    from api import loops
    monkeypatch.setitem(sys.modules, "hermes_cli", None)  # no agent installed: the loop import fails
    s = NS(session_id="s1", profile=None)
    loops.retag_session_profile(s, "b", lambda sess, p: (setattr(sess, "profile", p), (p, "retagged"))[1])
    assert s.profile == "b"


@requires_agent_modules
def test_loop_command_rechecks_profile_under_retag_lock(tmp_path, monkeypatch):
    session = NS(session_id="s1", profile=None, messages=[])
    agent, loops, started = _setup(tmp_path, monkeypatch, session, due=False)
    loops.run_loop_command("s1", "stop")
    # the retag lands after the route's ownership check, before the command
    loops.retag_session_profile(session, "b", lambda s, p: (setattr(s, "profile", p), (p, "retagged"))[1])
    assert "another profile" in loops.run_loop_command("s1", "5m check", request_profile="default")
    with loops._home("b"):
        assert agent.load_loop("s1") is None  # profile b's store was never written


@requires_agent_modules
@pytest.mark.parametrize("rotations", [1, 2])
def test_compression_mid_wakeup_keeps_turn_and_loop(tmp_path, monkeypatch, rotations):
    from api import config
    session = NS(session_id="s1", profile=None, messages=[])
    agent, loops, started = _setup(tmp_path, monkeypatch, session)
    with loops._home(None):
        loops.run_due_loops()
        db = agent._get_session_db()
        db.create_session("s1", "webui")
        db.create_session("c1", "webui", parent_session_id="s1")
        assert agent.migrate_loop_to_session("s1", "c1")  # agent compressed; WebUI hasn't rotated yet
        cur = "c1"
        if rotations == 2:  # two rotations in one wakeup leave the turn record on the grandparent
            db.create_session("c2", "webui", parent_session_id="c1")
            assert agent.migrate_loop_to_session("c1", "c2")
            cur = "c2"
    monkeypatch.setattr(config, "ACTIVE_RUNS", {"stream1": {"session_id": "s1"}})  # turn still running
    loops.run_due_loops()
    with loops._home(None):
        s = agent.load_loop(cur)
    assert s.status == "active" and s.awaiting_response  # not cleared, not judged mid-turn
    monkeypatch.setattr(config, "ACTIVE_RUNS", {})
    session.session_id = cur
    session.messages.append({"role": "assistant", "content": "done\nLOOP_COMPLETE"})
    loops.run_due_loops()
    with loops._home(None):
        assert agent.load_loop(cur).status == "done"
        if rotations == 2:
            assert db.get_meta(loops._TURN_PREFIX + cur)  # re-keyed to the current session
        _due(agent, cur)
    loops.run_due_loops()
    assert len(started) == 1


@requires_agent_modules
def test_message_after_stop_does_not_hide_stop(tmp_path, monkeypatch):
    session = NS(session_id="s1", profile=None, messages=[])
    agent, loops, started = _setup(tmp_path, monkeypatch, session)
    loops.run_due_loops()
    session.messages += [{"role": "assistant", "content": "Task cancelled."},
                         {"role": "user", "content": "next question", "timestamp": 2000.0},
                         {"role": "assistant", "content": "answer"}]
    loops.run_due_loops()
    with loops._home(None):
        s = agent.load_loop("s1")
    assert s.status == "paused" and s.paused_reason == "user-interrupted (Stop)"


def _deferred_stop_env(tmp_path, monkeypatch, session_source, worker_first):
    """Saved session in deferred save mode whose wakeup starts through the production chat-start preparer."""
    import queue
    from api import config, models, routes, streaming
    from api.models import Session
    monkeypatch.setattr(models, "SESSION_DIR", tmp_path / "sessions")
    (tmp_path / "sessions").mkdir()
    monkeypatch.setattr(routes, "get_webui_session_save_mode", lambda *a: "deferred")
    holder = {"s": Session(session_id="s1", title="loop", messages=[], session_source=session_source)}
    holder["s"].save()
    agent, loops, started = _setup(tmp_path, monkeypatch, holder["s"])
    monkeypatch.setattr(models, "get_session", lambda sid, **kw: holder["s"])
    monkeypatch.setattr(streaming, "get_session", lambda sid, **kw: holder["s"])

    def start(sid, msg, source):  # the prompt lives only in pending_* until the turn settles
        started.append(msg)
        routes._prepare_chat_start_session_for_stream(
            holder["s"], msg=msg, attachments=[], workspace=str(tmp_path), model="m",
            model_provider=None, stream_id="stream1", started_at=1000.75, source=source)
        config.STREAMS["stream1"], config.CANCEL_FLAGS["stream1"] = queue.Queue(), threading.Event()
        config.register_stream_owner("stream1", sid)
        if worker_first:
            config.register_session_writeback_owner(sid, "stream1")
        return {"stream_id": "stream1", "pending_started_at": 1000.75}

    monkeypatch.setattr(routes, "start_session_turn", start)
    return agent, loops, started, holder


@requires_agent_modules
@pytest.mark.parametrize("session_source,settled_by", [
    ("webui", "caller"), ("fork", "caller"), ("webui", "worker"), ("fork", "worker"),
])
def test_stop_on_wakeup_pauses_loop_after_reload(tmp_path, monkeypatch, session_source, settled_by):
    """Stop on a wakeup pauses the loop, whichever side persists the cancelled row first."""
    from api import config, streaming
    from api.models import Session
    from api.process_event_utils import build_active_turn_token
    agent, loops, started, holder = _deferred_stop_env(
        tmp_path, monkeypatch, session_source, worker_first=settled_by == "worker")
    with loops._home(None):
        _due(agent)
    loops.run_due_loops()
    assert holder["s"].messages == []  # deferred save: nothing written yet
    if session_source == "fork":
        assert holder["s"].pending_user_source == "fork"
    if settled_by == "worker":  # the streaming worker's cancel finalizer wins the race
        config.CANCEL_FLAGS["stream1"].set()
        streaming._finalize_cancelled_turn(holder["s"], stream_id="stream1")
        assert holder["s"].pending_user_message is None and holder["s"].active_stream_id is None
    assert streaming.cancel_stream("stream1")
    config.ACTIVE_RUNS.pop("stream1", None)
    holder["s"] = Session.load("s1")  # judge the durably settled transcript
    users = [m for m in holder["s"].messages if m.get("role") == "user"]
    assert len(users) == 1
    assert users[0]["_active_turn_token"] == build_active_turn_token("stream1", 1000.75)
    if settled_by == "caller":
        assert users[0]["timestamp"] == 1000.75  # Stop's own recovery keeps the exact pending time
    if session_source == "fork":
        assert users[0]["_source"] == "fork"
    loops.run_due_loops()
    with loops._home(None):
        s = agent.load_loop("s1")
        assert s.status == "paused" and s.paused_reason == "user-interrupted (Stop)"
        _due(agent)
    loops.run_due_loops()
    assert len(started) == 1  # no further provider call after a forced due


def test_loop_scheduler_kill_switch(monkeypatch):
    from api import loops
    spawned = []
    monkeypatch.setattr(loops.threading, "Thread", lambda *a, **kw: spawned.append(kw) or NS(start=lambda: None))
    monkeypatch.setenv("HERMES_WEBUI_LOOP_SCHEDULER", "0")
    assert loops.start_loop_scheduler() is False and spawned == []
    monkeypatch.delenv("HERMES_WEBUI_LOOP_SCHEDULER")
    assert loops.start_loop_scheduler() is True and spawned[0]["name"] == "webui-loop-scheduler"


def test_tokenless_row_is_never_matched_by_prompt():
    # Token-only recognizer: an older token-less row with the same prompt must not shadow the real wakeup.
    from api import loops
    turn = {"token": "tok", "started_at": 1000.75, "prompt": "check"}  # records written before this fix carry prompt
    msgs = [{"role": "user", "content": "check", "timestamp": 1000},
            {"role": "assistant", "content": "old reply"},
            {"role": "user", "content": "check", "timestamp": 1000.75, "_active_turn_token": "tok"},
            {"role": "assistant", "content": "done\nLOOP_COMPLETE"}]
    assert loops._wakeup_outcome(msgs, turn) == (False, "done\nLOOP_COMPLETE")
    assert loops._wakeup_outcome(msgs[:2], turn) == (False, "")


def test_turn_record_parent_cycle_terminates():
    from api import loops
    parents = {"a": "b", "b": "a"}
    db = NS(get_meta=lambda k: None, get_session=lambda sid: {"parent_session_id": parents.get(sid)},
            set_meta=lambda k, v: (_ for _ in ()).throw(AssertionError("no record to copy")))
    assert loops._turn_record(db, "a") == {}


def test_launch_response_keeps_started_at_when_worker_clears_pending(issue7193_env, tmp_path, monkeypatch):
    # greptile 4237618938: the response must carry the launch timestamp even if the worker
    # (or Stop) cleared s.pending_* before the launcher read it back.
    from api import routes
    from api.models import new_session
    cleared = threading.Event()

    def worker(*_a, **_kw):
        session.pending_started_at = None
        session.pending_user_message = None
        cleared.set()

    monkeypatch.setattr(routes, "_run_agent_streaming", worker)
    # runs after the session lock is released, before the response is built: let the worker win
    monkeypatch.setattr(routes, "set_last_workspace", lambda *_a, **_kw: cleared.wait(2))
    session = new_session(workspace=str(tmp_path / "workspace"))
    resp = _start7193(session, workspace=tmp_path / "workspace")
    assert cleared.wait(2) and session.pending_started_at is None
    assert resp["pending_started_at"]  # captured under the lock, not re-read after the worker ran


@requires_agent_modules
def test_archived_session_loop_pauses_instead_of_firing(tmp_path, monkeypatch):
    session = NS(session_id="s1", profile=None, messages=[], archived=True)
    agent, loops, started = _setup(tmp_path, monkeypatch, session)
    loops.run_due_loops()
    assert started == []
    with loops._home(None):
        s = agent.load_loop("s1")
        assert s.status == "paused" and s.paused_reason == "session archived"


@requires_agent_modules
def test_archive_guard_pauses_loop_under_loop_lock(tmp_path, monkeypatch):
    session = NS(session_id="s1", profile=None, messages=[], archived=False)
    agent, loops, started = _setup(tmp_path, monkeypatch, session)
    with loops.archive_guard("s1") as guard:
        assert loops._session_lock("s1").locked()  # held before the caller takes the agent lock
        session.archived = True
        guard["session"] = session
    with loops._home(None):
        assert agent.load_loop("s1").paused_reason == "session archived"
    with loops.archive_guard("s1") as guard:  # unarchive does not touch the loop
        session.archived = False
        guard["session"] = session
    with loops._home(None):
        assert agent.load_loop("s1").status == "paused"


def test_only_loop_command_carries_session_id():
    body = _src("static/commands.js")
    body = body[body.index("async function _runAgentCommandTransport"):]
    body = body[:body.index("\n}\n")]
    assert "/^\\/?loop(\\s|$)/i.test(command)?{command,session_id:" in body and ":{command})" in body


def test_busy_loop_reply_stays_in_owning_chat():
    src = _src("static/messages.js")
    busy = src[src.index("_pc.name==='loop'"):src.index("const defaultMessageMode=")]
    busy = busy[:busy.index("return;")]
    assert "_loopMsgs=S.messages" in busy and "_loopMsgs.push({role:'assistant'" in busy
    assert "S.messages.push({role:'assistant'" not in busy
    assert "S.messages===_loopMsgs&&(S.session&&S.session.session_id)===_loopSid" in busy
