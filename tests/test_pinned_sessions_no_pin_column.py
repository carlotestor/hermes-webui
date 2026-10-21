"""An Agent state.db without ``sessions.pinned`` keeps master's sidecar pin path (review 5593764346)."""

import sqlite3

import pytest

from tests import test_pinned_sessions_upgrade_migration as upgrade
from tests._pin_helpers import PinSess, patch_pin_endpoint

upgrade_env = upgrade.upgrade_env


def _columns(path):
    conn = sqlite3.connect(str(path))
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
    finally:
        conn.close()


def _old_agent_db(path, ids):
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT)")
    conn.executemany("INSERT INTO sessions VALUES (?, 'webui')", [(sid,) for sid in ids])
    conn.commit()
    conn.close()


def _upgrade_agent(path):
    conn = sqlite3.connect(str(path))
    conn.execute("ALTER TABLE sessions ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
    conn.commit()
    conn.close()


def _pins(path):
    conn = sqlite3.connect(str(path))
    try:
        return {r[0]: bool(r[1]) for r in conn.execute("SELECT id, pinned FROM sessions")}
    finally:
        conn.close()


@pytest.fixture
def old_agent(upgrade_env, monkeypatch):
    env = upgrade_env
    _old_agent_db(env.db, ["a", "b"])
    env.sidecars.update({"a": False, "b": False})
    sessions = {}

    def on_save(sess):
        env.sidecars[sess.session_id] = bool(sess.pinned)

    for sid in ("a", "b"):
        sessions[sid] = PinSess(sid, on_save=on_save)
    real_get = env.routes.get_session
    post = patch_pin_endpoint(monkeypatch, sessions, limit=5)

    def build():
        stub = env.routes.get_session
        monkeypatch.setattr(env.routes, "get_session", real_get)
        try:
            return upgrade._sidebar_build(env)
        finally:
            monkeypatch.setattr(env.routes, "get_session", stub)

    env.post, env.build, env.sessions = post, build, sessions
    return env


def test_pin_and_unpin_return_200_and_save_the_sidecar(old_agent):
    env = old_agent
    assert env.post("a", True)[0] == 200
    assert env.sidecars["a"] is True
    assert env.post("a", False)[0] == 200
    assert env.sidecars["a"] is False


def test_no_pinned_column_is_added(old_agent):
    env = old_agent
    env.post("a", True)
    env.post("b", True)
    env.post("b", False)
    assert env.build() == {"a": True, "b": False}
    assert "pinned" not in _columns(env.db)


def test_queued_pin_lands_after_the_agent_adds_the_column(old_agent):
    env = old_agent
    assert env.post("a", True)[0] == 200
    env.build()
    _upgrade_agent(env.db)
    assert env.build() == {"a": True, "b": False}
    assert _pins(env.db) == {"a": True, "b": False}


def test_unpin_before_the_upgrade_is_not_reapplied(old_agent):
    env = old_agent
    assert env.post("a", True)[0] == 200
    env.build()
    assert env.post("a", False)[0] == 200
    _upgrade_agent(env.db)
    assert env.build() == {"a": False, "b": False}
    assert _pins(env.db) == {"a": False, "b": False}
    assert env.sidecars["a"] is False


def test_unreadable_existing_state_db_still_fails_closed(upgrade_env, monkeypatch):
    env = upgrade_env
    env.db.write_bytes(b"not a sqlite database" * 64)
    sess = PinSess("a", on_save=lambda s: env.sidecars.__setitem__("a", bool(s.pinned)))
    env.sidecars["a"] = False
    post = patch_pin_endpoint(monkeypatch, {"a": sess}, limit=5)
    assert post("a", True)[0] == 503
    assert env.sidecars["a"] is False
