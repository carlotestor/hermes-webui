"""Legacy WebUI pins survive the move of the pin record into state.db.

Before state.db became the pin store, a WebUI pin lived only in the session
sidecar (``pinned: true``) while ``state.db.sessions.pinned`` stayed 0. The
first sidebar build after upgrading must copy those pins into state.db, verify
them, and mark the profile migrated before state.db is allowed to win.
"""

import sqlite3
import sys
import types

import pytest


def _make_db(path, ids, pinned=()):
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, pinned INTEGER NOT NULL DEFAULT 0,"
        " parent_session_id TEXT, end_reason TEXT)"
    )
    conn.executemany(
        "INSERT INTO sessions (id, pinned) VALUES (?, ?)",
        [(sid, 1 if sid in pinned else 0) for sid in ids],
    )
    conn.commit()
    conn.close()


def _db_pins(path):
    conn = sqlite3.connect(str(path))
    try:
        return {row[0]: bool(row[1]) for row in conn.execute("SELECT id, pinned FROM sessions")}
    finally:
        conn.close()


class _SqliteSessionDB:
    """Minimal ``hermes_state.SessionDB`` over the real sqlite file."""

    fail_writes = False

    def __init__(self, db_path):
        self._conn = sqlite3.connect(str(db_path))

    def get_session(self, sid):
        row = self._conn.execute("SELECT id, pinned FROM sessions WHERE id = ?", (sid,)).fetchone()
        return {"id": row[0], "pinned": row[1]} if row else None

    def set_session_pinned(self, sid, pinned):
        if _SqliteSessionDB.fail_writes:
            return False
        cur = self._conn.execute("UPDATE sessions SET pinned = ? WHERE id = ?", (int(pinned), sid))
        self._conn.commit()
        return cur.rowcount > 0

    def close(self):
        self._conn.close()


@pytest.fixture
def upgrade_env(tmp_path, monkeypatch):
    """One profile home with a state.db and a WebUI sidecar store."""
    from api import models, routes

    home = tmp_path / "home"
    home.mkdir()
    db = home / "state.db"
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()

    fake_mod = types.ModuleType("hermes_state")
    fake_mod.SessionDB = _SqliteSessionDB
    monkeypatch.setitem(sys.modules, "hermes_state", fake_mod)
    _SqliteSessionDB.fail_writes = False
    monkeypatch.setattr("api.profiles._resolve_profile_home_for_name", lambda _n: home, raising=False)
    monkeypatch.setattr("api.profiles._is_root_profile", lambda n: n == "default", raising=False)
    monkeypatch.setattr(models, "_get_profile_home", lambda _p: home)
    monkeypatch.setattr(routes, "SESSION_DIR", session_dir)

    sidecars = {}

    class _Sess:
        def __init__(self, sid):
            self.session_id = sid
            self.pinned = sidecars[sid]

        def save(self, **_kw):
            sidecars[self.session_id] = bool(self.pinned)

    monkeypatch.setattr(routes, "get_session", lambda sid, *a, **kw: _Sess(sid))
    monkeypatch.setattr(routes, "_ensure_full_session_before_mutation", lambda sid, s: s)
    return types.SimpleNamespace(db=db, sidecars=sidecars, session_dir=session_dir, routes=routes)


def _sidebar_build(env):
    rows = [
        {"session_id": sid, "pinned": pinned, "profile": "default"}
        for sid, pinned in env.sidecars.items()
    ]
    env.routes._reconcile_sidebar_pins_with_state_db(rows)
    return {row["session_id"]: row["pinned"] for row in rows}


def test_legacy_sidecar_pins_survive_first_sidebar_build(upgrade_env):
    env = upgrade_env
    # Pre-upgrade install: pins exist only in sidecars; state.db has pinned=0.
    _make_db(env.db, ["legacy_a", "legacy_b", "plain"])
    env.sidecars.update({"legacy_a": True, "legacy_b": True, "plain": False, "webui_only": True})

    shown = _sidebar_build(env)

    assert shown == {"legacy_a": True, "legacy_b": True, "plain": False, "webui_only": True}
    assert env.sidecars == {"legacy_a": True, "legacy_b": True, "plain": False, "webui_only": True}
    assert _db_pins(env.db) == {"legacy_a": True, "legacy_b": True, "plain": False}


def test_after_migration_state_db_wins(upgrade_env):
    env = upgrade_env
    _make_db(env.db, ["legacy_a", "desktop_pin"], pinned={"desktop_pin"})
    env.sidecars.update({"legacy_a": True, "desktop_pin": False})
    _sidebar_build(env)
    assert _db_pins(env.db) == {"legacy_a": True, "desktop_pin": True}

    # Desktop unpins after the migration: that now propagates to the sidecar.
    conn = sqlite3.connect(str(env.db))
    conn.execute("UPDATE sessions SET pinned = 0 WHERE id = 'legacy_a'")
    conn.commit()
    conn.close()

    shown = _sidebar_build(env)
    assert shown == {"legacy_a": False, "desktop_pin": True}
    assert env.sidecars == {"legacy_a": False, "desktop_pin": True}


def test_failed_migration_write_leaves_sidecar_pins_and_retries(upgrade_env):
    env = upgrade_env
    _make_db(env.db, ["legacy_a"])
    env.sidecars.update({"legacy_a": True})
    _SqliteSessionDB.fail_writes = True

    shown = _sidebar_build(env)
    assert shown == {"legacy_a": True}
    assert env.sidecars == {"legacy_a": True}
    assert _db_pins(env.db) == {"legacy_a": False}

    _SqliteSessionDB.fail_writes = False
    _sidebar_build(env)
    assert _db_pins(env.db) == {"legacy_a": True}
    assert env.sidecars == {"legacy_a": True}


def test_migration_is_recorded_once_per_state_db(upgrade_env):
    env = upgrade_env
    _make_db(env.db, ["legacy_a"])
    env.sidecars.update({"legacy_a": True})
    _sidebar_build(env)
    markers = list(env.session_dir.glob("_pin*migration*"))
    assert len(markers) == 1, markers

    writes = []
    orig = _SqliteSessionDB.set_session_pinned

    def _count(self, sid, pinned):
        writes.append((sid, pinned))
        return orig(self, sid, pinned)

    _SqliteSessionDB.set_session_pinned = _count
    try:
        _sidebar_build(env)
        _sidebar_build(env)
    finally:
        _SqliteSessionDB.set_session_pinned = orig
    assert writes == []
