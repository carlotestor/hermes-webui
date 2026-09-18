"""Pinned sessions share ``sessions.pinned`` in state.db with Hermes Desktop.

Desktop and ``hermes sessions pin`` persist pins in ``state.db.sessions.pinned``
(and back-fill pinned rows past the list LIMIT). WebUI must read that flag on
its agent-session projection, keep pinned rows in the sidebar regardless of the
recency window, and mirror its own pin toggles back to state.db.
"""

import pathlib
import sqlite3
import time

import pytest

import api.agent_sessions as agent_sessions

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _make_state_db(path, *, sessions=40, pinned_ids=()):
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            session_source TEXT,
            title TEXT,
            model TEXT,
            started_at REAL NOT NULL,
            message_count INTEGER DEFAULT 0,
            parent_session_id TEXT,
            ended_at REAL,
            end_reason TEXT,
            pinned INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp REAL
        );
        CREATE INDEX idx_messages_session ON messages(session_id, timestamp);
        """
    )
    base = time.time() - sessions * 10
    for i in range(sessions):
        sid = f"cli_{i:04d}"
        started = base + i * 10
        conn.execute(
            "INSERT INTO sessions (id, source, session_source, title, model, started_at, message_count, pinned)"
            " VALUES (?, 'cli', 'cli', ?, 'openai/gpt-5', ?, 2, ?)",
            (sid, sid, started, 1 if sid in pinned_ids else 0),
        )
        for j in range(2):
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (?, ?, ?, 'hi', ?)",
                (f"m_{i:04d}_{j}", sid, "user" if j == 0 else "assistant", started + j),
            )
    conn.commit()
    conn.close()


def test_projection_carries_state_db_pinned_flag(tmp_path):
    db = tmp_path / "state.db"
    _make_state_db(db, sessions=5, pinned_ids={"cli_0003"})

    rows = agent_sessions.read_importable_agent_session_rows(db, limit=20, exclude_sources=None)
    by_id = {row["id"]: row for row in rows}

    assert by_id["cli_0003"]["pinned"] is True or by_id["cli_0003"]["pinned"] == 1
    assert not by_id["cli_0000"]["pinned"]


def test_pinned_row_survives_recency_window(tmp_path):
    db = tmp_path / "state.db"
    # cli_0000 is the OLDEST row: far outside a 5-row window (and the 8x oversample).
    _make_state_db(db, sessions=60, pinned_ids={"cli_0000"})

    rows = agent_sessions.read_importable_agent_session_rows(db, limit=5, exclude_sources=None)
    ids = [row["id"] for row in rows]

    assert "cli_0000" in ids
    assert len([i for i in ids if i != "cli_0000"]) == 5


def test_projection_without_pinned_column_defaults_false(tmp_path):
    db = tmp_path / "state.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, title TEXT, model TEXT,
                               started_at REAL NOT NULL, message_count INTEGER DEFAULT 0);
        CREATE TABLE messages (id TEXT PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp REAL);
        INSERT INTO sessions VALUES ('old_0', 'cli', 'old', 'm', 1.0, 1);
        INSERT INTO messages VALUES ('m0', 'old_0', 'user', 'hi', 1.5);
        """
    )
    conn.commit()
    conn.close()

    rows = agent_sessions.read_importable_agent_session_rows(db, limit=5, exclude_sources=None)
    assert [row["id"] for row in rows] == ["old_0"]
    assert not rows[0]["pinned"]


def test_sidebar_row_reads_pinned_from_state_db(tmp_path, monkeypatch):
    from api import models

    db = tmp_path / "state.db"
    _make_state_db(db, sessions=3, pinned_ids={"cli_0001"})
    monkeypatch.setattr(models, "get_last_workspace", lambda *_a, **_kw: str(tmp_path))
    monkeypatch.setattr(models, "ensure_cron_project", lambda *_a, **_kw: None, raising=False)
    monkeypatch.setattr(models, "ensure_webhook_project", lambda *_a, **_kw: None, raising=False)
    monkeypatch.setattr(models, "_profile_has_user_projects", lambda *_a, **_kw: False, raising=False)

    rows = models._load_cli_sessions_uncached(tmp_path, db, None)
    by_id = {row["session_id"]: row for row in rows}

    assert by_id["cli_0001"]["pinned"] is True
    assert by_id["cli_0000"]["pinned"] is False


class _FakeSessionDB:
    calls = []

    def __init__(self, db_path):
        self.db_path = db_path

    def get_session(self, _session_id):
        return None

    def set_session_pinned(self, session_id, pinned):
        _FakeSessionDB.calls.append((str(self.db_path), session_id, pinned))
        return True

    def close(self):
        pass


def test_sync_session_pinned_writes_through_set_session_pinned(tmp_path, monkeypatch):
    import sys
    import types

    from api import state_sync

    fake_mod = types.ModuleType("hermes_state")
    fake_mod.SessionDB = _FakeSessionDB
    monkeypatch.setitem(sys.modules, "hermes_state", fake_mod)
    (tmp_path / "state.db").write_bytes(b"")
    monkeypatch.setattr(
        "api.profiles._resolve_profile_home_for_name", lambda _name: tmp_path, raising=False
    )
    _FakeSessionDB.calls.clear()

    assert state_sync.sync_session_pinned("abc123", True, profile="default") is True
    assert state_sync.sync_session_pinned("abc123", False, profile="default") is True

    assert _FakeSessionDB.calls == [
        (str(tmp_path / "state.db"), "abc123", True),
        (str(tmp_path / "state.db"), "abc123", False),
    ]


def test_pin_route_writes_state_db_before_sidecar():
    routes_py = (ROOT / "api" / "routes.py").read_text(encoding="utf-8")
    pin_block = routes_py.split('if parsed.path == "/api/session/pin":', 1)[1]
    pin_block = pin_block.split('if parsed.path == "/api/session/archive":', 1)[0]
    assert pin_block.count("_write_pin_to_state_db(") == 2
    # Both branches refuse before the sidecar save when state.db rejects the pin.
    assert pin_block.index("_write_pin_to_state_db(s, True)") < pin_block.index("s.save()")
    assert "503" in pin_block


def test_write_pin_refuses_when_state_db_write_fails(monkeypatch):
    from api import routes, state_sync
    from types import SimpleNamespace

    monkeypatch.setattr(state_sync, "state_db_knows_session", lambda *_a, **_kw: True)
    monkeypatch.setattr(state_sync, "sync_session_pinned", lambda *_a, **_kw: False)
    s = SimpleNamespace(session_id="known", profile="default")
    assert routes._write_pin_to_state_db(s, True) is False

    monkeypatch.setattr(state_sync, "sync_session_pinned", lambda *_a, **_kw: True)
    assert routes._write_pin_to_state_db(s, True) is True


def test_write_pin_accepts_sidecar_only_session(monkeypatch):
    from api import routes, state_sync
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(state_sync, "state_db_knows_session", lambda *_a, **_kw: False)
    monkeypatch.setattr(state_sync, "sync_session_pinned", lambda *a, **kw: calls.append(a) or True)
    s = SimpleNamespace(session_id="webui-only", profile="default")
    assert routes._write_pin_to_state_db(s, True) is True
    assert calls == []


def test_reconcile_state_db_wins_over_sidecar(monkeypatch):
    from api import routes

    saved = []

    class _Session:
        pinned = True
        _loaded_metadata_only = False

        def save(self, **kw):
            saved.append((self.pinned, kw))

    monkeypatch.setattr(routes, "get_session", lambda *_a, **_kw: _Session())

    # Unpinned in Desktop -> sidecar pin dropped.
    row = {"session_id": "s1", "pinned": True, "profile": "default"}
    routes._reconcile_sidebar_pin_with_state_db(row, {"pinned": False})
    assert row["pinned"] is False
    assert saved == [(False, {"touch_updated_at": False})]

    # Pinned in Desktop -> sidecar row pinned.
    _Session.pinned = False
    saved.clear()
    row = {"session_id": "s2", "pinned": False, "profile": "default"}
    routes._reconcile_sidebar_pin_with_state_db(row, {"pinned": True})
    assert row["pinned"] is True
    assert saved == [(True, {"touch_updated_at": False})]


def test_write_pin_fails_closed_when_state_db_lookup_fails(monkeypatch):
    from api import routes, state_sync
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(state_sync, "state_db_knows_session", lambda *_a, **_kw: None)
    monkeypatch.setattr(state_sync, "sync_session_pinned", lambda *a, **kw: calls.append(a) or True)
    s = SimpleNamespace(session_id="unknown-state", profile="default")
    assert routes._write_pin_to_state_db(s, True) is False
    assert calls == []


def test_state_db_knows_session_reports_lookup_failure(tmp_path, monkeypatch):
    import sys
    import types

    from api import state_sync

    class _BrokenDB:
        def __init__(self, *_a, **_kw):
            pass

        def get_session(self, _sid):
            raise RuntimeError("database is locked")

        def close(self):
            pass

    fake_mod = types.ModuleType("hermes_state")
    fake_mod.SessionDB = _BrokenDB
    monkeypatch.setitem(sys.modules, "hermes_state", fake_mod)
    (tmp_path / "state.db").write_bytes(b"")
    monkeypatch.setattr(
        "api.profiles._resolve_profile_home_for_name", lambda _name: tmp_path, raising=False
    )
    assert state_sync.state_db_knows_session("abc", profile="default") is None

    fake_mod.SessionDB = _FakeSessionDB
    assert state_sync.state_db_knows_session("missing", profile="default") is False


def test_sidebar_build_reconciles_pins_without_show_cli_sessions(monkeypatch):
    from api import routes

    seen = []
    monkeypatch.setattr(
        routes, "agent_session_pinned_flags",
        lambda ids, profile=None: seen.append((sorted(ids), profile)) or {"a": True, "b": False},
    )
    reconciled = []
    monkeypatch.setattr(
        routes, "_reconcile_sidebar_pin_with_state_db",
        lambda row, meta: reconciled.append((row["session_id"], meta["pinned"])),
    )
    monkeypatch.setattr(routes, "get_cli_sessions", lambda *a, **kw: pytest.fail("cli listing must not run"))
    rows = [
        {"session_id": "a", "profile": "default", "pinned": False, "session_source": "webui"},
        {"session_id": "b", "profile": "default", "pinned": True, "session_source": "webui"},
        {"session_id": "c", "profile": "default", "pinned": False, "session_source": "webui"},
    ]
    routes._reconcile_sidebar_pins_with_state_db(rows)
    assert seen == [(["a", "b", "c"], "default")]
    # Rows unknown to state.db ("c") are left alone.
    assert reconciled == [("a", True), ("b", False)]

    # And the sidebar payload builder invokes it on the show_cli_sessions=False path.
    src = (ROOT / "api" / "routes.py").read_text(encoding="utf-8")
    body = src.split("def _build_session_list_cache_payload", 1)[1]
    assert body.index("_reconcile_sidebar_pins_with_state_db(webui_sessions)") < body.index("if show_cli_sessions:")


def test_agent_session_pinned_flags_reads_state_db(tmp_path, monkeypatch):
    import sqlite3
    from api import models

    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, pinned INTEGER DEFAULT 0)")
    conn.executemany("INSERT INTO sessions VALUES (?, ?)", [("p", 1), ("u", 0)])
    conn.commit(); conn.close()
    monkeypatch.setattr(models, "_agent_state_db_path", lambda profile=None: db)
    assert models.agent_session_pinned_flags(["p", "u", "missing"]) == {"p": True, "u": False}
    monkeypatch.setattr(models, "_agent_state_db_path", lambda profile=None: None)
    assert models.agent_session_pinned_flags(["p"]) == {}
