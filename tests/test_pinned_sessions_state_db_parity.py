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
SESSIONS_JS = (ROOT / "static" / "sessions.js").read_text(encoding="utf-8")


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


def test_pin_route_mirrors_to_state_db():
    routes_py = (ROOT / "api" / "routes.py").read_text(encoding="utf-8")
    pin_block = routes_py.split('if parsed.path == "/api/session/pin":', 1)[1]
    pin_block = pin_block.split('if parsed.path == "/api/session/archive":', 1)[0]
    assert "sync_session_pinned(" in pin_block


def test_sidebar_shift_click_toggles_pin():
    assert "_toggleSessionPinned" in SESSIONS_JS
    assert "e.shiftKey" in SESSIONS_JS
    # The action menu and shift-click share one code path.
    assert SESSIONS_JS.count("/api/session/pin") == 1


def test_reconcile_keeps_sidecar_pin_when_state_db_write_fails(monkeypatch):
    from api import routes, state_sync

    routes._REASSERTED_SIDECAR_PINS.discard("keep-me")
    monkeypatch.setattr(state_sync, "sync_session_pinned", lambda *_a, **_kw: False)
    monkeypatch.setattr(routes, "get_session", lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("sidecar must not be rewritten")))
    row = {"session_id": "keep-me", "pinned": True, "profile": "default"}

    routes._reconcile_sidebar_pin_with_state_db(row, {"pinned": False})

    assert row["pinned"] is True
    # A failed push is retried on the next sidebar build, not settled.
    assert "keep-me" not in routes._REASSERTED_SIDECAR_PINS


def test_reconcile_adopts_state_db_pin_into_unpinned_sidecar(monkeypatch):
    from api import routes

    saved = []

    class _Session:
        pinned = False
        _loaded_metadata_only = False

        def save(self, **kw):
            saved.append((self.pinned, kw))

    monkeypatch.setattr(routes, "get_session", lambda *_a, **_kw: _Session())
    row = {"session_id": "adopt-me", "pinned": False, "profile": "default"}

    routes._reconcile_sidebar_pin_with_state_db(row, {"pinned": True})

    assert row["pinned"] is True
    assert saved == [(True, {"touch_updated_at": False})]


def test_sidebar_shift_click_respects_read_only_rows():
    idx = SESSIONS_JS.find("e.shiftKey")
    assert idx != -1
    assert "!readOnly" in SESSIONS_JS[idx: idx + 120]
