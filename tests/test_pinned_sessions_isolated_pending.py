"""Isolated profile mode: a foreign alias bucket must not drain the isolated owner's pending pins.

Review 5593764346 (Should fix) and comment 6305058288.
"""

import io
import json
import sqlite3
from urllib.parse import urlparse

import pytest

from api import profiles
from tests._pin_helpers import db_pins, install_sqlite_session_db


class _Handler:
    def __init__(self):
        self.status, self.headers, self.wfile = None, {}, io.BytesIO()

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.headers[key] = value

    def end_headers(self):
        pass


@pytest.fixture
def isolated_work(tmp_path, monkeypatch):
    from api import routes

    base = tmp_path / ".hermes"
    work = base / "profiles" / "work"
    work.mkdir(parents=True)
    db = work / "state.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, pinned INTEGER NOT NULL DEFAULT 0,"
                 " parent_session_id TEXT, end_reason TEXT)")
    conn.executemany("INSERT INTO sessions VALUES (?, 0, NULL, NULL)", [("listed",), ("empty",)])
    conn.commit()
    conn.close()
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()

    monkeypatch.setenv("HERMES_WEBUI_ISOLATED_PROFILE", "1")
    monkeypatch.setattr(profiles, "_INITIAL_ISOLATED_PROFILE_OPT_IN", "1")
    monkeypatch.setattr(profiles, "_INITIAL_HERMES_HOME", str(work))
    monkeypatch.setattr(profiles, "_DEFAULT_HERMES_HOME", base)
    monkeypatch.setattr(profiles, "_LIST_PROFILES_CACHE", None)
    install_sqlite_session_db(monkeypatch)
    monkeypatch.setattr(routes, "SESSION_DIR", session_dir)
    monkeypatch.setattr(routes, "list_profiles_api", lambda: [{"name": "work"}])
    assert profiles._is_isolated_profile_mode() is True

    sidecars = {"listed": False, "empty": True, "foreign": False}
    owners = {"listed": "work", "empty": "work", "foreign": "default"}

    class _Sess:
        def __init__(self, sid):
            self.session_id, self.pinned, self.profile = sid, sidecars[sid], owners[sid]

        def save(self, **_kw):
            sidecars[self.session_id] = bool(self.pinned)

    monkeypatch.setattr(routes, "get_session", lambda sid, *a, **kw: _Sess(sid))
    monkeypatch.setattr(routes, "_ensure_full_session_before_mutation", lambda sid, s: s)
    (session_dir / "empty.json").write_text(json.dumps(
        {"session_id": "empty", "title": "Untitled", "pinned": True, "messages": [], "profile": "work"}))

    # Queue ``empty``'s pin while a real SQLite writer holds the lock.
    writer = sqlite3.connect(str(db), isolation_level=None)
    writer.execute("BEGIN IMMEDIATE")
    try:
        routes._reconcile_sidebar_pins_with_state_db(
            [{"session_id": "listed", "pinned": False, "profile": "work"}], profiles=["work"])
    finally:
        writer.execute("ROLLBACK")
        writer.close()
    assert _pending(routes, db) == {"empty"}
    assert db_pins(db) == {"listed": False, "empty": False}
    return routes, db, sidecars


def _pending(routes, db):
    with routes._PIN_MIGRATION_LOCK:
        state = routes._load_pin_migration_state()
    return set((state.get("pending_pins") or {}).get(str(db)) or [])


def test_profiles_none_claims_the_isolated_key_for_its_owner_only(isolated_work):
    routes, db, sidecars = isolated_work
    assert routes._profiles_with_pending_pins(None) == ["work"]
    routes._reconcile_sidebar_pins_with_state_db([], profiles=None)
    assert db_pins(db) == {"listed": False, "empty": True}
    assert _pending(routes, db) == set()
    rows = [{"session_id": "empty", "pinned": True, "profile": "work"}]
    routes._reconcile_sidebar_pins_with_state_db(rows, profiles=None)
    assert rows[0]["pinned"] is True and sidecars["empty"] is True


def test_foreign_alias_bucket_does_not_consume_the_owners_pending_pin(isolated_work):
    routes, db, sidecars = isolated_work
    foreign = [{"session_id": "foreign", "pinned": False, "profile": "default"}]
    routes._reconcile_sidebar_pins_with_state_db(foreign, profiles=None)
    assert db_pins(db)["empty"] is True
    assert _pending(routes, db) == set()
    assert sidecars["empty"] is True


def test_real_sessions_get_foreign_only_row_then_owner_sidecar_visible(isolated_work, monkeypatch):
    """Real GET dispatcher/cache/builder: only a ``default`` row is displayed in isolated ``work``."""
    routes, db, sidecars = isolated_work
    visible = [{"session_id": "foreign", "title": "Foreign", "profile": "default", "pinned": False,
                "archived": False, "message_count": 3, "updated_at": 10, "last_message_at": 10}]
    monkeypatch.setattr(routes, "all_sessions", lambda **_kw: [dict(r) for r in visible])
    monkeypatch.setattr(routes, "_enrich_sidebar_lineage_metadata", lambda rows: None)
    monkeypatch.setattr(routes, "_reconcile_stale_stream_state_for_session_rows", lambda rows: False)
    monkeypatch.setattr(routes, "load_settings", lambda: {"show_cli_sessions": False})
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "work")

    for query in ("", "?all_profiles=1"):
        routes._session_list_cache_clear()
        handler = _Handler()
        routes.handle_get(handler, urlparse(f"http://example.com/api/sessions{query}"))
        assert handler.status == 200

    # The original sidecar gains a message and becomes visible.
    visible.append({"session_id": "empty", "title": "Now titled", "profile": "work", "pinned": True,
                    "archived": False, "message_count": 1, "updated_at": 11, "last_message_at": 11})
    routes._session_list_cache_clear()
    handler = _Handler()
    routes.handle_get(handler, urlparse("http://example.com/api/sessions"))
    assert handler.status == 200
    assert _pending(routes, db) == set()
    assert db_pins(db)["empty"] is True
    assert sidecars["empty"] is True
    shown = {r["session_id"]: r.get("pinned") for r in json.loads(handler.wfile.getvalue())["sessions"]}
    assert shown.get("empty") is True
