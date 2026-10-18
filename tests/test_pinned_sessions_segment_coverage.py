"""Pin writes cover every sidebar segment, and migration reads pins from unlisted sidecars."""

import json
import sqlite3

from tests._pin_helpers import db_pins, install_sqlite_session_db
from tests.test_pinned_sessions_upgrade_migration import _sidebar_build, upgrade_env  # noqa: F401


def _db(path, rows):
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, pinned INTEGER NOT NULL DEFAULT 0,"
        " parent_session_id TEXT, end_reason TEXT)"
    )
    conn.executemany("INSERT INTO sessions VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


def _home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    install_sqlite_session_db(monkeypatch)
    monkeypatch.setattr("api.profiles._resolve_profile_home_for_name", lambda _n: home, raising=False)
    monkeypatch.setattr("api.profiles._is_root_profile", lambda n: n == "default", raising=False)
    return home / "state.db"


def test_unpin_after_cli_close_clears_the_whole_conversation(tmp_path, monkeypatch):
    from api.state_sync import sync_session_pinned

    db = _home(tmp_path, monkeypatch)
    _db(db, [("old", 1, None, "cli_close"), ("new", 1, "old", None)])
    assert sync_session_pinned("new", False, profile="default") is True
    assert db_pins(db) == {"old": False, "new": False}


def test_pin_reaches_sibling_continuations_of_a_shared_parent(tmp_path, monkeypatch):
    from api.state_sync import sync_session_pinned

    db = _home(tmp_path, monkeypatch)
    _db(db, [("root", 0, None, "compression"), ("a", 0, "root", None), ("b", 0, "root", None)])
    assert sync_session_pinned("a", True, profile="default") is True
    assert db_pins(db) == {"root": True, "a": True, "b": True}


def test_migration_copies_pin_of_an_unlisted_empty_sidecar(upgrade_env):
    env = upgrade_env
    _db(env.db, [("listed", 0, None, None), ("empty", 0, None, None)])
    # ``empty`` is a pinned Untitled sidecar that all_sessions() leaves out of the sidebar rows.
    (env.session_dir / "empty.json").write_text(
        json.dumps({"session_id": "empty", "title": "Untitled", "pinned": True, "messages": []})
    )
    env.sidecars.update({"listed": False, "empty": True})
    rows = [{"session_id": "listed", "pinned": False, "profile": "default"}]
    env.routes._reconcile_sidebar_pins_with_state_db(rows)
    assert db_pins(env.db) == {"listed": False, "empty": True}
    # Once it gains a message and is listed, state.db keeps the pin.
    assert _sidebar_build(env) == {"listed": False, "empty": True}
