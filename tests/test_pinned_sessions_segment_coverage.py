"""Pin writes cover every sidebar segment, and migration reads pins from unlisted sidecars."""

import json
import sqlite3

from tests._pin_helpers import db_pins, install_sqlite_session_db
from tests import test_pinned_sessions_upgrade_migration as upgrade

upgrade_env = upgrade.upgrade_env


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
    assert upgrade._sidebar_build(env) == {"listed": False, "empty": True}


def _pending(env):
    with env.routes._PIN_MIGRATION_LOCK:
        state = env.routes._load_pin_migration_state()
    return state, set((state.get("pending_pins") or {}).get(str(env.db)) or [])


def _unlisted_empty_under_held_writer(env, sidecar):
    _db(env.db, [("listed", 0, None, None), ("empty", 0, None, None)])
    (env.session_dir / "empty.json").write_text(json.dumps(sidecar))
    env.sidecars.update({"listed": False, "empty": True})
    rows = [{"session_id": "listed", "pinned": False, "profile": "default"}]
    writer = sqlite3.connect(str(env.db), isolation_level=None)
    writer.execute("BEGIN IMMEDIATE")  # a concurrent Agent writer holds the lock
    try:
        env.routes._reconcile_sidebar_pins_with_state_db(rows)
    finally:
        writer.execute("ROLLBACK")
        writer.close()
    state, pending = _pending(env)
    assert str(env.db) in state.get("migrated_state_dbs", [])
    assert pending == {"empty"} and env.sidecars["empty"] is True
    assert db_pins(env.db) == {"listed": False, "empty": False}
    return rows


def test_pending_unlisted_pin_retries_before_it_is_listed(upgrade_env):
    env = upgrade_env
    sidecar = {"session_id": "empty", "title": "Untitled", "pinned": True, "messages": []}
    rows = _unlisted_empty_under_held_writer(env, sidecar)
    # Writer released; ``empty`` is still unlisted, yet its pending pin lands.
    env.routes._reconcile_sidebar_pins_with_state_db(rows)
    assert db_pins(env.db) == {"listed": False, "empty": True}
    assert _pending(env)[1] == set()


def test_pending_unlisted_pin_is_dropped_after_an_authoritative_unpin(upgrade_env):
    env = upgrade_env
    sidecar = {"session_id": "empty", "title": "Untitled", "pinned": True, "messages": []}
    rows = _unlisted_empty_under_held_writer(env, sidecar)
    env.sidecars["empty"] = False  # /api/session/pin unpinned it meanwhile
    env.routes._reconcile_sidebar_pins_with_state_db(rows)
    assert db_pins(env.db) == {"listed": False, "empty": False}
    assert _pending(env)[1] == set()


def test_pending_unlisted_pin_of_another_profile_is_not_applied(upgrade_env, monkeypatch):
    env = upgrade_env
    sidecar = {"session_id": "empty", "title": "Untitled", "pinned": True, "messages": []}
    rows = _unlisted_empty_under_held_writer(env, sidecar)
    # The sidecar now belongs to another profile: this state.db must not take its pin.
    get = env.routes.get_session

    def _other(sid, *a, **kw):
        sess = get(sid, *a, **kw)
        sess.profile = "work"
        return sess

    monkeypatch.setattr(env.routes, "get_session", _other)
    env.routes._reconcile_sidebar_pins_with_state_db(rows)
    assert db_pins(env.db) == {"listed": False, "empty": False}
    assert _pending(env)[1] == set()


def _released_pending_with_zero_owner_rows(env, monkeypatch, rows):
    """Held writer → queue → release, then the owner has no displayed row left."""
    work_home = env.db.parent.parent / "work_home"
    work_home.mkdir()
    homes = {"work": work_home}
    monkeypatch.setattr(
        "api.profiles._resolve_profile_home_for_name",
        lambda n: homes.get(n, env.db.parent),
        raising=False,
    )
    sidecar = {"session_id": "empty", "title": "Untitled", "pinned": True, "messages": []}
    _unlisted_empty_under_held_writer(env, sidecar)
    monkeypatch.setattr(
        env.routes, "list_profiles_api", lambda: [{"name": "default"}, {"name": "work"}]
    )
    for _ in range(2):
        env.routes._reconcile_sidebar_pins_with_state_db(rows)
    assert db_pins(env.db) == {"listed": False, "empty": True}
    assert _pending(env)[1] == set()


def test_pending_pin_retries_when_owner_has_no_displayed_rows(upgrade_env, monkeypatch):
    _released_pending_with_zero_owner_rows(upgrade_env, monkeypatch, [])


def test_pending_pin_retries_when_only_foreign_profile_rows_are_displayed(upgrade_env, monkeypatch):
    foreign = [{"session_id": "foreign", "pinned": False, "profile": "work"}]
    _released_pending_with_zero_owner_rows(upgrade_env, monkeypatch, foreign)
    assert foreign[0]["pinned"] is False


def test_single_profile_scope_does_not_retry_another_profiles_pending_pins(upgrade_env, monkeypatch):
    env = upgrade_env
    sidecar = {"session_id": "empty", "title": "Untitled", "pinned": True, "messages": []}
    _unlisted_empty_under_held_writer(env, sidecar)
    work_home = env.db.parent.parent / "work_home"
    work_home.mkdir()
    monkeypatch.setattr(
        "api.profiles._resolve_profile_home_for_name",
        lambda n: work_home if n == "work" else env.db.parent,
        raising=False,
    )
    env.routes._reconcile_sidebar_pins_with_state_db([], profiles=["work"])
    assert db_pins(env.db) == {"listed": False, "empty": False}
    assert _pending(env)[1] == {"empty"}
    env.routes._reconcile_sidebar_pins_with_state_db([], profiles=["default"])
    assert db_pins(env.db) == {"listed": False, "empty": True}
