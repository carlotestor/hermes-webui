"""A pin survives a compression that happens after the pin (real SessionDB lineage).

The Agent inserts a later compression child with ``pinned=0``; the WebUI must
pin it when a pinned session rotates, or the auto-archive sweep takes the tip.
"""

import pathlib
import time

import pytest

from tests._pin_helpers import db_pins, install_sqlite_session_db

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def real_db(tmp_path, monkeypatch):
    hermes_state = pytest.importorskip("hermes_state")
    import api.profiles as profiles

    home = tmp_path / "home"
    home.mkdir()
    db = hermes_state.SessionDB(home / "state.db")
    db.close()
    monkeypatch.setattr(profiles, "_resolve_profile_home_for_name", lambda _n: home)
    monkeypatch.setattr(profiles, "_is_root_profile", lambda n: n == "default")
    return hermes_state, home / "state.db"


def _row(hermes_state, path, sid):
    db = hermes_state.SessionDB(path)
    try:
        return db.get_session(sid)
    finally:
        db.close()


def _sweep(hermes_state, path):
    db = hermes_state.SessionDB(path)
    try:
        # idle_days=0: every unpinned tip idle before "now" is a candidate.
        return db.archive_stale_sessions(0)
    finally:
        db.close()


def test_pin_survives_later_compression_then_unpin_archives(real_db):
    from api.state_sync import sync_session_pinned
    from api.streaming import _carry_pin_to_compression_child

    hermes_state, path = real_db
    db = hermes_state.SessionDB(path)
    db.create_session("root", "webui")
    db.close()
    assert sync_session_pinned("root", True, profile="default") is True

    # Compression happens after the pin: the Agent ends the parent and inserts the child.
    db = hermes_state.SessionDB(path)
    db.end_session("root", "compression")
    db.create_session("child", "webui", parent_session_id="root")
    db.close()
    assert not _row(hermes_state, path, "child")["pinned"]

    assert _carry_pin_to_compression_child("root", "child", "default", False) is True
    assert _row(hermes_state, path, "child")["pinned"]

    time.sleep(0.01)
    _sweep(hermes_state, path)
    assert not _row(hermes_state, path, "child")["archived"]

    assert sync_session_pinned("child", False, profile="default") is True
    assert not _row(hermes_state, path, "root")["pinned"]
    _sweep(hermes_state, path)
    assert _row(hermes_state, path, "child")["archived"]


def test_compression_rotation_carries_the_pin():
    src = (ROOT / "api" / "streaming.py").read_text(encoding="utf-8")
    block = src.split("if _agent_sid and _agent_sid != session_id:", 1)[1]
    block = block.split("_compressed = True", 1)[0]
    assert "_carry_pin_to_compression_child(\n                        old_sid, new_sid," in block
    assert "if getattr(s, 'pinned', False):" not in block


def _sqlite_pins(tmp_path, monkeypatch, rows):
    import sqlite3
    from api import models

    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, pinned INTEGER NOT NULL DEFAULT 0)")
    conn.executemany("INSERT INTO sessions VALUES (?, ?)", rows)
    conn.commit()
    conn.close()
    install_sqlite_session_db(monkeypatch)
    monkeypatch.setattr(models, "_pin_state_db_path", lambda profile=None: db)
    monkeypatch.setattr("api.state_sync._resolve_state_db_path", lambda profile=None: db)
    return lambda: db_pins(db)


def test_carry_uses_state_db_pin_not_stale_sidecar(tmp_path, monkeypatch):
    from api.streaming import _carry_pin_to_compression_child

    # Desktop pinned the parent in state.db; the WebUI sidecar still says unpinned.
    pins = _sqlite_pins(tmp_path, monkeypatch, [("root", 1), ("child", 0)])
    assert _carry_pin_to_compression_child("root", "child", "default", False) is True
    assert pins()["child"] is True


def test_carry_skips_when_state_db_parent_unpinned(tmp_path, monkeypatch):
    from api.streaming import _carry_pin_to_compression_child

    # Desktop unpinned the parent; a stale sidecar pin must not re-pin the child.
    pins = _sqlite_pins(tmp_path, monkeypatch, [("root", 0), ("child", 0)])
    assert _carry_pin_to_compression_child("root", "child", "default", True) is None
    assert pins()["child"] is False


def test_failed_carry_keeps_child_pinned_and_retries(tmp_path, monkeypatch):
    import api.state_sync as state_sync
    from api import routes
    from api.streaming import _carry_pin_to_compression_child

    # Parent pinned in state.db, but the child write does not land.
    pins = _sqlite_pins(tmp_path, monkeypatch, [("root", 1), ("child", 0)])
    monkeypatch.setattr(routes, "SESSION_DIR", tmp_path)
    real_sync = state_sync.sync_session_pinned
    monkeypatch.setattr(state_sync, "sync_session_pinned", lambda *a, **kw: False)
    assert _carry_pin_to_compression_child("root", "child", "default", False) is False
    assert pins()["child"] is False

    # The rotation keeps the sidecar pin whenever the parent was pinned.
    src = (ROOT / "api" / "streaming.py").read_text(encoding="utf-8")
    block = src.split("if _agent_sid and _agent_sid != session_id:", 1)[1]
    block = block.split("_compressed = True", 1)[0]
    assert "s.pinned = _carry_pin_to_compression_child(" in block
    assert ") is not None\n" in block

    # The next sidebar build retries the carry; state.db must not unpin the sidecar first.
    monkeypatch.setattr(state_sync, "sync_session_pinned", real_sync)
    sidecar = {"root": True, "child": True}
    monkeypatch.setattr(routes, "_reconcile_sidebar_pin_with_state_db",
                        lambda row, meta: sidecar.__setitem__(row["session_id"], meta["pinned"]))
    routes._reconcile_sidebar_pins_with_state_db(
        [{"session_id": sid, "pinned": p, "profile": "default"} for sid, p in sidecar.items()])
    assert pins() == {"root": True, "child": True}
    assert sidecar == {"root": True, "child": True}
