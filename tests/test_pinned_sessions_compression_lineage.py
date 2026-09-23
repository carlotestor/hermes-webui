"""A pin survives a compression that happens after the pin (real SessionDB lineage).

The Agent inserts a later compression child with ``pinned=0``; the WebUI must
pin it when a pinned session rotates, or the auto-archive sweep takes the tip.
"""

import pathlib
import time

import pytest

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

    assert _carry_pin_to_compression_child("child", "default") is True
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
    assert "if getattr(s, 'pinned', False):" in block
    assert "_carry_pin_to_compression_child(new_sid," in block
