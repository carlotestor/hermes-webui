"""The pin quota counts state.db pins by compression lineage and fails closed."""

import sqlite3
import threading
from types import SimpleNamespace


def _state_db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT,"
        " end_reason TEXT, session_source TEXT, pinned INTEGER NOT NULL DEFAULT 0)"
    )
    conn.executemany("INSERT INTO sessions VALUES (?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


def test_state_db_only_compression_segments_share_one_lineage(tmp_path, monkeypatch):
    from api import models, routes

    db = tmp_path / "state.db"
    # One pinned conversation (root -> mid -> tip) plus a pinned fork of the tip.
    _state_db(db, [
        ("root", None, "compression", None, 1),
        ("mid", "root", "compression", None, 1),
        ("tip", "mid", None, None, 1),
        ("fork", "tip", None, "fork", 1),
    ])
    monkeypatch.setattr(models, "_pin_state_db_path", lambda profile=None: db)
    # WebUI storage holds only the tip; root and mid exist only in state.db.
    webui = [{"session_id": "tip", "pinned": True, "profile": "default", "parent_session_id": "mid"}]
    rows = routes._pin_quota_rows_from_state_db(webui)
    assert routes._visible_pinned_lineage_ids(rows) == {"root", "fork"}


def test_quota_rows_fail_closed_when_state_db_unreadable(tmp_path, monkeypatch):
    from api import models, routes

    monkeypatch.setattr(models, "_pin_state_db_path", lambda profile=None: tmp_path / "missing.db")
    assert routes._pin_quota_rows_from_state_db([{"session_id": "a", "pinned": False, "profile": "p"}]) is None


def test_pin_refused_when_a_profile_pin_db_is_unreadable(monkeypatch):
    from api import routes

    responses = {}
    writes = []

    class _Sess:
        profile = "default"
        archived = False
        session_id = "new_pin"
        pinned = False

        def compact(self):
            return {"session_id": self.session_id, "pinned": bool(self.pinned), "profile": "default"}

        def save(self, touch_updated_at=True):
            pass

    sess = _Sess()
    body = {"session_id": "new_pin", "pinned": True}
    # "work" profile's pins cannot be read; its sidecar says nothing is pinned.
    sidecar = [{"session_id": "w1", "pinned": False, "profile": "work"}, sess.compact()]
    monkeypatch.setattr(routes, "agent_session_pinned_ids", lambda profile=None: None if profile == "work" else set())
    monkeypatch.setattr(routes, "agent_session_pinned_flags", lambda ids, profile=None: None if profile == "work" else {})
    monkeypatch.setattr(routes, "_write_pin_to_state_db", lambda s, p: writes.append(p) or True)
    monkeypatch.setattr(routes, "_get_session_agent_lock", lambda sid: threading.RLock())
    monkeypatch.setattr(routes, "_get_or_materialize_session", lambda sid, **kw: sess)
    monkeypatch.setattr(routes, "get_session", lambda sid, **kw: sess)
    monkeypatch.setattr(routes, "_ensure_full_session_before_mutation", lambda _sid, s: s)
    monkeypatch.setattr(routes, "_session_is_subagent_view_only", lambda _sid: False)
    monkeypatch.setattr(routes, "all_sessions", lambda *a, **kw: [dict(r) for r in sidecar])
    monkeypatch.setattr(routes, "SESSIONS", {})
    monkeypatch.setattr(routes, "_PIN_QUOTA_RESERVATIONS", {})
    monkeypatch.setattr(routes, "load_settings", lambda: {"pinned_sessions_limit": 3})
    monkeypatch.setattr(routes, "publish_session_list_changed", lambda *a, **kw: None)
    monkeypatch.setattr(routes, "_check_csrf", lambda handler: True)
    monkeypatch.setattr(routes, "read_body", lambda handler: body)
    monkeypatch.setattr(routes, "j", lambda h, p, status=200, extra_headers=None: responses.setdefault("r", status) or True)
    monkeypatch.setattr(routes, "bad", lambda h, m, status=400: responses.setdefault("r", status) or True)

    routes.handle_post(object(), SimpleNamespace(path="/api/session/pin"))
    assert responses["r"] == 503
    assert writes == [] and sess.pinned is False
    assert routes._PIN_QUOTA_RESERVATIONS == {}
