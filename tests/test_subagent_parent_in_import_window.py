"""Regression tests: a recency-sliced subagent leaf must keep its parent row.

`read_importable_agent_session_rows()` applied the visible-window limit as a
flat per-row recency slice. Subagent rows only render as children when their
parent row is in the same payload, so a frozen orchestrator (no longer writing)
lost the recency race against its own still-streaming leaves and fell out of the
window — promoting the leaves to top-level sidebar rows. The projection now
re-adds subagent parents that the oversampled candidate set already projected.
"""
import sqlite3

from api.agent_sessions import read_importable_agent_session_rows


def _make_db(path, sessions, messages):
    """sessions: (id, title, source, parent_session_id); messages: (session_id, role, ts)."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT, model TEXT, "
        "message_count INTEGER, started_at REAL, source TEXT, parent_session_id TEXT)"
    )
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, timestamp REAL)")
    for sid, title, source, parent in sessions:
        conn.execute(
            "INSERT INTO sessions (id, title, model, message_count, started_at, source, parent_session_id) "
            "VALUES (?,?,?,?,?,?,?)",
            (sid, title, "gpt", 2, 1000.0, source, parent),
        )
    for sid, role, ts in messages:
        conn.execute("INSERT INTO messages (session_id, role, timestamp) VALUES (?,?,?)", (sid, role, ts))
    conn.commit()
    conn.close()


def _lineage_db(path, parent_source="subagent"):
    """One quiet orchestrator + 3 hot leaves + an unrelated hot subagent row."""
    sessions: list[tuple[str, str, str, str | None]] = [
        ("orch", "Orchestrator", parent_source, None),
        ("loner", "Unrelated", "subagent", None),
    ]
    messages = [("orch", "user", 100.0), ("orch", "assistant", 101.0)]  # went quiet early
    for i in range(3):
        sessions.append((f"leaf{i}", f"Leaf {i}", "subagent", "orch"))
        messages += [(f"leaf{i}", "user", 900.0 + i), (f"leaf{i}", "assistant", 901.0 + i)]
    messages += [("loner", "user", 800.0), ("loner", "assistant", 801.0)]
    _make_db(path, sessions, messages)


def test_evicted_subagent_parent_is_kept_in_window(tmp_path):
    db = tmp_path / "state.db"
    _lineage_db(db)

    # limit=3 is exactly the three hot leaves; the quiet parent is sliced off.
    rows = read_importable_agent_session_rows(db, limit=3, exclude_sources=None)
    by_id = {r["id"]: r for r in rows}

    assert "orch" in by_id, "quiet subagent parent must survive the recency slice"
    for i in range(3):
        leaf = by_id[f"leaf{i}"]
        assert leaf["parent_session_id"] == "orch"
        assert leaf["relationship_type"] == "child_session"


def test_unrelated_older_subagent_row_stays_excluded(tmp_path):
    db = tmp_path / "state.db"
    _lineage_db(db)

    rows = read_importable_agent_session_rows(db, limit=3, exclude_sources=None)

    # Only ancestors are rescued — an unrelated quieter subagent row stays out.
    assert "loner" not in {r["id"] for r in rows}


def test_webui_parent_is_not_force_imported(tmp_path):
    db = tmp_path / "state.db"
    _lineage_db(db, parent_source="webui")

    rows = read_importable_agent_session_rows(db, limit=3, exclude_sources=None)

    # The webui sidebar bucket already owns its own rows; don't duplicate them.
    assert "orch" not in {r["id"] for r in rows}
    assert {f"leaf{i}" for i in range(3)} <= {r["id"] for r in rows}
