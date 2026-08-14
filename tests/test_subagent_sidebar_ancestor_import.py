"""Regression tests: a visible subagent leaf must keep its orchestrator parent row.

Mechanism being pinned (sidebar "orphan promotion"):

``api/models.py: get_cli_sessions()`` builds the interactive sidebar window by
calling ``agent_sessions.read_importable_agent_session_rows(..., limit=
CLI_VISIBLE_SESSION_LIMIT)``. That projection orders rows by real recency
(``MAX(messages.timestamp)``) and then HARD-SLICES the result to ``limit``.

A ``delegate_task`` orchestrator is frozen while its delegated children run: the
orchestrator's own last message is old, while its subagent leaves are still
writing messages *right now*. So the leaves win the recency race, survive the
slice, and the quieter orchestrator PARENT is evicted from the payload.

The leaf row still carries ``relationship_type='child_session'`` and
``parent_session_id=<orchestrator>``, but with no parent ROW in the payload the
client (``static/sessions.js: _attachChildSessionsToSidebarRows()``) has nothing
to nest it under, so it promotes the orphan to a TOP-LEVEL "Subagent Session"
row in the sidebar. From the user's point of view a background delegation
suddenly appears as a peer conversation next to their real chats.

The contract these tests pin: after the recency slice, every selected
``source='subagent'`` row's missing ancestors are ADDED back (never swapped in
by evicting a selected row), walking ``parent_session_id`` upward and stopping
at a ``webui`` ancestor, a missing parent, a cycle, or a source the caller's own
filters exclude. Raising ``CLI_VISIBLE_SESSION_LIMIT`` is explicitly NOT the
fix, so every test here runs at a small, deterministic limit.

Black-box by construction: these tests only touch the public surface
``read_importable_agent_session_rows()`` and ``get_cli_sessions()``.
"""

import sqlite3
import time

import pytest

from api import models
from api.agent_sessions import read_importable_agent_session_rows

# Fixed epoch base so ordering is explicit and never depends on the real clock.
T = 1_760_000_000.0

# Mirrors the exclusions api/models.py uses for the interactive sidebar window,
# i.e. background sources are handled by their own bounded passes while
# ``webui`` rows stay eligible. Keeping webui eligible is what makes the
# "do not force-import the webui ancestor" assertion in this file meaningful.
SIDEBAR_EXCLUDES = ("cron", "webhook", "kanban")


def _session(sid, source, last_activity, *, parent=None, title=None,
             ended_at=None, end_reason=None, session_source=None):
    """Describe one state.db session row plus the recency that ranks it."""
    return {
        "id": sid,
        "source": source,
        "session_source": session_source,
        "title": title or f"{sid} title",
        "started_at": last_activity - 100.0,
        "parent_session_id": parent,
        "ended_at": ended_at,
        "end_reason": end_reason,
        "last_activity": last_activity,
    }


def _make_state_db(db_path, rows):
    """Create a state.db with the given session rows and real message rows.

    Real ``messages`` rows are mandatory, not decoration: the projection drops
    any row with ``actual_message_count <= 0``, and the recency window is
    ordered by ``MAX(messages.timestamp)``. A session with no messages is
    silently invisible and would quietly void these tests.
    """
    conn = sqlite3.connect(str(db_path))
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
            end_reason TEXT
        );
        CREATE INDEX idx_sessions_parent ON sessions(parent_session_id);
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp REAL
        );
        CREATE INDEX idx_messages_session ON messages(session_id, timestamp);
        """
    )
    for row in rows:
        conn.execute(
            """
            INSERT INTO sessions
                (id, source, session_source, title, model, started_at,
                 message_count, parent_session_id, ended_at, end_reason)
            VALUES (?, ?, ?, ?, 'openai/gpt-5', ?, 2, ?, ?, ?)
            """,
            (
                row["id"], row["source"], row["session_source"], row["title"],
                row["started_at"], row["parent_session_id"], row["ended_at"],
                row["end_reason"],
            ),
        )
        # One user + one assistant turn. The user turn matters: CLI-classified
        # rows need a user turn to stay visible in the sidebar projection.
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, 'user', 'hi', ?)",
            (row["id"], row["last_activity"] - 1.0),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, 'assistant', 'ok', ?)",
            (row["id"], row["last_activity"]),
        )
    conn.commit()
    conn.close()


def _ids(rows):
    return [row["id"] for row in rows]


def _by_id(rows):
    return {row["id"]: row for row in rows}


def _hot_leaf_orchestrator_rows(leaf_count=4, orchestrator="orch-frozen"):
    """The bug shape: one quiet orchestrator, N leaves with the newest activity.

    The orchestrator's last message is far older than every leaf's, so the
    recency slice keeps the leaves and evicts the parent.
    """
    rows = [_session(orchestrator, "subagent", T + 100.0, title="Delegated run")]
    rows += [
        _session(f"leaf-{i}", "subagent", T + 900.0 + i,
                 parent=orchestrator, title=f"Subagent leaf {i}")
        for i in range(leaf_count)
    ]
    return rows


@pytest.fixture
def fake_hermes_home(tmp_path, monkeypatch):
    """Point get_cli_sessions() at a temporary HERMES_HOME (see #3172 tests)."""
    home = tmp_path / "hermes"
    home.mkdir()

    import api.config as cfg
    import api.profiles as profiles
    monkeypatch.setattr(profiles, "get_active_hermes_home", lambda: home)
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: None)

    projects_file = tmp_path / "projects.json"
    projects_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(cfg, "PROJECTS_FILE", projects_file)
    monkeypatch.setattr(models, "PROJECTS_FILE", projects_file)
    monkeypatch.setattr(models, "_projects_migrated", True)

    # The CLI sessions projection is TTL-cached; drop anything a previous test
    # left behind so this test observes its own fixture.
    if hasattr(models, "clear_cli_sessions_cache"):
        models.clear_cli_sessions_cache()

    return home


def test_orchestrator_parent_rescued_when_recency_window_evicts_it(tmp_path):
    """4 hot leaves + 1 quiet orchestrator at limit=4: the parent must come back.

    Unfixed, ``projected[:limit]`` keeps exactly the 4 leaves and drops the
    orchestrator, which is what turns the leaves into top-level sidebar rows.
    """
    db = tmp_path / "state.db"
    rows = _hot_leaf_orchestrator_rows(leaf_count=4)
    # An older, unrelated CLI row that must not be dragged in by the rescue.
    rows.append(_session("cli-old", "cli", T + 50.0, title="Old CLI chat"))
    _make_state_db(db, rows)

    result = read_importable_agent_session_rows(
        db, limit=4, exclude_sources=SIDEBAR_EXCLUDES
    )
    got = _ids(result)
    leaves = [f"leaf-{i}" for i in range(4)]

    # add-not-evict: the rescue must not buy room for the parent by dropping a
    # leaf the recency window legitimately selected.
    missing_leaves = [leaf for leaf in leaves if leaf not in got]
    assert not missing_leaves, (
        f"Recency-selected subagent leaves {missing_leaves} disappeared from the payload; "
        f"the ancestor rescue must ADD the parent, never evict a selected child to make "
        f"room for it. Returned ids: {got}"
    )
    assert "orch-frozen" in got, (
        "The delegate_task orchestrator 'orch-frozen' was evicted by the recency window "
        f"(limit=4) and never restored. Returned ids: {got}. Without the parent row in the "
        "payload, static/sessions.js _attachChildSessionsToSidebarRows() cannot nest the "
        "leaves and promotes them to top-level 'Subagent Session' rows in the sidebar."
    )
    assert "cli-old" not in got, (
        "Unrelated older CLI row 'cli-old' was pulled into the window. The rescue must add "
        f"ancestors of selected subagent rows only, not widen the window. Returned ids: {got}"
    )


def test_rescued_parent_preserves_child_relationship(tmp_path):
    """The leaf keeps its lineage metadata AND its parent id resolves in-payload.

    This pair is the exact precondition the client needs to nest the row:
    ``relationship_type == 'child_session'`` plus a ``parent_session_id`` that
    is present in the returned id set.
    """
    db = tmp_path / "state.db"
    _make_state_db(db, _hot_leaf_orchestrator_rows(leaf_count=4))

    result = read_importable_agent_session_rows(
        db, limit=4, exclude_sources=SIDEBAR_EXCLUDES
    )
    got = _ids(result)
    rows_by_id = _by_id(result)

    leaf = rows_by_id.get("leaf-3")
    assert leaf is not None, (
        f"Hot leaf 'leaf-3' should always be inside the recency window. Returned ids: {got}"
    )
    assert leaf.get("relationship_type") == "child_session", (
        "Leaf lost its child_session relationship_type; the client keys nesting off this "
        f"field. Got relationship_type={leaf.get('relationship_type')!r}"
    )
    assert leaf.get("parent_session_id") == "orch-frozen", (
        "Leaf lost its parent_session_id pointer to the orchestrator. Got "
        f"{leaf.get('parent_session_id')!r}"
    )
    assert leaf["parent_session_id"] in got, (
        f"Leaf 'leaf-3' points at parent {leaf['parent_session_id']!r}, but that parent row is "
        f"NOT in the payload (ids: {got}). A child row whose parent id does not resolve inside "
        "the same payload is rendered as an orphan top-level sidebar row."
    )


def test_unrelated_rows_are_not_bulk_imported(tmp_path):
    """The rescue must stay surgical: only ancestors of selected subagent rows.

    Guards against the lazy fix of widening the query / dumping the table.
    """
    db = tmp_path / "state.db"
    rows = _hot_leaf_orchestrator_rows(leaf_count=3)
    rows.append(_session("subagent-unrelated", "subagent", T + 60.0,
                         title="Unrelated old subagent"))
    rows.append(_session("cli-unrelated", "cli", T + 40.0, title="Unrelated old CLI"))
    _make_state_db(db, rows)

    result = read_importable_agent_session_rows(
        db, limit=3, exclude_sources=SIDEBAR_EXCLUDES
    )
    got = _ids(result)

    assert "subagent-unrelated" not in got, (
        "An old subagent row that is NOT an ancestor of any selected row was imported. The "
        f"rescue must walk parent_session_id only, not re-admit every subagent row. Ids: {got}"
    )
    assert "cli-unrelated" not in got, (
        "An old unrelated CLI row was imported into a limit=3 window. The rescue must not "
        f"widen the recency window for non-ancestors. Ids: {got}"
    )


def test_missing_parent_does_not_break_listing(tmp_path):
    """A dangling parent_session_id must degrade quietly, not raise.

    Subagent rows can outlive their orchestrator row (pruned/rotated state.db),
    so the ancestor walk has to tolerate an id that resolves to nothing.
    """
    db = tmp_path / "state.db"
    rows = [
        _session("leaf-orphan", "subagent", T + 900.0,
                 parent="orchestrator-that-was-deleted", title="Orphan leaf"),
        _session("cli-live", "cli", T + 800.0, title="Live CLI chat"),
    ]
    _make_state_db(db, rows)

    result = read_importable_agent_session_rows(
        db, limit=3, exclude_sources=SIDEBAR_EXCLUDES
    )
    got = _ids(result)

    assert "leaf-orphan" in got, (
        "A subagent leaf whose parent row does not exist was dropped from the payload. A "
        f"dangling parent_session_id must never hide the child itself. Ids: {got}"
    )
    assert "orchestrator-that-was-deleted" not in got, (
        "A parent id that has no row in sessions must not be synthesised into the payload. "
        f"Ids: {got}"
    )
    assert "cli-live" in got, (
        f"Unrelated visible rows must keep listing normally alongside the orphan. Ids: {got}"
    )


@pytest.mark.timeout(30)
def test_parent_cycle_does_not_hang(tmp_path):
    """a -> b -> a lineage must terminate; the ancestor walk needs cycle protection.

    A naive `while parent_id:` walk over a corrupt/cyclic lineage spins forever
    and hangs the whole /api/sessions request, so this asserts both liveness
    (pytest-timeout) and a small wall-clock bound.
    """
    db = tmp_path / "state.db"
    rows = [
        _session("cycle-a", "subagent", T + 900.0, parent="cycle-b", title="Cycle A"),
        _session("cycle-b", "subagent", T + 890.0, parent="cycle-a", title="Cycle B"),
    ]
    _make_state_db(db, rows)

    started = time.monotonic()
    result = read_importable_agent_session_rows(
        db, limit=1, exclude_sources=SIDEBAR_EXCLUDES
    )
    elapsed = time.monotonic() - started

    assert elapsed < 10.0, (
        f"Listing a cyclic parent_session_id chain took {elapsed:.2f}s; the ancestor walk is "
        "not breaking the cycle and would stall the sidebar request."
    )
    got = _ids(result)
    assert "cycle-a" in got, (
        f"The recency-selected row 'cycle-a' vanished while resolving a cyclic lineage. Ids: {got}"
    )
    assert len(got) == len(set(got)), (
        f"Cyclic lineage produced duplicate rows in the payload: {got}"
    )
    assert len(got) <= 2, (
        f"Cyclic lineage over 2 rows returned {len(got)} rows ({got}); the walk is re-adding "
        "rows it already visited."
    )


def test_webui_ancestor_not_force_imported(tmp_path):
    """Depth-2: webui root -> orchestrator -> leaf. Rescue the orchestrator only.

    Rationale for stopping at the webui ancestor: the WebUI parent chat already
    reaches /api/sessions from its own JSON sidecar, so importing the state.db
    copy of it here would render a duplicate ghost row for the same
    conversation (see ``represented_webui_ids`` in api/routes.py). One hop is
    all the nesting contract needs.
    """
    db = tmp_path / "state.db"
    rows = [
        _session("webui-root", "webui", T + 50.0, title="Planning chat"),
        _session("orch-frozen", "subagent", T + 120.0,
                 parent="webui-root", title="Delegated run"),
    ]
    rows += [
        _session(f"leaf-{i}", "subagent", T + 900.0 + i,
                 parent="orch-frozen", title=f"Subagent leaf {i}")
        for i in range(3)
    ]
    _make_state_db(db, rows)

    # webui is deliberately NOT excluded here, so "webui-root is absent" proves
    # the walk stopped rather than proving the SQL filter removed it.
    result = read_importable_agent_session_rows(
        db, limit=3, exclude_sources=SIDEBAR_EXCLUDES
    )
    got = _ids(result)

    assert "orch-frozen" in got, (
        "The subagent orchestrator was not rescued in a depth-2 lineage "
        f"(webui-root -> orch-frozen -> leaf-*). Returned ids: {got}"
    )
    assert "webui-root" not in got, (
        "The webui ancestor was force-imported from state.db. The WebUI chat already reaches "
        "/api/sessions from its own JSON sidecar, so this creates a duplicate ghost row "
        f"(see represented_webui_ids in api/routes.py). Returned ids: {got}"
    )


def test_no_subagent_rows_leaves_window_unchanged(tmp_path):
    """With no subagent rows at all, the rescue is a strict no-op.

    Pins that the limit itself is not being quietly raised: exactly K rows, in
    unchanged recency order.
    """
    db = tmp_path / "state.db"
    rows = [
        _session(f"cli-{i}", "cli", T + 500.0 + i, title=f"CLI chat {i}")
        for i in range(6)
    ]
    _make_state_db(db, rows)

    result = read_importable_agent_session_rows(
        db, limit=4, exclude_sources=SIDEBAR_EXCLUDES
    )
    got = _ids(result)

    assert len(got) == 4, (
        f"limit=4 over a subagent-free db returned {len(got)} rows ({got}). The ancestor "
        "rescue must not change the window size when there is nothing to rescue."
    )
    assert got == ["cli-5", "cli-4", "cli-3", "cli-2"], (
        f"Recency order changed: expected newest-first ['cli-5','cli-4','cli-3','cli-2'], got {got}. "
        "The rescue must not reorder the normally-selected window."
    )


def test_get_cli_sessions_ships_parent_row_for_visible_subagent_child(
    fake_hermes_home, monkeypatch
):
    """END-TO-END sidebar contract through get_cli_sessions().

    Every visible subagent child must have its parent addressable in the SAME
    payload, unless the parent is a webui chat (delivered by its own sidecar)
    or is absent from state.db entirely. Anything else is an orphan the client
    promotes to a top-level sidebar row.

    Fixture is the real depth-2 delegation shape: webui root -> frozen
    orchestrator -> hot leaves.
    """
    monkeypatch.setattr(models, "CLI_VISIBLE_SESSION_LIMIT", 4)

    db = fake_hermes_home / "state.db"
    rows = [
        _session("webui-root", "webui", T + 50.0, title="Planning chat"),
        _session("orch-frozen", "subagent", T + 120.0,
                 parent="webui-root", title="Delegated run"),
        _session("cli-old", "cli", T + 70.0, title="Old CLI chat"),
    ]
    rows += [
        _session(f"leaf-{i}", "subagent", T + 900.0 + i,
                 parent="orch-frozen", title=f"Subagent leaf {i}")
        for i in range(4)
    ]
    _make_state_db(db, rows)
    source_by_id = {row["id"]: row["source"] for row in rows}

    # include_claude_code=False keeps the payload dependent on the fixture only.
    sessions = models.get_cli_sessions(include_claude_code=False)
    returned_ids = {s["session_id"] for s in sessions}

    subagent_children = [
        s for s in sessions
        if s.get("source_tag") == "subagent" and s.get("parent_session_id")
    ]
    assert subagent_children, (
        "Fixture failure: no visible subagent child rows came back from get_cli_sessions(), "
        f"so the sidebar contract was never exercised. Returned ids: {sorted(returned_ids)}"
    )

    for child in subagent_children:
        parent_id = child["parent_session_id"]
        parent_source = source_by_id.get(parent_id)
        allowed = (
            parent_id in returned_ids            # nestable: parent row shipped
            or parent_source == "webui"          # webui chat ships via its own sidecar
            or parent_source is None             # parent no longer exists in state.db
        )
        assert allowed, (
            f"Subagent session {child['session_id']!r} references parent {parent_id!r} "
            f"(source={parent_source!r}) which is NOT in the /api/sessions payload. "
            f"Allowed exceptions are ONLY: parent source == 'webui', or parent missing from "
            f"state.db. Returned ids: {sorted(returned_ids)}. Without the parent row the client "
            f"cannot nest this child and renders it as a top-level 'Subagent Session'."
        )

    assert "orch-frozen" in returned_ids, (
        "The frozen orchestrator is missing from the end-to-end sidebar payload at "
        f"CLI_VISIBLE_SESSION_LIMIT=4. Returned ids: {sorted(returned_ids)}"
    )
    assert "webui-root" not in returned_ids, (
        "The webui ancestor was force-imported from state.db into the sidebar payload, which "
        f"duplicates the sidecar-backed WebUI chat. Returned ids: {sorted(returned_ids)}"
    )
