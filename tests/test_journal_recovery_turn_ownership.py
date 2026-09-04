"""Turn-ownership gates for run-journal recovery dedupe (#7388 residuals).

1: window picks the current turn, indexed against the ORIGINAL messages array.
2: with a stream and no window, only a same-stream card may match.
3: the tool oracle must compare an actually-equal normalized name+preview."""
from __future__ import annotations

import pytest

import api.models as models
import api.profiles as profiles
import api.run_journal as run_journal
from api.models import (
    Session,
    _TURN_WINDOW_AMBIGUOUS,
    _append_journaled_partial_output,
    _find_existing_assistant_for_journal_content,
    _journal_tool_already_present,
    _normalize_journal_recovery_text,
    _pending_recovery_turn_window_start,
)
from api.process_event_utils import build_active_turn_token
from api.run_journal import append_run_event

STREAM_ID = "stream-current"
OTHER_STREAM = "stream-foreign"
PROMPT = "run the report"
ANSWER = "The report finished with 3 warnings and no errors."
TOOL_NAME = "terminal"
TOOL_PREVIEW = "ls"


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolate HERMES_HOME *and* the run-journal root.

    ``run_journal`` resolves its root via the session dir, so patching only the
    env var lets ``append_run_event`` write into the real ``~/.hermes``.
    """
    home = tmp_path / "hermes_home"
    home.mkdir()
    sessions = home / "sessions"
    sessions.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(profiles, "_DEFAULT_HERMES_HOME", home)
    monkeypatch.setattr(run_journal, "_default_session_dir", lambda: sessions)
    monkeypatch.setattr(models, "SESSION_DIR", sessions)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", sessions / "_index.json")
    return home


def _user(content, ts, **extra):
    row = {"role": "user", "content": content, "timestamp": ts}
    row.update(extra)
    return row


def _assistant(content, ts, **extra):
    row = {"role": "assistant", "content": content, "timestamp": ts}
    row.update(extra)
    return row


def _session(sid, messages, *, pending_started_at, tool_calls=None, active_stream_id=None):
    return Session(
        session_id=sid,
        title="gate",
        messages=list(messages),
        tool_calls=list(tool_calls or []),
        pending_user_message=PROMPT,
        pending_started_at=pending_started_at,
        pending_user_source="webui",
        pending_attachments=[],
        active_stream_id=active_stream_id,
    )


# ---------------------------------------------------------------------------
# Gate 1 — turn window selection
# ---------------------------------------------------------------------------


def test_recovered_echo_and_real_submission_open_the_same_turn(hermes_home):
    """Two checkpoint-identical user rows in one turn open at the EARLIEST.

    The repair path appends a ``_recovered`` echo, so one turn shows both rows;
    the core output between them must stay claimable (#3929).
    """
    ts = 1_700_000_000
    token = build_active_turn_token(STREAM_ID, ts)
    messages = [
        _user("an earlier, different prompt", ts - 500),   # 0
        _assistant(ANSWER, ts - 499),                      # 1 historical answer
        _user(PROMPT, ts, _active_turn_token=token),       # 2 real submission
        _assistant("core output for this turn", ts + 1),   # 3 current-turn core
        _user(PROMPT, ts, _recovered=True, _active_turn_token=token),  # 4 echo
    ]
    session = _session(
        "gate1-echo", messages, pending_started_at=ts, active_stream_id=STREAM_ID,
    )

    window = _pending_recovery_turn_window_start(session)

    assert window == 2, (
        f"window opened at {window}; both checkpoint rows belong to the current "
        "turn, so it must open at the earlier one (index 2) to keep this turn's "
        "core row at index 3 claimable"
    )
    # The historical answer at index 1 sits BEFORE the window and is a
    # different turn's output — it must not be claimable.
    assert (
        _find_existing_assistant_for_journal_content(
            session, ANSWER, stream_id=STREAM_ID, turn_start=window
        )
        is None
    ), "historical answer was claimed by the current turn — cross-turn data loss"


def test_historical_turn_with_different_checkpoint_never_wins_window(hermes_home):
    """An earlier turn repeating the prompt TEXT must not open the window.

    Only the text-only fallback could match it, and that runs solely when no
    exact checkpoint exists — here the current turn has one.
    """
    ts = 1_700_000_050
    messages = [
        _user(PROMPT, ts - 500),        # historical: same text, different ts
        _assistant(ANSWER, ts - 499),
        _user(PROMPT, ts),              # current turn: exact checkpoint
    ]
    session = _session("gate1-textdupe", messages, pending_started_at=ts)

    window = _pending_recovery_turn_window_start(session)

    assert window == 2, (
        f"window opened at {window}; the exact checkpoint at index 2 must beat "
        "the earlier text-only repeat at index 0"
    )
    assert (
        _find_existing_assistant_for_journal_content(
            session, ANSWER, stream_id=STREAM_ID, turn_start=window
        )
        is None
    ), "the earlier turn's answer must not be claimable"


def test_window_index_is_a_coordinate_in_original_messages(hermes_home):
    """A non-dict row before the checkpoint must not shift the window index.

    Compacting before ``enumerate()`` returns a compacted-list coordinate while
    ownership compares against the original array.
    """
    ts = 1_700_000_100
    messages = [
        None,                                   # 0 malformed/legacy row
        _user(PROMPT, ts - 50),                 # 1 historical submission
        _assistant(ANSWER, ts - 49),            # 2 historical answer
        _user(PROMPT, ts),                      # 3 current turn (exact ts)
    ]
    session = _session("gate1-nondict", messages, pending_started_at=ts)

    window = _pending_recovery_turn_window_start(session)

    assert window == 3, (
        f"window index {window} is a compacted-list coordinate; it must address "
        "the current checkpoint at index 3 of the original session.messages"
    )
    assert (
        _find_existing_assistant_for_journal_content(
            session, ANSWER, stream_id=STREAM_ID, turn_start=window
        )
        is None
    ), "index shift admitted the historical answer at index 2"


def test_ambiguous_boundary_fails_closed_toward_appending(hermes_home):
    """A pending turn with no matching row reports an AMBIGUOUS boundary.

    Distinct from a quiescent session (``None``), where legacy session-wide
    matching is still correct.
    """
    ts = 1_700_000_200
    messages = [
        _user("a completely different prompt", ts - 10),
        _assistant(ANSWER, ts - 9),
    ]
    session = _session("gate1-ambiguous", messages, pending_started_at=ts)

    window = _pending_recovery_turn_window_start(session)
    assert window is _TURN_WINDOW_AMBIGUOUS, (
        "a pending turn with no matching row must report an AMBIGUOUS boundary, "
        "not None — None means 'quiescent' and re-enables session-wide matching"
    )

    assert (
        _find_existing_assistant_for_journal_content(
            session, ANSWER, stream_id=STREAM_ID, turn_start=window
        )
        is None
    ), "an unprovable boundary must not claim a historical assistant row"


def test_quiescent_session_keeps_legacy_session_wide_match(hermes_home):
    """No pending turn: there is no current-turn boundary to protect.

    Legacy session-wide matching must survive here or the #3929
    reasoning-backfill contract breaks on pre-stream-id transcripts.
    """
    session = Session(
        session_id="gate1-quiescent",
        title="gate",
        messages=[
            _user("earlier", 1_700_000_250),
            _assistant(ANSWER, 1_700_000_251),
        ],
    )

    assert _pending_recovery_turn_window_start(session) is None
    assert (
        _find_existing_assistant_for_journal_content(session, ANSWER) == 1
    ), "a quiescent session must keep the pre-fix session-wide match"


# ---------------------------------------------------------------------------
# Gate 2 — tool ownership
# ---------------------------------------------------------------------------


def test_untagged_historical_tool_does_not_suppress_with_pending_turn(
    hermes_home,
):
    """Pending turn + unlocatable boundary: an untagged card must NOT match.

    Ownership is keyed on the WINDOW; nothing may be claimed by position.
    """
    session = _session(
        "gate2-untagged",
        [_user("unrelated", 1_700_000_300)],
        pending_started_at=1_700_000_300,
        tool_calls=[
            {
                "name": TOOL_NAME,
                "preview": TOOL_PREVIEW,
                "snippet": TOOL_PREVIEW,
                "assistant_msg_idx": 0,
            }
        ],
    )
    window = _pending_recovery_turn_window_start(session)
    assert window is _TURN_WINDOW_AMBIGUOUS

    assert not _journal_tool_already_present(
        session, TOOL_NAME, TOOL_PREVIEW, stream_id=STREAM_ID, turn_start=window
    ), (
        "an untagged historical tool card suppressed the current turn's card "
        "even though no turn window proved ownership"
    )


def test_same_stream_tagged_tool_still_matches(hermes_home):
    """Same-stream idempotency control: a card tagged with OUR stream matches."""
    session = _session(
        "gate2-same-stream",
        [_user(PROMPT, 1_700_000_400)],
        pending_started_at=1_700_000_400,
        tool_calls=[
            {
                "name": TOOL_NAME,
                "preview": TOOL_PREVIEW,
                "snippet": TOOL_PREVIEW,
                "_recovered_stream_id": STREAM_ID,
                "assistant_msg_idx": 0,
            }
        ],
    )

    assert _journal_tool_already_present(
        session, TOOL_NAME, TOOL_PREVIEW, stream_id=STREAM_ID, turn_start=None
    ), "same-stream replay must collapse onto its own earlier recovered card"


def test_foreign_stream_tagged_tool_never_matches(hermes_home):
    """A card tagged with a DIFFERENT stream belongs to another turn."""
    session = _session(
        "gate2-foreign",
        [_user(PROMPT, 1_700_000_500)],
        pending_started_at=1_700_000_500,
        tool_calls=[
            {
                "name": TOOL_NAME,
                "preview": TOOL_PREVIEW,
                "snippet": TOOL_PREVIEW,
                "_recovered_stream_id": OTHER_STREAM,
                "assistant_msg_idx": 0,
            }
        ],
    )

    assert not _journal_tool_already_present(
        session, TOOL_NAME, TOOL_PREVIEW, stream_id=STREAM_ID, turn_start=0
    ), "a foreign-stream card must never suppress this stream's tool"


def test_untagged_tool_inside_current_window_matches(hermes_home):
    """Untagged card whose owner sits inside the window IS current-turn output."""
    session = _session(
        "gate2-in-window",
        [_user(PROMPT, 1_700_000_600), _assistant("", 1_700_000_601)],
        pending_started_at=1_700_000_600,
        tool_calls=[
            {
                "name": TOOL_NAME,
                "preview": TOOL_PREVIEW,
                "snippet": TOOL_PREVIEW,
                "assistant_msg_idx": 1,
            }
        ],
    )

    assert _journal_tool_already_present(
        session, TOOL_NAME, TOOL_PREVIEW, stream_id=STREAM_ID, turn_start=0
    ), "a card owned by a row inside the window is the current turn's own card"


def test_legacy_session_wide_match_only_without_both_inputs(hermes_home):
    """Both ownership inputs absent -> documented pre-fix session-wide behavior."""
    session = _session(
        "gate2-legacy",
        [_user(PROMPT, 1_700_000_700)],
        pending_started_at=1_700_000_700,
        tool_calls=[
            {
                "name": TOOL_NAME,
                "preview": TOOL_PREVIEW,
                "snippet": TOOL_PREVIEW,
                "assistant_msg_idx": 0,
            }
        ],
    )

    assert _journal_tool_already_present(
        session, TOOL_NAME, TOOL_PREVIEW, stream_id=None, turn_start=None
    ), "legacy callers must keep the pre-fix session-wide match"


# ---------------------------------------------------------------------------
# Gate 3 — the oracle must compare an actually-equal normalized preview
# ---------------------------------------------------------------------------


def _write_tool_journal(sid, stream_id, *, preview=TOOL_PREVIEW):
    """Emit a tool pair whose normalized preview EQUALS the historical card's.

    The prior oracle emitted ``args={"cmd": "ls"}`` with no ``preview``, so the
    signatures were unequal and the assertion could not fail.
    """
    append_run_event(sid, stream_id, "tool", {"name": TOOL_NAME, "preview": preview})
    append_run_event(
        sid, stream_id, "tool_complete", {"name": TOOL_NAME, "preview": preview}
    )


def test_oracle_preview_is_actually_equal(hermes_home):
    """Guard the oracle itself: the emitted preview must match the card."""
    sid = "gate3-oracle"
    _write_tool_journal(sid, STREAM_ID)
    session = _session(
        sid,
        [_user(PROMPT, 1_700_000_800)],
        pending_started_at=1_700_000_800,
        tool_calls=[
            {
                "name": TOOL_NAME,
                "preview": TOOL_PREVIEW,
                "snippet": TOOL_PREVIEW,
                "assistant_msg_idx": 0,
                "_recovered_stream_id": STREAM_ID,
            }
        ],
    )
    assert _journal_tool_already_present(
        session, TOOL_NAME, TOOL_PREVIEW, stream_id=STREAM_ID, turn_start=None
    ), "fixture preview must be signature-equal or the repeated-tool gate is vacuous"


def test_repeated_untagged_tool_survives_then_replay_is_idempotent(hermes_home):
    """Two turns each keep their own card; replaying the stream adds no third."""
    sid = "gate3-two-turns"
    _write_tool_journal(sid, STREAM_ID)
    ts = 1_700_000_900
    session = _session(
        sid,
        [
            _user(PROMPT, ts - 100),
            _assistant("earlier answer", ts - 99),
            _user(PROMPT, ts),
        ],
        pending_started_at=ts,
        tool_calls=[
            {
                "name": TOOL_NAME,
                "preview": TOOL_PREVIEW,
                "snippet": TOOL_PREVIEW,
                "assistant_msg_idx": 1,
            }
        ],
    )

    _append_journaled_partial_output(session, STREAM_ID, dedupe_existing=True)
    after_first = [
        tc for tc in session.tool_calls
        if tc.get("name") == TOOL_NAME
    ]
    assert len(after_first) == 2, (
        f"expected the historical card plus the current turn's card, got "
        f"{len(after_first)} — the historical card suppressed current output"
    )

    for _ in range(3):
        _append_journaled_partial_output(session, STREAM_ID, dedupe_existing=True)
    after_replay = [
        tc for tc in session.tool_calls
        if tc.get("name") == TOOL_NAME
    ]
    assert len(after_replay) == 2, (
        f"replaying the same stream grew tool cards to {len(after_replay)}; "
        "same-stream recovery must be idempotent"
    )


def test_repeated_answer_across_turns_survives(hermes_home):
    """An identical answer in an earlier turn must not suppress the current one."""
    sid = "gate3-answer"
    append_run_event(sid, STREAM_ID, "token", {"text": ANSWER})
    ts = 1_700_001_000
    session = _session(
        sid,
        [
            _user(PROMPT, ts - 100),
            _assistant(ANSWER, ts - 99),
            _user(PROMPT, ts),
        ],
        pending_started_at=ts,
    )

    _append_journaled_partial_output(session, STREAM_ID, dedupe_existing=True)

    answers = [
        m for m in session.messages
        if m.get("role") == "assistant"
        and str(m.get("content") or "").strip() == ANSWER
    ]
    assert len(answers) == 2, (
        f"expected the historical answer plus the recovered current answer, got "
        f"{len(answers)} — the earlier turn suppressed current recovery output"
    )

    for _ in range(3):
        _append_journaled_partial_output(session, STREAM_ID, dedupe_existing=True)
    answers_after = [
        m for m in session.messages
        if m.get("role") == "assistant"
        and str(m.get("content") or "").strip() == ANSWER
    ]
    assert len(answers_after) == 2, (
        f"replay grew answers to {len(answers_after)}; recovery must be idempotent"
    )


def test_same_second_historical_real_and_current_echo_does_not_claim_history(
    hermes_home,
):
    """Same-second historical real + current echo must not claim history."""
    sid = "gate4-collision"
    append_run_event(sid, STREAM_ID, "token", {"text": ANSWER})
    _write_tool_journal(sid, STREAM_ID)
    ts = 1_700_001_100
    session = _session(
        sid,
        [
            _user(PROMPT, ts),
            _assistant(ANSWER, ts),
            _user(PROMPT, ts, _recovered=True),
        ],
        pending_started_at=ts,
        tool_calls=[
            {
                "name": TOOL_NAME,
                "preview": TOOL_PREVIEW,
                "snippet": TOOL_PREVIEW,
                "assistant_msg_idx": 1,
            }
        ],
    )
    assert _normalize_journal_recovery_text(TOOL_PREVIEW) == (
        _normalize_journal_recovery_text(session.tool_calls[0].get("preview"))
    ), "fixture preview must be signature-equal or this gate is vacuous"

    window = _pending_recovery_turn_window_start(session)
    assert window is _TURN_WINDOW_AMBIGUOUS, (
        f"window opened at {window}; colliding exact checkpoints must fail closed"
    )

    _append_journaled_partial_output(session, STREAM_ID, dedupe_existing=True)
    answers = [
        m for m in session.messages
        if m.get("role") == "assistant"
        and str(m.get("content") or "").strip() == ANSWER
    ]
    tools = [tc for tc in session.tool_calls if tc.get("name") == TOOL_NAME]
    assert len(answers) == 2, (
        f"expected historical plus recovered current answer, got {len(answers)}"
    )
    assert len(tools) == 2, (
        f"expected historical plus recovered current tool, got {len(tools)}"
    )

    for _ in range(3):
        _append_journaled_partial_output(session, STREAM_ID, dedupe_existing=True)
    answers_after = [
        m for m in session.messages
        if m.get("role") == "assistant"
        and str(m.get("content") or "").strip() == ANSWER
    ]
    tools_after = [tc for tc in session.tool_calls if tc.get("name") == TOOL_NAME]
    assert len(answers_after) == 2, (
        f"replay grew answers to {len(answers_after)}; recovery must be idempotent"
    )
    assert len(tools_after) == 2, (
        f"replay grew tools to {len(tools_after)}; recovery must be idempotent"
    )


def test_tool_only_dedupe_does_not_allocate_blank_anchor(hermes_home):
    """A current-window untagged tool must not grow messages on replay."""
    sid = "gate4-tool-anchor"
    _write_tool_journal(sid, STREAM_ID)
    ts = 1_700_001_200
    session = _session(
        sid,
        [_user(PROMPT, ts), _assistant("", ts + 1)],
        pending_started_at=ts,
        tool_calls=[
            {
                "name": TOOL_NAME,
                "preview": TOOL_PREVIEW,
                "snippet": TOOL_PREVIEW,
                "assistant_msg_idx": 1,
            }
        ],
    )
    assert _normalize_journal_recovery_text(TOOL_PREVIEW) == (
        _normalize_journal_recovery_text(session.tool_calls[0].get("preview"))
    ), "fixture preview must be signature-equal or this gate is vacuous"

    before_messages = len(session.messages)
    before_tools = len(session.tool_calls)
    for _ in range(4):
        _append_journaled_partial_output(session, STREAM_ID, dedupe_existing=True)
    assert len(session.tool_calls) == before_tools, (
        f"tool cards grew {before_tools} -> {len(session.tool_calls)}"
    )
    assert len(session.messages) == before_messages, (
        f"messages grew {before_messages} -> {len(session.messages)}; "
        "deduped tool recovery must not allocate a blank assistant anchor"
    )
