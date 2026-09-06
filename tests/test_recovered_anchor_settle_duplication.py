"""Geometric duplication of recovered reasoning-only rows at turn settle.

Refs #7388 (empty ``_recovered_from_run_journal`` rows) and #7416 (linear
journal-replay growth). This is the INDEPENDENT geometric leg: every settled
turn re-appended each empty recovered anchor once more, doubling per turn.
"""
from __future__ import annotations

import copy
import json
import os
import pathlib

import pytest

from api.streaming import (
    _merge_display_messages_after_agent_result,
    _restore_display_reasoning_metadata,
)

STREAM_ID = "34d53038e2d3485cb67cd90ef8408faf"


def _recovered_anchor(reasoning: str, ts: int = 1788356394) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "timestamp": ts,
        "_recovered_from_run_journal": True,
        "_recovered_stream_id": STREAM_ID,
        "reasoning": reasoning,
    }


def _is_clone(message) -> bool:
    return bool(
        isinstance(message, dict)
        and message.get("_recovered_from_run_journal")
        and message.get("role") == "assistant"
        and not (message.get("content") or "")
    )


def _clone_blocks(messages) -> list[int]:
    blocks, run = [], 0
    for message in messages:
        if _is_clone(message):
            run += 1
        elif run:
            blocks.append(run)
            run = 0
    if run:
        blocks.append(run)
    return blocks


def _compacted_history() -> tuple[list[dict], list[dict]]:
    """Display transcript with a recovered anchor + a COMPACTED model context.

    The context no longer contains the turn the anchor belonged to (it was
    compressed away), which is exactly the production shape: the anchor's
    API-safe position is beyond the end of ``result_messages``.
    """
    display = [
        {"role": "user", "content": "reload the webui", "timestamp": 1788356170},
        {"role": "assistant", "content": "Restarting the WebUI unit.", "timestamp": 1788356390},
        _recovered_anchor("The command was killed with SIGTERM"),
        {"role": "user", "content": "did it come back?", "timestamp": 1788357000},
        {"role": "assistant", "content": "Yes, it is up.", "timestamp": 1788357010},
    ]
    context = [
        {"role": "user", "content": "did it come back?", "timestamp": 1788357000},
        {"role": "assistant", "content": "Yes, it is up.", "timestamp": 1788357010},
    ]
    return display, context


def _settle_turn(display, context, turn: int):
    msg_text = f"follow-up {turn}"
    ts = 1788440000 + turn * 10
    result = copy.deepcopy(context) + [
        {"role": "user", "content": msg_text, "timestamp": ts},
        {"role": "assistant", "content": f"reply {turn}", "timestamp": ts + 5},
    ]
    restored = _restore_display_reasoning_metadata(copy.deepcopy(display), copy.deepcopy(result))
    merged = _merge_display_messages_after_agent_result(
        copy.deepcopy(display), copy.deepcopy(context), restored, msg_text, source="webui",
    )
    return merged, result


def test_restore_display_reasoning_does_not_append_unaligned_rows_past_result_end():
    display, context = _compacted_history()
    result = copy.deepcopy(context) + [
        {"role": "user", "content": "follow-up 1"},
        {"role": "assistant", "content": "reply 1"},
    ]

    restored = _restore_display_reasoning_metadata(display, copy.deepcopy(result))

    # The anchor's API-safe slot (2) is beyond the compacted result's own
    # rows, so nothing may be appended after the new reply.
    assert [m.get("content") for m in restored] == [m.get("content") for m in result]
    assert not any(_is_clone(m) for m in restored)


def test_settled_turns_do_not_double_recovered_anchors():
    display, context = _compacted_history()
    assert sum(_is_clone(m) for m in display) == 1
    for turn in range(1, 5):
        display, context = _settle_turn(display, context, turn)
        assert sum(_is_clone(m) for m in display) == 1, (
            f"turn {turn}: clone blocks {_clone_blocks(display)}"
        )
    # Original anchor stays where it was, right after its own reply.
    assert _is_clone(display[2])
    assert display[-1]["content"] == "reply 4"


def test_reasoning_only_row_still_restored_when_its_successor_is_present():
    previous = [
        {"role": "user", "content": "old turn", "timestamp": 1},
        {"role": "assistant", "content": "", "timestamp": 2, "reasoning": "visible thinking card"},
        {"role": "assistant", "content": "old answer", "timestamp": 3},
    ]
    updated = [
        {"role": "user", "content": "old turn"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new turn"},
        {"role": "assistant", "content": "new answer"},
    ]

    restored = _restore_display_reasoning_metadata(previous, updated)

    assert [m.get("content") for m in restored] == [
        "old turn", "", "old answer", "new turn", "new answer",
    ]
    assert restored[1]["reasoning"] == "visible thinking card"


_REAL_SIDECAR = os.environ.get("HERMES_WEBUI_REAL_CLONE_SIDECAR", "")


@pytest.mark.skipif(
    not _REAL_SIDECAR or not pathlib.Path(_REAL_SIDECAR).is_file(),
    reason="set HERMES_WEBUI_REAL_CLONE_SIDECAR to a captured sidecar to run",
)
def test_real_captured_sidecar_does_not_double_per_settled_turn():
    data = json.loads(pathlib.Path(_REAL_SIDECAR).read_text())
    display = copy.deepcopy(data["messages"])
    context = copy.deepcopy(data.get("context_messages") or [])
    before = sum(_is_clone(m) for m in display)
    assert before > 0
    for turn in range(1, 3):
        display, context = _settle_turn(display, context, turn)
        assert sum(_is_clone(m) for m in display) == before, _clone_blocks(display)
