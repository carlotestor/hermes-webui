"""A still-running transcript read without a live stream keeps its worklog open.

Delegated subagent sessions are loaded read-only from state.db with no
WebUI stream attached. While the subagent is still working, its last turn
ends in tool activity and has no final answer. The settled renderer used to
collapse that worklog unconditionally, which hid every thinking and tool card
behind a bare "Processed" chip.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UI_JS = (ROOT / "static" / "ui.js").read_text(encoding="utf-8")


def _extract_function(name: str) -> str:
    start = UI_JS.index(f"function {name}(")
    depth = 0
    for i in range(UI_JS.index("{", start), len(UI_JS)):
        if UI_JS[i] == "{":
            depth += 1
        elif UI_JS[i] == "}":
            depth -= 1
            if depth == 0:
                return UI_JS[start : i + 1]
    raise AssertionError(f"unterminated {name}")


def _run(messages, *, busy=False, anchor_is_last=True):
    node = shutil.which("node")
    if not node:
        pytest.skip("node executable is required")
    script = (
        "const turnA={id:'a'}, turnB={id:'b'};\n"
        "const inner={querySelectorAll:()=>[turnA,turnB]};\n"
        f"const S={{busy:{json.dumps(busy)},messages:{json.dumps(messages)}}};\n"
        + _extract_function("_settledTurnAwaitingAnswer")
        + f"\nprocess.stdout.write(String(_settledTurnAwaitingAnswer(inner,{'turnB' if anchor_is_last else 'turnA'})));\n"
    )
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10)
    assert out.returncode == 0, out.stderr
    return out.stdout == "true"


USER = {"role": "user", "content": "go"}
CALL = {"role": "assistant", "content": "", "reasoning": "r", "tool_calls": [{"id": "t1"}]}
RESULT = {"role": "tool", "tool_call_id": "t1", "content": "{}"}
ANSWER = {"role": "assistant", "content": "done", "reasoning": "r"}


def test_running_transcript_ending_in_tool_result_stays_open():
    assert _run([USER, CALL, RESULT]) is True


def test_running_transcript_ending_in_tool_call_stays_open():
    assert _run([USER, CALL, RESULT, CALL]) is True


def test_answered_turn_still_collapses():
    assert _run([USER, CALL, RESULT, ANSWER]) is False


def test_earlier_turns_still_collapse():
    assert _run([USER, CALL, RESULT], anchor_is_last=False) is False


def test_live_stream_path_is_untouched():
    assert _run([USER, CALL, RESULT], busy=True) is False


def test_settled_worklog_builder_uses_the_guard():
    block = UI_JS[UI_JS.index("const activityKey=`assistant:${aIdx}`;") :][:400]
    assert re.search(r"collapsed:!_settledTurnAwaitingAnswer\(inner,anchorTurn\)", block)
