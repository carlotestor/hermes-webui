"""delegate_task cards name their work instead of a bare "Delegated a task"."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

UI = (Path(__file__).resolve().parent.parent / "static" / "ui.js").read_text()


def _fn(name):
    start = UI.index(f"function {name}(")
    depth = 0
    for i in range(UI.index("{", start), len(UI)):
        depth += {"{": 1, "}": -1}.get(UI[i], 0)
        if depth == 0:
            return UI[start : i + 1]
    raise AssertionError(name)


def _labels(cards):
    if not shutil.which("node"):
        pytest.skip("node not available")
    names = ["_toolActionKind", "_toolTargetLabel", "_toolVisibleTargetLabel", "_toolActionLabelText", "_shortToolLabel"]
    if "function _toolDelegateTargetLabel(" in UI:
        names.append("_toolDelegateTargetLabel")
    stubs = """
const esc=s=>s; const _toolI18n=(k,f,...a)=>f(...a); const _redactToolTargetLabel=s=>s;
const _decodeToolLabelEntities=s=>s; const _toolDisplayName=tc=>tc.name; const _toolPathBasename=s=>s;
const _toolQueryTitle=s=>s; const _toolReadRangeLabel=()=>'';
"""
    src = stubs + "\n".join(_fn(n) for n in names)
    src += f"\nconsole.log(JSON.stringify({json.dumps(cards)}.map(tc=>_toolActionLabelText(tc,{{limit:112}}))));"
    out = subprocess.run(["node", "-e", src], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def test_delegate_task_cards_name_their_work():
    batch = {"name": "delegate_task", "done": True,
             "snippet": '{"status": "dispatched", "mode": "background", "count": 64, "delegation_id": "d"}',
             "args": {"tasks": "[{'goal': 'hx11 wave1 unit binance: type ts/src/binance.ts', '..."}}
    listing = {"name": "delegate_task", "done": True, "snippet": '{"action": "list", "count": 0}',
               "args": {"tasks": "[{'goal': 'WAVE template', 'context': 'x'}]", "action": "list"}}
    single = {"name": "delegate_task", "done": False, "args": {"goal": "Fix the sidebar leak"}}
    labels = _labels([batch, listing, single])
    assert labels == ["Delegated 64 subagents", "Delegated subagents (list)", "Delegating Fix the sidebar leak"]
    assert not any(re.fullmatch(r"Delegat\w+ a task", label) for label in labels)
