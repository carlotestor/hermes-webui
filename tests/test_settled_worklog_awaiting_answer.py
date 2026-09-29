"""A delegated subagent transcript shows its thinking and tool cards between messages.

Delegated subagent sessions are loaded read-only from state.db with no WebUI
stream attached, and the whole task is one long turn. Its settled worklogs are
therefore open and split at each visible interim text, so thinking stays in place
between the messages instead of being folded into one "Processed" block. Every
other session keeps one collapsed worklog per turn.
"""

from __future__ import annotations

import json
import sqlite3
import textwrap

import pytest

from tests.test_anchor_fallback_ownership import _render_messages_harness, _run_node_script
from tests import test_5307_subagent_child_transcript as _t5307

_make_state_db = _t5307._make_state_db
routes_module = _t5307.routes_module
isolated_state_db = _t5307.isolated_state_db

USER = {"role": "user", "content": "go"}
CALL = {
    "role": "assistant",
    "content": "",
    "reasoning": "r",
    "tool_calls": [{"id": "t1", "function": {"name": "terminal", "arguments": "{}"}}],
}
RESULT = {"role": "tool", "tool_call_id": "t1", "content": "{}"}
ANSWER = {"role": "assistant", "content": "done", "reasoning": "r"}

RUNNING_SUBAGENT = {
    "session_id": "child-1",
    "parent_session_id": "parent-1",
    "relationship_type": "child_session",
    "source_tag": "subagent",
    "raw_source": "subagent",
    "read_only": True,
    "is_cli_session": False,
    "active": True,
}
ENDED_SUBAGENT = {**RUNNING_SUBAGENT, "active": False}
WEBUI_SESSION = {"session_id": "webui-1", "source_tag": "webui", "raw_source": "webui"}
ENDED_FOREIGN_CLI = {
    "session_id": "cli-1",
    "source_tag": "cli",
    "raw_source": "cli",
    "is_cli_session": True,
    "read_only": True,
    "end_reason": "user_cancelled",
}

TWO_TURNS_RUNNING = [USER, CALL, RESULT, ANSWER, USER, CALL, RESULT]


def _collapsed_flags(session, messages, *, busy=False):
    """Render through the real renderMessages() and return each worklog's collapsed flag."""
    script = textwrap.dedent(
        f"""{_render_messages_harness()}
        S = {{
          session: {json.dumps(session)},
          messages: {json.dumps(messages)},
          toolCalls: [],
          busy: {json.dumps(busy)},
        }};
        renderMessages();
        console.log(JSON.stringify(
          elements.msgInner.querySelectorAll('.tool-worklog-group')
            .map((g) => g.getAttribute('data-collapsed') === 'true')
        ));
        """
    )
    return json.loads(_run_node_script(script))


def _call(i, text=""):
    return {**CALL, "content": text, "tool_calls": [{**CALL["tool_calls"][0], "id": f"t{i}"}]}


def _result(i):
    return {**RESULT, "tool_call_id": f"t{i}"}


INTERLEAVED = [
    USER, _call(1), _result(1), _call(2, "Now C#."), _result(2),
    _call(3), _result(3), _call(4, "Restoring it."), _result(4), _call(5), _result(5),
]


def _last_turn_layout(session, messages, *, busy=False):
    """Render via renderMessages() and list the last turn's children: G<collapsed> or T<idx>."""
    script = textwrap.dedent(
        f"""{_render_messages_harness()}
        S = {{ session: {json.dumps(session)}, messages: {json.dumps(messages)},
              toolCalls: [], busy: {json.dumps(busy)} }};
        renderMessages();
        const turnEls = elements.msgInner.querySelectorAll('.assistant-turn');
        const blocks = _assistantTurnBlocks(turnEls[turnEls.length - 1]);
        console.log(JSON.stringify(blocks.children.map((el) =>
          el.className.includes('tool-worklog-group')
            ? 'G' + el.getAttribute('data-collapsed')
            : 'T' + el.getAttribute('data-msg-idx'))));
        """
    )
    return json.loads(_run_node_script(script))


@pytest.mark.parametrize("session", [RUNNING_SUBAGENT, ENDED_SUBAGENT])
def test_subagent_worklogs_are_open(session):
    assert _collapsed_flags(session, TWO_TURNS_RUNNING) == [False, False]


def test_subagent_worklog_splits_at_each_interim_text():
    assert _last_turn_layout(RUNNING_SUBAGENT, INTERLEAVED) == [
        "Gfalse", "T1", "Gfalse", "T3", "Gfalse", "T5", "Gfalse", "T7", "Gfalse", "T9",
    ]


def test_webui_session_keeps_one_collapsed_worklog_per_turn():
    assert _collapsed_flags(WEBUI_SESSION, TWO_TURNS_RUNNING) == [True, True]
    assert _last_turn_layout(WEBUI_SESSION, INTERLEAVED)[0] == "Gtrue"
    assert _last_turn_layout(WEBUI_SESSION, INTERLEAVED).count("Gtrue") == 1


def test_foreign_cli_session_stays_collapsed():
    assert _collapsed_flags(ENDED_FOREIGN_CLI, TWO_TURNS_RUNNING) == [True, True]


def test_live_stream_path_is_untouched():
    script = textwrap.dedent(
        f"""{_render_messages_harness()}
        S = {{ session: {json.dumps(RUNNING_SUBAGENT)}, messages: [], toolCalls: [], busy: false }};
        const idle = _isDelegatedSubagentTranscript();
        S.busy = true;
        console.log(JSON.stringify([idle, _isDelegatedSubagentTranscript()]));
        """
    )
    assert json.loads(_run_node_script(script)) == [True, False]


def _seed_child(db, *, ended_at):
    _make_state_db(db, "parent-1", source="cli")
    _make_state_db(db, "child-1", source="subagent", message_count=3)
    conn = sqlite3.connect(str(db))
    conn.execute(
        "UPDATE sessions SET parent_session_id='parent-1', ended_at=? WHERE id='child-1'",
        (ended_at,),
    )
    conn.commit()
    conn.close()


@pytest.mark.parametrize("ended_at, active", [(None, True), (1781024999.0, False)])
def test_synthesized_subagent_carries_lineage_and_state_db_lifecycle(
    routes_module, isolated_state_db, ended_at, active
):
    _seed_child(isolated_state_db["db"], ended_at=ended_at)
    sess, reason = routes_module._claim_or_synthesize_cli_session("child-1")
    assert reason == "not_claimable"
    assert sess.read_only is True
    assert sess.parent_session_id == "parent-1"
    assert sess.relationship_type == "child_session"
    assert sess.active is active


def test_synthesized_foreign_session_has_no_lifecycle_marker(
    routes_module, isolated_state_db
):
    _make_state_db(isolated_state_db["db"], "cli-1", source="claude_code")
    sess, reason = routes_module._claim_or_synthesize_cli_session("cli-1")
    assert reason == "not_claimable"
    assert getattr(sess, "active", None) is None
    assert getattr(sess, "relationship_type", None) is None


def test_get_session_projects_subagent_lineage_and_lifecycle(
    routes_module, isolated_state_db, monkeypatch
):
    from types import SimpleNamespace

    import api.config

    _seed_child(isolated_state_db["db"], ended_at=None)
    monkeypatch.setattr(routes_module, "_session_visible_to_active_profile", lambda *_: True)
    monkeypatch.setattr(routes_module, "_lookup_cli_session_metadata", lambda *_: {})
    monkeypatch.setattr(api.config, "load_settings", lambda: {"api_redact_enabled": False})
    response = {}
    monkeypatch.setattr(
        routes_module, "j", lambda _h, payload, status=200, **_k: response.update(payload=payload)
    )
    routes_module._handle_session_get(
        None, SimpleNamespace(path="/api/session", query="session_id=child-1")
    )
    sess = response["payload"]["session"]
    assert (sess["parent_session_id"], sess["relationship_type"], sess["active"]) == (
        "parent-1",
        "child_session",
        True,
    )
    assert sess["read_only"] is True and sess["source_tag"] == "subagent"
