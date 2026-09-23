"""Settlement must keep the agent's per-message reasoning over drifted stream segments.

Adaptive-thinking models skip reasoning on some tool-call steps. The live
segment index then drifts, so segment k can hold step k+1's thinking; the
agent-authored ``reasoning`` field on each message is the source of truth.
"""

import pathlib
import textwrap

from api.streaming import _split_thinking_from_content

REPO = pathlib.Path(__file__).parent.parent


def _settlement_fn():
    src = (REPO / 'api' / 'streaming.py').read_text(encoding='utf-8')
    start = src.find('# #3587: use per-message segments')
    assert start >= 0
    start = src.rfind('\n', 0, start) + 1
    end = src.find('                try:\n                    _turn_duration_seconds', start)
    assert end > start
    body = textwrap.indent(textwrap.dedent(src[start:end]), '    ')
    ns = {'_split_thinking_from_content': _split_thinking_from_content}
    exec('def settle(s, _previous_messages, _reasoning_segments):\n' + body, ns)
    return ns['settle']


class _S:
    def __init__(self, messages):
        self.messages = messages


def _tool_step(call_id, reasoning):
    return {
        'role': 'assistant', 'content': '', 'reasoning': reasoning,
        'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': 'terminal', 'arguments': '{}'}}],
    }


def test_agent_reasoning_wins_over_drifted_segments():
    settle = _settlement_fn()
    prior = [{'role': 'user', 'content': 'q0'}, {'role': 'assistant', 'content': 'a0', 'reasoning': 'old'}]
    messages = prior + [
        {'role': 'user', 'content': 'q1'},
        _tool_step('c1', 'think A'),
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'},
        _tool_step('c2', None),  # adaptive thinking: no reasoning on this step
        {'role': 'tool', 'tool_call_id': 'c2', 'content': 'ok'},
        _tool_step('c3', 'think C'),
        {'role': 'tool', 'tool_call_id': 'c3', 'content': 'ok'},
        {'role': 'assistant', 'content': 'done', 'reasoning': None},
    ]
    # Drifted live segments: the no-thinking step did not advance the index.
    segments = {0: 'think A', 1: 'think C'}
    s = _S(messages)
    settle(s, prior, segments)
    asst = [m for m in s.messages if m.get('role') == 'assistant']
    assert [m.get('reasoning') for m in asst] == ['old', 'think A', None, 'think C', None]


def test_segments_still_fill_messages_without_agent_reasoning_key():
    settle = _settlement_fn()
    messages = [
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c1'}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'},
        {'role': 'assistant', 'content': 'answer'},
    ]
    s = _S(messages)
    settle(s, [], {0: 'seg0', 1: 'seg1'})
    asst = [m for m in s.messages if m.get('role') == 'assistant']
    assert [m.get('reasoning') for m in asst] == ['seg0', 'seg1']


def test_inline_think_still_split_from_content():
    settle = _settlement_fn()
    messages = [
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': '<think>plan</think>\nanswer', 'reasoning': None},
    ]
    s = _S(messages)
    settle(s, [], {})
    assert s.messages[1]['content'] == 'answer'
    assert s.messages[1]['reasoning'] == 'plan'
