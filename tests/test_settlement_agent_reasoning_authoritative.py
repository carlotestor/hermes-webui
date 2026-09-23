"""Settlement must keep the agent's per-message reasoning over drifted stream segments.

Adaptive-thinking models skip reasoning on some tool-call steps. The live
segment index then drifts, so segment k can hold step k+1's thinking; the
agent-authored ``reasoning`` field on each message is the source of truth.
"""

from types import SimpleNamespace

from api.streaming import _settle_turn_reasoning


def _tool_step(call_id, reasoning):
    return {
        'role': 'assistant', 'content': '', 'reasoning': reasoning,
        'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': 'terminal', 'arguments': '{}'}}],
    }


def _reasonings(s):
    return [m.get('reasoning') for m in s.messages if m.get('role') == 'assistant']


def test_agent_reasoning_wins_over_drifted_segments():
    prior = [{'role': 'user', 'content': 'q0'}, {'role': 'assistant', 'content': 'a0', 'reasoning': 'old'}]
    s = SimpleNamespace(messages=prior + [
        {'role': 'user', 'content': 'q1'},
        _tool_step('c1', 'think A'),
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'},
        _tool_step('c2', None),  # adaptive thinking: no reasoning on this step
        {'role': 'tool', 'tool_call_id': 'c2', 'content': 'ok'},
        _tool_step('c3', 'think C'),
        {'role': 'tool', 'tool_call_id': 'c3', 'content': 'ok'},
        {'role': 'assistant', 'content': 'done', 'reasoning': None},
    ])
    # Drifted live segments: the no-thinking step did not advance the index.
    _settle_turn_reasoning(s, prior, {0: 'think A', 1: 'think C'})
    assert _reasonings(s) == ['old', 'think A', None, 'think C', None]


def test_explicit_none_reasoning_is_not_backfilled_from_segment():
    s = SimpleNamespace(messages=[
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': 'answer', 'reasoning': None},
    ])
    _settle_turn_reasoning(s, [], {0: 'belongs to another step'})
    assert _reasonings(s) == [None]


def test_segments_still_fill_messages_without_agent_reasoning_key():
    s = SimpleNamespace(messages=[
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c1'}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'},
        {'role': 'assistant', 'content': 'answer'},
    ])
    _settle_turn_reasoning(s, [], {0: 'seg0', 1: 'seg1'})
    assert _reasonings(s) == ['seg0', 'seg1']


def test_inline_think_still_split_from_content():
    s = SimpleNamespace(messages=[
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': '<think>plan</think>\nanswer', 'reasoning': None},
    ])
    _settle_turn_reasoning(s, [], {})
    assert s.messages[1]['content'] == 'answer'
    assert s.messages[1]['reasoning'] == 'plan'
