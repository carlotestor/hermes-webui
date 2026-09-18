"""CLI_VISIBLE_SESSION_LIMIT is overridable via HERMES_WEBUI_VISIBLE_SESSION_LIMIT.

The sidebar recency window also bounds how many delegated subagent children can
render at once, since a child only nests when its row wins a slot in the same
payload. Operators running wide fan-outs need to raise it without editing code.
"""

import importlib


def _reload_limit(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("HERMES_WEBUI_VISIBLE_SESSION_LIMIT", raising=False)
    else:
        monkeypatch.setenv("HERMES_WEBUI_VISIBLE_SESSION_LIMIT", value)
    import api.models as models

    return importlib.reload(models).CLI_VISIBLE_SESSION_LIMIT


def test_defaults_to_20_when_unset(monkeypatch):
    assert _reload_limit(monkeypatch, None) == 20


def test_env_override_raises_the_window(monkeypatch):
    assert _reload_limit(monkeypatch, "64") == 64


def test_invalid_value_falls_back_to_default(monkeypatch):
    assert _reload_limit(monkeypatch, "bogus") == 20


def test_zero_and_negative_fall_back_to_default(monkeypatch):
    assert _reload_limit(monkeypatch, "0") == 20
    assert _reload_limit(monkeypatch, "-5") == 20
