"""A Gateway run whose owning worker epoch died reports status "unknown" and refuses Stop with
``unknown_execution``. Neither may leave the WebUI session busy forever."""
import io
import json
import urllib.error
from email.message import Message
from unittest.mock import patch

import pytest

import api.gateway_chat as gateway_chat
from tests.test_gateway_run_reattach_after_restart import (  # noqa: F401  (fixture)
    _orphaned_gateway_turn,
    _saved,
    _wait_for_reattach_threads,
    isolated_sessions,
)


def test_reattach_settles_a_run_the_gateway_reports_unknown(  # noqa: F811 (pytest fixture import)
    isolated_sessions, monkeypatch
):
    sid, _stream_id = _orphaned_gateway_turn()
    polls = []

    def unknown(base_url, api_key, run_id):
        polls.append(run_id)
        return {"run_id": run_id, "status": "unknown"}

    monkeypatch.setattr(gateway_chat, "_get_gateway_run_status", unknown)
    assert gateway_chat.resume_gateway_runs_after_restart() == [sid]
    _wait_for_reattach_threads()

    assert len(polls) == 1, "an unknown run must settle, not be polled again"
    saved = _saved(sid)
    assert saved["messages"][-2]["content"] == "long task"
    assert saved["messages"][-1]["_error"] is True
    assert "Gateway lost this run" in saved["messages"][-1]["content"]
    assert saved["active_stream_id"] is None and saved["gateway_run"] is None
    assert saved["pending_user_message"] is None


def _http_error(code, error_code):
    body = json.dumps({"error": {"message": error_code, "code": error_code}}).encode()
    return urllib.error.HTTPError(
        "http://gw:8642/v1/runs/r/stop", code, "conflict", Message(), io.BytesIO(body))


@pytest.mark.parametrize("code, error_code, expected", [
    (409, "unknown_execution", True),
    (409, "session_busy", False),
    (500, "unknown_execution", False),
])
def test_stop_treats_only_unknown_execution_as_nothing_left_to_stop(code, error_code, expected):
    class _Opener:
        def open(self, req, *, timeout=None):
            raise _http_error(code, error_code)

    with patch("urllib.request.build_opener", return_value=_Opener()), \
         patch("api.gateway_chat.gateway_run_endpoint", return_value=("http://gw:8642", "k")):
        assert gateway_chat.stop_gateway_run("run-dead") is expected
