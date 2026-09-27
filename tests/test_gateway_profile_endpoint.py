"""A named-profile session must reach its own profile on a shared Gateway listener, never the root's."""
from collections import OrderedDict
import io

import pytest

import api.gateway_chat as gateway_chat
import api.models as models
import api.streaming as streaming
from api import profiles
from api.config import STREAMS, STREAMS_LOCK, create_stream_channel
from api.models import new_session

SHARED = "http://127.0.0.1:8642"


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    work = root / "profiles" / "work"
    work.mkdir(parents=True)
    (root / ".env").write_text("API_SERVER_KEY=root-key-0123456789\n")
    monkeypatch.setattr(profiles, "_DEFAULT_HERMES_HOME", root)
    monkeypatch.setattr(profiles, "_active_profile", "default")
    monkeypatch.setattr(profiles, "_loaded_profile_env_keys", set())
    monkeypatch.delenv("HERMES_WEBUI_ISOLATED_PROFILE", raising=False)
    monkeypatch.setenv("HERMES_WEBUI_GATEWAY_BASE_URL", SHARED)
    monkeypatch.setenv("HERMES_WEBUI_GATEWAY_API_KEY", "root-key-0123456789")
    return root, work


def test_named_profile_without_own_url_uses_its_prefix_and_key(homes):
    _, work = homes
    (work / ".env").write_text("API_SERVER_KEY=work-key-0123456789\n")

    assert gateway_chat._gateway_endpoint_for_profile("work") == (f"{SHARED}/p/work", "work-key-0123456789")


def test_named_profile_without_key_never_borrows_the_root_key(homes):
    assert gateway_chat._gateway_endpoint_for_profile("work") == (f"{SHARED}/p/work", "")


def test_shared_url_loaded_from_root_env_reaches_named_profile(homes, monkeypatch):
    root, work = homes
    (root / ".env").write_text("HERMES_WEBUI_GATEWAY_BASE_URL=http://shared-gw:7000\nAPI_SERVER_KEY=root-key-0123456789\n")
    (work / ".env").write_text("API_SERVER_KEY=work-key-0123456789\n")
    monkeypatch.setenv("HERMES_WEBUI_GATEWAY_BASE_URL", "http://shared-gw:7000")
    monkeypatch.setattr(profiles, "_loaded_profile_env_keys", {"HERMES_WEBUI_GATEWAY_BASE_URL", "API_SERVER_KEY"})

    assert gateway_chat._gateway_endpoint_for_profile("work") == ("http://shared-gw:7000/p/work", "work-key-0123456789")


@pytest.mark.parametrize("source", ["env", "config"])
def test_named_profile_with_own_url_is_used_verbatim(homes, source):
    _, work = homes
    if source == "env":
        (work / ".env").write_text("HERMES_WEBUI_GATEWAY_BASE_URL=http://work-gw:9000/\nHERMES_WEBUI_GATEWAY_API_KEY=work-key-0123456789\n")
    else:
        (work / "config.yaml").write_text("webui_gateway_base_url: http://work-gw:9000/\n")
        (work / ".env").write_text("HERMES_WEBUI_GATEWAY_API_KEY=work-key-0123456789\n")

    assert gateway_chat._gateway_endpoint_for_profile("work") == ("http://work-gw:9000", "work-key-0123456789")


@pytest.mark.parametrize("name", [None, "", "default"])
def test_root_profile_stays_unprefixed(homes, name):
    assert gateway_chat._gateway_endpoint_for_profile(name) == (SHARED, "root-key-0123456789")


def test_isolated_profile_deployment_stays_unprefixed(homes, monkeypatch):
    monkeypatch.setattr(profiles, "_is_isolated_profile_mode", lambda: True)

    base_url, _ = gateway_chat._gateway_endpoint_for_profile("work")

    assert base_url == SHARED


def test_live_turn_of_named_profile_session_goes_to_its_profile(homes, tmp_path, monkeypatch):
    _, work = homes
    (work / ".env").write_text("API_SERVER_KEY=work-key-0123456789\n")
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", session_dir / "_index.json")
    monkeypatch.setattr(models, "SESSIONS", OrderedDict())
    monkeypatch.setenv("HERMES_WEBUI_CHAT_BACKEND", "gateway")
    monkeypatch.setenv("HERMES_WEBUI_GATEWAY_USE_RUNS_API", "1")
    monkeypatch.setattr(gateway_chat, "_gateway_reasoning_effort_for_request", lambda *a, **k: None)
    monkeypatch.setattr(streaming, "_load_webui_prefill_context", lambda cfg: {
        "status": "not_configured", "source": "none", "label": "", "message_count": 0, "messages": [],
    })
    monkeypatch.setattr(streaming, "_prefill_messages_with_webui_context", lambda ctx, cfg: [])
    s = new_session()
    s.profile = "work"
    stream_id = "stream-work"
    s.active_stream_id = stream_id
    s.pending_user_message = "hi"
    s.pending_attachments = []
    s.pending_started_at = 1.0
    s.save()
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append((req.get_method(), req.full_url, req.get_header("Authorization")))
        if req.get_method() == "POST":
            return io.BytesIO(b'{"run_id":"run_work"}')
        return io.BytesIO(
            b'data: {"event":"run.completed","output":"done"}\n'
            b"data: [DONE]\n"
        )

    monkeypatch.setattr(gateway_chat, "gateway_supports_approval", lambda *a, **k: True)
    monkeypatch.setattr(gateway_chat.urllib.request, "urlopen", fake_urlopen)
    with STREAMS_LOCK:
        STREAMS[stream_id] = create_stream_channel()

    gateway_chat._run_gateway_chat_streaming(s.session_id, "hi", "test-model", "/tmp", stream_id, [])

    assert seen and all(url.startswith(f"{SHARED}/p/work/v1/runs") for _, url, _ in seen), seen
    assert {auth for _, _, auth in seen} == {"Bearer work-key-0123456789"}
