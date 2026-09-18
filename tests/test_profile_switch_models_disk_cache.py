"""Profile switch must not discard the per-profile models disk cache.

``POST /api/profile/switch`` calls ``invalidate_models_cache()`` so the next
``/api/models`` re-resolves the new profile's catalog (#1200). That helper also
deleted the on-disk ``models_cache.<profile>.json`` — a test-isolation
behaviour — which forced a full cold rebuild (live provider ``fetch_models``
HTTPS calls, ~490k deepcopy calls) on every switch even though no source had
changed. The disk cache is already keyed per profile and rejected on read when
``_models_cache_source_fingerprint()`` differs, so deleting it buys no
correctness on the switch path and only costs seconds of latency.

The switch route now uses ``invalidate_models_cache(delete_disk=False)``.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _catalog(label: str) -> dict:
    return {
        "active_provider": "openai",
        "default_model": label,
        "configured_model_badges": {},
        "groups": [
            {
                "provider": "OpenAI",
                "provider_id": "openai",
                "models": [{"id": label, "label": label, "supports_fast_tier": False}],
            }
        ],
        "aliases": {},
    }


def _prime_memory_cache(monkeypatch, catalog: dict):
    import api.config as cfg

    monkeypatch.setattr(cfg, "_available_models_cache", catalog, raising=False)
    monkeypatch.setattr(cfg, "_available_models_cache_ts", 1.0, raising=False)
    monkeypatch.setattr(cfg, "_available_models_cache_source_fingerprint", {"p": "x"}, raising=False)
    monkeypatch.setattr(cfg, "_cache_build_in_progress", False, raising=False)


def test_invalidate_keeps_disk_cache_when_delete_disk_false(tmp_path, monkeypatch):
    import api.config as cfg

    cache_path = tmp_path / "models_cache.profile.json"
    cache_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cfg, "_get_models_cache_path", lambda: cache_path)
    _prime_memory_cache(monkeypatch, _catalog("old-profile-model"))

    cfg.invalidate_models_cache(delete_disk=False)

    # In-memory snapshot dropped so the next request re-resolves the profile...
    assert cfg._available_models_cache is None
    assert cfg._available_models_cache_source_fingerprint is None
    assert cfg._models_cache_provenance is None
    # ...but the fingerprint-guarded disk cache survives.
    assert cache_path.exists(), "delete_disk=False must not unlink the per-profile disk cache"


def test_invalidate_default_still_deletes_disk_cache(tmp_path, monkeypatch):
    """Test-isolation contract is unchanged for every existing caller."""
    import api.config as cfg

    cache_path = tmp_path / "models_cache.profile.json"
    cache_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cfg, "_get_models_cache_path", lambda: cache_path)
    _prime_memory_cache(monkeypatch, _catalog("old-profile-model"))

    cfg.invalidate_models_cache()

    assert cfg._available_models_cache is None
    assert not cache_path.exists()


def test_memory_only_invalidate_serves_next_request_from_disk_without_rebuild(tmp_path, monkeypatch):
    """After a memory-only drop, get_available_models() must reload the disk
    snapshot instead of running the live rebuild."""
    import api.config as cfg

    cache_path = tmp_path / "models_cache.profile.json"
    cache_path.write_text("{}", encoding="utf-8")
    disk_catalog = _catalog("disk-model")
    monkeypatch.setattr(cfg, "_get_models_cache_path", lambda: cache_path)
    monkeypatch.setattr(cfg, "_load_models_cache_from_disk", lambda: disk_catalog if cache_path.exists() else None)
    monkeypatch.setattr(cfg, "_load_stale_models_cache_from_disk", lambda: None)
    monkeypatch.setattr(cfg, "_models_cache_source_fingerprint", lambda: {"profile": "demo"})
    # Pin config mtime tracking so an unrelated config change (or leaked
    # state from an earlier test) cannot force the reload branch.
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("model: {}\n", encoding="utf-8")
    monkeypatch.setattr(cfg, "_get_config_path", lambda: cfg_file)
    monkeypatch.setattr(cfg, "_cfg_path", cfg_file, raising=False)
    monkeypatch.setattr(cfg, "_cfg_mtime", cfg_file.stat().st_mtime, raising=False)

    def _unexpected_rebuild(*_a, **_kw):
        raise AssertionError("profile switch must not trigger a live models rebuild")

    monkeypatch.setattr(cfg, "_invoke_models_rebuild", _unexpected_rebuild)
    _prime_memory_cache(monkeypatch, _catalog("stale-memory-model"))

    cfg.invalidate_models_cache(delete_disk=False)
    result = cfg.get_available_models()

    assert result["default_model"] == "disk-model"


def test_profile_switch_route_preserves_disk_cache():
    """The switch handler must opt out of the disk delete."""
    src = (REPO / "api" / "routes.py").read_text(encoding="utf-8")
    start = src.index('if parsed.path == "/api/profile/switch":')
    end = src.index('if parsed.path == "/api/profile/create":', start)
    block = src[start:end]

    calls = re.findall(r"invalidate_models_cache\(([^)]*)\)", block)
    assert calls, "switch route must still invalidate the in-memory models cache (#1200)"
    for args in calls:
        assert "delete_disk=False" in args, (
            "/api/profile/switch must call invalidate_models_cache(delete_disk=False); "
            "deleting the disk cache forces a multi-second cold rebuild on every switch"
        )
