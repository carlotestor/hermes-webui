# Models catalog cache invalidation contract (`/api/models`)

This document records the current identity and invalidation contract for the
`/api/models` catalog cache in `api/config.py`. It describes shipped behavior and
changes no runtime behavior. It was added after #7556 shipped in `exp-v0.52.303`,
which the #7556 review flagged as an undocumented runtime contract.

## What is cached

- **In memory:** `_available_models_cache` plus `_available_models_cache_ts`,
  with `_AVAILABLE_MODELS_CACHE_TTL` set to 24 hours.
- **On disk:** one `models_cache.json` per profile under the WebUI state
  directory (`_get_models_cache_path()`), stamped with `_schema_version` and
  `_webui_version`.
- **Cold path:** `get_available_models(prefer_cache=...)`; the `prefer_cache`
  branch never starts a live provider rebuild.
- **Hot path:** `_endpoint_advertised_model_ids()` reads only the published
  in-memory snapshot through the lock-free `_models_cache_provenance` tuple and
  validates it against the current source fingerprint before trusting it.
  `_sync_models_cache_provenance()` must run at every site that publishes or
  invalidates the snapshot, so the tuple can never tear.

## The three source axes

`_models_cache_source_fingerprint()` is the single chokepoint. A cache is served
only when every axis matches the value recorded in the cache — the disk reader
compares it in `_is_loadable_disk_cache()`, and the hot path compares it against
the fingerprint captured at publish time.

| Axis | Identity | Why it is fingerprinted this way |
| --- | --- | --- |
| `config_yaml` | stat identity: `mtime_ns` + size (`_models_cache_file_fingerprint`) | The file is rewritten only on deliberate user edits, and any edit can change the provider/model set, so the cheap conservative identity wins. |
| `auth_json` | content hash with a volatile-key deny-list (`_auth_store_semantic_fingerprint`, `_AUTH_FINGERPRINT_VOLATILE_KEYS`) | The credential store is rewritten roughly every 14 minutes by credential-pool / OAuth refresh; none of those rotating fields feed `detected_providers` or the returned catalog, and stat identity made the 24h cache churn on every refresh (RCA `t_d127953d` / `t_16551f61`). |
| `catalog` | baked-in provider catalog sha256 (`_PROVIDER_MODELS` + `_PROVIDER_DISPLAY`) plus the Codex local catalog (`_codex_models_cache_fingerprint`, `_CODEX_CACHE_FINGERPRINT_VOLATILE_KEYS`) | A restart after a catalog change must not keep serving a persisted payload for up to 24h (#2443). Codex rewrites `~/.codex/models_cache.json` on its own timer, bumping `mtime_ns` and size while models, `etag`, and `client_version` stay identical, so the Codex axis hashes **content** with only the refresh timestamps (`fetched_at`, `updated_at`) removed (#7540, #7556). |

## Invariant: deny-lists are one-directional

- Both volatile-key sets are deny-lists, never allow-lists. They may remove only
  fields that provably do not gate the provider/model set.
- Every other field — including fields that Codex or the auth store may add in
  the future — stays in the fingerprint
  (`test_unknown_codex_field_stays_in_fingerprint`).
- Consequence: excluding a volatile key can only make the fingerprint **more
  stable**; it can never hide a genuine catalog or provider change. When in
  doubt, keep the key in.

## Invariant: fallbacks are never less safe than stat

- Missing file → recorded as missing, and the fingerprint stays stable
  (`test_missing_codex_cache_fingerprint_is_stable_and_marked`).
- Unreadable, corrupt, or mid-write JSON → stat identity, marked
  `unparsed-fallback`.
- Transform failure — including `RecursionError` on a pathologically deep tree →
  stat identity, marked `encode-fallback`
  (`test_deeply_nested_codex_cache_degrades_to_stat_fallback_without_crashing`).
  The fingerprint must never raise into `/api/models`; a real rewrite still
  changes the stat identity, so the fallback is strictly no less safe than the
  pre-#7556 behavior.

## Version stamps (independent of the fingerprint)

- `_schema_version` must equal `_MODELS_CACHE_SCHEMA_VERSION` (currently `3`).
  Bump it when the cached payload shape changes incompatibly.
- `_webui_version` must equal the running version, which forces a rebuild after
  every release so picker-shape fixes appear immediately instead of after the
  TTL expires. When the runtime version cannot be resolved (early boot), that
  check is skipped rather than wedging the boot.

## Invalidation modes (`invalidate_models_cache`)

`invalidate_models_cache(*, delete_disk=True)` is the only entry point that
offers the `delete_disk` choice. It is **not** the only path that drops the
published in-memory snapshot: `invalidate_provider_models_cache()`,
`_get_fresh_memory_models_cache()` (on a fingerprint mismatch or an invalid
cached shape) and the config-reload branch inside
`get_available_models()` also reset `_available_models_cache` and call
`_sync_models_cache_provenance()`. None of those paths take a `delete_disk`
argument; they only clear memory. `invalidate_models_cache` has two modes:

| Mode | What is dropped | When to use |
| --- | --- | --- |
| `delete_disk=True` (default) | in-memory snapshot **and** the per-profile `models_cache.json` on disk | A source may have changed, or test isolation requires a guaranteed cold build. Every pre-existing caller keeps this mode. |
| `delete_disk=False` | in-memory snapshot only; the disk snapshot is left in place | The sources have **not** changed and the caller only needs the next request to re-resolve *which* profile's catalog to serve. |

`POST /api/profile/switch` uses `delete_disk=False`. The disk cache is already
keyed per profile (`_get_models_cache_path()`), and a stale or wrong-profile
snapshot is rejected on read by `_is_loadable_disk_cache()` via the source
fingerprint above, so deleting it on a switch bought no correctness — it only
forced a full cold rebuild (live provider `fetch_models` calls, several seconds)
on every switch. Because this mode leans entirely on the fingerprint check, it
is safe only while that check stays the single gate for serving a disk snapshot
(change-protocol items 1 and 4). Both modes must still call
`_sync_models_cache_provenance()` so the hot-path tuple cannot tear.

## Change protocol

1. Add or change a source axis in `_models_cache_source_fingerprint()` only —
   one chokepoint, so the disk reader and the published-snapshot reader always
   agree.
2. When adding a volatile key, name the value it derives from and prove it
   inert: one test that fails if the key is dropped from the deny-list, and one
   that shows a genuine catalog change still invalidates the cache.
3. Keep the fingerprint cheap, deterministic, and safe to run synchronously —
   it is recomputed on cache reads, including a per-turn hot path.
4. Do not add a second, parallel cache-identity mechanism. The stat-based
   `_models_cache_file_fingerprint()` stays only as the conservative fallback for
   the axes above.
5. This is a runtime contract: changes here update this document and are
   described in the PR body. Release-note wording belongs in the PR body, not in
   `CHANGELOG.md`, which release commits own.

## Tests

`tests/test_issue7540_codex_catalog_fingerprint.py` covers both invariant groups:
timestamp-only churn keeps the fingerprint identical (and a session visit after a
Codex refresh needs no live rebuild), while genuine changes — a new model, a
visibility change, any catalog field, any unknown field — still invalidate.

`tests/test_profile_switch_models_disk_cache.py` covers the invalidation modes:
`delete_disk=False` keeps the disk file and the next `get_available_models()`
reloads it without a live rebuild, the default still unlinks it, and the
executed `/api/profile/switch` route passes `delete_disk=False`.

## References

Issues/PRs: #2443, #7540, #7556, #7558. RCAs: `t_d127953d`, `t_16551f61`.
