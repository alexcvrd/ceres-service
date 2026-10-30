"""Configuration: placeholders, secret refusal, environment overrides, the API key."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ceres.config import SERVE_RIGHTS_BY_MODE, ConfigError, load_config, scan_secrets

from conftest import EXAMPLE_CONFIG, FIXTURES


def with_store_keys(tmp_path, **keys) -> "object":
    """The shipped example with its `store.serving_mode` line replaced by the given keys.

    JSON is valid YAML, so the values keep their types (a list stays a list). This is how an
    operator edits the store section, so a refusal test exercises the same path a real edit takes.
    """
    block = "".join(f"  {name}: {json.dumps(value)}\n" for name, value in keys.items())
    text = re.sub(r"^  serving_mode:.*$\n", block, EXAMPLE_CONFIG.read_text(encoding="utf-8"),
                  count=1, flags=re.MULTILINE)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_shipped_example_loads_under_two_different_roots(tmp_path, env):
    """The same file, two machine roots: nothing about a box is baked into the package."""
    first = load_config(EXAMPLE_CONFIG, env, require_api_key=False)
    other = dict(env, CERES_RUNTIME_ROOT=str(tmp_path / "second"))
    second = load_config(EXAMPLE_CONFIG, other, require_api_key=False)
    assert first.runtime_root != second.runtime_root
    assert first.store.dsn == second.store.dsn
    assert first.api.port == 8300


def test_unresolved_placeholder_is_refused(env):
    """A ${UPPER} the environment does not define stops startup and names itself."""
    stripped = {k: v for k, v in env.items() if k != "DDS_GPU_LOCK"}
    with pytest.raises(ConfigError) as exc:
        load_config(EXAMPLE_CONFIG, stripped, require_api_key=False)
    assert "DDS_GPU_LOCK" in str(exc.value)


def test_environment_override_wins(env):
    """`CERES_<SECTION>__<KEY>` overrides a scalar, and it is parsed as YAML."""
    values = dict(env, CERES_API__PORT="9301", CERES_API__DOCS="true",
                  CERES_STORE__STATEMENT_TIMEOUT_MS="1234")
    config = load_config(EXAMPLE_CONFIG, values, require_api_key=False)
    assert config.api.port == 9301
    assert config.api.docs is True
    assert config.store.statement_timeout_ms == 1234


def test_environment_override_reaches_a_nested_section(env):
    values = dict(env, CERES_API__ADMISSION__MAX_INFLIGHT="7")
    assert load_config(EXAMPLE_CONFIG, values, require_api_key=False).api.admission.max_inflight == 7


def test_a_non_override_variable_is_not_read_as_one(env):
    """`CERES_API_KEY` has no double underscore, so it never becomes `api.key`."""
    values = dict(env, CERES_API_KEY="k" * 32)
    config = load_config(EXAMPLE_CONFIG, values, require_api_key=True)
    assert config.api_key == "k" * 32
    assert "key" not in config.api.model_dump()


def test_missing_api_key_fails_startup(env):
    with pytest.raises(ConfigError) as exc:
        load_config(EXAMPLE_CONFIG, env, require_api_key=True)
    assert "CERES_API_KEY" in str(exc.value)


def test_no_auth_is_explicit_and_visible(env):
    config = load_config(EXAMPLE_CONFIG, env, require_api_key=False)
    assert config.auth_required is False and config.api_key is None


# ---------------------------------------------------------------- secrets (CLI rules)

@pytest.mark.parametrize("value", [
    "sk-abcdefgh12345678",
    "ghp_deadbeef",
    "postgresql://user:hunter2@host/db",
    "password = hunter2",
])
def test_secret_looking_values_are_refused(tmp_path, env, value):
    """A credential-shaped value stops startup; the message names the path, never the value."""
    path = tmp_path / "config.yaml"
    path.write_text(EXAMPLE_CONFIG.read_text(encoding="utf-8")
                    + f"\nextra_note: \"{value}\"\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(path, env, require_api_key=False)
    assert "secret-looking value" in str(exc.value)
    assert value not in str(exc.value)


def test_a_secret_looking_key_is_refused(tmp_path, env):
    path = tmp_path / "config.yaml"
    path.write_text(EXAMPLE_CONFIG.read_text(encoding="utf-8")
                    + "\nstore_password: something\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path, env, require_api_key=False)


def test_a_key_that_names_where_a_secret_lives_is_allowed():
    """`api_key_env` holds a variable name. Refusing it would make the config unwritable."""
    assert scan_secrets({"api": {"api_key_env": "CERES_API_KEY"}}) == []
    assert scan_secrets({"api": {"api_key": "CERES_API_KEY"}}) == ["api.api_key"]


def test_dsn_must_be_password_less(tmp_path, env):
    """The DSN regex is the last line before a password could reach a log or a process listing."""
    text = EXAMPLE_CONFIG.read_text(encoding="utf-8").replace(
        "postgresql://ceres_reader@db.invalid/veridian_rag",
        "postgres://ceres_reader@db.invalid/veridian_rag")
    assert text != EXAMPLE_CONFIG.read_text(encoding="utf-8"), "the replacement must hit"
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(path, env, require_api_key=False)
    assert "store.dsn" in str(exc.value)


def test_safe_dsn_drops_the_user(config):
    assert config.safe_dsn == "postgresql://db.invalid/veridian_rag"


# ---------------------------------------------------------------- serving_mode

def test_the_shipped_default_is_rnd_with_the_three_rights_states(config):
    """`rnd` is what ships, and it reads every rights state the archive declares."""
    assert config.store.serving_mode == "rnd"
    assert config.store.serve_rights == ("redistributable", "internal_research_only", "licensed")
    assert config.store.admitted_works == ()
    assert config.store.production_decision is None


def test_production_derives_the_narrow_list_and_keeps_the_decision(tmp_path, env):
    path = with_store_keys(tmp_path, serving_mode="production", production_decision="D0xx",
                           admitted_licensed_works=["w-b", "w-a"])
    store = load_config(path, env, require_api_key=False).store
    assert store.serve_rights == ("redistributable",)
    assert store.production_decision == "D0xx"
    # Sorted, because the digest below is a property of the admitted set, not of the typing order.
    assert store.admitted_works == ("w-a", "w-b")


def test_setting_serve_rights_directly_is_refused_and_the_message_names_d091(tmp_path, env):
    """The key an operator would reach for first. Refusing it silently would be the real failure."""
    path = with_store_keys(tmp_path, serving_mode="rnd",
                           serve_rights=["redistributable", "licensed"])
    with pytest.raises(ConfigError) as exc:
        load_config(path, env, require_api_key=False)
    message = str(exc.value)
    assert "serve_rights" in message and "production_decision" in message and "serving_mode" in message


def test_serve_rights_is_refused_as_an_environment_override_too(env):
    """`CERES_STORE__SERVE_RIGHTS` is the same edit by another route, and it is refused too.

    A list-valued override is already refused as a non-scalar, so this uses the scalar spelling —
    the one that would otherwise reach the store section.
    """
    values = dict(env, CERES_STORE__SERVE_RIGHTS="redistributable")
    with pytest.raises(ConfigError) as exc:
        load_config(EXAMPLE_CONFIG, values, require_api_key=False)
    assert "production_decision" in str(exc.value)


def test_admitted_works_are_refused_in_rnd(tmp_path, env):
    """rnd serves every licensed work already; a list there would read as a narrowing that is not."""
    path = with_store_keys(tmp_path, serving_mode="rnd", admitted_licensed_works=["w-a"])
    with pytest.raises(ConfigError) as exc:
        load_config(path, env, require_api_key=False)
    assert "admitted_licensed_works" in str(exc.value)


@pytest.mark.parametrize("decision", [None, "", "   "])
def test_production_without_a_recorded_decision_is_refused(tmp_path, env, decision):
    """The switch that makes Ceres customer-facing must name the compliance decision."""
    keys = {"serving_mode": "production"}
    if decision is not None:
        keys["production_decision"] = decision
    path = with_store_keys(tmp_path, **keys)
    with pytest.raises(ConfigError) as exc:
        load_config(path, env, require_api_key=False)
    assert "production_decision" in str(exc.value)


def test_an_unknown_mode_is_refused(tmp_path, env):
    path = with_store_keys(tmp_path, serving_mode="staging")
    with pytest.raises(ConfigError) as exc:
        load_config(path, env, require_api_key=False)
    assert "serving_mode" in str(exc.value)


def test_the_read_predicate_digest_differs_between_modes_and_admissions(tmp_path, env, config):
    """The mode and the admitted set are both inside the predicate identity."""
    from ceres.store import read_predicate_sha256

    def digest(store) -> str:
        return read_predicate_sha256(store.serve_rights, store.db_schema,
                                     admitted_licensed_works=store.admitted_works)

    production = load_config(
        with_store_keys(tmp_path / "a", serving_mode="production", production_decision="D0xx"),
        env, require_api_key=False).store
    admitted = load_config(
        with_store_keys(tmp_path / "b", serving_mode="production", production_decision="D0xx",
                        admitted_licensed_works=["w-a"]),
        env, require_api_key=False).store

    assert len({digest(config.store), digest(production), digest(admitted)}) == 3
    # The set, not the order: two spellings of the same admission are one predicate.
    reversed_order = load_config(
        with_store_keys(tmp_path / "c", serving_mode="production", production_decision="D0xx",
                        admitted_licensed_works=["w-b", "w-a"]),
        env, require_api_key=False).store
    same_set = load_config(
        with_store_keys(tmp_path / "d", serving_mode="production", production_decision="D0xx",
                        admitted_licensed_works=["w-a", "w-b"]),
        env, require_api_key=False).store
    assert digest(reversed_order) == digest(same_set)


def test_every_mode_has_a_derived_rights_list():
    """No mode may exist without a rights list; a new mode without one would serve nothing."""
    assert set(SERVE_RIGHTS_BY_MODE) == {"rnd", "production"}
    assert all(rights for rights in SERVE_RIGHTS_BY_MODE.values())


def test_the_removed_exclusion_key_is_rejected(tmp_path, env):
    """`store.evidence_exclude_reasons` was removed; a config still carrying it must not load."""
    path = tmp_path / "config.yaml"
    path.write_text(EXAMPLE_CONFIG.read_text(encoding="utf-8").replace(
        "  statement_timeout_ms: 5000",
        "  evidence_exclude_reasons: [table_unvalidated]\n  statement_timeout_ms: 5000"),
        encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(path, env, require_api_key=False)
    assert "evidence_exclude_reasons" in str(exc.value)


# ---------------------------------------------------------------- served route identity

R1_ID = "73c2e7911d632ad5602bb72f869d5be81a920343fdca9c204fe2a7d7a6bf7315"


def _r1_profile_overrides() -> dict[str, str]:
    profiles = FIXTURES / "profiles"
    files = {"chunk": "chunk-bridge-v3.1.yaml", "represent": "verbatim-v1.yaml",
             "embed": "jina-768-v1.yaml", "index": "inmem-exact-v1.yaml",
             "retrieve": "dense-top10-v2.yaml", "rerank": "qwen3-4b-rerank-v1.yaml",
             "answer": "grounded-qwen38-v1.yaml"}
    return {f"CERES_ROUTE__PROFILES__{slot.upper()}": str(profiles / name)
            for slot, name in files.items()}


def test_the_seven_r1_profiles_do_not_label_the_service_r1(env):
    """Hashing r1's profiles must not turn the service into r1."""
    config = load_config(EXAMPLE_CONFIG, dict(env, **_r1_profile_overrides()),
                         require_api_key=False)
    assert config.route.served_route_id is None
    assert config.route.served_route_id != config.route.judged_route_id


def test_a_served_route_declaration_is_not_ignored(env, tmp_path):
    """The served id must come from the frozen declaration file, not be left null."""
    declared = "ab" * 32
    route_file = tmp_path / "s1.route.yaml"
    route_file.write_text(f"route_name: s1\nroute_id: {declared}\n", encoding="utf-8")
    config = load_config(EXAMPLE_CONFIG, dict(env, CERES_ROUTE__SERVED_ROUTE_FILE=str(route_file)),
                         require_api_key=False)
    assert config.route.served_route_id == declared


def test_a_declaration_file_that_names_r1_is_refused(env, tmp_path):
    route_file = tmp_path / "r1.route.yaml"
    route_file.write_text(f"route_name: r1\nroute_id: {R1_ID}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="judged"):
        load_config(EXAMPLE_CONFIG, dict(env, CERES_ROUTE__SERVED_ROUTE_FILE=str(route_file)),
                    require_api_key=False)


def test_a_hand_written_served_route_id_is_refused(env):
    with pytest.raises(ConfigError, match="served_route_file"):
        load_config(EXAMPLE_CONFIG, dict(env, CERES_ROUTE__SERVED_ROUTE_ID="cd" * 32),
                    require_api_key=False)


# ---------------------------------------------------------------- release: bundled served route (2026-10-23)

SERVED_EXAMPLE = Path(__file__).resolve().parents[1] / "config.served.example.yaml"


def _served_env(tmp_path):
    return {"CERES_STORE_DSN": "postgresql://reader@db.invalid/veridian_rag", "CERES_GPU_LOCK": str(tmp_path / "gpu.lock"),
            "CERES_RUNTIME_ROOT": str(tmp_path / "rt")}


def test_bundled_route_assets_recompute_to_the_declared_route_id(tmp_path):
    """Bug: a bundled profile is edited (or a file goes missing) and the served route id no longer matches its declaration,
    so a fresh checkout refuses to start in live mode."""
    import yaml
    from ceres.config import compute_route_id, load_config
    cfg = load_config(SERVED_EXAMPLE, _served_env(tmp_path), require_api_key=False)
    declared = yaml.safe_load(cfg.resolve(cfg.route.served_route_file).read_text())["route_id"]
    assert compute_route_id(cfg.profile_paths()) == declared
    assert cfg.resolve(cfg.route.prompt_template).is_file()


def test_store_guarded_profiles_are_bundled_byte_identical():
    """Bug: chunk/represent/embed are rewritten while bundling, so the route–store guard refuses every real store."""
    import hashlib, json
    pins = json.loads((Path(__file__).parent / "release_fixtures" / "store-guarded-profiles.json").read_text())
    root = Path(__file__).resolve().parents[1] / "ceres" / "assets" / "loop-v0"
    for rel, sha in pins.items():
        assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == sha, rel


def test_model_cache_override_resolves_the_hugging_face_layout(monkeypatch, tmp_path):
    """Bug: models resolve only from absolute paths on the build machine; CERES_MODEL_CACHE must redirect them."""
    from ceres.models import hf_snapshot, model_cache_override
    monkeypatch.delenv("CERES_MODEL_CACHE", raising=False)
    assert model_cache_override() is None
    monkeypatch.setenv("CERES_MODEL_CACHE", str(tmp_path))
    assert model_cache_override() == tmp_path
    assert hf_snapshot(tmp_path, "Qwen/Qwen3-Reranker-0.6B", "abc") == tmp_path / "hub" / "models--Qwen--Qwen3-Reranker-0.6B" / "snapshots" / "abc"


def test_census_root_is_preferred_over_loop_root_for_source_titles(tmp_path):
    """Bug: readable sources look for the census inside the bundled assets instead of the configured census_root."""
    from ceres.config import load_config
    cfg = load_config(SERVED_EXAMPLE, _served_env(tmp_path), require_api_key=False)
    assert cfg.paths.census_root is None
    assets = SERVED_EXAMPLE.parent / "ceres" / "assets" / "loop-v0"
    text = (SERVED_EXAMPLE.read_text().replace("  # census_root: /path/to/dds/engine", f"  census_root: {tmp_path}")
            .replace("loop_root: ceres/assets/loop-v0", f"loop_root: {assets}"))
    alt = tmp_path / "served.yaml"; alt.write_text(text)
    assert load_config(alt, _served_env(tmp_path), require_api_key=False).paths.census_root == str(tmp_path)


def test_bundled_embed_platform_config_matches_its_pin():
    """Bug: the shipped embed platform config edited without re-pinning, so the live embedder refuses to load
    (models.py checks `platform_config_sha256`) although every offline test passes."""
    import hashlib
    import yaml
    root = Path(__file__).resolve().parents[1] / "ceres" / "assets" / "loop-v0"
    profile = yaml.safe_load((root / "04_rag/profiles/embed/jina-768-v1.yaml").read_text())
    config = root / profile["platform_config"]
    assert hashlib.sha256(config.read_bytes()).hexdigest() == profile["platform_config_sha256"]
