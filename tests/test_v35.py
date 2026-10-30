"""The v3.5 switch on the Ceres side: one token floor, `rerank_text: represented`, `ceres-eval --set`.

How it can fail, written down before the tests (testing rule):
  V1  a v3.5 route whose retrieve `min_context_tokens` differs from its chunk `min_child_tokens` is
      served or evaluated (chunks under the floor occupy pool slots, or chunks above it are dropped);
      or a v3.4 route (no `min_child_tokens`) is refused although the loop freezes it;
  V2  the Ceres rule drifts from the loop's rule (accepts what `run_route.load_route` refuses, or
      refuses what it accepts), for the floor or for the rerank_text agreement;
  V3  a route and its rerank profile disagree on `rerank_text` (or the identity profile declares it)
      and the route runs anyway — scoring one text under a route id frozen for the other;
  V4  a represented route silently FALLS BACK to the payload when a candidate's `represented_text` is
      NULL or the release has no column — another route under this route's id;
  V5  the represented text leaks into what is citable: packet `evidence_text`, quotes, validation;
      or the reranker is not given the represented text at all;
  V6  a default (payload) route changes: it reranks on represented_text when the store has the
      column, or its eval rows / cache keys gain a key;
  V7  the store mis-reads the column: bytea not decoded, a NUL dropped, bad UTF-8 guessed, a missing
      column not a typed NULL, an unsupported type decoded anyway;
  V8  the text Ceres sends the reranker differs from the text the loop scored for the same candidates
      (loop window-2 s1-wide-v35hr rerank attempts, represent-v2 output);
  V9  `--set` writes outside `<route>-<set>`, into the ingest-hardening folder without a set or a
      pinned release, mixes two sets in one work dir, or its MANIFEST does not record the set and the
      stage-5 answer contract (prompt v1, validator r3, context lines off) — or writes one that breaks it;
  V10 the order-parity report attributes a difference to the wrong cause.
No test loads a model or calls an answerer.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from ceres import eval as ceval
from ceres import pipeline
from ceres.config import AnswerSection, compute_route_id
from ceres.llm import FakeLLM
from ceres.route_contract import RouteContractError, check_route_contract, check_token_floor, rerank_text_choice
from ceres.store import FixtureStore, Item, PgStore, StoreError, decode_represented_text
from ceres.testing import EN_QUESTION, EvalFixtureStore, build_inline_fixture

from conftest import FIXTURES, ROOT, fixture_rows
from test_eval import CountingModels, build_loop, fixture_models, make_ctx

LOOP = ROOT.parent.parent / "workspace" / "dds" / "archive" / "loop-v0"
LOOP_PYTHON = LOOP / "00_control" / ".venv" / "bin" / "python"
BANK_V5 = LOOP / "03_eval" / "runs" / "all" / "bank-select-attempt-0001" / "route-questions.jsonl"
DATED_PLACE = Path(os.environ.get("CERES_DATED_PLACE_BANK") or (LOOP / "04_rag" / "runs" / "all" / "dated-set" / "route-questions-dated-place.jsonl"))
RERANK_ATTEMPTS = LOOP / "04_rag" / "runs" / "slice-v0" / "rerank"
requires_loop = pytest.mark.skipif(not (LOOP / "04_rag" / "routes" / "s1-wide-v35hr.yaml").is_file(),
                                   reason="the DDS loop is not on this box")
requires_loop_python = pytest.mark.skipif(not LOOP_PYTHON.is_file(), reason="the loop's venv is not on this box")

CHUNK_V35 = """profile_schema_version: dds-loop-rag-slot-profile-v1
slot: chunk
profile_id: chunk-bridge-v3.5
min_child_tokens: 21
"""
RERANK_V3 = """profile_schema_version: dds-loop-rag-slot-profile-v1
slot: rerank
profile_id: qwen3-4b-rerank-v3
top_k_input: 10
top_k_output: 5
rerank_text: represented
hardware_flags: {device: cuda, dtype: bfloat16, deterministic_order: true}
"""


def retrieve_profile(floor: int | None) -> str:
    body = ("profile_schema_version: dds-loop-rag-slot-profile-v1\nslot: retrieve\nprofile_id: dense-top10-v2\n"
            "top_k_children: 10\ncandidate_pool: 50\n"
            "near_duplicate_rule: {normalization: norm-v3, token_jaccard_min: 0.6, keep: higher_rank}\n")
    return body + (f"min_context_tokens: {floor}\n" if floor is not None else "")


def add_route(root: Path, name: str, overrides: dict[str, str], files: dict[str, str] | None = None,
              **route_keys) -> Path:
    """One more route in a `build_loop` tree: extra profile files (relative path → text), slot
    overrides, extra route keys (e.g. rerank_text); the id is recomputed so only the contract can refuse."""
    for relative, text in (files or {}).items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    base = yaml.safe_load((root / "04_rag" / "routes" / "s1.yaml").read_text(encoding="utf-8"))["profiles"]
    rel = dict(base, **overrides)
    route_id = compute_route_id({slot: root / path for slot, path in rel.items()})
    path = root / "04_rag" / "routes" / f"{name}.yaml"
    path.write_text(yaml.safe_dump({"route_schema_version": "dds-loop-rag-route-v1", "route_name": name,
                                    "route_id": route_id, **route_keys, "profiles": rel}, sort_keys=False),
                    encoding="utf-8")
    return path


@pytest.fixture
def loop(tmp_path) -> dict[str, Path]:
    return build_loop(tmp_path / "loop")


@pytest.fixture
def v35_loop(loop, tmp_path):
    """build_loop + v3.5 shapes: `s1r` (v3.5 chunk, floor 21, rerank_text: represented, rerank 10→5)."""
    root = tmp_path / "loop"
    files = {"04_rag/profiles/chunk/chunk-bridge-v3.5.yaml": CHUNK_V35,
             "04_rag/profiles/rerank/qwen3-4b-rerank-v3.yaml": RERANK_V3}
    loop = dict(loop)
    loop["s1r"] = add_route(root, "s1r", {"chunk": "04_rag/profiles/chunk/chunk-bridge-v3.5.yaml",
                                          "rerank": "04_rag/profiles/rerank/qwen3-4b-rerank-v3.yaml"},
                            files, rerank_text="represented")
    return loop


# ================================================================ V1 — the floor check (route loader)

def test_a_v35_route_whose_retrieve_floor_differs_from_the_chunk_floor_is_refused(v35_loop, tmp_path):
    """V1: chunk min_child_tokens 21 + retrieve min_context_tokens 20 must not load."""
    root = tmp_path / "loop"
    path = add_route(root, "floor20", {"chunk": "04_rag/profiles/chunk/chunk-bridge-v3.5.yaml",
                                       "retrieve": "04_rag/profiles/retrieve/floor20.yaml"},
                     {"04_rag/profiles/retrieve/floor20.yaml": retrieve_profile(20)})
    with pytest.raises(ceval.Refused, match="min_context_tokens 20 differs .* min_child_tokens 21"):
        ceval.load_route(path)


def test_a_v35_route_whose_retrieve_profile_has_no_floor_is_refused(v35_loop, tmp_path):
    """V1: an absent min_context_tokens is 0 (the loop's default), which is not the chunk floor."""
    root = tmp_path / "loop"
    path = add_route(root, "nofloor", {"chunk": "04_rag/profiles/chunk/chunk-bridge-v3.5.yaml",
                                       "retrieve": "04_rag/profiles/retrieve/nofloor.yaml"},
                     {"04_rag/profiles/retrieve/nofloor.yaml": retrieve_profile(None)})
    with pytest.raises(ceval.Refused, match="min_context_tokens 0 differs"):
        ceval.load_route(path)


def test_a_chunk_profile_without_min_child_tokens_is_never_refused_for_its_floor(v35_loop, tmp_path):
    """V1: v3.4-and-earlier chunk profiles carry no floor; any retrieve floor must still load (v3.4 routes keep working)."""
    root = tmp_path / "loop"
    for floor in (None, 5, 21, 40):
        path = add_route(root, f"old{floor}", {"retrieve": f"04_rag/profiles/retrieve/old{floor}.yaml"},
                         {f"04_rag/profiles/retrieve/old{floor}.yaml": retrieve_profile(floor)})
        assert ceval.load_route(path).name == f"old{floor}"


def test_a_represent_body_floor_that_differs_from_the_chunk_floor_is_refused():
    """V1 (loop parity): run_route also refuses a represent min_body_tokens that is not the chunk floor."""
    with pytest.raises(RouteContractError, match="min_body_tokens 20"):
        check_token_floor({"min_child_tokens": 21}, {"min_context_tokens": 21}, {"min_body_tokens": 20})
    check_token_floor({"min_child_tokens": 21}, {"min_context_tokens": 21}, {"min_body_tokens": 21})
    check_token_floor({"min_child_tokens": 21}, {"min_context_tokens": 21}, {})


@requires_loop
def test_the_real_v35_routes_pass_the_floor_check_and_a_crafted_mismatch_of_them_refuses(tmp_path):
    """V1 on the real files: s0-v35h and s1-wide-v35hr load (floor 21 = 21); the same s0-v35h with its
    retrieve profile's floor edited to 20 is refused — by the eval loader and by the serving deps."""
    for name, text in (("s0-v35h", "payload"), ("s1-wide-v35hr", "represented"),
                       ("s0-v34", "payload"), ("s1-wide-v34", "payload")):
        route = ceval.load_route(LOOP / "04_rag" / "routes" / f"{name}.yaml")
        assert route.rerank_text == text, name
    real = yaml.safe_load((LOOP / "04_rag" / "routes" / "s0-v35h.yaml").read_text(encoding="utf-8"))
    root = tmp_path / "copy"
    rel = {}
    for slot, value in real["profiles"].items():
        target = root / value
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LOOP / value, target)
        rel[slot] = value
    retrieve = root / rel["retrieve"]
    retrieve.write_text(retrieve.read_text(encoding="utf-8").replace("min_context_tokens: 21",
                                                                      "min_context_tokens: 20"), encoding="utf-8")
    route_file = root / "04_rag" / "routes" / "s0-v35h-floor20.yaml"
    route_file.parent.mkdir(parents=True, exist_ok=True)
    route_file.write_text(yaml.safe_dump(dict(real, route_name="s0-v35h-floor20",
                                              route_id=compute_route_id({s: root / p for s, p in rel.items()})),
                                         sort_keys=False), encoding="utf-8")
    with pytest.raises(ceval.Refused, match="one floor"):
        ceval.load_route(route_file)
    loaded = {slot: (root / path).read_bytes() for slot, path in rel.items()}
    with pytest.raises(pipeline.PipelineError, match="one floor"):
        pipeline.profile_contract(loaded)


def test_the_serving_route_check_refuses_a_floor_mismatch(v35_loop, tmp_path, config):
    """V1: `ceres-api` refuses to start on such a route (served_route_check reads the route through
    eval.load_route), and Deps built from its profiles refuses too (the --fake/config-only path)."""
    from ceres.api import StartupRefused, served_route_check

    root = tmp_path / "loop"
    path = add_route(root, "floor20", {"chunk": "04_rag/profiles/chunk/chunk-bridge-v3.5.yaml",
                                       "retrieve": "04_rag/profiles/retrieve/floor20.yaml"},
                     {"04_rag/profiles/retrieve/floor20.yaml": retrieve_profile(20)})
    profiles = {slot: str(root / value) for slot, value in
                yaml.safe_load(path.read_text(encoding="utf-8"))["profiles"].items()}
    served = config.model_copy(update={"route": config.route.model_copy(
        update={"served_route_file": str(path), "profiles": profiles})})
    with pytest.raises(StartupRefused, match="one floor"):
        served_route_check(served)
    store, models, _ = build_inline_fixture()
    with pytest.raises(pipeline.PipelineError, match="one floor"):
        pipeline.Deps.from_files(config, store, models, FakeLLM(), {s: Path(p) for s, p in profiles.items()},
                                 root / "04_rag" / "prompts" / "grounded_answer_v1.md")


# ================================================================ V3 — rerank_text agreement

@pytest.mark.parametrize("route_value, profile_text, expected", [
    ("represented", RERANK_V3.replace("rerank_text: represented\n", ""), "differs from its rerank profile's 'payload'"),
    (None, RERANK_V3, "rerank_text 'payload' differs from its rerank profile's 'represented'"),
    ("payload", RERANK_V3, "differs from its rerank profile's 'represented'"),
    ("header", RERANK_V3, "route rerank_text must be one of"),
    ("represented", RERANK_V3.replace("rerank_text: represented", "rerank_text: header"), "rerank_text must be one of"),
])
def test_a_route_and_its_rerank_profile_that_disagree_on_rerank_text_are_refused(
        v35_loop, tmp_path, route_value, profile_text, expected):
    """V3: every error case of the loop's agreement rule refuses the route in the eval/serving loader."""
    root = tmp_path / "loop"
    keys = {} if route_value is None else {"rerank_text": route_value}
    path = add_route(root, "disagree", {"rerank": "04_rag/profiles/rerank/case.yaml"},
                     {"04_rag/profiles/rerank/case.yaml": profile_text}, **keys)
    with pytest.raises(ceval.Refused, match=expected):
        ceval.load_route(path)


@pytest.mark.parametrize("declared", ["payload", "represented"])
def test_the_identity_rerank_profile_must_not_declare_rerank_text(v35_loop, tmp_path, declared):
    """V3: none-v1 scores nothing; declaring rerank_text on it (even `payload`) is refused, as the loop does."""
    root = tmp_path / "loop"
    identity = (root / "04_rag/profiles/rerank/none-v1.yaml").read_text(encoding="utf-8")
    path = add_route(root, "idtext", {"rerank": "04_rag/profiles/rerank/none-text.yaml"},
                     {"04_rag/profiles/rerank/none-text.yaml": identity + f"rerank_text: {declared}\n"},
                     rerank_text=declared)
    with pytest.raises(ceval.Refused, match="identity rerank profile scores nothing"):
        ceval.load_route(path)
    with pytest.raises(RouteContractError):
        rerank_text_choice({"profile_id": "none-v1", "operation": "identity", "rerank_text": declared})


def test_a_route_that_declares_rerank_text_without_a_rerank_slot_is_refused():
    """V3 (loop parity): run_route refuses rerank_text on a route with no rerank slot."""
    with pytest.raises(RouteContractError, match="has no rerank slot"):
        check_route_contract({"route_name": "x", "rerank_text": "payload"}, {"rerank": None})


def test_the_represented_route_loads_and_deps_carry_its_choice(v35_loop, config):
    """V3/V6: agreement → the route loads; Deps read `represented` from the profile; default routes stay payload."""
    route = ceval.load_route(v35_loop["s1r"])
    assert route.rerank_text == "represented"
    store, models, _ = build_inline_fixture()
    deps = pipeline.Deps.from_files(config, store, models, FakeLLM(), dict(route.profiles), route.prompt_path)
    assert deps.rerank_text == "represented"
    for name in ("r1", "s0", "s1", "s1-wide"):
        base = ceval.load_route(v35_loop[name])
        assert pipeline.Deps.from_files(config, store, models, FakeLLM(), dict(base.profiles),
                                        base.prompt_path).rerank_text == "payload"


# ================================================================ V2 — the Ceres rule IS the loop's rule

MIRROR_SCRIPT = r"""
import json, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import yaml
import run_route, slot_adapter
out = []
for case in json.load(sys.stdin):
    paths = {slot: (Path(p) if p else None) for slot, p in case["paths"].items()}
    try:
        run_route.check_route_contract(case["route"], paths)
        rerank = paths.get("rerank")
        if rerank is not None:
            slot_adapter.rerank_text_choice(yaml.safe_load(rerank.read_text()) or {})
        out.append("ok")
    except slot_adapter.AdapterError as exc:
        out.append("refused")
print(json.dumps(out))
"""


@requires_loop_python
def test_the_route_contract_gives_the_loops_verdict_on_every_case(tmp_path):
    """V2: the same crafted routes through the loop's own `run_route.check_route_contract` +
    `slot_adapter.rerank_text_choice` (its venv, read-only, no bytecode) and through Ceres's mirror."""
    texts = {
        "chunk35": CHUNK_V35, "chunk34": "slot: chunk\nprofile_id: chunk-bridge-v3.4\n",
        "ret21": retrieve_profile(21), "ret20": retrieve_profile(20), "ret0": retrieve_profile(None),
        "rep21": "slot: represent\nprofile_id: represent-v2\nmin_body_tokens: 21\n",
        "rep20": "slot: represent\nprofile_id: represent-v2\nmin_body_tokens: 20\n",
        "rep": "slot: represent\nprofile_id: verbatim-v1\n",
        "v3": RERANK_V3, "v2": RERANK_V3.replace("rerank_text: represented\n", "").replace("v3", "v2"),
        "v3bad": RERANK_V3.replace("rerank_text: represented", "rerank_text: header"),
        "none": "slot: rerank\nprofile_id: none-v1\noperation: identity\n",
        "nonetext": "slot: rerank\nprofile_id: none-v1\noperation: identity\nrerank_text: payload\n",
    }
    files = {}
    for key, text in texts.items():
        files[key] = tmp_path / f"{key}.yaml"
        files[key].write_text(text, encoding="utf-8")
    grid = [(chunk, ret, rep, rerank, route_text)
            for chunk in ("chunk35", "chunk34") for ret in ("ret21", "ret20", "ret0")
            for rep in ("rep21", "rep20", "rep") for rerank in ("v3", "v2", "v3bad", "none", "nonetext")
            for route_text in (None, "payload", "represented", "header")]
    cases = []
    for chunk, ret, rep, rerank, route_text in grid:
        route = {"route_name": "case", **({} if route_text is None else {"rerank_text": route_text})}
        cases.append({"route": route, "paths": {"chunk": str(files[chunk]), "retrieve": str(files[ret]),
                                                "represent": str(files[rep]), "rerank": str(files[rerank])}})
    result = subprocess.run([str(LOOP_PYTHON), "-B", "-c", MIRROR_SCRIPT, str(LOOP / "04_rag")],
                            input=json.dumps(cases), capture_output=True, text=True, timeout=120,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert result.returncode == 0, result.stderr[-2000:]
    loop_verdicts = json.loads(result.stdout)
    ceres_verdicts = []
    for case in cases:
        parsed = {slot: yaml.safe_load(Path(p).read_text(encoding="utf-8")) for slot, p in case["paths"].items()}
        try:
            check_route_contract(case["route"], parsed)
            ceres_verdicts.append("ok")
        except RouteContractError:
            ceres_verdicts.append("refused")
    assert len(loop_verdicts) == len(cases) == 360
    assert ceres_verdicts == loop_verdicts
    assert 0 < ceres_verdicts.count("ok") < len(cases)          # both outcomes are exercised


# ================================================================ V7 — the store column

def test_represented_text_decodes_bytea_with_a_nul_and_text_as_is():
    """V7: bytea is strict UTF-8 like payload and keeps a NUL; text passes; NULL stays None; bad UTF-8 fails closed."""
    assert decode_represented_text(None, "c") is None
    assert decode_represented_text("issued July 2024 | Poland\nbody", "c") == "issued July 2024 | Poland\nbody"
    assert decode_represented_text("ţară\x00x".encode("utf-8"), "c") == "ţară\x00x"
    assert decode_represented_text(memoryview("ă".encode("utf-8")), "c") == "ă"
    with pytest.raises(StoreError, match="not valid UTF-8"):
        decode_represented_text(b"\xff\xfe", "c")
    with pytest.raises(StoreError, match="neither text nor bytea"):
        decode_represented_text({"a": 1}, "c")


def test_the_pg_select_names_represented_text_only_when_the_release_has_it():
    """V7: an absent column is a typed NULL placeholder; a present one is selected by name."""
    store = PgStore("postgresql://ceres_reader@127.0.0.1/veridian_rag", open_pool=False)
    assert "NULL::text AS represented_text" in store.search_sql(("context",), None)
    assert "represented_text AS represented_text" in store.search_sql(("context", "represented_text"), None)


def test_represented_text_stays_out_of_the_hit_record_and_asdict():
    """V6: `record()` (logs) and `asdict` (ceres-eval rows and their cache keys) never gain the key."""
    store = with_represented(build_inline_fixture()[0], lambda row: "HDR " + row["payload"])
    item = store.search(np.eye(1, 8)[0], store.active_release().release_attempt_id, ["redistributable"])[0]
    assert item.represented_text.startswith("HDR ")
    assert "represented_text" not in item.record()
    assert "represented_text" not in dataclasses.asdict(item)
    assert "represented_text" not in ceval.item_record(item)
    assert ceval.item_record(item, represented=True)["represented_text"] == item.represented_text
    back = ceval.item_from_record(ceval.item_record(item, represented=True))
    assert back.represented_text == item.represented_text and back.payload == item.payload


def test_the_fixture_store_infers_the_column_only_from_rows_that_carry_the_key():
    """V7: rows with the key → the release has the column (None = NULL); without it → no column."""
    plain = build_inline_fixture()[0]
    assert "represented_text" not in plain.active_release().optional_columns
    store = with_represented(plain, lambda row: None)
    assert "represented_text" in store.active_release().optional_columns
    withheld = store.withheld(store.active_release(), ["redistributable"])
    assert "represented_text" in withheld.optional_columns          # survives the rights count copy


def test_postgres_reads_represented_text_as_bytea_or_text_and_skips_an_unsupported_type(pg):
    """V7 against real PostgreSQL: bytea (with a NUL) and text are decoded; NULL is None; the detected
    type is recorded; a jsonb column is not selected (reads NULL) and a represented route is then refused."""
    from conftest import fixture_rows as rows_of, fixture_vectors

    RND = ["redistributable", "internal_research_only", "licensed"]
    picked = [i for i, row in enumerate(rows_of()) if row["evidence_eligible"] and row["rights_state"] in RND][:5]
    rows, vectors = [rows_of()[i] for i in picked], fixture_vectors()[picked]
    for schema, kind, value in (("rt_bytea", "bytea", "issued July 2024 | Poland\x00\nbody".encode("utf-8")),
                                ("rt_text", "text", "issued July 2024 | Poland\nbody"),
                                ("rt_jsonb", "jsonb", json.dumps({"x": 1}))):
        pg.create_schema(schema)
        with pg.admin() as conn:
            conn.execute(f"ALTER TABLE {schema}.items ADD COLUMN IF NOT EXISTS represented_text {kind}")
        pg.load(schema, "rel-rt", rows, vectors, active=True)
        with pg.admin() as conn:
            conn.execute(f"UPDATE {schema}.items SET represented_text = %s{'::jsonb' if kind == 'jsonb' else ''}"
                         " WHERE chunk_id = %s", (value, rows[0]["chunk_id"]))
        store = pg.store(schema)
        release = store.active_release()
        found = {item.chunk_id: item for item in store.search(vectors[0], "rel-rt", RND, 50)}
        others = [item.represented_text for cid, item in found.items() if cid != rows[0]["chunk_id"]]
        if kind == "jsonb":
            assert "represented_text" not in release.optional_columns
            assert found[rows[0]["chunk_id"]].represented_text is None
            deps = type("D", (), {"rerank_text": "represented"})()
            with pytest.raises(pipeline.RerankTextUnavailable):
                pipeline.require_rerank_text(deps, release)
            continue
        assert release.optional_columns == ("represented_text",)
        assert store.column_types == {"represented_text": kind}
        expected = value.decode("utf-8") if isinstance(value, bytes) else value
        assert found[rows[0]["chunk_id"]].represented_text == expected
        assert others and all(text is None for text in others)          # NULL rows stay None


# ================================================================ V4/V5/V6 — the pipeline

def with_represented(store: FixtureStore, text_of) -> FixtureStore:
    """The same fixture with a `represented_text` value on every row (None = SQL NULL)."""
    rows = []
    for row in store.rows:
        payload = row.get("payload")
        rows.append(dict(row, represented_text=text_of(dict(row, payload=payload))))
    return FixtureStore(rows, store.vectors, store.active_release())


class RecordingModels:
    """Wraps the inline FakeModels; records every text list the reranker receives."""

    def __init__(self, inner):
        self.inner, self.reranked = inner, []

    def embed_query(self, text, lang):
        return self.inner.embed_query(text, lang)

    def rerank(self, question, texts):
        self.reranked.append(list(texts))
        return self.inner.rerank(question, texts)

    def health(self):
        return self.inner.health()


def represented_deps(config, store, models):
    deps = pipeline.Deps.build(config, store, models, FakeLLM())
    deps.rerank_text = "represented"
    return deps


HEADER = "issued July 2024 | Poland | HEADERONLYTOKEN"


def test_the_reranker_receives_represented_text_while_evidence_text_stays_payload(config):
    """V5: the model scores header + text; the packet, evidence_text and the validated quote are the payload."""
    base, inner, _ = build_inline_fixture()
    store = with_represented(base, lambda row: f"{HEADER}\n{row['payload']}")
    models = RecordingModels(inner)
    response = pipeline.ask(represented_deps(config, store, models),
                            pipeline.AskRequest(question=EN_QUESTION, lang="en"))
    payloads = {row["chunk_id"]: row["payload"] for row in store.rows}
    assert models.reranked and all(text.startswith(HEADER + "\n") for text in models.reranked[0])
    assert all(text.split("\n", 1)[1] in payloads.values() for text in models.reranked[0])
    for context in response.packet["contexts"]:
        assert context["evidence_text"] == payloads[context["chunk_id"]]
    assert "HEADERONLYTOKEN" not in json.dumps(response.packet) + json.dumps(response.answer)
    assert response.answer["state"] == "VALIDATED"


def test_a_quote_from_the_header_only_never_validates():
    """V5: validation reads evidence_text (payload); text that exists only in represented_text is not a quote."""
    item = Item(chunk_id="c" * 64, parent_id="p", work_id="w", membership="m", rights_state="redistributable",
                payload="Winter wheat is sown at 450 to 550 viable seeds per square metre in the plain.",
                payload_sha256="0" * 64)
    item.represented_text = f"{HEADER}\n{item.payload}"
    packet = pipeline.build_packet("q?", "en", "q" * 64, [item]).to_dict()
    candidate = {"decision": "ANSWER", "answer_language": "en", "user_answer": "Sown in July 2024 in Poland.",
                 "citations": [{"claim": "Sown in July 2024 in Poland.", "evidence_id": "E1",
                                "support_quote": "issued July 2024 | Poland"}],
                 "missing_information": [], "confidence": "HIGH", "reason_code": "SUPPORTED"}
    assert "citation support_quote is not a continuous evidence substring" in pipeline.validate(candidate, packet)
    assert packet["contexts"][0]["evidence_text"] == item.payload


def test_a_null_represented_text_refuses_the_request_and_never_falls_back(config):
    """V4: one NULL candidate → RerankTextUnavailable; the reranker is never called with payloads."""
    base, inner, _ = build_inline_fixture()
    first = base.rows[0]["chunk_id"]
    store = with_represented(base, lambda row: None if row["chunk_id"] == first else f"{HEADER}\n{row['payload']}")
    models = RecordingModels(inner)
    with pytest.raises(pipeline.RerankTextUnavailable, match=r"represented_text_unavailable: 1 of 3 .*\(NULL\)"):
        pipeline.ask(represented_deps(config, store, models), pipeline.AskRequest(question=EN_QUESTION, lang="en"))
    assert models.reranked == []


def test_a_release_without_the_column_refuses_before_any_model_call(config):
    """V4: no column → refused by name at the release pin, before embedding or reranking."""
    base, inner, _ = build_inline_fixture()
    models = RecordingModels(inner)
    calls = []
    models.embed_query = lambda text, lang: calls.append(text)
    with pytest.raises(pipeline.RerankTextUnavailable, match="has no items.represented_text column"):
        pipeline.ask(represented_deps(config, base, models), pipeline.AskRequest(question=EN_QUESTION, lang="en"))
    assert calls == [] and models.reranked == []


def test_a_payload_route_still_reranks_on_payload_when_the_store_has_the_column(config):
    """V6: the column's presence changes nothing for a route that does not declare `represented`."""
    base, inner, _ = build_inline_fixture()
    store = with_represented(base, lambda row: f"{HEADER}\n{row['payload']}")
    models = RecordingModels(inner)
    deps = pipeline.Deps.build(config, store, models, FakeLLM())
    assert deps.rerank_text == "payload"
    response = pipeline.ask(deps, pipeline.AskRequest(question=EN_QUESTION, lang="en"))
    payloads = {row["payload"] for row in store.rows}
    assert models.reranked and all(text in payloads for text in models.reranked[0])
    reference = pipeline.ask(pipeline.Deps.build(config, base, RecordingModels(inner), FakeLLM()),
                             pipeline.AskRequest(question=EN_QUESTION, lang="en"))
    assert response.packet == reference.packet


def test_reranker_inputs_is_the_payload_list_for_payload_and_refuses_unknown_choices():
    """V6: the default branch is exactly `[hit.payload …]` (r1); an unknown choice is an error, not payload."""
    items = [Item(chunk_id=str(i), parent_id="p", work_id="w", membership="m", rights_state="r",
                  payload=f"payload {i}", payload_sha256="0") for i in range(3)]
    for item in items:
        item.represented_text = "rep " + item.chunk_id
    assert pipeline.reranker_inputs(items) == ["payload 0", "payload 1", "payload 2"]
    assert pipeline.reranker_inputs(items, "represented") == ["rep 0", "rep 1", "rep 2"]
    with pytest.raises(pipeline.PipelineError, match="unknown rerank_text"):
        pipeline.reranker_inputs(items, "header")


async def test_the_api_answers_503_with_the_public_code_when_the_text_is_unavailable(config):
    """V4: the refusal reaches the client as 503 `represented_text_unavailable`, detail in the log only."""
    import httpx

    from ceres.api import create_app

    base, inner, _ = build_inline_fixture()
    app = create_app(represented_deps(config, base, inner), config)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ceres.test") as http:
        reply = await http.post("/v1/ask", json={"question": EN_QUESTION, "lang": "en"})
    assert reply.status_code == 503
    assert reply.json()["detail"]["code"] == "represented_text_unavailable"
    assert base.active_release().release_attempt_id not in reply.text


# ================================================================ V4/V5/V6 — ceres-eval

class TextRecordingModels(CountingModels):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.texts: list[list[str]] = []

    def rerank(self, question, texts):
        self.texts.append(list(texts))
        return super().rerank(question, texts)


def recording_models() -> TextRecordingModels:
    base = fixture_models()
    models = TextRecordingModels(base.vectors, base.by_text, dim=base.dim)
    models.rerank_scores = base.rerank_scores
    return models


def represented_store(null_ids: frozenset = frozenset(), column: bool = True) -> EvalFixtureStore:
    rows = []
    for row in fixture_rows():
        row = dict(row)
        if column:
            row["represented_text"] = None if row["chunk_id"] in null_ids else f"REP {row['chunk_id'][:12]} {row['payload']}"
        rows.append(row)
    return EvalFixtureStore.from_fixture(rows=rows)


def test_the_eval_reranks_a_represented_route_on_the_store_text_and_packs_the_payload(tmp_path, config, v35_loop):
    """V5 (eval): rerank rows record what was scored; every scored text is the store's represented_text;
    packets carry the payload; the chain validates."""
    models = recording_models()
    store = represented_store()
    ctx = make_ctx(tmp_path, config, v35_loop["s1r"], store=store, models=models)
    ceval.run_chain(ctx)
    represented = {row["chunk_id"]: row["represented_text"] for row in store.rows}
    payload = {row["chunk_id"]: row["payload"] for row in store.rows}
    assert models.texts and all(text.startswith("REP ") for batch in models.texts for text in batch)
    rerank_rows = ceval.read_rows(ctx.phase_file("rerank"))
    for row in rerank_rows:
        for hit in row["hits"]:
            assert hit["represented_text"] == represented[hit["chunk_id"]]
    for packet in ceval.read_rows(ctx.out / "packets.jsonl"):
        for context in packet["contexts"]:
            assert context["evidence_text"] == payload[context["chunk_id"]]
    manifest = json.loads((ctx.out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["route"]["rerank_text"] == "represented"


def test_the_eval_retrieve_phase_refuses_a_release_without_the_column(tmp_path, config, v35_loop):
    """V4 (eval): no column → Refused naming the code; no retrieve row is written."""
    ctx = make_ctx(tmp_path, config, v35_loop["s1r"], store=represented_store(column=False))
    ceval.run_phase("embed", ctx)
    with pytest.raises(ceval.Refused, match="represented_text_unavailable: release .* has no"):
        ceval.run_phase("retrieve", ctx)
    assert ceval.read_rows(ctx.phase_file("retrieve")) == []


def test_the_eval_rerank_phase_refuses_a_null_candidate_and_never_scores_payloads(tmp_path, config, v35_loop):
    """V4 (eval): a NULL represented_text among the candidates refuses the phase; the model never sees a payload."""
    probe = make_ctx(tmp_path, config, v35_loop["s1r"], store=represented_store(), work="probe")
    for phase in ("embed", "retrieve"):
        ceval.run_phase(phase, probe)
    first_kept = ceval.read_rows(probe.phase_file("retrieve"))[0]["kept"][0]["chunk_id"]
    models = recording_models()
    store = represented_store(null_ids=frozenset({first_kept}))
    ctx = make_ctx(tmp_path, config, v35_loop["s1r"], store=store, models=models)
    for phase in ("embed", "retrieve"):
        ceval.run_phase(phase, ctx)
    with pytest.raises(ceval.Refused, match="represented_text_unavailable"):
        ceval.run_phase("rerank", ctx)
    payloads = {row["payload"] for row in store.rows}
    assert not any(text in payloads for batch in models.texts for text in batch)


def test_a_payload_routes_eval_rows_have_no_represented_key_even_when_the_store_has_the_column(tmp_path, config, v35_loop):
    """V6 (eval): s1's retrieve/rerank rows and their input shas are what they are without the column."""
    with_column = make_ctx(tmp_path, config, v35_loop["s1"], store=represented_store(), work="a")
    without = make_ctx(tmp_path, config, v35_loop["s1"], store=represented_store(column=False), work="b")
    for ctx in (with_column, without):
        for phase in ("embed", "retrieve", "rerank"):
            ceval.run_phase(phase, ctx)
    for phase in ("retrieve", "rerank"):
        a = ceval.read_rows(with_column.phase_file(phase))
        b = ceval.read_rows(without.phase_file(phase))
        assert a == b
        assert "represented_text" not in json.dumps(a)


# ================================================================ V9 — --set and the answers dir

SET_ANSWERS = Path("runs") / "set-answers"      # a neutral --answers-dir, relative to the loop root


def set_ctx(tmp_path, config, route_file, *, out_name, set_name="bank-v5", release="fixture", subdir=SET_ANSWERS, **kwargs):
    answers = tmp_path / "loop" / subdir
    ctx = make_ctx(tmp_path, config, route_file, out=answers / out_name, scratch=False, answers_dir=subdir, **kwargs)
    ctx.set_name = set_name
    ctx.release_id = release
    return ctx, answers


def test_a_set_folder_takes_route_dash_set_only_once_it_exists_and_with_release(tmp_path, config, loop):
    """V9: <answers>/<route>-<set> passes only when the folder exists, --set is given and --release pins."""
    ctx, answers = set_ctx(tmp_path, config, loop["s0"], out_name="s0-bank-v5")
    with pytest.raises(ceval.Refused, match="existing run folder"):
        ceval.guard_out(ctx)
    assert not answers.exists()
    answers.mkdir(parents=True)
    assert ceval.guard_out(ctx) == (answers / "s0-bank-v5").resolve()
    for out_name, set_name, release, message in (
            ("s0", "bank-v5", "fixture", "named after it"),                   # the bare route folder
            ("s0-dated-place", "bank-v5", "fixture", "named after it"),       # another set's folder
            ("s1-bank-v5", "bank-v5", "fixture", "named after it"),           # another route's folder
            ("s0-bank-v5", "bank-v5", None, "pins its release with --release")):
        ctx, _ = set_ctx(tmp_path, config, loop["s0"], out_name=out_name, set_name=set_name, release=release)
        with pytest.raises(ceval.Refused, match=message):
            ceval.guard_out(ctx)


@pytest.mark.parametrize("name", ["Bank", "../x", "a/b", "", "-x", "x" * 65])
def test_a_set_name_that_is_not_a_plain_lowercase_name_is_refused(name):
    """V9: the set name becomes part of a folder name inside the loop."""
    with pytest.raises(ceval.Refused, match="--set must match"):
        ceval.check_set_name(name)
    assert ceval.check_set_name("dated-place") == "dated-place"


def test_a_work_dir_holds_one_set(tmp_path, config, loop):
    """V9: RUN-STATE pins the set; another set, or none, in the same work dir is refused."""
    ctx = make_ctx(tmp_path, config, loop["s0"])
    ctx.set_name = "bank-v5"
    ceval.run_phase("embed", ctx)
    assert json.loads(ctx.state_path.read_text(encoding="utf-8"))["set"] == "bank-v5"
    for other in ("dated-place", None):
        ctx.set_name = other
        with pytest.raises(ceval.Refused, match="pinned to question set bank-v5"):
            ceval.run_phase("embed", ctx)
    flagless = make_ctx(tmp_path, config, loop["s0"], work="flagless")
    ceval.run_phase("embed", flagless)
    assert "set" not in json.loads(flagless.state_path.read_text(encoding="utf-8"))
    flagless.set_name = "bank-v5"
    with pytest.raises(ceval.Refused, match="pinned to question set <none>"):
        ceval.run_phase("embed", flagless)


def test_a_set_manifest_records_the_set_and_asserts_the_stage5_answer_contract(tmp_path, config, loop):
    """V9: MANIFEST.json says which set, the folder, and asserts prompt v1 / validator r3 / no context lines."""
    ctx = make_ctx(tmp_path, config, loop["s1-wide"], out=tmp_path / "out" / "s1-wide-dated-place")
    ctx.set_name = "dated-place"
    ceval.run_chain(ctx)
    manifest = json.loads((ctx.out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["set"] == {"name": "dated-place", "folder": "s1-wide-dated-place", "pinned": "dated-place"}
    asserted = manifest["asserted"]
    assert asserted["prompt"] == {"name": "grounded_answer_v1.md", "revision": "v1",
                                  "sha256": "7a37e08184ffd804fbc07da11296246ffe1479be74660cbb2be71a57ff7da0b4"}
    assert asserted["validator_revision"] == "ceres-validate-r3" == manifest["answer"]["validator_revision"]
    assert asserted["answer_context_lines"] is False


def test_a_flagless_manifest_has_no_set_or_asserted_keys(tmp_path, config, loop):
    """V6/V9: without --set the manifest keeps its keys (no set, no asserted, no rerank_text for payload)."""
    ctx = make_ctx(tmp_path, config, loop["s0"])
    ceval.run_chain(ctx)
    manifest = json.loads((ctx.out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert "set" not in manifest and "asserted" not in manifest and "rerank_text" not in manifest["route"]


def test_a_set_run_refuses_to_answer_with_context_lines_on(tmp_path, config, loop):
    """V9: ceres-eval never renders context lines; a --set run under a config that turns them on sends nothing."""
    llm = FakeLLM()
    lines_on = config.model_copy(update={"answer": AnswerSection(context_lines=True)})
    ctx = make_ctx(tmp_path, lines_on, loop["s0"], llm=llm)
    ctx.set_name = "bank-v5"
    for phase in ("embed", "retrieve", "rerank"):
        ceval.run_phase(phase, ctx)
    with pytest.raises(ceval.Refused, match="context lines"):
        ceval.run_phase("answer", ctx)
    assert llm.calls == []


def test_a_set_manifest_is_refused_when_the_recorded_validator_is_not_r3(tmp_path, config, loop):
    """V9: answers recorded under another validator revision get no stage-5 manifest."""
    ctx = make_ctx(tmp_path, config, loop["s0"])
    ctx.set_name = "bank-v5"
    for phase in ("embed", "retrieve", "rerank", "answer"):
        ceval.run_phase(phase, ctx)
    path = ctx.phase_file("answer")
    rows = [dict(row, validator_revision="ceres-validate-r2") for row in ceval.read_rows(path)]
    ceval.atomic_jsonl(path, rows)
    with pytest.raises(ceval.Invalid, match="validator recorded 'ceres-validate-r2'"):
        ceval.run_phase("manifest", ctx)
    assert not (ctx.out / "MANIFEST.json").exists()


def test_a_set_manifest_is_refused_for_a_prompt_that_is_not_v1(tmp_path, config, loop):
    """V9: another prompt file (even one its profile pins) is not prompt v1."""
    root = tmp_path / "loop"
    prompt = root / "04_rag" / "prompts" / "grounded_answer_v2.md"
    prompt.write_text("another prompt\n", encoding="utf-8")
    answer = (root / "04_rag/profiles/answer/grounded-qwen38-v1.yaml").read_text(encoding="utf-8")
    sha = hashlib.sha256(prompt.read_bytes()).hexdigest()
    answer = answer.replace("grounded_answer_v1.md", "grounded_answer_v2.md")
    answer = "\n".join(f"prompt_template_sha256: {sha}" if line.startswith("prompt_template_sha256") else line
                       for line in answer.splitlines()) + "\n"
    path = add_route(root, "s0v2", {"rerank": "04_rag/profiles/rerank/none-v1.yaml",
                                    "answer": "04_rag/profiles/answer/v2.yaml"},
                     {"04_rag/profiles/answer/v2.yaml": answer})
    route = ceval.load_route(path)
    ctx = make_ctx(tmp_path, config, path)
    ctx.set_name = "bank-v5"
    recorded = {"revision": "ceres-validate-r3"}
    deps = ctx.deps()
    with pytest.raises(ceval.Invalid, match="not prompt v1"):
        ceval.stage5_assertions(ctx, deps, recorded)
    assert route.prompt_path.name == "grounded_answer_v2.md"


@requires_loop
def test_the_two_stage5_question_files_load_as_banks():
    """V9: bank v5 (152) and the dated place+month set (60) load through the eval's bank loader."""
    bank, _ = ceval.load_bank(BANK_V5)
    dated, sha = ceval.load_bank(DATED_PLACE)
    assert len(bank) == 152
    assert len(dated) == 60 and {row["language"] for row in dated} == {"en"}
    assert sha == "0f5887547abb658b42069c72807fc3ba2d045ccd880609aa5f1deeca01e594cc"


# ================================================================ V8 — input parity with the loop (window 2)

LOOP_TEXTS_SCRIPT = r"""
import hashlib, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import yaml
import slot_adapter as sa
result = {}
cache = {}
for attempt in sys.argv[2:]:
    attempt = Path(attempt)
    manifest = json.loads((attempt / "MANIFEST.json").read_text())
    profile = yaml.safe_load(sa.resolve_loop_path(manifest["profile_path"]).read_text())
    choice = sa.rerank_text_choice(profile)
    top_k_input = int(profile["top_k_input"])
    input_manifest = sa.resolve_loop_path(manifest["input_manifest_path"])
    retrieve = sa.read_json(input_manifest)
    hits = sa.read_jsonl(sa.manifest_output(input_manifest, retrieve, "hits.jsonl"))
    by_question = {}
    for hit in hits:
        by_question.setdefault(hit["question_id"], []).append(hit)
    index_path = sa.linked_manifest(retrieve, "index_manifest_path")
    chain = sa.resolve_index_chain(index_path)
    represent_path = str(chain[3])
    if represent_path not in cache:
        cache[represent_path] = sa.representations_for_hits(input_manifest)
    representations = cache[represent_path]
    questions = {}
    for qid, rows in by_question.items():
        candidates = rows[:top_k_input]
        texts = sa.rerank_texts(candidates, {}, representations, choice)
        questions[qid] = [[row["chunk_id"], hashlib.sha256(text.encode("utf-8")).hexdigest()]
                          for row, text in zip(candidates, texts)]
    result[attempt.name] = {"choice": choice, "represent_manifest": represent_path, "questions": questions,
                            "questions_path": manifest.get("questions_path"),
                            "input_manifest_path": manifest["input_manifest_path"]}
print(json.dumps(result))
"""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@requires_loop
@requires_loop_python
def test_ceres_sends_the_reranker_the_text_the_loop_scored_window2():
    """V8: for loop window-2 s1-wide-v35hr rerank attempts 0001 (bank v5, the one matching the bank run
    `s1-wide-v35hr bank try1` of window2.log) and 0002 (dated-s1wide), the text Ceres would send the
    reranker for the SAME candidates — represent-v2 text as `dds load` stores it (bytea and text), read
    back by the store decoder, selected by `pipeline.reranker_inputs` — equals the text the loop scored
    (the loop's own `rerank_texts` over its own chain, run in its venv, compared by sha256)."""
    attempts = [RERANK_ATTEMPTS / f"s1-wide-v35hr-rerank-attempt-000{n}" for n in (1, 2)]
    result = subprocess.run([str(LOOP_PYTHON), "-B", "-c", LOOP_TEXTS_SCRIPT, str(LOOP / "04_rag"),
                             *map(str, attempts)], capture_output=True, text=True, timeout=600,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert result.returncode == 0, result.stderr[-2000:]
    loop = json.loads(result.stdout)
    assert loop["s1-wide-v35hr-rerank-attempt-0001"]["questions_path"].endswith(
        "bank-select-attempt-0001/route-questions.jsonl")
    # The store's source: the s0-v35h represent output the v3.5 load came from.
    store_source = sorted((LOOP / "04_rag/runs/slice-v0/represent").glob("s0-v35h-represent-attempt-*"))[-1]
    source_file = store_source / "representations.jsonl"
    needed = {cid for attempt in loop.values() for rows in attempt["questions"].values() for cid, _ in rows}
    texts: dict[str, str] = {}
    with source_file.open(encoding="utf-8") as handle:
        for line in handle:
            if '"chunk_id"' not in line:
                continue
            row = json.loads(line)
            if row["chunk_id"] in needed:
                texts[row["chunk_id"]] = row["representation_text"]
    assert needed <= set(texts)
    compared = 0
    for name, attempt in loop.items():
        assert attempt["choice"] == "represented"
        chain_file = Path(attempt["represent_manifest"]).parent / "representations.jsonl"
        assert file_sha256(chain_file) == file_sha256(source_file)     # the loop scored the store's source
        reranked = {}
        for line in (RERANK_ATTEMPTS / name / "reranked-hits.jsonl").read_text(encoding="utf-8").splitlines():
            hit = json.loads(line)
            reranked.setdefault(hit["question_id"], set()).add(hit["chunk_id"])
        for qid, rows in attempt["questions"].items():
            assert reranked[qid] <= {cid for cid, _ in rows}      # the loop's top-5 came from these candidates
            for stored in ("bytea", "text"):
                items = []
                for cid, _ in rows:
                    item = Item(chunk_id=cid, parent_id="p", work_id="w", membership="m", rights_state="r",
                                payload="unused", payload_sha256="0")
                    raw = texts[cid].encode("utf-8") if stored == "bytea" else texts[cid]
                    item.represented_text = decode_represented_text(raw, cid)
                    items.append(item)
                ceres = [hashlib.sha256(text.encode("utf-8")).hexdigest()
                         for text in pipeline.reranker_inputs(items, "represented")]
                assert ceres == [sha for _, sha in rows], (name, qid, stored)
            compared += len(rows)
    assert len(loop["s1-wide-v35hr-rerank-attempt-0001"]["questions"]) == 152
    assert len(loop["s1-wide-v35hr-rerank-attempt-0002"]["questions"]) == 120
    assert compared == 152 * 100 + 120 * 100


# ================================================================ V10 — the order-parity tool

sys.path.insert(0, str(ROOT / "tools"))
import rerank_order_parity as rop  # noqa: E402


def _hits(qid, ids, scores=None):
    return [{"question_id": qid, "chunk_id": cid, "rank": i + 1, "score": (scores or {}).get(cid, 10 - i)}
            for i, cid in enumerate(ids)]


def test_the_order_report_attributes_matches_filter_first_score_and_missing():
    """V10: same candidates + same top-5 → match; different candidate sets → filter-first (parity rule),
    naming the loop's ineligible chunk; same candidates, other order → score with both sides' scores."""
    cands = [f"c{i}" for i in range(8)]
    loop_retrieved = {"q1": _hits("q1", cands), "q2": _hits("q2", cands), "q3": _hits("q3", cands)}
    loop_reranked = {"q1": _hits("q1", cands[:5]), "q2": _hits("q2", ["x", *cands[:4]]),
                     "q3": _hits("q3", ["c1", "c0", "c2", "c3", "c4"], {"c1": 5.01, "c0": 5.0}),
                     "q4": _hits("q4", cands[:5])}
    loop_reranked["q2"][0]["evidence_eligible"] = False
    loop_retrieved["q2"] = _hits("q2", ["x", *cands[:7]])
    ceres_retrieve = {q: {"question_id": q, "kept": [{"chunk_id": c} for c in cands]} for q in ("q1", "q2", "q3")}
    ceres_rerank = {q: {"question_id": q, "scored": 8,
                        "hits": [{"chunk_id": c, "rank": i + 1, "rerank_score": 9 - i}
                                 for i, c in enumerate(cands[:5])]} for q in ("q1", "q2", "q3")}
    report = rop.compare(ceres_rerank, ceres_retrieve, loop_reranked, loop_retrieved, top_k=5, top_k_input=8)
    rows = {row["question_id"]: row for row in report["rows"]}
    assert rows["q1"]["status"] == "match"
    assert rows["q2"]["status"] == "filter-first"
    assert rows["q2"]["loop_only_candidates"] == ["x"] and rows["q2"]["loop_top_not_scored_by_ceres"] == ["x"]
    assert rows["q2"]["loop_top_evidence_ineligible"] == ["x"]
    assert rows["q3"]["status"] == "score" and rows["q3"]["first_difference_rank"] == 1
    assert rows["q3"]["pair"] == ["c0", "c1"] and rows["q3"]["loop_scores"] == [5.0, 5.01]
    assert rows["q4"]["status"] == "missing"
    assert report["counts"] == {"filter-first": 1, "match": 1, "missing": 1, "score": 1}


def test_the_order_tool_reads_the_loop_attempt_chain_and_refuses_output_inside_the_loop(tmp_path, capsys):
    """V10: the CLI resolves the loop's retrieve hits from the rerank MANIFEST and never writes into the loop."""
    loop = tmp_path / "loop"
    attempt = loop / "04_rag/runs/slice-v0/rerank/r-attempt-0001"
    retrieve = loop / "04_rag/runs/slice-v0/retrieve/r-attempt-0001"
    for folder in (attempt, retrieve):
        folder.mkdir(parents=True)
    (attempt / "MANIFEST.json").write_text(json.dumps({"input_manifest_path": "04_rag/runs/slice-v0/retrieve/r-attempt-0001/MANIFEST.json",
                                                       "counts": {"rerank_text": "represented"}}))
    (attempt / "reranked-hits.jsonl").write_text("".join(json.dumps(h) + "\n" for h in _hits("q1", ["a", "b"])))
    (retrieve / "hits.jsonl").write_text("".join(json.dumps(h) + "\n" for h in _hits("q1", ["a", "b", "c"])))
    work = tmp_path / "work" / "routes" / "r"
    work.mkdir(parents=True)
    (work / "rerank.jsonl").write_text(json.dumps({"question_id": "q1", "scored": 3, "hits": [
        {"chunk_id": "a", "rank": 1}, {"chunk_id": "b", "rank": 2}]}) + "\n")
    (work / "retrieve.jsonl").write_text(json.dumps({"question_id": "q1", "kept": [
        {"chunk_id": c} for c in "abc"]}) + "\n")
    args = ["--ceres-work-dir", str(tmp_path / "work"), "--route", "r", "--loop-rerank-attempt", str(attempt)]
    assert rop.main(args + ["--out", str(tmp_path / "report.json")]) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["counts"] == {"match": 1} and report["loop"]["rerank_text"] == "represented"
    assert rop.main(args + ["--out", str(loop / "report.json")]) == 2
    assert not (loop / "report.json").exists()
