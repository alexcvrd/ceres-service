"""`ceres-eval`: the phased answer run, over the fixture store and fakes only.

How the eval can fail, written down before the tests (testing rule):
  - a route's widths are ignored (s0 calls the reranker; s1-wide scores 10, not 100);
  - the eval's packets drift from what `/v1/ask` serves for the same route and question;
  - the request body, the raw event names or `request_sha256` drift from the loop's, or the real
      HTTP client mishandles a recorded reply;
  - a crash mid-answer duplicates rows or re-calls the answerer for finished questions;
  - a transport failure writes a fake row, or leaves no resumable state;
  - the active release changes between phases and the answers mix two releases;
  - a missing, duplicated, extra or rewritten bank question passes validate;
  - a route whose declared id is not its profiles' id labels the answers;
  - the OCR-citation count is wrong (counts cited-but-not-OCR, or packed-but-not-cited);
  - a real run writes outside the answers folder, or anything lands elsewhere in the loop;
  - a cached row is reused after its inputs changed, or `--force-phase` does not recompute;
  - the shipped answers schema is not the loop's byte for byte;
  - raw events of a question that never finished survive into the answers folder.
  - the answerer runs another context size than the answer profile and
      the run answers anyway, or records only the declared size; or an unknown `/props` shape
      breaks an answer run that would otherwise work;
  - MANIFEST stamps the CODE's validator revision onto answers that
      were produced under another one (r1 answers relabelled r3 by a manifest rerun), or
      validate re-checks old answers with new rules without saying so.
  - `--release` reads the ACTIVE release instead of the named inactive
      one, or pins a release id the schema does not hold, or misses a reload of the named release
      between phases; or the flagless run changes (another release, extra pin keys); or serving
      starts reading a named release;
  - the output allowlist admits another route's folder, a run folder the caller has not
      created, or any other loop path;
  - MANIFEST/RUN-STATE do not say which schema was read and that the release was pinned by name,
      or `--schema` lets a non-identifier reach SQL.
No test calls a model server: answers come from `FakeLLM`, or from the fixture's recorded replies
behind `httpx.MockTransport`.
"""

from __future__ import annotations

import hashlib
import json

import ceres
import shutil
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml

from ceres import eval as ceval
from ceres import pipeline
from ceres.config import ROUTE_SLOT_ORDER, compute_route_id
from ceres.contracts import ANSWER_SCHEMA, AskRequest
from ceres.llm import FakeLLM, LLMTransportError, OpenAIChatLLM
from ceres.models import FakeModels
from ceres.store import Release
from ceres.testing import EvalFixtureStore, fixture_bank, golden_rerank_scores

from conftest import FIXTURES, ROOT, fixture_rows, fixture_vectors, read_jsonl

LOOP_ANSWERS_SCHEMA = (ROOT.parent.parent / "workspace" / "dds" / "archive" / "loop-v0"
                       / "00_control" / "schemas" / "answers.schema.json")

# Profiles the fixture does not carry, written the way the loop's files state them (only the keys
# Ceres reads matter; the ids are recomputed from these bytes, so they are self-consistent).
NONE_V1 = """profile_schema_version: dds-loop-rag-slot-profile-v1
slot: rerank
profile_id: none-v1
operation: identity
hardware_flags: {device: cpu, deterministic: true}
"""
TOP100 = """profile_schema_version: dds-loop-rag-slot-profile-v1
slot: retrieve
profile_id: dense-top100-v2
top_k_children: 100
parent_deduplication: keep-best
near_duplicate_rule: {normalization: norm-v3, token_jaccard_min: 0.6, keep: higher_rank}
min_context_tokens: 21
candidate_pool: 500
"""
RERANK_V2 = """profile_schema_version: dds-loop-rag-slot-profile-v1
slot: rerank
profile_id: qwen3-4b-rerank-v2
top_k_input: 100
top_k_output: 5
hardware_flags: {device: cuda, dtype: bfloat16, deterministic_order: true}
"""
R1_FILES = {"chunk": "chunk-bridge-v3.1.yaml", "represent": "verbatim-v1.yaml",
            "embed": "jina-768-v1.yaml", "index": "inmem-exact-v1.yaml",
            "retrieve": "dense-top10-v2.yaml", "rerank": "qwen3-4b-rerank-v1.yaml",
            "answer": "grounded-qwen38-v1.yaml"}


def build_loop(root: Path) -> dict[str, Path]:
    """A throw-away loop tree: the fixture's r1 profiles + prompt, and four routes over them.

    Returns route name → route file. Shapes: r1 and s1 (rerank 10→5), s0 (identity), s1-wide
    (pool 500 → 100 children, rerank 100→5).
    """
    profiles = root / "04_rag" / "profiles"
    for slot, name in R1_FILES.items():
        target = profiles / slot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FIXTURES / "profiles" / name, target)
    (profiles / "rerank" / "none-v1.yaml").write_text(NONE_V1, encoding="utf-8")
    (profiles / "retrieve" / "dense-top100-v2.yaml").write_text(TOP100, encoding="utf-8")
    (profiles / "rerank" / "qwen3-4b-rerank-v2.yaml").write_text(RERANK_V2, encoding="utf-8")
    prompt = root / "04_rag" / "prompts" / "grounded_answer_v1.md"
    prompt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURES / "profiles" / "grounded_answer_v1.md", prompt)
    shapes = {
        "r1": {},
        "s0": {"rerank": "04_rag/profiles/rerank/none-v1.yaml"},
        "s1": {},
        "s1-wide": {"retrieve": "04_rag/profiles/retrieve/dense-top100-v2.yaml",
                    "rerank": "04_rag/profiles/rerank/qwen3-4b-rerank-v2.yaml"},
    }
    routes: dict[str, Path] = {}
    for name, overrides in shapes.items():
        rel = {slot: f"04_rag/profiles/{slot}/{R1_FILES[slot]}" for slot in ROUTE_SLOT_ORDER}
        rel.update(overrides)
        route_id = compute_route_id({slot: root / path for slot, path in rel.items()})
        path = root / "04_rag" / "routes" / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"route_schema_version": "dds-loop-rag-route-v1",
                                        "route_name": name, "route_id": route_id,
                                        "profiles": rel}, sort_keys=False), encoding="utf-8")
        routes[name] = path
    return routes


class CountingModels(FakeModels):
    """Fixture query vectors + the loop's golden rerank scores, counting calls per question."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rerank_widths: list[int] = []
        self.embedded_rows = 0

    def embed_queries(self, rows):
        self.embedded_rows += len(rows)
        return super().embed_queries(rows)

    def rerank(self, question, texts):
        self.rerank_widths.append(len(texts))
        return super().rerank(question, texts)


def fixture_models() -> CountingModels:
    questions = fixture_bank()
    vectors = np.load(FIXTURES / "query-vectors.npy").astype(np.float32)
    return CountingModels({row["question_id"]: vectors[i] for i, row in enumerate(questions)},
                          {row["text"].strip(): row["question_id"] for row in questions},
                          dim=vectors.shape[1], rerank_scores=golden_rerank_scores(FIXTURES))


class CrashingLLM(FakeLLM):
    """A FakeLLM that dies (a process crash, not a transport error) on call `crash_on`."""

    def __init__(self, crash_on: int, *, transport: bool = False) -> None:
        super().__init__()
        self.crash_on, self.transport = crash_on, transport

    def chat(self, request_body):
        if len(self.calls) + 1 == self.crash_on:
            self.calls.append(request_body)
            if self.transport:
                raise LLMTransportError("connection refused (test)")
            raise RuntimeError("simulated crash mid-answer")
        return super().chat(request_body)


class DownLLM(FakeLLM):
    def health(self):
        return "down", "ConnectError", None


@pytest.fixture
def loop(tmp_path) -> dict[str, Path]:
    return build_loop(tmp_path / "loop")


@pytest.fixture
def bank_file() -> Path:
    return FIXTURES / "questions.jsonl"


def make_ctx(tmp_path, config, route_file, *, store=None, models=None, llm=None, out=None,
             scratch=True, bank=None, work="work", force=(), allow_unknown_release_profiles=True,
             answers_dir=None):
    # Fixture stores carry no profile shas, so tests allow unknown ones unless a test checks the refusal.
    route = ceval.load_route(route_file)
    bank_path = bank or FIXTURES / "questions.jsonl"
    rows, sha = ceval.load_bank(bank_path)
    stack = ceval.Stack(store=store if store is not None else EvalFixtureStore.from_fixture(),
                        models=models if models is not None else fixture_models(),
                        llm=llm if llm is not None else FakeLLM(),
                        endpoint={"base_url": "http://answerer.test/v1"})
    return ceval.EvalContext(route=route, bank_path=bank_path, bank=rows, bank_sha256=sha,
                             work_dir=tmp_path / work, config=config, stack=stack,
                             out=out if out is not None else tmp_path / "out" / route.name,
                             scratch=scratch, force=set(force), log=lambda message: None,
                             allow_unknown_release_profiles=allow_unknown_release_profiles,
                             answers_dir=answers_dir)


# ================================================================ the three route shapes

@pytest.mark.parametrize("shape, pool, children, scored_widths", [
    ("s0", 50, 10, []),                 # identity: the reranker is never called
    ("s1", 50, 10, [10, 10, 10]),       # one call per question, 10 candidates each
    ("s1-wide", 500, 100, [100, 100, 100]),
])
def test_full_chain_writes_schema_valid_files_with_the_route_widths(tmp_path, config, loop, shape,
                                                                    pool, children, scored_widths):
    """every shape runs embed→manifest; widths come from the route's profiles, not r1's."""
    models = fixture_models()
    ctx = make_ctx(tmp_path, config, loop[shape], models=models)
    ceval.run_chain(ctx)
    out = ctx.out
    assert ceval.validate_outputs(out, ctx.bank) == []
    packets, answers = read_jsonl(out / "packets.jsonl"), read_jsonl(out / "answers.jsonl")
    assert [p["question_id"] for p in packets] == [a["question_id"] for a in answers] == ctx.bank_ids()
    assert all(1 <= len(p["contexts"]) <= 5 for p in packets)
    retrieved = read_jsonl(ctx.route_dir / "retrieve.jsonl")
    assert {r["candidate_count"] for r in retrieved} == {pool}
    assert {len(r["kept"]) for r in retrieved} == {children}
    assert models.rerank_widths == scored_widths
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["route"]["route_id"] == manifest["route"]["route_id_recomputed"] == ctx.route.route_id
    assert manifest["counts"]["questions"] == 3 and manifest["counts"]["VALIDATED"] == 3
    assert (manifest["reranker"] == "identity") is (shape == "s0")
    assert manifest["bank"]["sha256"] == ctx.bank_sha256
    assert manifest["validate"] == "passed"


@pytest.mark.parametrize("shape", ["s0", "s1", "s1-wide"])
def test_eval_packets_are_the_packets_v1_ask_serves(tmp_path, config, loop, shape):
    """the eval composes the SAME pipeline functions as `/v1/ask` — packets are identical."""
    bank = None
    if shape == "s1-wide":
        # One question keeps the 500→100 dedup affordable here (~4 s per question on CPU); the
        # full three-question chain of this shape runs in the width test above.
        bank = tmp_path / "one-question.jsonl"
        _rewrite(bank, fixture_bank()[:1])
    ctx = make_ctx(tmp_path, config, loop[shape], bank=bank)
    ceval.run_chain(ctx, ("embed", "retrieve", "rerank", "answer"))
    route = ctx.route
    deps = pipeline.Deps.from_files(config, EvalFixtureStore.from_fixture(), fixture_models(),
                                    FakeLLM(), dict(route.profiles), route.prompt_path)
    ours = {row["question_id"]: row for row in read_jsonl(ctx.out / "packets.jsonl")}
    for row in ctx.bank:
        served = pipeline.ask(deps, AskRequest(question=row["text"], lang=row["language"],
                                               question_id=row["question_id"]))
        assert served.packet == ours[row["question_id"]], row["question_id"]


# ================================================================ golden replay, real HTTP client

def replay_llm(golden_raw: dict[int, dict]) -> tuple[OpenAIChatLLM, list[dict]]:
    """`OpenAIChatLLM` over `httpx.MockTransport` answering with the fixture's recorded replies."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "qwen38"}]})
        body = json.loads(request.content)
        seen.append(body)
        event = golden_raw[body["seed"]]
        return httpx.Response(200, content=json.dumps(event["response"]).encode("utf-8"),
                              headers={"Content-Type": "application/json"})

    return OpenAIChatLLM("http://answerer.test/v1", expect_model="qwen38",
                         transport=httpx.MockTransport(handler)), seen


def test_the_r1_shape_replays_the_loops_answer_run_through_the_real_client(tmp_path, config, loop):
    """bodies hash to the loop's `request_sha256`, raw files carry the loop's names, and the
    answer rows equal the loop's golden rows — through `OpenAIChatLLM`, no model server."""
    raw_events = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((FIXTURES / "golden" / "raw").glob("*.json"))]
    by_seed = {event["request"]["seed"]: event for event in raw_events}
    llm, seen = replay_llm(by_seed)
    ctx = make_ctx(tmp_path, config, loop["r1"], llm=llm)
    ceval.run_chain(ctx)
    # The loop's rows, unchanged: r3 reverts the one r2 change (conftest R2_ONLY_GOLDEN_RESCUE).
    golden = {row["question_id"]: row for row in read_jsonl(FIXTURES / "golden" / "answers.jsonl")}
    assert {row["question_id"]: row for row in read_jsonl(ctx.out / "answers.jsonl")} == golden
    # r3 validates nothing earlier than the loop did, so every recorded attempt is sent and written.
    sent = raw_events
    assert len(raw_events) == 6 and len(seen) == len(sent) == 6
    written = sorted(p.name for p in (ctx.out / "raw").glob("*.json"))
    assert written == sorted(ceval._raw_name(e["question_id"], int(e["attempt"])) for e in sent)
    for event in sent:
        ours = json.loads((ctx.out / "raw" / ceval._raw_name(event["question_id"], event["attempt"]))
                          .read_text(encoding="utf-8"))
        assert ours["request_sha256"] == event["request_sha256"]
        assert ours["structural_errors"] == event["structural_errors"]
    manifest = json.loads((ctx.out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["answer"]["models_id"] == "qwen38"
    assert manifest["counts"]["STRUCTURE_FAILED"] == 2          # as the loop's r1; r3 reverts r2's one lift
    assert manifest["counts"]["structural_error_histogram"]["all_attempts"]


def _status_llm(status: int) -> OpenAIChatLLM:
    """`OpenAIChatLLM` whose server lists qwen38 and answers every chat call with HTTP `status`."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "qwen38"}]})
        return httpx.Response(status, json={"error": {"message": "request exceeds the available context size"}})

    return OpenAIChatLLM("http://answerer.test/v1", expect_model="qwen38", transport=httpx.MockTransport(handler))


def test_a_request_rejected_with_http_400_fails_its_question_instead_of_wedging_the_run(tmp_path, config, loop):
    """a 400 (e.g. context overflow) was a transport stop, so every resume stopped at the
    same question. Now each question is written STRUCTURE_FAILED with `answerer_rejected_request` and the run ends."""
    ctx = make_ctx(tmp_path, config, loop["r1"], llm=_status_llm(400))
    ceval.run_chain(ctx)
    rows = read_jsonl(ctx.out / "answers.jsonl")
    assert rows and all(row["state"] == "STRUCTURE_FAILED" for row in rows)
    assert all(pipeline.ANSWERER_REJECTED_CODE in (row["errors"] or []) for row in rows)
    assert not (ctx.route_dir / "transport-failures.jsonl").exists()


@pytest.mark.parametrize("status", [404, 429, 500])
def test_endpoint_level_http_errors_still_stop_the_run_as_transport_failures(tmp_path, config, loop, status):
    """a wrong URL (404), rate limit (429) or server error (500) must stop the run, never be written
    as a failed answer for every question."""
    ctx = make_ctx(tmp_path, config, loop["r1"], llm=_status_llm(status))
    ceval.run_chain(ctx, ("embed", "retrieve", "rerank"))
    with pytest.raises(ceval.TransportStopped):
        ceval.run_phase("answer", ctx)
    assert not (ctx.out / "answers.jsonl").exists() or not read_jsonl(ctx.out / "answers.jsonl")


def test_the_client_refuses_to_start_on_the_wrong_model(tmp_path, config, loop):
    """`/v1/models` listing another model stops the phase before any chat call."""
    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "some-other-model"}]})

    llm = OpenAIChatLLM("http://answerer.test/v1", expect_model="qwen38",
                        transport=httpx.MockTransport(handler))
    ctx = make_ctx(tmp_path, config, loop["s0"], llm=llm)
    ceval.run_chain(ctx, ("embed", "retrieve", "rerank"))
    with pytest.raises(ceval.Refused, match="not ready"):
        ceval.run_phase("answer", ctx)
    assert not (ctx.out / "answers.jsonl").exists()


# ================================================================ crash, transport, resume

def test_a_crash_mid_answer_resumes_without_duplicates_or_re_calls(tmp_path, config, loop):
    """two questions done, the process dies on the third; the rerun calls only the third."""
    ctx = make_ctx(tmp_path, config, loop["s1"], llm=CrashingLLM(crash_on=3))
    ceval.run_chain(ctx, ("embed", "retrieve", "rerank"))
    with pytest.raises(RuntimeError, match="simulated crash"):
        ceval.run_phase("answer", ctx)
    assert len(read_jsonl(ctx.phase_file("answer"))) == 2
    assert not (ctx.out / "answers.jsonl").exists()          # the loop files appear only when complete
    resumed = FakeLLM()
    ctx.stack = ceval.Stack(store=EvalFixtureStore.from_fixture(), models=fixture_models(), llm=resumed)
    ceval.run_phase("answer", ctx)
    assert len(resumed.calls) == 1
    work_ids = [row["question_id"] for row in read_jsonl(ctx.phase_file("answer"))]
    assert sorted(work_ids) == sorted(ctx.bank_ids()) and len(work_ids) == 3
    ceval.run_phase("validate", ctx)
    # A second complete rerun is free: nothing is re-asked.
    again = FakeLLM()
    ctx.stack = ceval.Stack(store=EvalFixtureStore.from_fixture(), models=fixture_models(), llm=again)
    ceval.run_phase("answer", ctx)
    assert again.calls == []


def test_a_transport_failure_stops_with_resumable_state_and_no_fake_row(tmp_path, config, loop):
    """the answerer drops on question 2 → exit 75, no row for it, the event kept in the work
    dir; the rerun asks only questions 2 and 3."""
    ctx = make_ctx(tmp_path, config, loop["s0"], llm=CrashingLLM(crash_on=2, transport=True))
    ceval.run_chain(ctx, ("embed", "retrieve", "rerank"))
    with pytest.raises(ceval.TransportStopped) as stopped:
        ceval.run_phase("answer", ctx)
    assert stopped.value.exit_code == ceval.EXIT_TRANSPORT == 75
    rows = read_jsonl(ctx.phase_file("answer"))
    assert [row["answer"]["state"] for row in rows] == ["VALIDATED"]
    assert not any(row["answer"]["state"] == "TRANSPORT_FAILED" for row in rows)
    failures = read_jsonl(ctx.route_dir / "transport-failures.jsonl")
    assert len(failures) == 1 and "connection refused" in failures[0]["event"]["transport_error"]
    assert not (ctx.out / "answers.jsonl").exists()
    assert len(list((ctx.out / "raw").glob("*.json"))) == 1      # only the finished question's
    resumed = FakeLLM()
    ctx.stack = ceval.Stack(store=EvalFixtureStore.from_fixture(), models=fixture_models(), llm=resumed)
    ceval.run_phase("answer", ctx)
    assert len(resumed.calls) == 2
    assert ceval.validate_outputs(ctx.out, ctx.bank) == []


def test_an_unreachable_answerer_is_detected_before_any_call(tmp_path, config, loop):
    """health "down" stops the phase with exit 75 and zero chat calls."""
    llm = DownLLM()
    ctx = make_ctx(tmp_path, config, loop["s0"], llm=llm)
    ceval.run_chain(ctx, ("embed", "retrieve", "rerank"))
    with pytest.raises(ceval.TransportStopped):
        ceval.run_phase("answer", ctx)
    assert llm.calls == []


def test_raw_events_of_an_unfinished_question_are_replaced_not_kept(tmp_path, config, loop):
    """a stale `attempt-02` from an earlier run is removed when the question is answered
    in one attempt, so raw/ holds exactly the events of the rows written."""
    ctx = make_ctx(tmp_path, config, loop["s0"])
    ceval.run_chain(ctx, ("embed", "retrieve", "rerank"))
    qid = ctx.bank_ids()[0]
    stale = ctx.out / "raw" / ceval._raw_name(qid, 2)
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("{}", encoding="utf-8")
    ceval.run_phase("answer", ctx)
    assert not stale.exists()
    assert ceval.validate_outputs(ctx.out, ctx.bank) == []


# ================================================================ the release pin

def test_a_release_change_between_phases_is_refused(tmp_path, config, loop):
    """retrieve pins release A; the store switches to B; rerank refuses and writes nothing."""
    store = EvalFixtureStore.from_fixture()
    ctx = make_ctx(tmp_path, config, loop["s1"], store=store)
    ceval.run_chain(ctx, ("embed", "retrieve"))
    store.switch_release("all/release-attempt-9999")
    with pytest.raises(ceval.Refused, match="active release changed"):
        ceval.run_phase("rerank", ctx)
    assert not ctx.phase_file("rerank").exists()


def test_a_reload_under_the_same_release_id_is_also_a_change(tmp_path, config, loop):
    """`dds load --replace` keeps the id and changes the manifest — still refused."""
    store = EvalFixtureStore.from_fixture()
    ctx = make_ctx(tmp_path, config, loop["s1"], store=store)
    ceval.run_chain(ctx, ("embed", "retrieve"))
    store.switch_release(store.active_release().release_attempt_id, manifest_sha256="f" * 64)
    with pytest.raises(ceval.Refused, match="active release changed"):
        ceval.run_phase("retrieve", ctx)


def test_a_release_built_from_other_chunks_is_refused_for_the_route(tmp_path, config, loop):
    """a release whose manifest names another chunk profile is not this route's store."""
    store = EvalFixtureStore.from_fixture(manifest={
        "product_manifest_sha256": "a" * 64,
        "attempts": {"chunk": {"profile_sha256": "b" * 64}}})
    ctx = make_ctx(tmp_path, config, loop["s1"], store=store)
    ceval.run_phase("embed", ctx)
    with pytest.raises(ceval.Refused, match="other chunk profile"):
        ceval.run_phase("retrieve", ctx)


def test_a_release_that_names_no_profile_shas_is_refused_unless_explicitly_allowed(tmp_path, config, loop):
    """missing shas were 'unknown' and passed, so a staged release built with
    another profile could be measured and labelled as this route. Now unknown is refused by default."""
    store = EvalFixtureStore.from_fixture(manifest={"product_manifest_sha256": "a" * 64})
    ctx = make_ctx(tmp_path, config, loop["s1"], store=store, allow_unknown_release_profiles=False)
    ceval.run_phase("embed", ctx)
    with pytest.raises(ceval.Refused, match="does not name its chunk, represent, embed profile"):
        ceval.run_phase("retrieve", ctx)
    allowed = make_ctx(tmp_path, config, loop["s1"], store=EvalFixtureStore.from_fixture(), work="allowed")
    ceval.run_chain(allowed, ("embed", "retrieve"))


def test_the_manifest_carries_the_pinned_release_and_product_sha(tmp_path, config, loop):
    """MANIFEST names the release and the product manifest sha the answers came from."""
    route = ceval.load_route(loop["s1"])
    store = EvalFixtureStore.from_fixture(manifest={
        "product_manifest_sha256": "a" * 64,
        "attempts": {slot: {"profile_sha256": route.profile_sha256[slot]}
                     for slot in ("chunk", "represent", "embed")}})
    ctx = make_ctx(tmp_path, config, loop["s1"], store=store)
    ceval.run_chain(ctx)
    manifest = json.loads((ctx.out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["release"]["product_manifest_sha256"] == "a" * 64
    assert manifest["release"]["release_attempt_id"] == store.active_release().release_attempt_id


def test_the_manifest_counts_answers_that_needed_the_second_attempt(tmp_path, config, loop):
    """The MANIFEST must say which validator judged the answers (ceres-validate-r3
    with its guards) and how many answers it sent to the second attempt; losing either
    hides the validator's effect."""
    ctx = make_ctx(tmp_path, config, loop["s0"], llm=FakeLLM(failures=1))
    ceval.run_chain(ctx)
    manifest = json.loads((ctx.out / "MANIFEST.json").read_text(encoding="utf-8"))
    counts = manifest["counts"]
    assert manifest["answer"]["validator_revision"] == "ceres-validate-r3"
    assert manifest["answer"]["validator_guards"]["max_gap_chars"] == 200
    assert "gap_forbids_capitalized_word" in manifest["answer"]["validator_guards"]
    # FakeLLM(failures=1) fails the run's first call only: one question needs a second attempt.
    assert counts["answers_by_attempts"] == {"1": counts["questions"] - 1, "2": 1}
    assert counts["validated_on_second_attempt"] == 1


def test_the_manifest_states_where_the_read_predicate_runs_and_carries_the_notes(tmp_path, config, loop):
    """Parity: the scorecard label depends on the predicate running
    before dedup/top-k, and ruling notes (e.g. a question the loop found no hits for) must reach the
    MANIFEST; a malformed notes file must be refused, never silently dropped."""
    notes = tmp_path / "notes.json"
    notes.write_text(json.dumps(["bank question c114b6c7 has no hits in the loop run"]), encoding="utf-8")
    ctx = make_ctx(tmp_path, config, loop["s0"])
    ctx.notes_file = notes
    ceval.run_chain(ctx)
    manifest = json.loads((ctx.out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["store"]["read_predicate_applied"].startswith("before dedup and top-k")
    assert manifest["notes"] == ["bank question c114b6c7 has no hits in the loop run"]
    notes.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ceval.Refused, match="JSON list of strings"):
        ceval.run_phase("manifest", ctx)


# ================================================================ validate against the bank

@pytest.fixture
def finished(tmp_path, config, loop):
    ctx = make_ctx(tmp_path, config, loop["s0"])
    ceval.run_chain(ctx, ("embed", "retrieve", "rerank", "answer"))
    assert ceval.validate_outputs(ctx.out, ctx.bank) == []
    return ctx


def _rewrite(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.parametrize("damage, expected", [
    ("missing", "bank ids missing"),
    ("duplicated", "duplicated question ids"),
    ("extra", "not in the bank"),
])
def test_validate_fails_on_a_missing_duplicated_or_extra_id(finished, damage, expected):
    """the judge joins on the bank's ids; any id drift must fail validate (exit 1)."""
    answers = read_jsonl(finished.out / "answers.jsonl")
    if damage == "missing":
        answers = answers[:-1]
    elif damage == "duplicated":
        answers = answers + [answers[0]]
    else:
        answers = answers + [{**answers[0], "question_id": "e" * 64}]
    _rewrite(finished.out / "answers.jsonl", answers)
    findings = ceval.validate_outputs(finished.out, finished.bank)
    assert any(expected in finding for finding in findings), findings
    with pytest.raises(ceval.Invalid):
        ceval.run_phase("validate", finished)


def test_validate_fails_when_a_packet_rewrites_the_question_or_language(finished):
    """same ids but other text/language is a different question to the judge."""
    packets = read_jsonl(finished.out / "packets.jsonl")
    packets[0]["question"] += " (edited)"
    packets[1]["language"] = "ro" if packets[1]["language"] == "en" else "en"
    _rewrite(finished.out / "packets.jsonl", packets)
    findings = ceval.validate_outputs(finished.out, finished.bank)
    assert any("question text differs" in f for f in findings)
    assert any("language differs" in f for f in findings)


def test_validate_rechecks_a_validated_candidate(finished):
    """a VALIDATED row whose citation is not in its packet fails validate."""
    answers = read_jsonl(finished.out / "answers.jsonl")
    answers[0]["candidate"]["citations"][0]["evidence_id"] = "E9"
    _rewrite(finished.out / "answers.jsonl", answers)
    assert any("VALIDATED but the validator" in f for f in ceval.validate_outputs(finished.out, finished.bank))


def test_the_manifest_carries_the_tool_written_live_server_block_and_an_empty_attested_list(finished):
    """Contract: the tool's own live reading sits under `live_server` with `tool` naming ceres-eval
    and its version; hand-recorded values must never appear there, so a fresh tool run writes `attested: []`."""
    ceval.run_phase("validate", finished)
    ceval.run_phase("manifest", finished)
    manifest = json.loads((finished.out / "MANIFEST.json").read_text(encoding="utf-8"))
    live = manifest["live_server"]
    assert set(live) >= {"n_ctx", "models", "read_at", "tool"}
    assert live["tool"].startswith("ceres-eval ") and live["tool"].split()[1] == ceres.__version__
    assert manifest["attested"] == []


def test_a_hand_corrected_manifest_is_never_overwritten(finished):
    """A rerun of `manifest` must not silently undo an owner's hand correction."""
    ceval.run_phase("validate", finished)
    ceval.run_phase("manifest", finished)
    path = finished.out / "MANIFEST.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["hand_corrected"] = {"at": "2026-09-26", "by": "owner", "fields": ["answer.validator_revision"]}
    path.write_text(json.dumps(document), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ceval.Refused, match="hand-corrected"):
        ceval.run_phase("manifest", finished)
    assert path.read_bytes() == before


def test_the_manifest_is_refused_when_validate_fails(finished):
    """MANIFEST.json exists only for validated outputs."""
    answers = read_jsonl(finished.out / "answers.jsonl")
    _rewrite(finished.out / "answers.jsonl", answers[:-1])
    with pytest.raises(ceval.Invalid, match="refusing to write MANIFEST"):
        ceval.run_phase("manifest", finished)
    assert not (finished.out / "MANIFEST.json").exists()


def test_the_cli_validate_exits_non_zero_on_a_mismatch(finished, loop, bank_file):
    """the CLI's exit code is what the owner's script checks."""
    answers = read_jsonl(finished.out / "answers.jsonl")
    _rewrite(finished.out / "answers.jsonl", answers[:-1])
    code = ceval.main(["validate", "--fake", "--scratch", "--route", str(loop["s0"]),
                       "--bank", str(bank_file), "--work-dir", str(finished.work_dir),
                       "--out", str(finished.out)])
    assert code == ceval.EXIT_INVALID


def test_a_bank_with_a_duplicated_id_is_refused(tmp_path):
    """the eval never answers a bank whose ids are not unique."""
    rows = fixture_bank()
    path = tmp_path / "bank.jsonl"
    _rewrite(path, rows + [rows[0]])
    with pytest.raises(ceval.Refused, match="duplicate question_id"):
        ceval.load_bank(path)


# ================================================================ route identity

def test_a_route_whose_id_does_not_match_its_profiles_is_refused(tmp_path, loop, bank_file):
    """a declared id that its seven profiles do not hash to never labels answers."""
    doc = yaml.safe_load(loop["s1"].read_text(encoding="utf-8"))
    doc["route_id"] = "0" * 64
    loop["s1"].write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ceval.Refused, match="does not match its seven profiles"):
        ceval.load_route(loop["s1"])
    code = ceval.main(["embed", "--fake", "--scratch", "--route", str(loop["s1"]), "--bank",
                       str(bank_file), "--work-dir", str(tmp_path / "w")])
    assert code == ceval.EXIT_REFUSED


def test_an_edited_profile_changes_the_recomputed_id_and_is_refused(loop):
    """editing a profile after the route was frozen breaks the id check."""
    profile = ceval.load_route(loop["s1-wide"]).profiles["retrieve"]
    profile.write_text(profile.read_text(encoding="utf-8").replace("candidate_pool: 500", "candidate_pool: 400"),
                       encoding="utf-8")
    with pytest.raises(ceval.Refused):
        ceval.load_route(loop["s1-wide"])


# ================================================================ OCR citation counts

def test_the_ocr_citation_count_counts_cited_ocr_contexts_only(tmp_path, config, loop):
    """mark q1's E1 (FakeLLM cites E1) and q2's E2 (packed, never cited) as OCR-admitted.
    Cited counts follow E1 only; the packet count follows every packed OCR context."""
    probe = make_ctx(tmp_path, config, loop["s0"], work="probe", out=tmp_path / "probe-out")
    ceval.run_chain(probe, ("embed", "retrieve", "rerank", "answer"))
    packets = read_jsonl(probe.out / "packets.jsonl")
    marked = {packets[0]["contexts"][0]["chunk_id"], packets[1]["contexts"][1]["chunk_id"]}
    expected_cited = sum(p["contexts"][0]["chunk_id"] in marked for p in packets)
    expected_packed = sum(any(c["chunk_id"] in marked for c in p["contexts"]) for p in packets)
    assert packets[1]["contexts"][0]["chunk_id"] not in marked    # q2 packs OCR but cites none
    assert 1 <= expected_cited < expected_packed                  # the test can tell them apart
    rows = [dict(row, evidence_basis="ocr_admitted_unverified") if row["chunk_id"] in marked else row
            for row in read_jsonl(FIXTURES / "items.jsonl")]
    store = EvalFixtureStore.from_fixture(rows=rows)
    ctx = make_ctx(tmp_path, config, loop["s0"], store=store)
    ceval.run_chain(ctx)
    counts = json.loads((ctx.out / "MANIFEST.json").read_text(encoding="utf-8"))["counts"]
    assert counts["answers_citing_ocr_admitted_unverified"] == expected_cited
    assert counts["cited_ocr_admitted_unverified_contexts"] == expected_cited
    assert counts["packets_with_ocr_admitted_unverified_contexts"] == expected_packed
    assert len(read_jsonl(ctx.route_dir / "ocr-citations.jsonl")) == expected_cited
    evidence = [c for p in read_jsonl(ctx.out / "packets.jsonl") for c in p["contexts"]
                if c.get("evidence_basis")]
    assert {c["chunk_id"] for c in evidence} == marked


# ================================================================ where a run may write

RUN_ANSWERS = Path("runs") / "answers"      # a neutral --answers-dir, relative to the loop root


def test_a_real_run_writes_only_to_the_answers_dir_folder_of_its_route(tmp_path, config, loop):
    """without --scratch, --out must be <loop>/<--answers-dir>/<route>."""
    answers_root = tmp_path / "loop" / RUN_ANSWERS
    for wrong in (tmp_path / "elsewhere", answers_root / "s1",            # another route's folder
                  tmp_path / "loop" / "04_rag" / "runs" / "s0"):
        ctx = make_ctx(tmp_path, config, loop["s0"], out=wrong, scratch=False, answers_dir=RUN_ANSWERS)
        with pytest.raises(ceval.Refused, match="real run writes only"):
            ceval.guard_out(ctx)
    ctx = make_ctx(tmp_path, config, loop["s0"], out=answers_root / "s0", scratch=False,
                   answers_dir=RUN_ANSWERS)
    answers_root.mkdir(parents=True)           # the caller creates the run folder
    ceval.run_chain(ctx, ("embed", "retrieve", "rerank", "answer", "validate"))
    assert (answers_root / "s0" / "answers.jsonl").is_file()
    with pytest.raises(ceval.Refused, match="answerer-cmd-file"):
        ceval.run_phase("manifest", ctx)
    command = tmp_path / "answerer.cmd"
    command.write_text("llama-server -c 16384 --cache-type-k f16 --cache-type-v f16\n", encoding="utf-8")
    ctx.answerer_cmd_file = command
    ceval.run_phase("manifest", ctx)
    manifest = json.loads((answers_root / "s0" / "MANIFEST.json").read_text(encoding="utf-8"))
    assert "-c 16384" in manifest["answer"]["answerer_start_command"]["text"]
    assert manifest["scratch"] is False


def test_scratch_never_writes_inside_the_loop_and_the_work_dir_lies_outside(tmp_path, config, loop):
    """--scratch widens where --out may go, but never into the loop; the work dir neither."""
    ctx = make_ctx(tmp_path, config, loop["s0"], out=tmp_path / "loop" / "scratch-out")
    with pytest.raises(ceval.Refused, match="outside the loop"):
        ceval.guard_out(ctx)
    ctx = make_ctx(tmp_path, config, loop["s0"], work="loop/work")
    with pytest.raises(ceval.Refused, match="--work-dir must lie outside the loop"):
        ceval.run_phase("embed", ctx)


def test_fake_answers_need_scratch(tmp_path, loop, bank_file):
    """the CLI's fake stack cannot write into the real answers folder."""
    code = ceval.main(["chain", "--fake", "--route", str(loop["s0"]), "--bank", str(bank_file),
                       "--work-dir", str(tmp_path / "w"), "--out", str(tmp_path / "o")])
    assert code == ceval.EXIT_REFUSED


# ================================================================ caching

def test_embed_is_shared_across_routes_and_retrieve_rows_are_cached(tmp_path, config, loop):
    """routes with one embed profile share one embedding; a rerun computes nothing."""
    models = fixture_models()
    for shape in ("s0", "s1", "r1"):
        ctx = make_ctx(tmp_path, config, loop[shape], models=models)
        ceval.run_chain(ctx, ("embed", "retrieve"))
    assert models.embedded_rows == 3
    assert ceval.run_phase("retrieve", ctx) == {"computed": 0, "cached": 3}


def test_changed_query_vectors_invalidate_the_downstream_rows(tmp_path, config, loop):
    """`--force-phase embed` with other vectors makes retrieve recompute instead of reusing."""
    ctx = make_ctx(tmp_path, config, loop["s1"])
    ceval.run_chain(ctx, ("embed", "retrieve", "rerank"))
    shifted = fixture_models()
    shifted.vectors = {k: np.roll(v, 1) for k, v in shifted.vectors.items()}
    ctx.stack = ceval.Stack(store=EvalFixtureStore.from_fixture(), models=shifted, llm=FakeLLM())
    ctx.force = {"embed"}
    assert ceval.run_phase("embed", ctx)["computed"] == 3
    assert list(ctx.embed_dir.glob("embed.jsonl.forced-*"))          # the old rows are kept aside
    assert ceval.run_phase("retrieve", ctx)["computed"] == 3
    assert ceval.run_phase("rerank", ctx)["computed"] == 3


def test_rerank_refuses_to_run_before_retrieve_is_complete(tmp_path, config, loop):
    """a phase never runs on a partial upstream (it would silently answer fewer questions)."""
    ctx = make_ctx(tmp_path, config, loop["s1"])
    ceval.run_phase("embed", ctx)
    with pytest.raises(ceval.Refused, match="retrieve phase is not complete"):
        ceval.run_phase("rerank", ctx)


def test_a_torn_last_line_is_dropped_but_a_torn_middle_line_is_corruption(tmp_path):
    """a crash mid-append tears only the last line; anything else is refused."""
    path = tmp_path / "phase.jsonl"
    path.write_text('{"question_id": "a"}\n{"question_id": "b', encoding="utf-8")
    assert [row["question_id"] for row in ceval.read_rows(path)] == ["a"]
    path.write_text('{"question_id": "a\n{"question_id": "b"}\n', encoding="utf-8")
    with pytest.raises(ceval.Refused, match="corrupt"):
        ceval.read_rows(path)


def test_a_different_bank_in_the_same_work_dir_is_refused(tmp_path, config, loop):
    """one work dir, one bank — caches keyed by question id must not cross banks."""
    ctx = make_ctx(tmp_path, config, loop["s0"])
    ceval.run_phase("embed", ctx)
    other = tmp_path / "other-bank.jsonl"
    _rewrite(other, fixture_bank()[:2])
    ctx2 = make_ctx(tmp_path, config, loop["s0"], bank=other)
    with pytest.raises(ceval.Refused, match="pinned to bank"):
        ceval.run_phase("embed", ctx2)


# ================================================================ the shipped schema copy

def test_the_shipped_answers_schema_is_the_loops_byte_for_byte():
    """validate uses `ceres/schemas/answers.schema.json`; it must be the loop's file."""
    shipped = (ROOT / "ceres" / "schemas" / ANSWER_SCHEMA).read_bytes()
    assert hashlib.sha256(shipped).hexdigest() == ceval.ANSWERS_SCHEMA_SHA256
    if LOOP_ANSWERS_SCHEMA.is_file():                   # the loop is on this box: compare bytes too
        assert LOOP_ANSWERS_SCHEMA.read_bytes() == shipped


# ================================================================ the answerer's LIVE context

def props_llm(props_reply, golden: bool = True) -> tuple[OpenAIChatLLM, dict[str, list]]:
    """The golden replay client, plus a `/props` reply: a dict (200 JSON), an int (that status),
    bytes (a 200 non-JSON body) or an exception class (the transport raises it)."""
    raw_events = [json.loads(p.read_text(encoding="utf-8"))
                  for p in sorted((FIXTURES / "golden" / "raw").glob("*.json"))]
    by_seed = {event["request"]["seed"]: event for event in raw_events}
    seen: dict[str, list] = {"paths": [], "chats": []}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["paths"].append(request.url.path)
        if request.method == "GET" and request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "qwen38"}]})
        if request.method == "GET" and request.url.path.endswith("/props"):
            if isinstance(props_reply, type) and issubclass(props_reply, Exception):
                raise props_reply("props probe failed (test)", request=request)
            if isinstance(props_reply, int):
                return httpx.Response(props_reply, text="Not Found")
            if isinstance(props_reply, bytes):
                return httpx.Response(200, content=props_reply)
            return httpx.Response(200, json=props_reply)
        body = json.loads(request.content)
        seen["chats"].append(body)
        return httpx.Response(200, content=json.dumps(by_seed[body["seed"]]["response"]).encode("utf-8"),
                              headers={"Content-Type": "application/json"})

    return OpenAIChatLLM("http://answerer.test/v1", expect_model="qwen38",
                         transport=httpx.MockTransport(handler)), seen


LLAMA_PROPS_16K = {"default_generation_settings": {"n_ctx": 16384, "temperature": 0.8},
                   "total_slots": 1, "model_path": "/models/qwen3.8-q8_0.gguf",
                   "build_info": "b7777-abcdef0", "chat_template": "{{ messages }}"}


def test_a_live_context_that_differs_from_the_profile_refuses_the_answer_phase(tmp_path, config, loop):
    """the server runs n_ctx 8192, the profile declares 16384 → exit 2 with both numbers, zero
    chat calls, no answer files (the earlier defect would have answered on 8192)."""
    llm, seen = props_llm({"default_generation_settings": {"n_ctx": 8192}})
    ctx = make_ctx(tmp_path, config, loop["r1"], llm=llm)
    ceval.run_chain(ctx, ("embed", "retrieve", "rerank"))
    with pytest.raises(ceval.Refused) as refused:
        ceval.run_phase("answer", ctx)
    assert refused.value.exit_code == ceval.EXIT_REFUSED == 2
    assert "8192" in str(refused.value) and "16384" in str(refused.value)
    assert seen["chats"] == []
    assert not (ctx.out / "answers.jsonl").exists()
    assert not ctx.phase_file("answer").exists()


@pytest.mark.parametrize("props_reply", [
    404,                                                   # an OpenAI server without /props (vLLM)
    {"slots": 1, "model": "qwen38"},                       # JSON of an unknown shape
    b"<html>props</html>",                                 # a 200 that is not JSON
    [16384],                                               # JSON, but not an object
    {"default_generation_settings": {"n_ctx": True}},      # a bool is not a context size
    httpx.ReadTimeout,                                     # the probe itself times out
])
def test_an_unknown_props_shape_does_not_break_answering_and_is_recorded_unavailable(
        tmp_path, config, loop, props_reply):
    """a server whose /props we cannot read still answers; the run records `available: false`
    with a reason, and the manifest says the live context is unknown (null), not "matches"."""
    llm, seen = props_llm(props_reply)
    ctx = make_ctx(tmp_path, config, loop["r1"], llm=llm)
    ceval.run_chain(ctx)
    assert len(seen["chats"]) == 6
    endpoint = json.loads((ctx.route_dir / "answer-endpoint.json").read_text(encoding="utf-8"))
    assert endpoint["server_props"]["available"] is False and endpoint["server_props"]["reason"]
    answer = json.loads((ctx.out / "MANIFEST.json").read_text(encoding="utf-8"))["answer"]
    assert answer["live_props"]["available"] is False
    assert answer["live_n_ctx"] is None and answer["live_n_ctx_matches_profile"] is None
    assert answer["profile_context_tokens"] == 16384


def test_a_matching_live_context_passes_and_is_recorded(tmp_path, config, loop):
    """n_ctx 16384 = the profile → the run answers, and answer-endpoint.json + MANIFEST carry
    the live snapshot (n_ctx, model path, build) read from /props at the server ROOT, not /v1."""
    llm, seen = props_llm(LLAMA_PROPS_16K)
    ctx = make_ctx(tmp_path, config, loop["r1"], llm=llm)
    ceval.run_chain(ctx)
    assert "/props" in seen["paths"] and "/v1/props" not in seen["paths"]
    endpoint = json.loads((ctx.route_dir / "answer-endpoint.json").read_text(encoding="utf-8"))
    assert endpoint["models_id"] == "qwen38" and endpoint["server_props"]["n_ctx"] == 16384
    answer = json.loads((ctx.out / "MANIFEST.json").read_text(encoding="utf-8"))["answer"]
    assert answer["live_n_ctx"] == 16384 and answer["live_n_ctx_matches_profile"] is True
    assert answer["live_props"]["model_path"] == "/models/qwen3.8-q8_0.gguf"
    assert answer["live_props"]["build_info"] == "b7777-abcdef0"
    assert answer["live_props"]["fetched_at"]
    assert "chat_template" not in answer["live_props"]         # a snapshot, not the whole body


@pytest.mark.parametrize("payload, expected", [
    ({"default_generation_settings": {"n_ctx": 16384}, "n_ctx": 4096}, 16384),  # dgs wins
    ({"default_generation_settings": {}, "n_ctx": 4096}, 4096),                  # top-level fallback
    ({"n_ctx": 0}, None),                                                        # 0 is not a context
    ({"default_generation_settings": {"n_ctx": "16384"}}, None),                 # a string is not a number
])
def test_props_parsing_takes_the_tasks_n_ctx_order_and_never_guesses(payload, expected):
    """n_ctx = default_generation_settings.n_ctx, else top-level; anything else is unknown."""
    from ceres.llm import parse_server_props

    props = parse_server_props(payload, "2026-09-26T00:00:00Z")
    assert props.get("n_ctx") == expected
    assert props["available"] is (expected is not None)


@pytest.mark.parametrize("props_reply", [404, 500, b"not json", httpx.ConnectError])
def test_the_clients_server_props_itself_never_raises(props_reply):
    """`OpenAIChatLLM.server_props()` returns `available: false` on a non-2xx, a non-JSON body
    or a transport error — callers other than `live_server_props` rely on it not raising."""
    llm, _seen = props_llm(props_reply)
    props = llm.server_props()
    assert props["available"] is False and props["reason"] and props["fetched_at"]


def test_a_client_without_server_props_is_unknown_not_a_crash():
    """FakeLLM (and any client without `server_props`, or one that raises) → unavailable."""
    from ceres.llm import context_check, live_server_props

    class Raising(FakeLLM):
        def server_props(self):
            raise RuntimeError("boom")

    for llm in (FakeLLM(), Raising()):
        props = live_server_props(llm)
        assert props["available"] is False
        assert context_check(props, {"hardware_flags": {"context_tokens": 16384}})[
            "live_n_ctx_matches_profile"] is None


# ================================================================ the validator revision a run used

ELLIPSIS_PASSAGE = ("Winter wheat is sown in autumn on well prepared seedbeds and around the Danube "
                    "plain it is sown later in the season when soils stay warm.")


def _ellipsis_candidate(quote: str) -> tuple[dict, dict]:
    packet = {"language": "en", "contexts": [{"evidence_id": "E1", "evidence_text": ELLIPSIS_PASSAGE}]}
    candidate = {"decision": "ANSWER", "answer_language": "en",
                 "user_answer": "Winter wheat is sown in autumn.",
                 "citations": [{"claim": "Winter wheat is sown in autumn", "evidence_id": "E1",
                                "support_quote": quote}],
                 "missing_information": [], "confidence": "MEDIUM", "reason_code": "SUPPORTED"}
    return candidate, packet


def test_every_shipped_validator_revision_can_still_be_rechecked():
    """r1, r2 and r3 each resolve to their OWN rules with no fallback note. Fails when the code
    moves to r4 without adding r3 to `_historical_validators`."""
    for revision in ("ceres-validate-r1", "ceres-validate-r2", "ceres-validate-r3"):
        _validate, used, note = ceval.validator_for(revision)
        assert used == revision and note is None


def test_the_recheck_rules_differ_by_revision_exactly_where_the_revisions_differ():
    """a clean ellipsis quote passes r2/r3 but not r1; a gap holding "Danube" passes r2 only.
    Catches a historical validator that is silently the current one."""
    rules = {rev: ceval.validator_for(rev)[0]
             for rev in ("ceres-validate-r1", "ceres-validate-r2", "ceres-validate-r3")}
    clean = _ellipsis_candidate("Winter wheat is sown ... on well prepared seedbeds")
    proper_noun = _ellipsis_candidate("prepared seedbeds and ... plain it is sown later")
    assert [bool(rules[r](*clean)) for r in sorted(rules)] == [True, False, False]
    assert [bool(rules[r](*proper_noun)) for r in sorted(rules)] == [True, False, True]


def test_the_guards_copy_equals_serving_validate_with_the_same_guards(finished):
    """`_validate_with_guards(QUOTE_GUARDS)` is a copy of `pipeline.validate`; drift between
    them would re-check old answers with rules that never ran."""
    copy = ceval._validate_with_guards(pipeline.QUOTE_GUARDS)
    packets = {row["question_id"]: row for row in read_jsonl(finished.out / "packets.jsonl")}
    cases = [(row["candidate"], packets[row["question_id"]])
             for row in read_jsonl(finished.out / "answers.jsonl") if row.get("candidate")]
    cases += [_ellipsis_candidate("Winter wheat is sown ... on well prepared seedbeds"),
              _ellipsis_candidate("prepared seedbeds and ... plain it is sown later"),
              _ellipsis_candidate(""), ({"decision": "ANSWER"}, packets[next(iter(packets))])]
    for candidate, packet in cases:
        assert copy(candidate, packet) == pipeline.validate(candidate, packet)


def test_rerunning_manifest_after_a_validator_bump_keeps_the_recorded_revision(tmp_path, config, loop,
                                                                              monkeypatch):
    """the earlier defect — a manifest rerun under newer code must report the revision the
    answers were produced under, with its guards, not the code's constant."""
    ctx = make_ctx(tmp_path, config, loop["s0"])
    ceval.run_chain(ctx)
    before = json.loads((ctx.out / "MANIFEST.json").read_text(encoding="utf-8"))["answer"]
    assert before["validator_revision"] == "ceres-validate-r3"
    assert before["validator_revision_status"] == "recorded by the answer phase"
    assert before["validator_guards"] == pipeline.QUOTE_GUARDS.as_dict()
    monkeypatch.setattr(ceval, "VALIDATOR_REVISION", "ceres-validate-r4")
    ceval.run_phase("manifest", ctx)
    after = json.loads((ctx.out / "MANIFEST.json").read_text(encoding="utf-8"))["answer"]
    assert after["validator_revision"] == "ceres-validate-r3"
    assert after["validator_guards"] == before["validator_guards"]
    # r4 code no longer carries r3 in this simulation: it says so and names both revisions.
    assert after["validated_with"]["revision"] == "ceres-validate-r4"
    assert "ceres-validate-r3" in after["validated_with"]["note"]


def test_a_legacy_run_without_a_recorded_revision_reports_null_not_the_codes_value(tmp_path, config, loop):
    """work rows and answer-endpoint.json from before the fix carry no revision → the manifest
    says null + "not recorded by this run", never the current code's r3."""
    ctx = make_ctx(tmp_path, config, loop["s0"])
    ceval.run_chain(ctx, ("embed", "retrieve", "rerank", "answer"))
    rows = read_jsonl(ctx.phase_file("answer"))
    _rewrite(ctx.phase_file("answer"), [{k: v for k, v in row.items() if k != "validator_revision"}
                                        for row in rows])
    endpoint_path = ctx.route_dir / "answer-endpoint.json"
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    endpoint.pop("validator")
    endpoint.pop("server_props")
    endpoint_path.write_text(json.dumps(endpoint), encoding="utf-8")
    ceval.run_phase("manifest", ctx)
    answer = json.loads((ctx.out / "MANIFEST.json").read_text(encoding="utf-8"))["answer"]
    assert answer["validator_revision"] is None
    assert answer["validator_revision_status"] == "not recorded by this run"
    assert answer["validator_guards"] is None
    assert "recorded no validator revision" in answer["validated_with"]["note"]
    assert answer["live_props"] is None and answer["live_n_ctx"] is None


def test_validate_rechecks_with_the_recorded_r1_rules_not_the_current_ones(finished):
    """answers recorded as r1 whose quote passes only via the r2/r3 ellipsis rule must FAIL
    validate — the current rules would wave them through."""
    answers = read_jsonl(finished.out / "answers.jsonl")
    packets = {row["question_id"]: row for row in read_jsonl(finished.out / "packets.jsonl")}
    row = next(r for r in answers if r["state"] == "VALIDATED")
    packet = packets[row["question_id"]]
    citation = row["candidate"]["citations"][0]
    words = next(c["evidence_text"] for c in packet["contexts"]
                 if c["evidence_id"] == citation["evidence_id"]).split()
    r1 = ceval.validator_for("ceres-validate-r1")[0]
    quote = None
    for start in range(len(words) - 12):
        for gap in range(1, 4):
            trial = " ".join(words[start:start + 4]) + " ... " + " ".join(words[start + 4 + gap:start + 8 + gap])
            citation["support_quote"] = trial
            if not pipeline.validate(row["candidate"], packet) and r1(row["candidate"], packet):
                quote = trial
                break
        if quote:
            break
    assert quote, "no ellipsis quote in the fixture passage passes r3 but not r1"
    _rewrite(finished.out / "answers.jsonl", answers)
    assert ceval.validate_outputs(finished.out, finished.bank) == []          # current rules: fine
    ceval.run_phase("validate", finished)                                      # recorded r3: fine
    work = read_jsonl(finished.phase_file("answer"))
    _rewrite(finished.phase_file("answer"), [{**r, "validator_revision": "ceres-validate-r1"} for r in work])
    with pytest.raises(ceval.Invalid, match="validated with ceres-validate-r1"):
        ceval.run_phase("validate", finished)


# ================================================================ explicit release pin, --schema

ACTIVE_ELSEWHERE = "all/release-attempt-active-elsewhere"
# The MANIFEST `release` keys of a flagless run, as they were before --schema/--release existed.
FLAGLESS_RELEASE_KEYS = {"release_attempt_id", "manifest_sha256", "product_manifest_sha256", "route_id",
                         "row_count", "profile_sha256", "pinned_at"}


class StagedReleaseStore(EvalFixtureStore):
    """The chunk-v34 shape: the fixture's rows sit under a STAGED release (active=false) while another
    release, holding no rows here, is the active one. Counts named-release lookups."""

    def __init__(self, items, vectors, release, manifest=None):
        super().__init__(items, vectors, release, manifest)
        self.staged = release
        self._release = Release(ACTIVE_ELSEWHERE, "other-route", 0, {}, manifest_sha256="a" * 64)
        self.named_lookups: list[str] = []

    def release(self, release_attempt_id):
        self.named_lookups.append(release_attempt_id)
        return next((r for r in (self.staged, self._release)
                     if r.release_attempt_id == release_attempt_id), None)

    def release_manifest(self, release_attempt_id):
        known = (self.staged.release_attempt_id, self._release.release_attempt_id)
        return dict(self.manifest) if release_attempt_id in known else None

    def reload_staged(self, manifest_sha256):
        """`dds load --replace` of the staged release: same id, another manifest."""
        from dataclasses import replace

        self.staged = replace(self.staged, manifest_sha256=manifest_sha256)


def staged_ctx(tmp_path, config, loop, store, **kwargs):
    ctx = make_ctx(tmp_path, config, loop["s1"], store=store, **kwargs)
    ctx.release_id = store.staged.release_attempt_id
    return ctx


def test_an_explicit_pin_reads_the_named_inactive_release_not_the_active_one(tmp_path, config, loop):
    """--release must search the staged release (and record the pin), not the active one."""
    store = StagedReleaseStore.from_fixture()
    ctx = staged_ctx(tmp_path, config, loop, store)
    ceval.run_chain(ctx)
    retrieved = read_jsonl(ctx.phase_file("retrieve"))
    assert {row["release_attempt_id"] for row in retrieved} == {store.staged.release_attempt_id}
    assert all(row["kept"] for row in retrieved)                  # the staged rows were searched
    state = json.loads(ctx.state_path.read_text(encoding="utf-8"))["release"]
    assert state["release_attempt_id"] == store.staged.release_attempt_id
    assert state["pinned_explicitly"] is True and state["schema"] == config.store.db_schema
    manifest = json.loads((ctx.out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["release"]["release_attempt_id"] == store.staged.release_attempt_id
    assert manifest["release"]["pinned_explicitly"] is True
    assert manifest["release"]["schema"] == config.store.db_schema == manifest["store"]["schema"]


def test_without_flags_the_active_release_is_pinned_and_the_pin_has_no_new_keys(tmp_path, config, loop):
    """a flagless run must keep reading the ACTIVE release and never look a release up by name;
    its RUN-STATE and MANIFEST release blocks carry exactly the keys they carried before."""
    store = StagedReleaseStore.from_fixture()
    ctx = make_ctx(tmp_path, config, loop["s1"], store=store)
    ceval.run_chain(ctx, ("embed", "retrieve"))
    assert {row["release_attempt_id"] for row in read_jsonl(ctx.phase_file("retrieve"))} == {ACTIVE_ELSEWHERE}
    assert store.named_lookups == []
    state = json.loads(ctx.state_path.read_text(encoding="utf-8"))["release"]
    assert not set(ceval.PIN_MODE_KEYS) & set(state)
    ctx = make_ctx(tmp_path, config, loop["s1"], work="work-default")        # the plain fixture store
    ceval.run_chain(ctx)
    manifest = json.loads((ctx.out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert set(manifest["release"]) == FLAGLESS_RELEASE_KEYS


def test_an_explicit_pin_with_a_wrong_id_refuses_and_pins_nothing(tmp_path, config, loop):
    """a --release id the schema does not hold must refuse, never fall back to the active one."""
    store = StagedReleaseStore.from_fixture()
    ctx = staged_ctx(tmp_path, config, loop, store)
    ctx.release_id = "all/release-attempt-9999"
    ceval.run_phase("embed", ctx)
    with pytest.raises(ceval.Refused, match="does not exist in schema"):
        ceval.run_phase("retrieve", ctx)
    assert json.loads(ctx.state_path.read_text(encoding="utf-8"))["release"] is None
    assert not ctx.phase_file("retrieve").exists()


def test_a_reload_of_the_explicitly_pinned_release_between_phases_is_refused(tmp_path, config, loop):
    """the named release reloaded under the same id (another manifest sha) must stop the run."""
    store = StagedReleaseStore.from_fixture()
    ctx = staged_ctx(tmp_path, config, loop, store)
    ceval.run_chain(ctx, ("embed", "retrieve"))
    store.reload_staged("f" * 64)
    with pytest.raises(ceval.Refused, match="pinned release changed between phases"):
        ceval.run_phase("rerank", ctx)
    assert not ctx.phase_file("rerank").exists()


@pytest.mark.parametrize("first, second", [("explicit", "flagless"), ("flagless", "explicit"),
                                           ("explicit", "other-id")])
def test_a_work_dir_refuses_a_phase_pinned_another_way(tmp_path, config, loop, first, second):
    """an explicit pin followed by a flagless phase (or the reverse, or another id) would mix the
    staged and the active release in one answer set."""
    store = StagedReleaseStore.from_fixture()
    ids = {"explicit": store.staged.release_attempt_id, "flagless": None, "other-id": ACTIVE_ELSEWHERE}
    ctx = make_ctx(tmp_path, config, loop["s1"], store=store)
    ctx.release_id = ids[first]
    ceval.run_chain(ctx, ("embed", "retrieve"))
    ctx.release_id = ids[second]
    with pytest.raises(ceval.Refused, match="pinned its release with|names another"):
        ceval.run_phase("rerank", ctx)


def test_serving_never_reads_a_named_release(config, loop):
    """`/v1/ask` must answer from the active release only, never through `release(id)`."""
    store = StagedReleaseStore.from_fixture()
    store.staged, store._release = store._release, store.staged      # fixture release active now
    route = ceval.load_route(loop["s1"])
    deps = pipeline.Deps.from_files(config, store, fixture_models(), FakeLLM(),
                                    dict(route.profiles), route.prompt_path)
    row = fixture_bank()[0]
    pipeline.ask(deps, AskRequest(question=row["text"], lang=row["language"],
                                  question_id=row["question_id"]))
    assert store.named_lookups == []
    api_source = (ROOT / "ceres" / "api.py").read_text(encoding="utf-8")
    assert ".release(" not in api_source and "store.release(" not in api_source


@pytest.mark.parametrize("schema", ["ceres_other_v0; drop", "Ceres", "", "1abc"])
def test_a_schema_that_is_not_a_plain_identifier_is_refused(tmp_path, loop, bank_file, schema):
    """--schema is interpolated into SQL (schema names cannot be parameters): refuse, exit 2."""
    with pytest.raises(ceval.Refused, match="plain identifier"):
        ceval.with_schema(None, schema)
    code = ceval.main(["embed", "--fake", "--scratch", "--schema", schema, "--route", str(loop["s0"]),
                       "--bank", str(bank_file), "--work-dir", str(tmp_path / "w")])
    assert code == ceval.EXIT_REFUSED


def test_the_schema_override_changes_the_effective_config_and_nothing_else(config):
    """--schema must reach the store section (and the effective-config sha), and nothing else."""
    staged = ceval.with_schema(config, "ceres_other_v0")   # a sample name; no second live schema exists
    assert staged.store.db_schema == "ceres_other_v0" and config.store.db_schema != "ceres_other_v0"
    assert staged.config_sha256() != config.config_sha256()
    assert staged.store.model_dump(exclude={"db_schema"}) == config.store.model_dump(exclude={"db_schema"})
    assert staged.model_dump(exclude={"store"}) == config.model_dump(exclude={"store"})


# ---------------------------------------------------------------- the release pin over real PostgreSQL (disposable)

def _staged_schema(pg, schema, route_file=None):
    """`rel-active` (5 fixture rows, active) and `rel-staged` (every fixture row, active=false).

    With `route_file`, `rel-staged`'s row carries that route's real chunk/represent/embed shas. Needed since
    The eval pin now reads the row's columns, and the loader's placeholder shas
    ("0"*64, …) are, correctly, another route's store."""
    rows, vectors = fixture_rows(), fixture_vectors()
    pg.create_schema(schema)
    pg.load(schema, "rel-active", rows[:5], vectors[:5], active=True)
    shas = None
    if route_file is not None:
        route = ceval.load_route(route_file)
        shas = {slot: route.profile_sha256[slot] for slot in ("chunk", "represent", "embed")}
    pg.load(schema, "rel-staged", rows, vectors, profile_shas=shas)


def pg_ctx(tmp_path, config, loop, pg, schema):
    staged_config = ceval.with_schema(config, schema)
    ctx = make_ctx(tmp_path, staged_config, loop["s1"], store=pg.store(schema))
    ctx.release_id, ctx.schema = "rel-staged", schema
    return ctx


def test_pgstore_explicit_pin_answers_from_the_inactive_release_and_says_so(tmp_path, config, loop, pg):
    """On SQL: a --schema/--release run reads the inactive release of the named schema, and
    MANIFEST names that schema, the release and `pinned_explicitly: true`."""
    _staged_schema(pg, "v34_pin_ok", loop["s1"])
    ctx = pg_ctx(tmp_path, config, loop, pg, "v34_pin_ok")
    ceval.run_chain(ctx)
    assert {row["release_attempt_id"] for row in read_jsonl(ctx.phase_file("retrieve"))} == {"rel-staged"}
    manifest = json.loads((ctx.out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["release"]["release_attempt_id"] == "rel-staged"
    assert manifest["release"]["pinned_explicitly"] is True
    assert manifest["release"]["schema"] == manifest["store"]["schema"] == "v34_pin_ok"


def test_pgstore_reload_of_the_pinned_inactive_release_is_refused(tmp_path, config, loop, pg):
    """On SQL: `dds load --replace` of the staged release between phases must stop the run."""
    _staged_schema(pg, "v34_pin_reload", loop["s1"])
    ctx = pg_ctx(tmp_path, config, loop, pg, "v34_pin_reload")
    ceval.run_chain(ctx, ("embed", "retrieve"))
    with pg.admin() as conn:
        conn.execute("UPDATE v34_pin_reload.releases SET manifest = manifest || '{\"reloaded\": 1}'"
                     " WHERE release_attempt_id = 'rel-staged'")
    with pytest.raises(ceval.Refused, match="pinned release changed between phases"):
        ceval.run_phase("rerank", ctx)


# ================================================================ the output allowlist

def test_a_real_run_without_an_answers_dir_is_refused_even_for_an_existing_folder(tmp_path, config, loop):
    """Bug caught: a non-scratch run that writes into the loop with no declared boundary (the default is
    none, so scratch is required)."""
    answers = tmp_path / "loop" / RUN_ANSWERS
    answers.mkdir(parents=True)
    ctx = make_ctx(tmp_path, config, loop["s0"], out=answers / "s0", scratch=False)
    with pytest.raises(ceval.Refused, match="no --answers-dir"):
        ceval.guard_out(ctx)


def test_the_answers_dir_is_writable_for_the_matching_route_once_it_exists(tmp_path, config, loop):
    """<loop>/<--answers-dir>/<route> is writable only for that route and only when the caller already
    created the answers dir; the guard never creates it."""
    answers = tmp_path / "loop" / RUN_ANSWERS
    ctx = make_ctx(tmp_path, config, loop["s0"], out=answers / "s0", scratch=False, answers_dir=RUN_ANSWERS)
    with pytest.raises(ceval.Refused, match="existing run folder"):
        ceval.guard_out(ctx)
    assert not answers.exists()
    answers.mkdir(parents=True)
    assert ceval.guard_out(ctx) == (answers / "s0").resolve()
    ctx = make_ctx(tmp_path, config, loop["s0"], out=answers / "s1", scratch=False, answers_dir=RUN_ANSWERS)
    with pytest.raises(ceval.Refused, match="named after it"):
        ceval.guard_out(ctx)


@pytest.mark.parametrize("relative", [
    "runs/answers",                 # the answers dir itself
    "runs/s0",                      # beside answers/, not in it
    "runs/answers/s0/deeper",       # below the route folder
    "runs/other-answers/s0",        # another run folder
    "runs/answers-r2/s0",           # a sibling of the answers dir
    "04_rag/runs/s0",
])
def test_any_other_loop_path_is_refused(tmp_path, config, loop, relative):
    """only <answers dir>/<route> passes; every other loop path is refused, even when it exists."""
    (tmp_path / "loop" / RUN_ANSWERS).mkdir(parents=True, exist_ok=True)
    target = tmp_path / "loop" / relative
    target.mkdir(parents=True, exist_ok=True)
    ctx = make_ctx(tmp_path, config, loop["s0"], out=target, scratch=False, answers_dir=RUN_ANSWERS)
    with pytest.raises(ceval.Refused, match="real run writes only"):
        ceval.guard_out(ctx)


@pytest.mark.parametrize("answers_dir", ["../outside", ".", "/tmp"])
def test_an_answers_dir_outside_the_loop_or_the_loop_itself_is_refused(tmp_path, config, loop, answers_dir):
    """Bug caught: --answers-dir used to move the boundary out of the loop (or to its root), which would
    let a real run write anywhere."""
    ctx = make_ctx(tmp_path, config, loop["s0"], out=tmp_path / "x" / "s0", scratch=False,
                   answers_dir=Path(answers_dir))
    with pytest.raises(ceval.Refused, match="--answers-dir must be a folder inside the loop"):
        ceval.guard_out(ctx)


def test_a_set_run_needs_route_dash_set_and_a_pinned_release(tmp_path, config, loop):
    """with --set the answers folder is <route>-<set> and --release must pin the release."""
    answers = tmp_path / "loop" / RUN_ANSWERS
    answers.mkdir(parents=True)
    ctx = make_ctx(tmp_path, config, loop["s0"], out=answers / "s0", scratch=False, answers_dir=RUN_ANSWERS)
    ctx.set_name = "bank-v5"
    with pytest.raises(ceval.Refused, match="named after it"):
        ceval.guard_out(ctx)
    ctx.out = answers / "s0-bank-v5"
    with pytest.raises(ceval.Refused, match="pins its release with --release"):
        ceval.guard_out(ctx)
    ctx.release_id = "fixture"
    assert ceval.guard_out(ctx) == (answers / "s0-bank-v5").resolve()


def test_toggling_the_short_circuit_changes_the_answer_identity_but_off_keeps_the_old_one(tmp_path, config, loop):
    """An independent review found that the answer-phase identity omitted no_evidence_short_circuit, so
    resume could reuse answers made under the other setting. Off must keep the pre-fix identity (old runs resume)."""
    from ceres.contracts import VALIDATOR_REVISION
    ctx = make_ctx(tmp_path, config, loop["s0"])
    reranked = {qid: {"hits": [], "question_id": qid} for qid in ctx.bank_ids()}
    off = ceval.answer_phase_inputs(ctx, reranked, "p" * 64, False)
    on = ceval.answer_phase_inputs(ctx, reranked, "p" * 64, True)
    legacy = {qid: ceval.item_sha256({"rerank_row_sha256": ceval.item_sha256(reranked[qid]),
                                      "answer_profile_sha256": ctx.route.profile_sha256["answer"],
                                      "prompt_sha256": "p" * 64, "validator_revision": VALIDATOR_REVISION})
              for qid in ctx.bank_ids()}
    assert off == legacy
    assert all(on[qid] != off[qid] for qid in ctx.bank_ids())


def test_a_same_manifest_profile_column_change_between_phases_is_refused(tmp_path, config, loop):
    """An independent review found that after the first pin, later phases compared only (id, manifest
    sha), so a replace that changed the row's profile columns under the same manifest went unnoticed."""
    import dataclasses
    route = ceval.load_route(loop["s1"])
    shas = {slot: route.profile_sha256[slot] for slot in ("chunk", "represent", "embed")}
    store = EvalFixtureStore.from_fixture(manifest={
        "product_manifest_sha256": "a" * 64,
        "attempts": {slot: {"profile_sha256": sha} for slot, sha in shas.items()}})
    ctx = make_ctx(tmp_path, config, loop["s1"], store=store)
    ceval.run_chain(ctx, ("embed", "retrieve"))
    store._release = dataclasses.replace(store._release, profile_sha256=dict(shas, chunk="e" * 64))
    with pytest.raises(ceval.Refused, match="chunk profile sha changed between phases"):
        ceval.run_phase("rerank", ctx)
