"""The recipe: the read predicate, the three dedup rules, the request body, the validator, the
answer loop. Behaviour is asserted against the crafted inline fixture, where each rule fires once.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx
import pytest
import yaml

from ceres import pipeline
from ceres.contracts import AskRequest, VALIDATOR_REVISION
from ceres.llm import FakeLLM, LLMTransportError, OpenAIChatLLM, build_request_body, stable_seed
from ceres.normalize import normalize
from ceres.store import Item, StoreError, read_predicate_sha256

from ceres.testing import CHUNK, EN_QUESTION, LABEL_BY_CHUNK, RO_QUESTION

from conftest import EXAMPLE_CONFIG, FIXTURES, R2_ONLY_GOLDEN_RESCUE, read_jsonl, requires_fixtures


def labels(chunk_ids) -> list[str]:
    """The inline fixture's readable labels for a list of 64-hex chunk ids."""
    return [LABEL_BY_CHUNK.get(chunk_id, chunk_id) for chunk_id in chunk_ids]


def ask(deps, question: str = EN_QUESTION, lang: str = "en"):
    return pipeline.ask(deps, AskRequest(question=question, lang=lang))


# ---------------------------------------------------------------- the read predicate

def test_rnd_mode_serves_the_internal_chunk(deps):
    """The shipped default is `rnd`, and rnd reads `internal_research_only` too.

    c5 is the inline fixture's internal-only chunk and it outranks most of the pool, so rnd must
    serve it and withhold nothing. This is the whole point of the default: R&D reads everything.
    """
    response = ask(deps)
    served = set(labels(context["chunk_id"] for context in response.packet["contexts"]))
    assert deps.config.store.serving_mode == "rnd"
    assert deps.config.store.serve_rights == \
        ("redistributable", "internal_research_only", "licensed")
    assert "c5" in served
    assert response.meta.serving_mode == "rnd"
    assert response.meta.filtered.rights == 0


def test_production_mode_keeps_the_internal_chunk_out_of_the_pool(production_deps):
    """The same fixture in production mode: c5 never leaves the store, and it is counted."""
    response = ask(production_deps)
    served = set(labels(context["chunk_id"] for context in response.packet["contexts"]))
    assert production_deps.config.store.serve_rights == ("redistributable",)
    assert "c5" not in served
    assert response.meta.serving_mode == "production"
    assert response.meta.filtered.rights == 1


def test_the_two_modes_do_not_share_a_read_predicate_identity(deps, production_deps):
    """A rnd answer and a production answer are distinguishable in the record, by digest."""
    assert ask(deps).meta.read_predicate_sha256 != ask(production_deps).meta.read_predicate_sha256


def test_a_flagged_chunk_is_served_with_its_flag_in_warnings(deps):
    """`table_unvalidated` no longer filters. It reaches the judge as a warning instead."""
    contexts = {LABEL_BY_CHUNK[c["chunk_id"]]: c for c in ask(deps).packet["contexts"]}
    assert "c6" in contexts
    assert contexts["c6"]["warnings"] == ["table_unvalidated"]
    assert ask(deps).meta.filtered.excluded == 0


def test_meta_reports_the_predicate_identity_and_an_unassigned_route(deps):
    meta = ask(deps).meta
    assert meta.read_predicate_sha256 == read_predicate_sha256(
        deps.config.store.serve_rights, deps.config.store.db_schema,
        admitted_licensed_works=deps.config.store.admitted_works)
    # The served route is not frozen yet, and an answer is never labelled r1.
    assert meta.route_id is None and meta.route_status == "unassigned"
    assert meta.judged_route_id.startswith("73c2e791")
    assert meta.validator_revision == VALIDATOR_REVISION
    assert meta.lang_source == "client"


def test_a_store_with_no_active_release_stops_the_request(deps):
    class Broken:
        def active_release(self):
            raise StoreError("no active release")

    deps.store = Broken()
    with pytest.raises(StoreError):
        ask(deps)


# ---------------------------------------------------------------- dedup (slot_adapter port)

def test_the_three_dedup_rules_each_fire_once(production_deps, inline_stack):
    """c2 parent-duplicate, c3 near-duplicate, c4 too short — only c1 and c6 survive.

    Production mode, because the inline fixture was crafted so that exactly one rule fires per
    item and c5's rule is the rights predicate. In rnd, c5 is served and the dedup rules are
    unaffected by it (`test_rnd_mode_serves_the_internal_chunk` covers that side).
    """
    store, models, _ = inline_stack
    release = store.active_release()
    qvec = models.embed_query(EN_QUESTION, "en")
    candidates = pipeline.retrieve(store, qvec, release,
                                   serve_rights=production_deps.config.store.serve_rights, limit=50)
    assert labels(item.chunk_id for item in candidates) == ["c1", "c6", "c3", "c4", "c2"]
    texts = {item.chunk_id: item.payload for item in candidates}
    dropped: list[dict] = []
    kept = pipeline.dedup(candidates, texts, pipeline.NEAR_DUPLICATE_RULE, dropped=dropped)
    assert labels(item.chunk_id for item in kept) == ["c1", "c6"]
    assert [(LABEL_BY_CHUNK[row["dropped_chunk_id"]], row["reason"]) for row in dropped] == \
           [("c3", "near_duplicate"), ("c4", "tiny")]
    assert dropped[0]["kept_chunk_id"] == CHUNK["c1"] and dropped[0]["jaccard"] >= 0.6
    # c2 is dropped silently by keep-best, exactly as the reference pipeline does it (no `dropped` row).
    assert "c2" not in set(labels(item.chunk_id for item in kept))


def test_ranks_are_renumbered_after_dedup(production_deps, inline_stack):
    store, models, _ = inline_stack
    qvec = models.embed_query(EN_QUESTION, "en")
    items = pipeline.retrieve(store, qvec, store.active_release(),
                              serve_rights=production_deps.config.store.serve_rights, limit=50)
    kept = pipeline.dedup(items, {i.chunk_id: i.payload for i in items}, pipeline.NEAR_DUPLICATE_RULE)
    assert [item.rank for item in kept] == [1, 2]


def _licensed_store():
    """Two licensed works and one redistributable chunk, on one axis so the order is obvious."""
    import numpy as np

    from ceres.store import FixtureStore, Release

    def row(label: str, work_id: str, rights: str, text: str) -> dict:
        import hashlib

        chunk_id = hashlib.sha256(f"ceres/licensed-fixture/{label}".encode()).hexdigest()
        return {"release_attempt_id": "licensed-0001", "chunk_id": chunk_id,
                "parent_id": f"p-{label}", "work_id": work_id, "membership": "train_eligible",
                "rights_state": rights, "evidence_eligible": True, "payload": text,
                "payload_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "locator_ids": [], "lineage_ids": [], "excluded_reasons": []}

    rows = [
        row("open", "w-open", "redistributable", "An open passage about sowing density. " * 4),
        row("admitted", "w-admitted", "licensed", "A licensed passage the decision admitted. " * 4),
        row("other", "w-other", "licensed", "A licensed passage nobody admitted. " * 4),
    ]
    vectors = np.array([[1.0, 0.0], [0.99, 0.14], [0.98, 0.2]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    store = FixtureStore(rows, vectors, Release("licensed-0001", "fixture", len(rows), {}))
    return store, rows, np.array([1.0, 0.05], dtype=np.float32)


def test_production_serves_a_licensed_work_only_when_it_is_admitted_by_id():
    """Second arm: `rights_state = 'licensed' AND work_id = ANY(admitted)`.

    The admission is per work id, not per rights state: admitting one licensed work must not make
    every licensed work servable.
    """
    store, rows, query = _licensed_store()
    labels_by_chunk = {rows[index]["chunk_id"]: name
                       for index, name in enumerate(("open", "admitted", "other"))}

    served = store.search(query, "licensed-0001", ["redistributable"], 50, ["w-admitted"])
    assert [labels_by_chunk[item.chunk_id] for item in served] == ["open", "admitted"]

    without = store.search(query, "licensed-0001", ["redistributable"], 50, [])
    assert [labels_by_chunk[item.chunk_id] for item in without] == ["open"]


def test_an_admitted_work_is_not_counted_as_withheld():
    """`meta.filtered.rights` must count what the whole clause withholds, not only the first arm."""
    store, _rows, _query = _licensed_store()
    release = store.active_release()
    assert store.withheld(release, ["redistributable"], ["w-admitted"]).withheld_rights == 1
    assert store.withheld(release, ["redistributable"], []).withheld_rights == 2


def test_the_pg_predicate_interpolates_no_value_into_the_sql_text():
    """Bug caught: a rights value, an admitted work id or the evidence basis formatted into the SQL
    (injection, and a predicate that differs from FixtureStore's). Execution itself is tested
    against real Postgres in `test_store.py`; this pins the statement's shape.
    """
    from ceres.store import PgStore

    clause = PgStore.RIGHTS_CLAUSE
    assert "rights_state = ANY(%(serve_rights)s::text[])" in clause
    assert "rights_state = 'licensed' AND work_id = ANY(%(admitted)s::text[])" in clause
    store = PgStore("postgresql://ceres_reader@127.0.0.1/veridian_rag", open_pool=False)
    body = store.search_sql(("evidence_basis", "page_start", "page_end"), "ocr_admitted_unverified")
    assert "AND evidence_eligible" in body and "release_attempt_id = %(active)s" in body
    assert "evidence_basis IS DISTINCT FROM %(refused_basis)s::text" in body
    assert "ocr_admitted_unverified" not in body       # the basis is a bound parameter
    assert body.count("%(") == 7        # q ×2, active, serve_rights, admitted, refused_basis, limit


def test_dedup_refuses_a_candidate_whose_text_it_does_not_have(deps, inline_stack):
    store, models, _ = inline_stack
    items = pipeline.retrieve(store, models.embed_query(EN_QUESTION, "en"), store.active_release(),
                              serve_rights=deps.config.store.serve_rights, limit=50)
    with pytest.raises(pipeline.PipelineError):
        pipeline.dedup(items, {}, pipeline.NEAR_DUPLICATE_RULE)


# ---------------------------------------------------------------- the request body

def test_the_request_body_is_the_loop_body(deps):
    """Ten keys, the profile's decoding, the reference pipeline's seed, and the thinking switch."""
    response = ask(deps)
    body = deps.llm.calls[-1]
    assert set(body) == {"model", "messages", "temperature", "top_p", "top_k", "presence_penalty",
                         "max_tokens", "seed", "stream", "chat_template_kwargs"}
    assert body["model"] == deps.answer_profile["model_alias"]
    decoding = deps.answer_profile["decoding"]
    for field in ("temperature", "top_p", "top_k", "presence_penalty", "max_tokens"):
        assert body[field] == decoding[field]
    assert body["stream"] is False
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["seed"] == stable_seed(response.packet["question_id"], 1)
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == deps.prompt
    user = body["messages"][1]["content"]
    assert user.startswith("Required answer language: en\n\nQuestion: ")
    assert "PRIVATE REFERENCE MATERIAL START" in user and user.endswith(
        "Return only the required JSON object.")


def test_the_seed_changes_with_the_attempt_and_the_question():
    assert stable_seed("a" * 64, 1) != stable_seed("a" * 64, 2)
    assert stable_seed("a" * 64, 1) != stable_seed("b" * 64, 1)
    assert 0 <= stable_seed("a" * 64, 1) < 2 ** 31


def test_the_prompt_must_match_the_profile_hash(config, inline_stack, tmp_path):
    """A prompt that is not the profile's prompt is a different route, so startup refuses it."""
    store, models, _ = inline_stack
    prompt = tmp_path / "grounded_answer_v1.md"
    prompt.write_text("not the r1 prompt", encoding="utf-8")
    config.route.prompt_template = str(prompt)
    with pytest.raises(pipeline.PipelineError) as exc:
        pipeline.Deps.build(config, store, models, FakeLLM())
    assert "prompt_template_sha256" in str(exc.value)


# ---------------------------------------------------------------- validation

def _candidate(**overrides):
    base = {"decision": "ANSWER", "answer_language": "en", "user_answer": "Wheat is sown at 450.",
            "citations": [{"claim": "Wheat is sown at 450.", "evidence_id": "E1",
                           "support_quote": "Winter wheat is sown at a density of 450"}],
            "missing_information": [], "confidence": "HIGH", "reason_code": "SUPPORTED"}
    base.update(overrides)
    return base


def _fixture_packet(deps) -> dict:
    return ask(deps).packet


def test_the_serving_revision_rejects_an_empty_claim_that_the_loop_accepts(deps):
    """Counterexample: the empty string is a substring of everything."""
    packet = _fixture_packet(deps)
    quote = packet["contexts"][0]["evidence_text"].split()[:8]
    candidate = _candidate(user_answer="Something true.",
                           citations=[{"claim": "", "evidence_id": "E1",
                                       "support_quote": " ".join(quote)}])
    assert pipeline.structural_answer(candidate, packet) == []          # the frozen reference validator
    errors = pipeline.validate(candidate, packet)                        # ceres-validate-r1
    assert errors and all("ceres-validate-r1" in error for error in errors)


def test_the_validator_still_enforces_the_loop_rules(deps):
    packet = _fixture_packet(deps)
    quote = " ".join(packet["contexts"][0]["evidence_text"].split()[:8])
    good = _candidate(user_answer=f"Answer: {quote}",
                      citations=[{"claim": quote, "evidence_id": "E1", "support_quote": quote}])
    assert pipeline.validate(good, packet) == []
    assert pipeline.validate({**good, "decision": "ANSWER", "missing_information": ["x"]}, packet)
    assert pipeline.validate({**good, "answer_language": "ro"}, packet)
    assert pipeline.validate({**good, "reason_code": "PARTIAL_SUPPORT"}, packet)
    assert pipeline.validate({**good, "citations": [{"claim": quote, "evidence_id": "E9",
                                                     "support_quote": quote}]}, packet)
    assert pipeline.validate({k: v for k, v in good.items() if k != "confidence"}, packet)


def test_a_quote_that_is_not_in_the_evidence_is_rejected(deps):
    packet = _fixture_packet(deps)
    claim = "Invented."
    bad = _candidate(user_answer="Invented.", citations=[{"claim": claim, "evidence_id": "E1",
                                                          "support_quote": "never written anywhere"}])
    assert pipeline.validate(bad, packet)


# ---------------------------------------------------------------- the answer loop

def test_a_first_structural_failure_is_retried_once(config, inline_stack):
    store, models, _ = inline_stack
    deps = pipeline.Deps.build(config, store, models, FakeLLM(failures=1))
    response = ask(deps)
    assert response.answer["state"] == "VALIDATED"
    assert response.answer["attempts"] == 2
    assert len(deps.llm.calls) == 2
    assert deps.llm.calls[0]["seed"] != deps.llm.calls[1]["seed"]


def test_two_structural_failures_end_as_structure_failed_without_an_answer(config, inline_stack):
    store, models, _ = inline_stack
    deps = pipeline.Deps.build(config, store, models, FakeLLM(failures=2))
    answer = ask(deps).answer
    assert answer["state"] == "STRUCTURE_FAILED"
    assert answer["attempts"] == 2 and "candidate" not in answer
    assert answer["errors"]


def _feedback_run(config, inline_stack, mode):
    """One ask() with a first structural failure, under answer-profile `retry_feedback: mode` (None = absent)."""
    store, models, _ = inline_stack
    deps = pipeline.Deps.build(config, store, models, FakeLLM(failures=1))
    if mode is not None:
        deps.answer_profile = {**deps.answer_profile, "retry_feedback": mode}
    return ask(deps), deps.llm.calls


def test_feedback_retry_leaves_attempt_one_byte_identical(config, inline_stack):
    """Turning the option on must not change the attempt-1 request (first-attempt successes stay as they were)."""
    _r_off, off = _feedback_run(config, inline_stack, None)
    _r_on, on = _feedback_run(config, inline_stack, "validator_errors")
    assert json.dumps(on[0], sort_keys=True) == json.dumps(off[0], sort_keys=True)


def test_feedback_retry_sends_the_reply_and_its_errors_in_plain_words(config, inline_stack):
    """Attempt 2 = attempt 1's messages + the rejected reply + one user message naming what failed; new seed."""
    response, calls = _feedback_run(config, inline_stack, "validator_errors")
    first, second = calls
    assert second["messages"][:2] == first["messages"] and len(second["messages"]) == 4
    assert second["messages"][2]["role"] == "assistant" and second["messages"][2]["content"] == '{"decision": "ANSWER"}'
    feedback = second["messages"][3]
    assert feedback["role"] == "user" and "failed the format check" in feedback["content"]
    assert "candidate keys do not match the profile schema" in feedback["content"]       # unmapped error: verbatim
    assert second["seed"] != first["seed"] and response.answer["state"] == "VALIDATED"


def test_without_the_option_the_retry_is_a_plain_resample(config, inline_stack):
    """Default off: attempt 2 carries only the two original messages (every earlier run's recipe)."""
    _r, calls = _feedback_run(config, inline_stack, None)
    assert len(calls[1]["messages"]) == 2


def test_an_unknown_retry_feedback_mode_is_refused(config, inline_stack):
    """A typo in the profile must not silently run the plain retry."""
    with pytest.raises(pipeline.PipelineError, match="retry_feedback"):
        _feedback_run(config, inline_stack, "validator-errors")


def test_feedback_lines_name_each_rule_once():
    """The plain-word rules, deduplicated (two bad claims → one line)."""
    message = pipeline.retry_feedback_message([
        "citation claim is not a continuous user_answer substring",
        "citation claim is not a continuous user_answer substring",
        "citation support_quote is invalid or exceeds 25 words", "citations must be an array of at most two",
        "decision/reason_code mismatch"])
    assert message.count("character for character") == 1
    assert "at most 25 words" in message and "at most two citations" in message and "PARTIAL_SUPPORT" in message


def _two_passage_packet():
    return {"question_id": "q", "language": "en", "contexts": [
        {"evidence_id": "E1", "evidence_text": "Wheat is sown in October in the south."},
        {"evidence_id": "E2", "evidence_text": "Light  intensity deepens the tillering node."}]}


def test_located_line_names_the_passage_a_misattributed_quote_comes_from():
    """A quote verbatim in E2 but cited as E1 gets a line naming E2 (whitespace/case folded like the validator)."""
    candidate = {"citations": [{"claim": "x", "evidence_id": "E1", "support_quote": "light intensity deepens the tillering node"}]}
    lines = pipeline.located_quote_lines(candidate, _two_passage_packet())
    assert lines == ['The quote "light intensity deepens the tillering node" occurs in E2, not in E1: cite E2, or quote E1.']


def test_located_line_is_silent_for_a_correct_or_a_paraphrased_quote():
    """No diagnosis when the quote is in its own passage, or in no passage at all (nothing true to say)."""
    own = {"citations": [{"claim": "x", "evidence_id": "E1", "support_quote": "Wheat is sown in October"}]}
    nowhere = {"citations": [{"claim": "x", "evidence_id": "E1", "support_quote": "wheat gets planted in autumn"}]}
    assert pipeline.located_quote_lines(own, _two_passage_packet()) == []
    assert pipeline.located_quote_lines(nowhere, _two_passage_packet()) == []
    assert pipeline.located_quote_lines(None, _two_passage_packet()) == []        # unparseable reply


def test_located_mode_runs_and_plain_feedback_mode_stays_p3(config, inline_stack):
    """The located-lines option is its own value; `validator_errors` never carries located lines."""
    _r, calls = _feedback_run(config, inline_stack, "validator_errors_located")
    assert len(calls[1]["messages"]) == 4
    assert pipeline.retry_feedback_message(["decision/reason_code mismatch"]) == \
        pipeline.retry_feedback_message(["decision/reason_code mismatch"], [])


def test_a_transport_failure_terminates_the_loop_immediately(config, inline_stack):
    store, models, _ = inline_stack
    deps = pipeline.Deps.build(config, store, models, FakeLLM(transport_error=True))
    answer = ask(deps).answer
    assert answer["state"] == "TRANSPORT_FAILED"
    assert answer["attempts"] == 1            # no retry against a server that is down
    assert len(deps.llm.calls) == 1


def test_raw_events_are_written_under_the_execution_id(deps):
    """Every attempt's bytes land on disk before the candidate is judged."""
    response = ask(deps)
    raw_dir = deps.config.raw_executions_dir / response.meta.execution_id / "raw"
    files = sorted(raw_dir.glob("*.json"))
    assert len(files) == response.answer["attempts"]
    event = json.loads(files[0].read_text(encoding="utf-8"))
    assert event["request"]["seed"] == stable_seed(response.packet["question_id"], 1)
    assert "response" in event and "structural_errors" in event


def test_both_languages_produce_a_validated_answer(deps):
    for question, lang in ((EN_QUESTION, "en"), (RO_QUESTION, "ro")):
        response = ask(deps, question, lang)
        assert response.answer["state"] == "VALIDATED"
        assert response.answer["candidate"]["answer_language"] == lang
        assert response.packet["language"] == lang


def test_latency_is_reported_per_stage(deps):
    latency = ask(deps).meta.latency_ms
    assert latency.total >= latency.answer >= 0
    assert latency.embed >= 0 and latency.retrieve >= 0 and latency.rerank >= 0


def test_an_empty_store_is_not_a_failure(config, inline_stack):
    """A healthy store with nothing admissible answers UNSUPPORTED — it does not 503."""
    import numpy as np

    from ceres.store import FixtureStore, Release

    empty = FixtureStore([], np.zeros((0, 8), dtype=np.float32),
                         Release("empty-0001", "fixture", 0, {}))
    _store, models, _ = inline_stack
    deps = pipeline.Deps.build(config, empty, models, FakeLLM())
    response = ask(deps)
    assert response.packet["contexts"] == []
    assert response.answer["state"] == "VALIDATED"
    assert response.answer["candidate"]["decision"] == "UNSUPPORTED"


# ---------------------------------------------------------------- OCR-admitted evidence
# The inline fixture with some items marked `ocr_admitted_unverified` — a test-only copy, built in
# memory from `build_inline_fixture()`; `tests/fixtures/` is not touched. In rnd the packet is
# E1 = c1, E2 = c5, E3 = c6, and FakeLLM cites E1 only.

OCR = "ocr_admitted_unverified"


class CitingLLM(FakeLLM):
    """Cites the given evidence ids in order, each with the first words of that context."""

    def __init__(self, cite, **kwargs):
        super().__init__(**kwargs)
        self.cite = list(cite)

    def _compose(self, user_prompt):
        from ceres.llm import _REFERENCE

        references = _REFERENCE.findall(user_prompt)
        citations, claims = [], []
        for number, (evidence_id, words) in enumerate(self.cite):
            quote = " ".join(references[int(evidence_id[1:]) - 1].split()[number:number + words])
            citations.append({"claim": quote, "evidence_id": evidence_id, "support_quote": quote})
            claims.append(quote)
        return {"decision": "ANSWER", "answer_language": "en", "user_answer": " ".join(claims),
                "citations": citations, "missing_information": [], "confidence": "MEDIUM",
                "reason_code": "SUPPORTED"}


def ocr_deps(config, inline_stack, marks, llm=None):
    """`marks`: label → (page_start, page_end), or None for an unknown page base, for the items
    to mark as OCR-admitted."""
    from ceres.store import FixtureStore

    store, models, _ = inline_stack
    rows = [dict(row) for row in store.rows]
    for row in rows:
        label = LABEL_BY_CHUNK[row["chunk_id"]]
        if label in marks:
            start, end = marks[label] if marks[label] is not None else (None, None)
            row.update(evidence_basis=OCR, page_start=start, page_end=end)
    marked = FixtureStore(rows, store.vectors, store.active_release())
    return pipeline.Deps.build(config, marked, models, llm or FakeLLM())


def ocr_log(config) -> list[dict]:
    path = config.ocr_citation_log
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_cited_ocr_context_is_marked_in_the_packet_and_in_meta(config, inline_stack):
    """The basis must reach the packet context and `meta.citation_marks` for a cited E1."""
    response = ask(ocr_deps(config, inline_stack, {"c1": (3, 4)}))
    first = response.packet["contexts"][0]
    assert response.answer["state"] == "VALIDATED"
    assert LABEL_BY_CHUNK[first["chunk_id"]] == "c1" and first["evidence_basis"] == OCR
    assert all("evidence_basis" not in c for c in response.packet["contexts"][1:])
    assert [m.model_dump() for m in response.meta.citation_marks] == [
        {"evidence_id": "E1", "chunk_id": CHUNK["c1"], "work_id": "w1", "evidence_basis": OCR}]


def test_only_cited_ocr_contexts_are_marked_and_logged(config, inline_stack):
    """E1 (OCR) cited twice → one mark, one log line; E3 (OCR) not cited → neither;
    E2 (not OCR) cited → no mark. Every field of the log line is checked."""
    llm = CitingLLM([("E1", 6), ("E2", 6)])
    deps = ocr_deps(config, inline_stack, {"c1": (3, 4), "c6": (9, 9)}, llm=llm)
    response = ask(deps)
    assert [LABEL_BY_CHUNK[c["chunk_id"]] for c in response.packet["contexts"]] == ["c1", "c5", "c6"]
    assert [m.evidence_id for m in response.meta.citation_marks] == ["E1"]

    twice = CitingLLM([("E1", 6), ("E1", 5)])
    again = ask(ocr_deps(config, inline_stack, {"c1": (3, 4), "c6": (9, 9)}, llm=twice))
    assert [m.evidence_id for m in again.meta.citation_marks] == ["E1"]

    lines = ocr_log(config)
    assert len(lines) == 2                            # one per answer, E3 never, E1 once each
    line = lines[0]
    assert set(line) == {"at", "execution_id", "question_id", "lang", "chunk_id", "work_id",
                         "evidence_id", "locator_ids", "support_quote", "page_start",
                         "page_end"}
    assert line["execution_id"] == response.meta.execution_id
    assert line["question_id"] == response.packet["question_id"] and line["lang"] == "en"
    assert (line["chunk_id"], line["work_id"], line["evidence_id"]) == (CHUNK["c1"], "w1", "E1")
    assert line["locator_ids"] == response.packet["contexts"][0]["locator_ids"]
    assert line["support_quote"] == response.answer["candidate"]["citations"][0]["support_quote"]
    assert (line["page_start"], line["page_end"]) == (3, 4)
    assert lines[1]["support_quote"] == again.answer["candidate"]["citations"][0]["support_quote"]


def test_the_logged_page_span_is_the_stores_and_null_when_absent_never_guessed(config, inline_stack):
    """`page_start`/`page_end` are the store's span, and null — never a guess (e.g. page 0 or
    a span derived from locators) — when the page base is unknown. Page 0 must survive as 0."""
    ask(ocr_deps(config, inline_stack, {"c1": None}))
    ask(ocr_deps(config, inline_stack, {"c1": (0, 0)}))
    ask(ocr_deps(config, inline_stack, {"c1": (2, 5)}))
    assert [(line["page_start"], line["page_end"]) for line in ocr_log(config)] == [
        (None, None), (0, 0), (2, 5)]


def test_a_structure_failed_answer_marks_and_logs_nothing(config, inline_stack):
    """No candidate reached anyone, so nothing was cited: no mark, no log line."""
    response = ask(ocr_deps(config, inline_stack, {"c1": (1, 1)}, llm=FakeLLM(failures=2)))
    assert response.answer["state"] == "STRUCTURE_FAILED"
    assert response.meta.citation_marks == [] and ocr_log(config) == []


def test_production_mode_never_packs_an_ocr_context(production_config, inline_stack):
    """The mode's arm reaches the store; no OCR context, no mark, no log."""
    response = ask(ocr_deps(production_config, inline_stack, {"c1": (1, 1), "c6": None}))
    assert all("evidence_basis" not in c for c in response.packet["contexts"])
    assert CHUNK["c1"] not in {c["chunk_id"] for c in response.packet["contexts"]}
    assert response.meta.citation_marks == [] and ocr_log(production_config) == []


def test_an_unloggable_ocr_citation_is_not_served(env, inline_stack, tmp_path):
    """With no runtime root, or an unwritable log, an OCR citation must not be served."""
    from ceres.config import load_config

    no_root = load_config(EXAMPLE_CONFIG, dict(env, CERES_PATHS__RUNTIME_ROOT=""),
                          require_api_key=False)
    assert no_root.ocr_citation_log is None
    with pytest.raises(pipeline.PipelineError, match="citation log"):
        ask(ocr_deps(no_root, inline_stack, {"c1": (1, 1)}))

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "logs").write_text("a file where the log folder should be", encoding="utf-8")
    unwritable = load_config(EXAMPLE_CONFIG, dict(env, CERES_PATHS__RUNTIME_ROOT=str(blocked)),
                             require_api_key=False)
    with pytest.raises(pipeline.PipelineError, match="not writable"):
        ask(ocr_deps(unwritable, inline_stack, {"c1": (1, 1)}))
    # An answer that cites no OCR context needs no log and is served.
    assert ask(ocr_deps(no_root, inline_stack, {"c6": (1, 1)})).answer["state"] == "VALIDATED"


# ---------------------------------------------------------------- LLM failure classes

def _mock_llm(handler) -> OpenAIChatLLM:
    return OpenAIChatLLM("http://answerer.test/v1", transport=httpx.MockTransport(handler))


def _llm_deps(config, inline_stack, llm):
    store, models, _ = inline_stack
    return pipeline.Deps.build(config, store, models, llm)


def test_http_200_with_invalid_json_is_a_structural_failure_that_retries(config, inline_stack):
    """The reference pipeline treats undecodable 200s as structural and keeps its attempts."""
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=b"not json")

    deps = _llm_deps(config, inline_stack, _mock_llm(handler))
    response = pipeline.ask(deps, AskRequest(question=EN_QUESTION, lang="en"))
    assert response.answer["state"] == "STRUCTURE_FAILED"
    assert response.answer["attempts"] == 2
    assert len(calls) == 2


def test_a_malformed_first_reply_does_not_end_the_attempt_loop(config, inline_stack):
    """An undecodable first reply must spend one attempt, not end the loop."""
    fake = FakeLLM()
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, content=b"\xff\xfe not utf-8")
        payload, raw = fake.chat(json.loads(request.content))
        return httpx.Response(200, content=raw)

    deps = _llm_deps(config, inline_stack, _mock_llm(handler))
    response = pipeline.ask(deps, AskRequest(question=EN_QUESTION, lang="en"))
    assert response.answer["state"] == "VALIDATED"
    assert response.answer["attempts"] == 2


def test_a_non_object_json_reply_is_structural_and_does_not_crash_meta(config, inline_stack):
    deps = _llm_deps(config, inline_stack,
                     _mock_llm(lambda request: httpx.Response(200, json=[1, 2, 3])))
    response = pipeline.ask(deps, AskRequest(question=EN_QUESTION, lang="en"))
    assert response.answer["state"] == "STRUCTURE_FAILED"
    assert response.answer["attempts"] == 2
    assert response.meta.model_reported is None


def test_http_and_network_errors_stay_transport_failures(config, inline_stack):
    for handler in (lambda request: httpx.Response(500, text="upstream exploded"),
                    lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused"))):
        deps = _llm_deps(config, inline_stack, _mock_llm(handler))
        response = pipeline.ask(deps, AskRequest(question=EN_QUESTION, lang="en"))
        assert response.answer["state"] == "TRANSPORT_FAILED"
        assert response.answer["attempts"] == 1
        assert response.answer["errors"] == ["transport_failed"]


# ---------------------------------------------------------------- route widths
# Route settings the pipeline must honour instead of silently hard-coding 
#
# Each test names the bug it catches:
#   * the rerank slot ignoring `top_k_input` and always scoring 10 (the reference pipeline had this bug;
#     it was found while drafting s1-wide, and Ceres had copied it);
#   * a `none-v1` (operation: identity) rerank profile still calling the reranker — s0 must never
#     touch the reranker, or it cannot serve on a 24 GB GPU beside the answer model;
#   * a retrieve profile with keys missing being filled with r1's values instead of the reference pipeline's own
#     defaults (`slot_adapter.build_retrieval`: pool = top_k_children, no near-duplicate rule,
#     min_context_tokens 0).

class CountingModels:
    """Records how many texts each rerank call received; scores reverse the input order."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def embed_query(self, text, lang):  # pragma: no cover - not used here
        raise AssertionError("embed_query must not be called")

    def rerank(self, question, texts):
        self.calls.append(len(texts))
        return [float(i) for i in range(len(texts))]  # last input scores highest


def _hits(n: int) -> list[Item]:
    hits = []
    for i in range(n):
        hits.append(Item(chunk_id=f"{i:064x}", parent_id=f"p{i}", work_id="w", membership="train_eligible",
                         rights_state="redistributable", payload=f"text {i}",
                         payload_sha256="0" * 64, locator_ids=[], lineage_ids=[], excluded_reasons=[],
                         score=1.0 - i / 1000, dense_rank=i + 1))
    return hits


def test_the_reranker_scores_top_k_input_candidates_not_a_hard_coded_ten():
    models = CountingModels()
    top = pipeline.rerank(models, "q", _hits(100), top_k=5, top_k_input=100)
    assert models.calls == [100]
    assert [h.dense_rank for h in top] == [100, 99, 98, 97, 96]


def test_an_identity_rerank_profile_never_calls_the_reranker_and_keeps_dense_order():
    models = CountingModels()
    top = pipeline.rerank(models, "q", _hits(10), top_k=5, top_k_input=10, identity=True)
    assert models.calls == []
    assert [h.dense_rank for h in top] == [1, 2, 3, 4, 5]
    assert [h.rank for h in top] == [1, 2, 3, 4, 5]


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_route_widths_are_not_hard_coded_r1_values(tmp_path):
    """Every width comes from the route's own profiles (s1-wide pool 500, top_k 100)."""
    retrieve = {"slot": "retrieve", "profile_id": "dense-top100-v2", "top_k_children": 100, "candidate_pool": 500,
                "min_context_tokens": 21,
                "near_duplicate_rule": {"normalization": "norm-v3", "token_jaccard_min": 0.6, "keep": "higher_rank"}}
    rerank = {"slot": "rerank", "profile_id": "qwen3-4b-rerank-v2", "top_k_input": 100, "top_k_output": 5}
    widths = pipeline.route_widths({"retrieve": _write(tmp_path, "r.yaml", retrieve).read_bytes(),
                                    "rerank": _write(tmp_path, "k.yaml", rerank).read_bytes()})
    assert widths == {"candidate_pool": 500, "top_k_children": 100, "min_context_tokens": 21,
                      "near_duplicate_rule": retrieve["near_duplicate_rule"],
                      "rerank_identity": False, "rerank_top_k_input": 100, "rerank_top_k_output": 5}


def test_missing_retrieve_keys_take_the_loop_defaults_not_r1_values(tmp_path):
    retrieve = {"slot": "retrieve", "profile_id": "dense-top10-v1", "top_k_children": 10}
    rerank = {"slot": "rerank", "profile_id": "none-v1", "operation": "identity"}
    widths = pipeline.route_widths({"retrieve": _write(tmp_path, "r.yaml", retrieve).read_bytes(),
                                    "rerank": _write(tmp_path, "k.yaml", rerank).read_bytes()})
    assert widths["candidate_pool"] == 10            # slot_adapter: profile.get("candidate_pool", top_k_children)
    assert widths["near_duplicate_rule"] is None     # no rule → no near-duplicate drop
    assert widths["min_context_tokens"] == 0         # slot_adapter: int(profile.get("min_context_tokens", 0))
    assert widths["rerank_identity"] is True


def test_a_rerank_profile_without_widths_is_refused(tmp_path):
    rerank = {"slot": "rerank", "profile_id": "qwen3-4b-rerank-v1"}  # no top_k_input / top_k_output
    retrieve = {"slot": "retrieve", "profile_id": "x", "top_k_children": 10}
    with pytest.raises(pipeline.PipelineError, match="top_k_input"):
        pipeline.route_widths({"retrieve": _write(tmp_path, "r.yaml", retrieve).read_bytes(),
                               "rerank": _write(tmp_path, "k.yaml", rerank).read_bytes()})


# ---------------------------------------------------------------- ceres-validate-r2: ellipsis quotes
# How the rule can fail, each case a test below: an ellipsis hides a negation or a number that flips or
# rescales the claim; the pieces are stitched from far apart, from two passages, or out of order; a piece
# is so short it matches anywhere; a lone trailing "..." slips through as a "shortened" quote; one good
# ellipsis citation lifts the error of a second, genuinely misquoted one; the served quote is rewritten; a
# new key leaks into the reference pipeline's exact citation shape; an r1-valid answer stops validating.

WHEAT = ("Winter wheat is sown in early autumn after the previous crop is harvested. Seedbed preparation "
         "must leave a fine, firm surface. The crop overwinters as a rosette and resumes growth in spring.")
SUBSTRING = pipeline.QUOTE_SUBSTRING_ERROR


def _r2_packet(*texts: str, language: str = "en") -> dict:
    return {"question_id": "q" * 64, "language": language,
            "contexts": [{"evidence_id": f"E{i}", "chunk_id": f"{i}" * 64, "evidence_text": text}
                         for i, text in enumerate(texts, start=1)]}


def _cite(*quotes: str, evidence_ids=None, language: str = "en") -> dict:
    """An ANSWER whose claims are exact, so the quote is the only thing under test."""
    evidence_ids = evidence_ids or ["E1"] * len(quotes)
    claims = [f"Claim {n}." for n in range(len(quotes))]
    return _candidate(answer_language=language, user_answer=" ".join(claims),
                      citations=[{"claim": claim, "evidence_id": eid, "support_quote": quote}
                                 for claim, eid, quote in zip(claims, evidence_ids, quotes)])


def test_r2_accepts_a_faithful_ellipsis_quote_that_r1_refused():
    """The bug r2 fixes: 87 % of the quote failures in the evaluation run were faithful quotes shortened with "..."."""
    packet = _r2_packet(WHEAT)
    candidate = _cite("Winter wheat is sown in early autumn ... The crop overwinters as a rosette")
    assert SUBSTRING in pipeline.structural_answer(candidate, packet)        # the reference r1 refuses it
    assert pipeline.validate(candidate, packet) == []


def test_r2_accepts_the_unicode_ellipsis_too():
    """A model writing "…" instead of "..." must not be refused for the glyph alone."""
    packet = _r2_packet(WHEAT)
    assert pipeline.validate(_cite("Winter wheat is sown in early autumn… The crop overwinters as a rosette"),
                             packet) == []


@pytest.mark.parametrize("passage,quote", [
    ("Nitrogen is applied at tillering but not before the soil warms up in spring.",
     "Nitrogen is applied at tillering ... before the soil warms up"),                       # EN not
    ("Nitrogen is applied at tillering but NOT before the soil warms up in spring.",
     "Nitrogen is applied at tillering ... before the soil warms up"),                       # case-insensitive
    ("Nitrogen is applied at tillering, and farmers don't wait until the soil warms up in spring.",
     "Nitrogen is applied at tillering, ... wait until the soil warms up"),                  # EN n't
    ("Azotul se aplică la înfrățire, dar nu înainte ca solul să se încălzească primăvara.",
     "Azotul se aplică la înfrățire, ... înainte ca solul să se încălzească"),               # RO nu
    ("Grâul se seamănă toamna devreme fără pregătirea patului germinativ în zonele umede.",
     "Grâul se seamănă toamna devreme ... pregătirea patului germinativ în zonele umede"),   # RO fără
], ids=["en-not", "en-NOT", "en-nt", "ro-nu", "ro-fara"])
def test_r2_refuses_an_ellipsis_that_hides_a_negation_word(passage, quote):
    """The meaning-flip bug: "applied ... before the soil warms" drops the "not" that reverses it."""
    language = "ro" if "ă" in passage else "en"
    errors = pipeline.validate(_cite(quote, language=language), _r2_packet(passage, language=language))
    assert errors == [SUBSTRING]


def test_r2_negation_is_matched_as_a_whole_word_not_inside_other_words():
    """Over-refusal bug: "another", "notable" and "nunta" contain no/not/nu but negate nothing."""
    passage = "Winter wheat is sown after another notable harvest nunta over, then the crop overwinters well."
    assert pipeline.validate(_cite("Winter wheat is sown ... then the crop overwinters"), _r2_packet(passage)) == []


def test_r2_refuses_an_ellipsis_that_hides_a_digit():
    """The rescaling bug: "at a density of ... seeds per square metre" drops the 450."""
    passage = "Sow winter wheat at a density of 450 seeds per square metre in early autumn."
    errors = pipeline.validate(_cite("Sow winter wheat at a density of ... seeds per square metre"),
                               _r2_packet(passage))
    assert errors == [SUBSTRING]


def test_r2_refuses_a_gap_longer_than_200_characters_and_accepts_exactly_200():
    """The stitching bug: two true sentences far apart are not one quote. The limit is inclusive."""
    head, tail = "Winter wheat is sown early", "the crop overwinters as a rosette"
    for filler, ok in (("a" * 198, True), ("a" * 199, False)):           # gap = " " + filler + " "
        passage = f"{head} {filler} {tail}"
        assert len(normalize(passage)) - len(head) - len(tail) == len(filler) + 2
        errors = pipeline.validate(_cite(f"{head} ... {tail}"), _r2_packet(passage))
        assert errors == ([] if ok else [SUBSTRING]), len(filler) + 2


def test_r2_refuses_a_two_word_piece_and_accepts_three():
    """The fragment bug: a two-word piece ("Winter wheat") matches almost any agronomy passage."""
    packet = _r2_packet(WHEAT)
    assert pipeline.validate(_cite("Winter wheat ... The crop overwinters as a rosette"), packet) == [SUBSTRING]
    assert pipeline.validate(_cite("Winter wheat is ... The crop overwinters as a rosette"), packet) == []


def test_r2_refuses_pieces_out_of_order():
    """The reordering bug: both pieces exist, but the quote reads them in an order the source does not."""
    errors = pipeline.validate(_cite("The crop overwinters as a rosette ... Winter wheat is sown in early autumn"),
                               _r2_packet(WHEAT))
    assert errors == [SUBSTRING]


def test_r2_refuses_pieces_split_across_two_passages():
    """The cross-passage bug: piece one from E1, piece two from E2, cited as either one.

    The passages are short and clean on purpose: joined, they would pass every other guard (gap under
    200 characters, no digit, no negation), so only the one-passage rule can refuse this quote."""
    packet = _r2_packet("Winter wheat is sown in early autumn.", "The crop overwinters as a rosette in cold regions.")
    quote = "Winter wheat is sown in early autumn ... The crop overwinters as a rosette"
    assert pipeline.validate(_cite(quote, evidence_ids=["E1"]), packet) == [SUBSTRING]
    assert pipeline.validate(_cite(quote, evidence_ids=["E2"]), packet) == [SUBSTRING]
    joined = _r2_packet(" ".join(context["evidence_text"] for context in packet["contexts"]))
    assert pipeline.validate(_cite(quote), joined) == []                  # the same words in ONE passage pass


@pytest.mark.parametrize("quote", ["Winter wheat is sown in early autumn ...",
                                   "... Winter wheat is sown in early autumn",
                                   "Winter wheat is sown in early autumn…"])
def test_r2_refuses_a_single_piece_with_a_dangling_ellipsis(quote):
    """One piece is not an ellipsis quote: the marker omits nothing between two quoted pieces."""
    assert pipeline.validate(_cite(quote), _r2_packet(WHEAT)) == [SUBSTRING]


def test_r2_still_refuses_a_plain_misquote():
    """The rule must not relax r1 for a quote with no ellipsis at all."""
    assert pipeline.validate(_cite("Winter wheat is sown in late spring"), _r2_packet(WHEAT)) == [SUBSTRING]


def test_r2_lifts_one_error_per_faithful_citation_not_a_misquote_beside_it():
    """The error-count bug: a good ellipsis citation must not clear a second, misquoted citation."""
    candidate = _cite("Winter wheat is sown in early autumn ... The crop overwinters as a rosette",
                      "Winter wheat is sown in late spring")
    assert pipeline.structural_answer(candidate, _r2_packet(WHEAT)).count(SUBSTRING) == 2
    assert pipeline.validate(candidate, _r2_packet(WHEAT)) == [SUBSTRING]


def test_r2_keeps_the_loop_25_word_limit_for_ellipsis_quotes():
    """Only the substring rule is relaxed: a faithful ellipsis quote over 25 words still fails."""
    passage = " ".join(f"w{chr(97 + n % 26)}x" for n in range(40))          # no digits, no negation
    words = passage.split()
    quote = " ".join(words[:13]) + " ... " + " ".join(words[14:27])           # 26 words
    errors = pipeline.validate(_cite(quote), _r2_packet(passage))
    assert errors == ["citation support_quote is invalid or exceeds 25 words"]


def test_r2_keeps_every_r1_valid_candidate_valid(deps):
    """No answer r1 validates may fail r2: a continuous quote (any case) needs no ellipsis rule."""
    packet = _fixture_packet(deps)
    quote = " ".join(packet["contexts"][0]["evidence_text"].split()[:10])
    for q in (quote, quote.upper()):
        good = _candidate(user_answer=f"Answer: {quote}",
                          citations=[{"claim": quote, "evidence_id": "E1", "support_quote": q}])
        assert pipeline.structural_answer(good, packet) == [] and pipeline.validate(good, packet) == []
        assert pipeline.accepted_quote_spans(good, packet) == []


def test_r2_guards_are_one_frozen_record():
    """A guard edited in one place and not another would validate by one rule and report another."""
    with pytest.raises(Exception):
        pipeline.QUOTE_GUARDS.max_gap_chars = 500                            # frozen dataclass
    assert pipeline.QUOTE_GUARDS.as_dict()["max_gap_chars"] == 200
    assert VALIDATOR_REVISION == "ceres-validate-r3" == pipeline.QUOTE_GUARDS.as_dict()["revision"]
    assert pipeline.QUOTE_GUARDS.as_dict()["gap_forbids_capitalized_word"] == R3_MEASURED_PATTERN


# ---------------------------------------------------------------- ceres-validate-r3: proper names
# How the r3 guard can fail: a place/region/named entity in the omitted gap is dropped and the quote passes
# (EN, or RO with a diacritic capital the character class forgot, or a cedilla capital the fold missed); a
# sentence-initial capital inside the gap is refused although the measured lookbehind lets it through; the
# lookbehind is "fixed" to see past the gap, changing the measured rule without a re-measurement; the golden
# r2 rescue that dropped "except Ireland, southern Spain ..." still validates.

R3_MEASURED_PATTERN = r"(?<![.!?]\s)\b[A-ZĂÂÎȘŞȚŢ][a-zăâîșşțţ]+"     # write_r3_sets.py CAPITAL, verbatim


def test_r3_refuses_an_ellipsis_that_drops_a_place_name_like_around_the_adriatic_sea():
    """The contradicted-rescue bug: "... and around the Adriatic Sea ..." drops the place scope, so the
    quote claims for everywhere what the source says for one sea coast. The control, with the same gap
    lower-cased, passes every guard, so only the r3 guard refuses the first quote."""
    quote = "Heavy rainfall was recorded in the north ... over the last three weeks"
    for gap, expected in ((" and around the Adriatic Sea ", [SUBSTRING]), (" and around the inland coast ", [])):
        passage = f"Heavy rainfall was recorded in the north{gap}over the last three weeks."
        assert pipeline.validate(_cite(quote), _r2_packet(passage)) == expected, gap


@pytest.mark.parametrize("passage", [
    "Seceta a afectat culturile de porumb din nord, de-a lungul râului Șiret, în lunile de vară.",
    "Seceta a afectat culturile de porumb din nord, mai ales în Țara noastră, în lunile de vară.",
    "Seceta a afectat culturile de porumb din nord, de-a lungul râului Şiret, în lunile de vară.",   # cedilla Ş
], ids=["ro-S-comma", "ro-T-comma", "ro-S-cedilla"])
def test_r3_refuses_a_romanian_place_name_whose_capital_is_a_diacritic(passage):
    """The ASCII-only bug: a class of [A-Z] would let "Șiret" / "Țara" through. The gap holds no other
    capital, so only the diacritic letters in the measured class can refuse it."""
    quote = "Seceta a afectat culturile de porumb din nord, ... în lunile de vară"
    assert pipeline.validate(_cite(quote, language="ro"), _r2_packet(passage, language="ro")) == [SUBSTRING]
    clean = passage.replace("Șiret", "șiret").replace("Şiret", "șiret").replace("Țara", "țara")
    assert pipeline.validate(_cite(quote, language="ro"), _r2_packet(clean, language="ro")) == []


def test_r3_does_not_refuse_a_capital_that_only_starts_a_sentence_after_a_full_stop_inside_the_gap():
    """The over-refusal bug the measured lookbehind prevents: "The" and "Then" after ". " inside the gap
    start sentences, not names. The gap has no digit, no negation word and is short, so no other guard
    refuses it; the control adds one mid-sentence name to the same gap and is refused."""
    quote = "Winter wheat is sown in early autumn ... the crop overwinters as a rosette"
    passage = ("Winter wheat is sown in early autumn. The seedbed must be firm. "
               "Then the crop overwinters as a rosette.")
    assert pipeline.validate(_cite(quote), _r2_packet(passage)) == []
    named = passage.replace("must be firm", "near Brasov must be firm")
    assert pipeline.validate(_cite(quote), _r2_packet(named)) == [SUBSTRING]


def test_r3_lookbehind_sees_only_the_gap_as_measured_so_a_capital_at_the_gap_start_is_refused():
    """The silent-rule-change bug: the measured guard searches the gap string alone, so a sentence-initial
    capital that is the gap's FIRST word (its ". " sits in the quoted piece before it) is refused. Letting
    the lookbehind see the piece would narrow the guard without a re-measurement."""
    import dataclasses

    passage = ("Winter wheat is sown in early autumn. Seedbed preparation must leave a fine, firm surface. "
               "The crop overwinters as a rosette.")
    quote = "Winter wheat is sown in early autumn. ... The crop overwinters as a rosette"
    assert pipeline.validate(_cite(quote), _r2_packet(passage)) == [SUBSTRING]     # gap " Seedbed ... surface. "
    r2_only = dataclasses.replace(pipeline.QUOTE_GUARDS, gap_capitalized_word=r"(?!)")   # never matches
    assert pipeline.ellipsis_piece_spans(quote, passage, r2_only) is not None       # only r3 refuses it


@requires_fixtures
def test_r3_refuses_the_golden_r2_rescue_that_dropped_except_ireland_southern_spain():
    """The real-data bug r3 exists for: the golden attempt "Elsewhere in Europe... temperatures were
    mildly above average (up to 3 °C)" omits ", except Ireland, southern Spain and Portugal, the Balkans
    ...". r2's guards accept it; r3 must refuse it, exactly as the reference r1 did."""
    import dataclasses

    question_id, attempt = R2_ONLY_GOLDEN_RESCUE
    event = json.loads((FIXTURES / "golden" / "raw" / f"{question_id}-attempt-{attempt:02d}.json").read_text(encoding="utf-8"))
    candidate = json.loads(event["response"]["choices"][0]["message"]["content"])
    packet = next(row for row in read_jsonl(FIXTURES / "golden" / "packets.jsonl") if row["question_id"] == question_id)
    assert event["structural_errors"] == [SUBSTRING]
    assert pipeline.validate(candidate, packet) == [SUBSTRING]
    contexts = {c["evidence_id"]: c["evidence_text"] for c in packet["contexts"]}
    [shortened] = [c for c in candidate["citations"] if "..." in c["support_quote"]]
    r2_only = dataclasses.replace(pipeline.QUOTE_GUARDS, gap_capitalized_word=r"(?!)")   # never matches
    assert pipeline.ellipsis_piece_spans(shortened["support_quote"], contexts[shortened["evidence_id"]], r2_only)
    assert pipeline.ellipsis_piece_spans(shortened["support_quote"], contexts[shortened["evidence_id"]]) is None


class EllipsisLLM(FakeLLM):
    """Cites E1 with an ellipsis quote: words [0:4] "..." words [5:9], skipping one clean word."""

    def _compose(self, user_prompt):
        from ceres.llm import _REFERENCE

        words = _REFERENCE.findall(user_prompt)[0].split()
        assert not re.search(r"\d", words[4]) and words[4].lower() not in {"no", "not", "nu"}
        quote = " ".join(words[:4]) + " ... " + " ".join(words[5:9])
        claim = "Grounded claim."
        return {"decision": "ANSWER", "answer_language": "en", "user_answer": claim,
                "citations": [{"claim": claim, "evidence_id": "E1", "support_quote": quote}],
                "missing_information": [], "confidence": "MEDIUM", "reason_code": "SUPPORTED"}


def test_meta_quote_spans_record_an_accepted_ellipsis_quote_and_the_quote_is_served_unchanged(config, inline_stack):
    """The served quote keeps its "..." (never silently rewritten into a quote the model did not
    write), meta.quote_spans locates each piece in normalize(passage), and the reference pipeline's exact citation
    key set gains nothing."""
    store, models, _ = inline_stack
    llm = EllipsisLLM()
    response = ask(pipeline.Deps.build(config, store, models, llm))
    assert response.answer["state"] == "VALIDATED" and response.answer["attempts"] == 1
    served = response.answer["candidate"]["citations"][0]
    sent = llm._compose(llm.calls[0]["messages"][1]["content"])["citations"][0]["support_quote"]
    assert " ... " in served["support_quote"] and served["support_quote"] == sent
    assert set(served) == {"claim", "evidence_id", "support_quote"}
    assert set(response.answer["candidate"]) == {"decision", "answer_language", "user_answer", "citations",
                                                "missing_information", "confidence", "reason_code"}
    context = response.packet["contexts"][0]
    spans = [record.model_dump() for record in response.meta.quote_spans]
    assert len(spans) == 1
    record = spans[0]
    assert (record["evidence_id"], record["chunk_id"], record["support_quote"], record["norm_version"]) == \
           ("E1", context["chunk_id"], served["support_quote"], "norm-v3")
    text = normalize(context["evidence_text"])
    pieces = [normalize(piece) for piece in served["support_quote"].split("...")]
    assert [text[start:end] for start, end in record["piece_spans"]] == pieces


def test_meta_quote_spans_stay_empty_for_continuous_quotes_and_failed_answers(config, inline_stack):
    """quote_spans must mean "passed only through the ellipsis rule" — never a copy of every citation."""
    store, models, _ = inline_stack
    assert ask(pipeline.Deps.build(config, store, models, FakeLLM())).meta.quote_spans == []
    failed = ask(pipeline.Deps.build(config, store, models, FakeLLM(failures=2)))
    assert failed.answer["state"] == "STRUCTURE_FAILED" and failed.meta.quote_spans == []


# ---------------------------------------------------------------- regression: the saved evaluation attempts

C6_ANSWERS = Path(os.environ.get("CERES_C6_ANSWERS_DIR") or (
    Path(__file__).resolve().parents[3] / "path" / "to" / "saved-answers"))
CHANGED_R2 = Path(os.environ.get("CERES_CHANGED_R2_FILE") or (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "changed-answers-r2.json"))


C6_ROUTES = ("s0-v33", "s1-v33", "s1-wide-v33")


def _c6_selection(row: dict) -> tuple:
    """(state, selected attempt, candidate) — what a served answer is; `attempts` of a failed row is noise."""
    if row["state"] == "VALIDATED":
        return ("VALIDATED", row["attempts"], row["candidate"])
    return (row["state"], None, None)


@pytest.mark.skipif(not all((C6_ANSWERS / route / "answers-r3.jsonl").is_file()
                            and (C6_ANSWERS / route / "MANIFEST-r3.json").is_file() for route in C6_ROUTES)
                    or not CHANGED_R2.is_file(),
                    reason="the reference pipeline's saved attempts with their r3 sets (or the measurement file) "
                           "are not on this machine; set CERES_C6_ANSWERS_DIR to the saved answers folder")
def test_r3_replays_the_c6_run_reproducing_answers_r3_and_reverting_exactly_the_10_measured_r2_rescues():
    """r3 was cleared on a measurement: replaying the saved attempts under the
    serving validator must reproduce `answers-r3.jsonl` on all three routes (same state, same selected
    attempt, same candidate), and versus `answers-r2.jsonl` exactly the 10 (route, question) pairs of the
    MANIFEST-r3 `changed_vs_r2` move — each back to its r1 selection. A validator wider or narrower than
    the measured one changes a selection, and this replay catches it. r1-valid answers never get worse."""
    measured_r2 = json.loads(CHANGED_R2.read_text(encoding="utf-8"))
    reverted_expected, reverted_observed, still_changed_vs_r1 = set(), set(), set()
    for route in C6_ROUTES:
        folder = C6_ANSWERS / route
        manifest = json.loads((folder / "MANIFEST-r3.json").read_text(encoding="utf-8"))
        assert manifest["validator_revision"] == "ceres-validate-r3"
        assert manifest["guards"]["added"]["gap_forbids_capitalized_word"] == \
            pipeline.QUOTE_GUARDS.gap_capitalized_word == R3_MEASURED_PATTERN
        changed = manifest["changed_vs_r2"]
        assert changed["now_structure_failed"] == [] and changed["new_selection"] == []
        reverted_expected |= {(route, qid) for qid in changed["same_as_r1"]}
        packets = {row["question_id"]: row for row in read_jsonl(folder / "packets.jsonl")}
        r1 = {row["question_id"]: row for row in read_jsonl(folder / "answers.jsonl")}
        r2 = {row["question_id"]: row for row in read_jsonl(folder / "answers-r2.jsonl")}
        r3 = {row["question_id"]: row for row in read_jsonl(folder / "answers-r3.jsonl")}
        assert len(r3) == len(packets) == 152
        for question_id, packet in packets.items():
            selection = ("STRUCTURE_FAILED", None, None)
            for attempt in (1, 2):
                raw = folder / "raw" / f"{question_id}-attempt-{attempt:02d}.json"
                if not raw.is_file():
                    break
                event = json.loads(raw.read_text(encoding="utf-8"))
                try:
                    candidate = json.loads(event["response"]["choices"][0]["message"]["content"])
                except (KeyError, IndexError, TypeError, ValueError):
                    continue
                if not pipeline.validate(candidate, packet):
                    selection = ("VALIDATED", attempt, candidate)
                    break
            assert selection == _c6_selection(r3[question_id]), (route, question_id)
            if selection != _c6_selection(r2[question_id]):
                reverted_observed.add((route, question_id))
                assert selection == _c6_selection(r1[question_id]), (route, question_id)
            r1_attempt = _c6_selection(r1[question_id])[1]
            if selection[1] != r1_attempt:
                assert r1_attempt is None or (selection[1] or 99) < r1_attempt, (route, question_id)
                still_changed_vs_r1.add((route, question_id))
    assert len(reverted_expected) == 10
    assert reverted_observed == reverted_expected
    # What r3 still changes versus r1 is the r2 set (19) minus the reverted 10.
    r2_changes = {(route, row["question_id"]) for route, rows in measured_r2.items() for row in rows}
    assert len(r2_changes) == 19 and reverted_expected <= r2_changes
    assert still_changed_vs_r1 == r2_changes - reverted_expected


class _NeverCalledLLM:
    """An answerer that fails the test if it is ever asked ."""
    calls = 0

    def chat(self, body):
        type(self).calls += 1
        raise AssertionError("the answerer must not be called for an empty packet with the short circuit on")


def _empty_packet(lang):
    return {"question_id": "q" * 64, "question": "?", "language": lang, "contexts": []}


@pytest.mark.parametrize("lang", ["en", "ro"])
def test_with_the_short_circuit_on_an_empty_packet_is_refused_without_asking_the_answerer(deps, lang):
    """Bug: an empty retrieval still called the LLM, so a VALIDATED UNSUPPORTED row
    could carry model-memory advice. With `answer.no_evidence_short_circuit` the refusal is deterministic."""
    row = pipeline.answer(_NeverCalledLLM(), deps.answer_profile, deps.prompt, _empty_packet(lang),
                          no_evidence_short_circuit=True)
    assert row.state == "VALIDATED" and row.attempts == 0
    assert row.candidate["decision"] == "UNSUPPORTED" and row.candidate["reason_code"] == "NO_RELEVANT_SUPPORT"
    assert row.candidate["answer_language"] == lang and row.candidate["citations"] == []
    assert pipeline.validate(row.candidate, _empty_packet(lang)) == []


def test_the_short_circuit_is_off_by_default_so_served_answers_do_not_change(config):
    """The switch changes answers, so it must default to off (a measured candidate, never silent)."""
    assert pipeline.answer_enabled_short_circuit(config) is False
    assert config.answer.no_evidence_short_circuit is False


from ceres.validate_r4 import continuous_quote_r4_errors  # noqa: E402  (candidate module)


def _one_citation(quote, passage):
    candidate = {"decision": "ANSWER", "answer_language": "en", "user_answer": "Sow 15 kg per hectare.",
                 "citations": [{"claim": "Sow 15 kg per hectare.", "evidence_id": "E1", "support_quote": quote}],
                 "missing_information": [], "confidence": "HIGH", "reason_code": "SUPPORTED"}
    packet = {"question_id": "q" * 64, "question": "?", "language": "en",
              "contexts": [{"evidence_id": "E1", "evidence_text": passage}]}
    return candidate, packet


@pytest.mark.parametrize("quote", ["5", "5 kg", "ow 15 kg per hecta"])
def test_candidate_r4_rejects_fragment_quotes_that_r3_accepts(quote):
    """Bug: r3 accepts any casefolded substring, so "5" or a mid-word cut passes
    against "15 kg"; the candidate r4 rule refuses them. r4 is NOT served: this pins the candidate only."""
    candidate, packet = _one_citation(quote, "Farmers sow 15 kg per hectare in autumn.")
    assert continuous_quote_r4_errors(candidate, packet)


def test_candidate_r4_accepts_a_whole_word_quote_and_serving_still_runs_r3():
    """A normal word-bounded quote passes r4, and the served revision is still r3."""
    candidate, packet = _one_citation("sow 15 kg per hectare", "Farmers sow 15 kg per hectare in autumn.")
    assert continuous_quote_r4_errors(candidate, packet) == []
    assert VALIDATOR_REVISION == "ceres-validate-r3"


def test_candidate_r4_rejects_a_quote_that_differs_only_in_letter_case():
    """Bug: the r4 candidate casefolded, so "SOW 15 KG PER HECTARE" passed
    against "sow 15 kg per hectare" although it is not verbatim."""
    candidate, packet = _one_citation("SOW 15 KG PER HECTARE", "Farmers sow 15 kg per hectare in autumn.")
    assert any("letter case" in error for error in continuous_quote_r4_errors(candidate, packet))


def test_candidate_r4_allows_a_capitalised_first_letter_of_a_mid_sentence_quote():
    """Recorded case: answers quote "Încingerea în cuiburi …" from "… încingerea în cuiburi …"; a quote
    starting mid-sentence is often capitalised, so only the first letter's case is tolerated."""
    candidate, packet = _one_citation("Sow 15 kg per hectare", "Farmers sow 15 kg per hectare in autumn.")
    assert continuous_quote_r4_errors(candidate, packet) == []


# ---------------------------------------------------------------- citation switch (strict or soft)
# How the switch can fail, and the test that catches each:
#   * strict is no longer exactly r3 → test_strict_mode_is_r3_unchanged
#   * soft rejects a whole answer for one bad citation (the 7.4 % loss it exists to stop) → test_soft_drops_only_the_failing_citation
#   * soft serves more than two citations → test_soft_keeps_the_first_two_passing_citations
#   * soft refuses ANSWER/PARTIAL left with no citation → test_soft_serves_an_answer_whose_citations_all_failed
#   * soft lets a non-citation rule through (reason_code, missing_information) → test_soft_keeps_every_other_rule_hard
#   * soft drops a quote that r3's ellipsis rule accepts → test_soft_keeps_an_ellipsis_quote_r3_accepts
#   * soft still retries a reply that failed only on citations → test_soft_accepts_a_citation_only_failure_without_a_retry
#   * the served answer does not record what strict would have said → test_the_served_answer_records_mode_strict_errors_and_drops

SWITCH_PACKET = {"question_id": "q", "packet_id": "grounded-q", "question": "How is winter wheat sown?", "language": "en", "contexts": [
    {"evidence_id": "E1", "evidence_text": "Winter wheat is sown at a density of 450 seeds per square metre in October."},
    {"evidence_id": "E2", "evidence_text": "Barley tolerates salinity better than wheat on coastal soils of the region."}]}
GOOD = {"claim": "Wheat is sown at 450.", "evidence_id": "E1", "support_quote": "Winter wheat is sown at a density of 450"}


def _switch_candidate(**overrides):
    return _candidate(user_answer="Wheat is sown at 450. Barley tolerates salinity.", **overrides)


def test_strict_mode_is_r3_unchanged():
    bad = _switch_candidate(citations=[GOOD, {"claim": "invented", "evidence_id": "E1", "support_quote": "Winter wheat"}])
    verdict = pipeline.judge_citations(bad, SWITCH_PACKET, "strict")
    assert verdict.errors == pipeline.validate(bad, SWITCH_PACKET) == verdict.strict_errors
    assert verdict.candidate is bad and verdict.dropped == []


def test_soft_drops_only_the_failing_citation():
    bad_claim = {"claim": "not in the answer", "evidence_id": "E1", "support_quote": "Winter wheat is sown"}
    candidate = _switch_candidate(citations=[bad_claim, GOOD])
    verdict = pipeline.judge_citations(candidate, SWITCH_PACKET, "soft")
    assert verdict.errors == [] and verdict.candidate["citations"] == [GOOD]
    assert verdict.strict_errors == ["citation claim is not a continuous user_answer substring"]
    assert verdict.dropped == [{"index": 0, "evidence_id": "E1",
                                "reasons": ["citation claim is not a continuous user_answer substring"]}]


def test_soft_keeps_the_first_two_passing_citations():
    second = {"claim": "Barley tolerates salinity.", "evidence_id": "E2", "support_quote": "Barley tolerates salinity"}
    third = {"claim": "Wheat is sown", "evidence_id": "E1", "support_quote": "sown at a density of 450 seeds"}
    verdict = pipeline.judge_citations(_switch_candidate(citations=[GOOD, second, third]), SWITCH_PACKET, "soft")
    assert verdict.errors == [] and verdict.candidate["citations"] == [GOOD, second]
    assert verdict.dropped == [{"index": 2, "evidence_id": "E1", "reasons": ["over_limit"]}]


def test_soft_serves_an_answer_whose_citations_all_failed():
    paraphrase = {"claim": "Wheat is sown at 450.", "evidence_id": "E1", "support_quote": "wheat gets planted densely"}
    verdict = pipeline.judge_citations(_switch_candidate(citations=[paraphrase]), SWITCH_PACKET, "soft")
    assert verdict.errors == [] and verdict.candidate["citations"] == []
    assert verdict.dropped[0]["reasons"] == [pipeline.QUOTE_SUBSTRING_ERROR]


def test_soft_keeps_every_other_rule_hard():
    wrong_code = _switch_candidate(reason_code="PARTIAL_SUPPORT")
    assert pipeline.judge_citations(wrong_code, SWITCH_PACKET, "soft").errors == ["decision/reason_code mismatch"]
    missing = _switch_candidate(missing_information=["the sowing date"])
    assert pipeline.ANSWER_RULE in pipeline.judge_citations(missing, SWITCH_PACKET, "soft").errors
    no_keys = {"decision": "ANSWER"}
    assert pipeline.judge_citations(no_keys, SWITCH_PACKET, "soft").errors == ["candidate keys do not match the profile schema"]
    unsupported = _switch_candidate(decision="UNSUPPORTED", reason_code="NO_RELEVANT_SUPPORT",
                                    missing_information=["x"], citations=[GOOD])
    verdict = pipeline.judge_citations(unsupported, SWITCH_PACKET, "soft")
    assert verdict.errors == [] and verdict.candidate["citations"] == []
    assert verdict.dropped[0]["reasons"] == ["unsupported_has_citations"]


def test_soft_keeps_an_ellipsis_quote_r3_accepts():
    ellipsis = {"claim": "Wheat is sown at 450.", "evidence_id": "E1",
                "support_quote": "Winter wheat is sown at a density of 450... per square metre in October"}   # gap " seeds ": no digit
    assert pipeline.validate(_switch_candidate(citations=[ellipsis]), SWITCH_PACKET) == []   # r3 accepts it
    verdict = pipeline.judge_citations(_switch_candidate(citations=[ellipsis, {"claim": "x y", "evidence_id": "E9",
                                                                              "support_quote": "a b c"}]),
                                       SWITCH_PACKET, "soft")
    assert verdict.candidate["citations"] == [ellipsis] and [d["index"] for d in verdict.dropped] == [1]


class _ScriptedLLM:
    """Replies with the given candidates in order (one per attempt)."""

    def __init__(self, *candidates):
        self.candidates, self.calls = list(candidates), []

    def chat(self, body):
        self.calls.append(body)
        content = json.dumps(self.candidates[len(self.calls) - 1])
        envelope = {"choices": [{"message": {"content": content}}], "model": "fake"}
        return envelope, json.dumps(envelope).encode()


def test_soft_accepts_a_citation_only_failure_without_a_retry():
    citation_only = _switch_candidate(citations=[{"claim": "absent", "evidence_id": "E1", "support_quote": "Winter wheat"}])
    profile = {"model_alias": "m", "decoding": {"maximum_attempts": 2}}
    soft_llm, strict_llm = _ScriptedLLM(citation_only, _switch_candidate()), _ScriptedLLM(citation_only, _switch_candidate())
    soft = pipeline.answer(soft_llm, profile, "prompt", SWITCH_PACKET, citation_mode="soft")
    strict = pipeline.answer(strict_llm, profile, "prompt", SWITCH_PACKET, citation_mode="strict")
    assert (soft.state, soft.attempts, len(soft_llm.calls)) == ("VALIDATED", 1, 1)
    assert soft.candidate["citations"] == [] and soft.citation["citation_mode"] == "soft"
    assert (strict.attempts, len(strict_llm.calls)) == (2, 2)                       # strict retried, as today
    assert "citation" not in soft.to_dict()                                           # the reference row keeps its keys


def test_the_served_answer_records_mode_strict_errors_and_drops(config, inline_stack):
    store, models, _ = inline_stack
    soft_config = config.model_copy(update={"answer": config.answer.model_copy(update={"citation_mode": "soft"})})
    deps = pipeline.Deps.build(soft_config, store, models, FakeLLM())
    meta = ask(deps).meta
    assert meta.citation_mode == "soft" and meta.strict_errors == [] and meta.dropped_citations == []
    strict_meta = ask(pipeline.Deps.build(config, store, models, FakeLLM())).meta
    assert strict_meta.citation_mode == "strict"


# ---------------------------------------------------------------- MARS issue boost (qwen3-0.6b-rerank-v3)
# How the boost can fail, and the test that catches each:
#   * the vendored date reader drifts from the reference pipeline's / the profile's pin → test_the_date_reader_is_the_pinned_loop_file
#   * a window or reader Ceres does not implement runs anyway → test_a_boost_profile_ceres_does_not_implement_is_refused
#   * a wrong match (other month, non-MARS parent, undated question) gets the boost → test_only_a_matching_mars_issue_is_boosted
#   * equal final scores reorder by something other than dense order → test_boost_ties_keep_dense_order
#   * the served rerank never applies it, or applies it on the dense OOM fallback → test_the_served_rerank_lifts_the_dated_bulletin

from types import SimpleNamespace as _NS

from ceres import issue_boost as _ib
from ceres.store import Item as _Item


def test_the_date_reader_is_the_pinned_loop_file():
    import hashlib
    import os

    pin = "76afe7bfd1dc8352611297405ecd98362bbe0dd8709b113df083aa600ed44882"     # qwen3-0.6b-rerank-v3 date_reader_sha256
    assert _ib.date_reader_sha256() == pin
    loop = Path(os.environ.get("DDS_LOOP_ROOT", "/nonexistent")) / "loop-v0/04_rag/analytics/mars_dates.py"
    if loop.is_file():
        assert hashlib.sha256(loop.read_bytes()).hexdigest() == pin


@pytest.mark.parametrize("boost", [{"beta": 2.0, "window": "same+next", "date_reader_sha256": _ib.date_reader_sha256()},
                                   {"beta": 2.0, "window": "same", "date_reader_sha256": "0" * 64}], ids=["window", "reader"])
def test_a_boost_profile_ceres_does_not_implement_is_refused(boost):
    with pytest.raises(_ib.IssueBoostError):
        _ib.boost_config({"profile_id": "x", "issue_boost": boost})


def test_only_a_matching_mars_issue_is_boosted():
    parents = ["mars-europe-2012-07-vol20-no7:s1", "mars-europe-2012-08-vol20-no8:s1", "ebooks_2026-x:page-1"]
    order, target, n = _ib.boosted_order("What did the July 2012 MARS bulletin say?", parents, [0.0, 0.0, 1.0], 2.0, 3)
    assert target == [2012, 7] and n == 1 and order == [0, 2, 1]
    _o, target, n = _ib.boosted_order("What happened to maize in 2012?", parents, [0.0, 0.0, 1.0], 2.0, 3)
    assert target == [2012, None] and n == 2                                       # a year alone matches every 2012 issue
    order, target, n = _ib.boosted_order("How is wheat sown?", parents, [0.0, 0.5, 1.0], 2.0, 3)
    assert target is None and n == 0 and order == [2, 1, 0]


def test_boost_ties_keep_dense_order():
    order, _t, _n = _ib.boosted_order("What did the July 2012 bulletin say?", ["a", "b", "mars-europe-2012-07-v:s"],
                                      [2.0, 1.0, 0.0], 2.0, 3)
    assert order == [0, 2, 1]                     # the boosted third (0 + 2 = 2.0) ties the first and stays behind it


def _rerank_deps(scores, *, oom=False):
    class Models:
        def rerank(self, question, texts):
            if oom:
                raise type("OutOfMemoryError", (RuntimeError,), {})("CUDA out of memory")
            return list(scores)
    return _NS(models=Models(), rerank_top_k=2, rerank_top_k_input=3, rerank_identity=False, rerank_text="payload",
               rerank_tie_break="dense_rank", rerank_model={"model": "m"}, rerank_on_oom="dense_order",
               issue_boost={"beta": 2.0, "window": "same"})


def _boost_hits():
    parents = ["ebooks-a:p1", "ebooks-b:p1", "mars-europe-2012-07-vol20-no7:s12"]
    return [_Item(chunk_id=f"c{i}", parent_id=p, work_id="w", membership="m", rights_state="r", payload="text",
                  payload_sha256="s", dense_rank=i + 1) for i, p in enumerate(parents)]


def test_the_served_rerank_lifts_the_dated_bulletin():
    top, info = pipeline.rerank_with_status(_rerank_deps([3.0, 2.5, 1.5]), "How did the July 2012 heat wave hit maize?", _boost_hits())
    assert [h.chunk_id for h in top] == ["c2", "c0"] and info["issue_boost"] == {"target": [2012, 7], "boosted_count": 1}
    top, info = pipeline.rerank_with_status(_rerank_deps([3.0, 2.5, 1.5], oom=True), "How did the July 2012 heat wave hit maize?",
                                            _boost_hits())
    assert info["status"] == "fallback_oom" and "issue_boost" not in info and [h.chunk_id for h in top] == ["c0", "c1"]


# ---------------------------------------------------------------- premise-corrected ANSWER (revision ceres-validate-r3-pc1)
# How the gated revision can fail, and the test that catches each:
#   * today's profile starts accepting the code (r3 not unchanged) → test_todays_profile_rejects_premise_corrected_exactly_as_r3
#   * the code passes under PARTIAL/UNSUPPORTED → test_the_code_is_accepted_only_under_answer
#   * a "correction" with an invented quote or no citation passes → test_a_correction_still_needs_a_real_quote_and_a_citation
#   * a typo'd or unknown code in a profile is silently accepted → test_an_unknown_extra_reason_code_is_refused
#   * soft mode ignores the profile's codes (the reference pipeline's own slip) → test_soft_mode_uses_the_same_extra_codes
#   * the served answer is labelled r3 although judged with the codes → test_the_answer_loop_and_revision_follow_the_profile

PC = frozenset({"PREMISE_CORRECTED"})


def _premise(**over):
    return _switch_candidate(reason_code="PREMISE_CORRECTED", **over)


def test_todays_profile_rejects_premise_corrected_exactly_as_r3():
    for candidate in (_premise(), _switch_candidate(), _switch_candidate(citations=[])):
        assert pipeline.validate_profiled(candidate, SWITCH_PACKET) == pipeline.validate(candidate, SWITCH_PACKET)
    assert pipeline.validate_profiled(_premise(), SWITCH_PACKET) == ["decision/reason_code mismatch"]
    assert pipeline.validator_revision_for({}) == "ceres-validate-r3"


def test_the_code_is_accepted_only_under_answer():
    assert pipeline.validate_profiled(_premise(), SWITCH_PACKET, PC) == []
    partial = _premise(decision="PARTIAL", missing_information=["x"])
    unsupported = _premise(decision="UNSUPPORTED", missing_information=["x"], citations=[])
    assert "decision/reason_code mismatch" in pipeline.validate_profiled(partial, SWITCH_PACKET, PC)
    assert "decision/reason_code mismatch" in pipeline.validate_profiled(unsupported, SWITCH_PACKET, PC)


def test_a_correction_still_needs_a_real_quote_and_a_citation():
    invented = _premise(citations=[{"claim": "Wheat is sown at 450.", "evidence_id": "E1",
                                    "support_quote": "spring barley is sown in March"}])
    assert pipeline.QUOTE_SUBSTRING_ERROR in pipeline.validate_profiled(invented, SWITCH_PACKET, PC)
    assert pipeline.validate_profiled(_premise(citations=[]), SWITCH_PACKET, PC) == [pipeline.ANSWER_RULE]


def test_an_unknown_extra_reason_code_is_refused():
    with pytest.raises(pipeline.PipelineError, match="extra_answer_reason_codes"):
        pipeline.extra_answer_reasons({"extra_answer_reason_codes": ["PREMISE_CORECTED"]})


def test_soft_mode_uses_the_same_extra_codes():
    bad_cite = {"claim": "absent", "evidence_id": "E1", "support_quote": "Winter wheat"}
    candidate = _premise(citations=[bad_cite, GOOD])
    assert pipeline.judge_citations(candidate, SWITCH_PACKET, "soft", PC).errors == []
    assert pipeline.judge_citations(candidate, SWITCH_PACKET, "soft").errors == ["decision/reason_code mismatch"]


def test_the_answer_loop_and_revision_follow_the_profile():
    base = {"model_alias": "m", "decoding": {"maximum_attempts": 2}}
    gated = {**base, "extra_answer_reason_codes": ["PREMISE_CORRECTED"]}
    ok = pipeline.answer(_ScriptedLLM(_premise(), _premise()), gated, "prompt", SWITCH_PACKET)
    no = pipeline.answer(_ScriptedLLM(_premise(), _premise()), base, "prompt", SWITCH_PACKET)
    assert (ok.state, ok.attempts, ok.candidate["reason_code"]) == ("VALIDATED", 1, "PREMISE_CORRECTED")
    assert (no.state, no.attempts) == ("STRUCTURE_FAILED", 2)
    assert pipeline.validator_revision_for(gated) == "ceres-validate-r3-pc1"


# ---------------------------------------------------------------- packet width (answer profile max_contexts)
# Failure modes: (1) the packet is cut to 5 whatever the route says (a k8 route served 5
# passages under its own id); (2) a route whose reranker returns more passages than the packet holds starts anyway;
# (3) a prompt that tells the model "E1..E5" is given 8 passages; (4) a malformed width (0, a string, unbounded) loads;
# (5) the default changes and moves every existing route (covered by test_parity: packets equal the reference pipeline's E1..E5).

_LOOP = Path(os.environ.get("DDS_LOOP_ROOT")
             or Path(__file__).resolve().parents[3] / "path" / "to" / "reference-pipeline" / "loop-v0")
_PROFILES = _LOOP / "04_rag" / "profiles"
_requires_k8_profiles = pytest.mark.skipif(
    not (_PROFILES / "rerank" / "qwen3-0.6b-rerank-v4-k8.yaml").is_file(),
    reason=f"the reference pipeline's k8 rerank/answer profiles are not on this box ({_PROFILES})")
_V3_PROMPT = Path(__file__).resolve().parents[1] / "ceres" / "assets" / "loop-v0" / "04_rag" / "prompts" / "grounded_answer_v3.md"


def _width_hits(n):
    return [_Item(chunk_id=f"c{i}", parent_id=f"p{i}", work_id="w", membership="m", rights_state="redistributable",
                  payload=f"passage {i}", payload_sha256="0" * 64) for i in range(n)]


def test_packet_holds_max_contexts_passages_not_a_fixed_five():
    """Bug: build_packet stopped at 5 for every route, so a route asking for 8 passages silently served 5."""
    packet = pipeline.build_packet("q", "en", "a" * 64, _width_hits(10), max_contexts=8)
    assert [c.evidence_id for c in packet.contexts] == [f"E{i}" for i in range(1, 9)]
    assert len(pipeline.build_packet("q", "en", "a" * 64, _width_hits(10)).contexts) == 5   # the default is unchanged


@_requires_k8_profiles
def test_a_rerank_wider_than_the_packet_is_refused_at_startup(config, inline_stack):
    """Bug: the k8 route (rerank top_k_output 8, answer profile without max_contexts) started and served 5
    passages under the k8 route id. These are the real profiles of that route."""
    store, models, _ = inline_stack
    profiles = dict(config.profile_paths(), retrieve=_PROFILES / "retrieve" / "dense-top100-v2.yaml",
                    rerank=_PROFILES / "rerank" / "qwen3-0.6b-rerank-v4-k8.yaml",
                    answer=_PROFILES / "answer" / "grounded-qwen38-v3-ctx8k.yaml")
    with pytest.raises(pipeline.PipelineError, match=r"returns 8 passages but the answer packet holds 5"):
        pipeline.Deps.from_files(config, store, models, FakeLLM(), profiles, _V3_PROMPT)


@_requires_k8_profiles
def test_an_answer_profile_with_max_contexts_8_serves_8(config, inline_stack, tmp_path):
    """Bug: max_contexts in the answer profile was read but not used by the served packet width or /v1/info."""
    store, models, _ = inline_stack
    answer = yaml.safe_load((_PROFILES / "answer" / "grounded-qwen38-v3-ctx8k.yaml").read_text(encoding="utf-8"))
    answer["max_contexts"] = 8
    (tmp_path / "answer.yaml").write_text(yaml.safe_dump(answer), encoding="utf-8")
    profiles = dict(config.profile_paths(), retrieve=_PROFILES / "retrieve" / "dense-top100-v2.yaml",
                    rerank=_PROFILES / "rerank" / "qwen3-0.6b-rerank-v4-k8.yaml", answer=tmp_path / "answer.yaml")
    deps = pipeline.Deps.from_files(config, store, models, FakeLLM(), profiles, _V3_PROMPT)
    assert (deps.max_contexts, deps.rerank_top_k) == (8, 8)


def test_a_prompt_that_names_e1_to_e5_is_refused_for_a_wider_packet():
    """Bug: prompts v1/v2 tell the model the passages are "E1..E5"; given 8, it would treat E6..E8 as invalid ids."""
    with pytest.raises(pipeline.PipelineError, match=r"names passages E1..E5 but the packet holds 8"):
        pipeline.packet_width({"max_contexts": 8}, 'one of the passage identifiers E1..E5>')
    assert pipeline.packet_width({"max_contexts": 5}, 'identifiers E1..E5') == 5
    assert pipeline.packet_width({"max_contexts": 8}, _V3_PROMPT.read_text(encoding="utf-8")) == 8   # v3: "E1, E2, …"


@pytest.mark.parametrize("bad", [0, 13, "8", True, 8.0])
def test_a_malformed_packet_width_is_refused(bad):
    """Bug: a typo in max_contexts (0, a string, a float, a bool, beyond the limit) loads and serves an odd packet."""
    with pytest.raises(pipeline.PipelineError, match="max_contexts"):
        pipeline.packet_width({"max_contexts": bad})


# ---------------------------------------------------------------- malformed envelopes and answer-contract types

class _EnvelopeLLM:
    """Answers every call with the given (already parsed) envelope."""

    def __init__(self, envelope):
        self.envelope = envelope

    def chat(self, body):
        return self.envelope, json.dumps(self.envelope).encode()


MALFORMED_ENVELOPES = [{"choices": [None]}, {"choices": [{"message": "bad"}]}, {"choices": {"bad": 1}}]


@pytest.mark.parametrize("envelope", MALFORMED_ENVELOPES, ids=["null-choice", "string-message", "dict-choices"])
def test_a_malformed_model_envelope_is_a_recorded_structure_failure_not_a_crash(envelope):
    """Bug caught: the envelope was read a second time outside the protected block, so these replies
    raised AttributeError/KeyError out of answer() and no attempt was recorded."""
    events: list = []
    profile = {"model_alias": "m", "decoding": {"maximum_attempts": 2}}
    row = pipeline.answer(_EnvelopeLLM(envelope), profile, "prompt", SWITCH_PACKET, events=events)
    assert row.state == "STRUCTURE_FAILED" and row.attempts == 2
    assert row.errors == [pipeline.INVALID_RESPONSE_CODE]
    assert [event["attempt"] for event in events] == [1, 2]
    assert all(event["structural_errors"] for event in events)


def test_a_malformed_model_envelope_through_ask_returns_structure_failed(config, inline_stack):
    """Bug caught: the same malformed envelope escaped pipeline.ask() as an exception (an HTTP 500)."""
    deps = _llm_deps(config, inline_stack, _EnvelopeLLM({"choices": [None]}))
    response = pipeline.ask(deps, AskRequest(question=EN_QUESTION, lang="en"))
    assert response.answer["state"] == "STRUCTURE_FAILED"
    assert response.answer["attempts"] == 2


@pytest.mark.parametrize("bad", [[{"bad": "object"}, 12, None], [12], ["fine", None]])
def test_missing_information_with_non_string_entries_is_not_validated(bad):
    """Bug caught: only the list length was checked, so non-string entries came back VALIDATED
    although the API contract says missing_information holds 0-4 strings."""
    candidate = _switch_candidate(decision="PARTIAL", reason_code="PARTIAL_SUPPORT", missing_information=bad)
    assert "ceres-validate-r1: missing_information entries must be strings" in pipeline.validate(candidate, SWITCH_PACKET)
    profile = {"model_alias": "m", "decoding": {"maximum_attempts": 1}}
    row = pipeline.answer(_ScriptedLLM(candidate), profile, "prompt", SWITCH_PACKET)
    assert row.state == "STRUCTURE_FAILED"


def test_missing_information_of_strings_is_still_validated():
    """Guard against over-tightening: a PARTIAL with string entries passes the new rule."""
    candidate = _switch_candidate(decision="PARTIAL", reason_code="PARTIAL_SUPPORT", missing_information=["soil type"])
    assert pipeline.validate(candidate, SWITCH_PACKET) == []


# ---------------------------------------------------------------- health_url probe

def _health_llm(health_status, *, health_raises=False, expect="qwen38"):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "qwen38"}]})
        if health_raises:
            raise httpx.ConnectError("refused")
        return httpx.Response(health_status)

    llm = OpenAIChatLLM("http://answerer.test/v1", expect_model=expect, health_url="http://answerer.test/health",
                        transport=httpx.MockTransport(handler))
    return llm, seen


def test_health_probes_the_configured_health_url_and_is_ready_on_2xx():
    """Bug caught: health_url was stored but never requested."""
    llm, seen = _health_llm(200)
    assert llm.health()[0] == "ready"
    assert "http://answerer.test/health" in seen


@pytest.mark.parametrize("status,raises", [(503, False), (200, True)], ids=["http-503", "unreachable"])
def test_a_failing_health_url_makes_the_answerer_down_even_when_the_model_is_listed(status, raises):
    """Bug caught: /v1/models listed the model, so a dead health endpoint still reported ready."""
    llm, _ = _health_llm(status, health_raises=raises)
    assert llm.health()[0] == "down"


def test_without_a_health_url_only_the_models_endpoint_is_probed():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={"data": [{"id": "qwen38"}]})

    llm = OpenAIChatLLM("http://answerer.test/v1", expect_model="qwen38", transport=httpx.MockTransport(handler))
    assert llm.health()[0] == "ready" and seen == ["http://answerer.test/v1/models"]
