"""Checked chunk context for the answerer: store column, context line, flag, validator, fixture.

Failure modes, written before the tests:
  - the flag is off but the context still reaches the prompt, the request body or `meta`
      (a v3.4 answer is no longer byte-identical to r1);
  - `context_line` renders a value whose status is not "found" (ambiguous / not_found /
      extraction_failed / not_applicable) — a guess presented as metadata;
  - a "found" value that is malformed (impossible date, inverted period, empty place) is
      rendered or repaired instead of omitted;
  - the line takes the date from the passage (the running header "No 6 – 24 June 2024") instead
      of the row's checked issue date (22 July 2024);
  - the line is placed inside REFERENCE TEXT, or a value carries a newline into the prompt
      structure;
  - a support_quote copied from the SOURCE METADATA line validates (metadata becomes citable);
  - the flag changes the packet (the loop schema forbids extra context keys) or evidence_text
      (the header is stripped), or the quote check of a payload quote;
  - the announced names (issued_date, place_coverage, …) are silently mapped;
  - the store SELECTs an absent `context` column (500 on today's v3.4 store), ignores a present
      one, or fails a request on a malformed context (it is metadata, never citable);
  - `Item.context` leaks into `record()` or `dataclasses.asdict` (ceres-eval's cache key);
  - `config_sha256` moves for every config that does not set the new section.

Fixture: `tests/fixtures-ctx/` = ONE verbatim row of the v3.5 release-0008 chunk-context artifact
and its candidate chunk (MANIFEST.json there names source lines and sha256s).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pytest

from ceres import pipeline
from ceres.config import load_config
from ceres.contracts import AskRequest, validate_packet
from ceres.llm import FakeLLM, answer_user_prompt, build_request_body, context_line
from ceres.models import FakeModels
from ceres.store import FixtureStore, PgStore, Release

from conftest import EXAMPLE_CONFIG, FIXTURES, ROOT, read_jsonl

CTX = ROOT / "tests" / "fixtures-ctx"
ROW_LINE = (CTX / "chunk-context.jsonl").read_bytes().rstrip(b"\n")
CHUNK_LINE = (CTX / "candidate-chunks.jsonl").read_bytes().rstrip(b"\n")
ROW = json.loads(ROW_LINE)
CHUNK = json.loads(CHUNK_LINE)
HEADER = "No 6 – 24 June 2024"
EXPECTED_LINE = ("Source: JRC MARS Bulletin · issued 22 July 2024 · reporting period"
                 " 1 Jun–13 Jul 2024 · place: Poland")
RELEASE = "ctx-fixture/release-attempt-0001"
QUESTION = "How did the winter barley forecast for Poland change?"


def ctx_store(context=ROW) -> FixtureStore:
    """The fixture chunk as the one item of a release; `context` is what `items.context` holds."""
    row = {
        "release_attempt_id": RELEASE, "chunk_id": CHUNK["chunk_id"], "parent_id": CHUNK["parent_id"],
        "work_id": CHUNK["parent_id"].split(":")[0], "membership": "train_eligible",
        "rights_state": "redistributable", "evidence_eligible": True, "payload": CHUNK["payload"],
        "payload_sha256": CHUNK["payload_sha256"], "locator_ids": CHUNK["locator_ids"],
        "lineage_ids": CHUNK["lineage_ids"], "excluded_reasons": CHUNK["excluded_reasons"],
    }
    if context is not ...:
        row["context"] = context
    vectors = np.eye(1, 8, dtype=np.float32)
    return FixtureStore([row], vectors, Release(release_attempt_id=RELEASE, route_id="ctx", row_count=1))


def ctx_config(env, flag: bool):
    extra = {"CERES_ANSWER__CONTEXT_LINES": "true"} if flag else {}
    return load_config(EXAMPLE_CONFIG, dict(env, **extra), require_api_key=False)


def ctx_ask(env, flag: bool, *, context=ROW, llm=None):
    llm = llm or FakeLLM()
    deps = pipeline.Deps.build(ctx_config(env, flag), ctx_store(context), FakeModels(dim=8), llm)
    return pipeline.ask(deps, AskRequest(question=QUESTION, lang="en")), llm, deps


def with_field(key, **change):
    row = json.loads(ROW_LINE)
    row[key] = dict(row[key], **change)
    return row


# ---------------------------------------------------------------- the fixture itself

def test_the_fixture_is_the_poland_page_16_row_of_the_july_2024_bulletin():
    """Bug caught: a fixture edited by hand, or a different row than the task names. The copy must
    be byte-identical to the artifact row (MANIFEST sha), page_index 15 (0-based) = page 16."""
    manifest = json.loads((CTX / "MANIFEST.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(ROW_LINE).hexdigest() == manifest["files"]["chunk-context.jsonl"]["row_sha256"]
    assert hashlib.sha256(CHUNK_LINE).hexdigest() == manifest["files"]["candidate-chunks.jsonl"]["row_sha256"]
    assert ROW["chunk_id"] == CHUNK["chunk_id"] == manifest["chunk_id"]
    assert ROW["payload_sha256"] == CHUNK["payload_sha256"] == hashlib.sha256(
        CHUNK["payload"].encode("utf-8")).hexdigest()
    assert ROW["parent_id"].startswith("mars-europe-2024-07-vol32-no7:")
    assert [page["page_index"] for page in ROW["places_by_page"]] == [15]
    assert ROW["issue_date"]["value"] == "2024-07-22" and ROW["place"]["value"] == "Poland"
    assert HEADER in CHUNK["payload"]


# ---------------------------------------------------------------- the context line (pure)

def test_the_context_line_says_july_2024_not_the_running_header_june():
    """the line is built from the row's checked issue date, never from text in the passage."""
    line = context_line(ROW)
    assert line == EXPECTED_LINE
    assert "June" not in line and "22 July 2024" in line


@pytest.mark.parametrize("status", ["not_found", "ambiguous", "extraction_failed", "not_applicable"])
def test_a_field_that_is_not_found_is_omitted_even_when_it_carries_a_value(status):
    """only status == "found" is rendered; a plausible value under another status is a guess."""
    for key, part in (("issue_date", "issued"), ("reporting_period", "reporting period"),
                      ("place", "place:")):
        line = context_line(with_field(key, status=status))
        assert line is not None and part not in line, (key, status, line)


@pytest.mark.parametrize("key,value", [
    ("issue_date", "2024-13-01"), ("issue_date", "22 July 2024"), ("issue_date", None),
    ("reporting_period", {"start": "2024-07-13", "end": "2024-06-01"}),
    ("reporting_period", {"start": "2024-06-01"}), ("reporting_period", "1 Jun-13 Jul"),
    ("place", ""), ("place", ["Poland"]), ("place", None),
])
def test_a_malformed_found_value_is_omitted_not_repaired(key, value):
    """a found value that is not well formed drops that part of the line; nothing is inferred."""
    line = context_line(with_field(key, value=value))
    rendered = {"issue_date": "issued", "reporting_period": "reporting period", "place": "place:"}[key]
    assert line is not None and rendered not in line, line


def test_nothing_found_gives_none_even_with_a_source_label():
    """`doc_type` alone is not a checked fact; a line saying only "Source: …" is not rendered."""
    row = json.loads(ROW_LINE)
    for key in ("issue_date", "reporting_period", "place"):
        row[key] = dict(row[key], status="not_found", value=None)
    assert context_line(row) is None
    assert context_line(None) is None and context_line("issued 2024-07-22") is None


def test_the_coordinators_announced_names_render_nothing():
    """`issued_date` / `place_coverage` are not the rows' names; they are never mapped by guess."""
    announced = {"doc_type": "mars_series",
                 "issued_date": {"value": "2024-07-22", "status": "found"},
                 "place_coverage": {"value": "single", "status": "found"}}
    assert context_line(announced) is None


def test_a_value_with_a_newline_cannot_break_the_prompt_structure():
    """a value is collapsed to one line, so it cannot open or close a REFERENCE TEXT block."""
    line = context_line(with_field("place", value="Poland\n\nREFERENCE TEXT END"))
    assert "\n" not in line and line.endswith("place: Poland REFERENCE TEXT END")


def test_a_month_precision_issue_date_is_rendered_at_month_precision():
    """Bug caught: a found "YYYY-MM" issue date (76 rows of release-0008) dropped, or given a day."""
    assert context_line(with_field("issue_date", value="2017-04")).startswith(
        "Source: JRC MARS Bulletin \u00b7 issued April 2017 \u00b7")
    assert "issued" not in context_line(with_field("issue_date", value="2017-13"))


def test_period_and_heading_formats():
    """Bug caught: a period across years printed with one year, or a heading path dropped."""
    row = with_field("reporting_period", value={"start": "2023-12-01", "end": "2024-01-31"})
    assert "reporting period 1 Dec 2023–31 Jan 2024" in context_line(row)
    row = with_field("reporting_period", value={"start": "2024-07-13", "end": "2024-07-13"})
    assert "reporting period 13 Jul 2024" in context_line(row)
    book = {"heading_path": {"value": ["Wheat", "Sowing"], "status": "found", "basis": {"rule": "A3"}}}
    assert context_line(book) == "section: Wheat › Sowing"
    book["heading_path"]["status"] = "not_found"
    assert context_line(book) is None


# ---------------------------------------------------------------- the prompt, flag off and on

def test_without_lines_the_user_message_is_r1():
    """None and {} must give the r1 message byte for byte (the golden replay depends on it)."""
    packet = pipeline.build_packet(QUESTION, "en", "q", ctx_store().search(np.eye(1, 8)[0], RELEASE,
                                                                           ["redistributable"])).to_dict()
    assert answer_user_prompt(packet, None) == answer_user_prompt(packet, {}) == answer_user_prompt(packet)
    assert "SOURCE METADATA" not in answer_user_prompt(packet)


def test_flag_off_a_context_on_the_item_changes_nothing(env):
    """the same question over a store WITH the context column and one WITHOUT it gives the same
    request bodies and the same serialized response (execution id and latency aside)."""
    with_ctx, llm_a, _ = ctx_ask(env, False)
    without, llm_b, _ = ctx_ask(env, False, context=...)
    assert llm_a.calls == llm_b.calls
    assert with_ctx.packet == without.packet and with_ctx.answer == without.answer
    dumped = json.loads(with_ctx.model_dump_json())
    assert "context_lines" not in dumped["meta"]


def test_flag_on_the_line_precedes_reference_text_and_is_recorded(env):
    """The line sits between [E1] and REFERENCE TEXT START; the packet is unchanged and still
    schema-valid; the line is in the raw request (on disk) and in meta.context_lines."""
    on, llm, deps = ctx_ask(env, True)
    off, _, _ = ctx_ask(env, False)
    user = llm.calls[0]["messages"][1]["content"]
    assert (f"[E1]\n\nSOURCE METADATA: {EXPECTED_LINE}\n\nREFERENCE TEXT START\n\n{CHUNK['payload']}"
            f"\n\nREFERENCE TEXT END") in user
    assert on.packet == off.packet
    validate_packet(on.packet)
    assert on.packet["contexts"][0]["evidence_text"] == CHUNK["payload"]
    assert HEADER in on.packet["contexts"][0]["evidence_text"]
    assert [record.model_dump() for record in on.meta.context_lines] == [
        {"evidence_id": "E1", "line": EXPECTED_LINE}]
    raw_dir = deps.config.raw_executions_dir / on.meta.execution_id / "raw"
    recorded = json.loads(next(raw_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert EXPECTED_LINE in recorded["request"]["messages"][1]["content"]


def test_flag_on_a_context_without_found_fields_leaves_the_message_r1(env):
    """Flag on, but nothing found (or a v3.4 item) → no SOURCE METADATA, meta lists nothing."""
    empty = json.loads(ROW_LINE)
    for key in ("issue_date", "reporting_period", "place"):
        empty[key] = dict(empty[key], status="ambiguous", value=None)
    for context in (empty, ...):
        on, llm_on, _ = ctx_ask(env, True, context=context)
        off, llm_off, _ = ctx_ask(env, False, context=context)
        assert llm_on.calls == llm_off.calls
        assert on.meta.context_lines == []


# ---------------------------------------------------------------- the validator

class MetadataQuotingLLM(FakeLLM):
    """An answerer that cites the SOURCE METADATA line as if it were the passage."""

    def _compose(self, user_prompt):
        line = user_prompt.split("SOURCE METADATA: ", 1)[1].split("\n", 1)[0]
        quote = " ".join(line.split()[5:12])          # "issued 22 July 2024 · reporting period"
        return {"decision": "ANSWER", "answer_language": "en", "user_answer": f"It says {quote}",
                "citations": [{"claim": quote, "evidence_id": "E1", "support_quote": quote}],
                "missing_information": [], "confidence": "MEDIUM", "reason_code": "SUPPORTED"}


def test_a_quote_taken_from_the_metadata_line_is_rejected(env):
    """the validator reads evidence_text only; a quote that exists only in SOURCE METADATA fails,
    directly and through the whole answer loop (both attempts → STRUCTURE_FAILED)."""
    response, llm, _ = ctx_ask(env, True, llm=MetadataQuotingLLM())
    assert response.answer["state"] == "STRUCTURE_FAILED"
    assert pipeline.QUOTE_SUBSTRING_ERROR in response.answer["errors"]
    candidate = llm._compose(llm.calls[0]["messages"][1]["content"])
    quote = candidate["citations"][0]["support_quote"]
    assert quote in EXPECTED_LINE and quote not in CHUNK["payload"]
    assert pipeline.QUOTE_SUBSTRING_ERROR in pipeline.validate(candidate, response.packet)


def test_a_payload_quote_checks_the_same_with_the_flag_on_or_off(env):
    """a quote from the passage (here one that includes the June running header) validates
    identically with and without the metadata line."""
    on, _, _ = ctx_ask(env, True)
    off, _, _ = ctx_ask(env, False)
    assert on.answer["state"] == off.answer["state"] == "VALIDATED"
    assert on.answer["candidate"] == off.answer["candidate"]
    quote = on.answer["candidate"]["citations"][0]["support_quote"]
    assert HEADER in quote
    assert pipeline.validate(on.answer["candidate"], on.packet) == \
        pipeline.validate(off.answer["candidate"], off.packet) == []


class ReplayLLM:
    """Returns the golden response recorded for the request's seed; keeps every body it was sent."""

    MODEL_ID = "replay"

    def __init__(self, raw):
        self.by_seed = {event["request"]["seed"]: event for attempts in raw.values()
                        for event in attempts.values()}
        self.bodies = []

    def chat(self, request_body):
        self.bodies.append(request_body)
        event = self.by_seed[request_body["seed"]]
        return event["response"], json.dumps(event["response"], ensure_ascii=False).encode("utf-8")

    def health(self):
        return "ready", "replay", self.MODEL_ID


def test_flag_on_over_the_v34_fixture_still_replays_the_golden_request_bodies(env):
    """turning the flag on against a store without the context column (today's v3.4) must not
    change a single request byte: every golden body is reproduced to its sha."""
    questions = read_jsonl(FIXTURES / "questions.jsonl")
    items = {row["chunk_id"]: row.get("payload", "") for row in read_jsonl(FIXTURES / "items.jsonl")}
    vectors = np.load(FIXTURES / "query-vectors.npy").astype(np.float32)
    texts = {row["question_id"]: row["text"] for row in questions}
    scores = {(texts[row["question_id"]].strip(), items[row["chunk_id"]].strip()): float(row["score"])
              for row in read_jsonl(FIXTURES / "golden" / "reranked-top5.jsonl") if row["chunk_id"] in items}
    models = FakeModels({row["question_id"]: vectors[i] for i, row in enumerate(questions)},
                        {row["text"].strip(): row["question_id"] for row in questions},
                        dim=vectors.shape[1], rerank_scores=scores)
    raw: dict[str, dict[int, dict]] = {}
    for path in sorted((FIXTURES / "golden" / "raw").glob("*.json")):
        event = json.loads(path.read_text(encoding="utf-8"))
        raw.setdefault(event["question_id"], {})[int(event["attempt"])] = event
    llm = ReplayLLM(raw)
    deps = pipeline.Deps.build(ctx_config(env, True), FixtureStore.from_dir(FIXTURES), models, llm)
    for row in questions:
        response = pipeline.ask(deps, AskRequest(question=row["text"], lang=row["language"],
                                                 question_id=row["question_id"]))
        assert response.meta.context_lines == []
    assert len(llm.bodies) == 6
    for body in llm.bodies:
        canonical = json.dumps(body, ensure_ascii=False, allow_nan=False, sort_keys=True,
                               separators=(",", ":")).encode("utf-8")
        assert hashlib.sha256(canonical).hexdigest() == llm.by_seed[body["seed"]]["request_sha256"]


def test_build_request_body_with_lines_changes_only_the_user_message():
    """Bug caught: the flag leaking into decoding fields, the seed or the system prompt."""
    packet = pipeline.build_packet(QUESTION, "en", "q", ctx_store().search(np.eye(1, 8)[0], RELEASE,
                                                                           ["redistributable"])).to_dict()
    profile = {"model_alias": "m"}
    plain = build_request_body(packet, profile, "P", 1)
    lined = build_request_body(packet, profile, "P", 1, {"E1": EXPECTED_LINE})
    assert {k: v for k, v in plain.items() if k != "messages"} == \
        {k: v for k, v in lined.items() if k != "messages"}
    assert plain["messages"][0] == lined["messages"][0]
    assert plain["messages"][1] != lined["messages"][1]


# ---------------------------------------------------------------- store and config

def test_the_fixture_store_reads_the_context_and_tolerates_a_bad_one():
    """absent key → None; JSON text → dict; a non-object → None, and the search still serves."""
    query = np.eye(1, 8)[0]
    assert ctx_store(...).search(query, RELEASE, ["redistributable"])[0].context is None
    assert ctx_store(ROW).search(query, RELEASE, ["redistributable"])[0].context == ROW
    assert ctx_store(ROW_LINE.decode()).search(query, RELEASE, ["redistributable"])[0].context == ROW
    for bad in (["Poland"], "not json", 7):
        item = ctx_store(bad).search(query, RELEASE, ["redistributable"])[0]
        assert item.context is None and item.payload == CHUNK["payload"]


def test_the_context_stays_out_of_the_hit_record_and_the_eval_cache_key():
    """`record()` goes to logs and run records; `asdict` is ceres-eval's rerank record, whose
    sha is the answer phase's cache key. Neither may gain a `context` key."""
    item = ctx_store(ROW).search(np.eye(1, 8)[0], RELEASE, ["redistributable"])[0]
    assert item.context == ROW
    assert "context" not in item.record()
    assert "context" not in dataclasses.asdict(item)


def test_the_answer_section_leaves_the_config_digest_alone_until_it_is_set(env):
    """every existing config keeps its `config_sha256`; turning the flag on changes it."""
    off, on = ctx_config(env, False), ctx_config(env, True)
    assert off.answer.context_lines is False and on.answer.context_lines is True
    legacy = off.model_dump(mode="json", by_alias=True,
                            exclude={"api_key", "source_path", "auth_required", "answer"})
    canonical = json.dumps(legacy, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert off.config_sha256() == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert on.config_sha256() != off.config_sha256()


def test_the_pg_select_names_the_context_column_only_when_the_release_has_it():
    """an absent column is a typed NULL placeholder, a present one is selected by name."""
    store = PgStore("postgresql://ceres_reader@127.0.0.1/veridian_rag", open_pool=False)
    assert "NULL::jsonb AS context" in store.search_sql(("evidence_basis",), None)
    assert "context AS context" in store.search_sql(("evidence_basis", "context"), None)
    assert "NULL::jsonb" not in store.search_sql(("context",), None)


def test_postgres_reads_the_context_column_when_present_and_null_when_absent(pg):
    """The store against real PostgreSQL + pgvector: detection, jsonb decoding, and today's layout."""
    from psycopg.types.json import Jsonb

    from conftest import fixture_rows, fixture_vectors, fixture_query_vectors

    plain = pg.store("ceres_serve_v0")
    release = plain.active_release()
    assert "context" not in release.optional_columns
    items = plain.search(fixture_query_vectors()[0], release.release_attempt_id, list(RND), 50)
    assert items and all(item.context is None for item in items)

    picked = [i for i, row in enumerate(fixture_rows())
              if row["evidence_eligible"] and row["rights_state"] in RND][:6]
    rows, vectors = [fixture_rows()[i] for i in picked], fixture_vectors()[picked]
    pg.create_schema("ctx_present")
    with pg.admin() as conn:
        conn.execute("ALTER TABLE ctx_present.items ADD COLUMN IF NOT EXISTS context jsonb")
    pg.load("ctx_present", "rel-ctx", rows, vectors, active=True)
    with pg.admin() as conn:
        conn.execute("UPDATE ctx_present.items SET context = %s WHERE chunk_id = %s",
                     (Jsonb(ROW), rows[0]["chunk_id"]))
    store = pg.store("ctx_present")
    assert store.active_release().optional_columns == ("context",)
    found = {item.chunk_id: item for item in store.search(vectors[0], "rel-ctx", list(RND), 50)}
    assert found[rows[0]["chunk_id"]].context == ROW
    assert all(item.context is None for chunk_id, item in found.items() if chunk_id != rows[0]["chunk_id"])
    assert context_line(found[rows[0]["chunk_id"]].context) == EXPECTED_LINE


RND = ("redistributable", "internal_research_only", "licensed")


# ---------------------------------------------------------------- ruling on the context shape

def _ctx(**fields):
    base = {"doc_type": "mars_series"}
    base.update(fields)
    return base


def _f(value, **basis):
    return {"value": value, "status": "found", "basis": basis}


def test_a_found_document_scope_stands_in_for_a_missing_place_labelled_with_its_scope():
    """Regional bulletins (Ukraine, Russia, …): without it the line would drop the only place the chunk has."""
    from ceres.llm import context_line
    line = context_line(_ctx(issue_date=_f("2024-04-15"),
                             place={"value": None, "status": "extraction_failed", "basis": {}},
                             document_scope=_f("Türkiye", scope="document")))
    assert line is not None and "place: Türkiye (document scope)" in line


@pytest.mark.parametrize("basis", ["x", ["document"], 7])
def test_a_non_dict_document_scope_basis_never_crashes_the_context_line(basis):
    """a truthy non-dict basis raised AttributeError → every question citing the chunk got a 500."""
    from ceres.llm import context_line
    line = context_line(_ctx(place={"value": None, "status": "not_found", "basis": {}},
                             document_scope={"value": "Ukraine", "status": "found", "basis": basis}))
    assert line is not None and "place: Ukraine (document scope)" in line


def test_document_scope_never_overrides_a_found_place():
    """A document-level region must not replace or duplicate the page-level place the rows found."""
    from ceres.llm import context_line
    line = context_line(_ctx(place=_f("Poland"), document_scope=_f("North Africa", scope="document")))
    assert "place: Poland" in line and "North Africa" not in line


def test_extraction_failed_fields_are_not_shown():
    """extraction_failed is 'not shown' (by ruling); rendering its value would print a failed guess."""
    from ceres.llm import context_line
    line = context_line(_ctx(issue_date={"value": "2024-06-24", "status": "extraction_failed", "basis": {}},
                             place=_f("Poland")))
    assert "2024" not in line and "June" not in line and "place: Poland" in line
