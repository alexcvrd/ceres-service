"""Wire contracts and the loop-shaped records that travel inside them.

Two families live here on purpose:

  * pydantic models — what the HTTP API accepts and returns (`AskRequest`, `AskResponse`,
    `HealthResponse`, `InfoResponse`);
  * plain dicts in the loop's own shapes — the answer packet
    (`rag-answer-packets.schema.json`) and the answer row (`answers.schema.json`), validated
    against the schema files shipped in `ceres/schemas/` (byte-identical copies of
    `loop-v0/00_control/schemas/`). The packet schema is the loop's
    updated copy (sha256 `PACKET_SCHEMA_SHA256`, optional context field `evidence_basis`);
    `tests/fixtures/schemas/` keeps the
    earlier copy the first fixture was built against.

Ceres never invents a packet shape: the DDS evaluation step judges `packets.jsonl` + `answers.jsonl`
(the loop's answer contract), so what `/v1/ask` returns must be exactly what the eval hook writes.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import jsonschema
from pydantic import BaseModel, ConfigDict, Field

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
PACKET_SCHEMA = "rag-answer-packets.schema.json"
ANSWER_SCHEMA = "answers.schema.json"
# sha256 of the DDS engine's `00_control/schemas/rag-answer-packets.schema.json`
# (`contexts[].evidence_basis`, enum ["ocr_admitted_unverified"]). A test pins the shipped
# copy to it, so a silent edit of either side is visible.
PACKET_SCHEMA_SHA256 = "261ef189fae42284edca0cdbc54c4df929680203ea013299d2fccc60c51b1c5a"

# The serving-side validation revision. `structural_answer` is the loop's frozen validator; Ceres
# adds one check on top of it (non-empty citation claim and support_quote: the empty string is a substring of everything, so the loop validator accepts an
# empty claim). The revision id travels in `meta.validator_revision` so a judged number can always
# be attributed to the validator that produced it.
# r2: r1 plus ONE relaxation — a support_quote
# shortened with "..."/"…" passes when every guard in `pipeline.QUOTE_GUARDS` holds (pieces of >= 3
# words, in order, in the same cited passage, short gaps with no digit and no negation word). The
# guards live in one place (`pipeline.QuoteGuards`); `/v1/info` reports them.
# r3: r2 plus ONE guard — no omitted gap may hold a capitalized
# word that does not start a sentence (a blunt proxy for a dropped place/region/named-entity scope).
VALIDATOR_REVISION = "ceres-validate-r3"

# The question-id encoding, documented because a bank id must be reproducible by anyone:
#   sha256( "ceres/question/v1" ‹US› lang ‹US› NFC(question.strip()) ), UTF-8, lowercase hex.
# ‹US› is U+001F, the same unit separator the loop uses between identity fields
# (`slot_adapter.py:409 CHUNK_IDENTITY_SEPARATOR`). A question the DDS bank already identifies
# keeps the bank's id: the client sends it as `question_id` and Ceres does not recompute it.
QUESTION_ID_DOMAIN = "ceres/question/v1"
UNIT_SEPARATOR = "\x1f"

MAX_CONTEXTS = 5  # the DEFAULT packet width E1..E5 — what the loop's `build_packets` uses (`slot_adapter.py:3281`)
# An answer profile may set `max_contexts` (packet width); see pipeline.packet_width.
MAX_CONTEXTS_LIMIT = 12  # the widest packet any answer profile may ask for


def question_id_for(question: str, lang: str) -> str:
    """The deterministic ad-hoc question id (see QUESTION_ID_DOMAIN for the exact encoding)."""
    text = unicodedata.normalize("NFC", question.strip())
    joined = UNIT_SEPARATOR.join((QUESTION_ID_DOMAIN, lang, text))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- loop-shaped records

@dataclass
class Context:
    """One E1..E5 evidence context of an answer packet (`rag-answer-packets.schema.json`)."""

    evidence_id: str
    chunk_id: str
    parent_id: str
    evidence_text: str
    evidence_text_sha256: str
    locator_ids: list[str]
    lineage_ids: list[str]
    warnings: list[str]   # = the store's `excluded_reasons` for that chunk
    # Carried from the chunk when non-null (`ocr_admitted_unverified`). Omitted otherwise, so
    # an ordinary packet stays byte-identical to the loop's goldens.
    evidence_basis: str | None = None

    def to_dict(self) -> dict[str, Any]:
        context = {
            "evidence_id": self.evidence_id, "chunk_id": self.chunk_id, "parent_id": self.parent_id,
            "evidence_text": self.evidence_text, "evidence_text_sha256": self.evidence_text_sha256,
            "locator_ids": list(self.locator_ids), "lineage_ids": list(self.lineage_ids),
            "warnings": list(self.warnings),
        }
        if self.evidence_basis is not None:
            context["evidence_basis"] = self.evidence_basis
        return context


@dataclass
class Packet:
    """The DDS answer packet handed to the answer model."""

    packet_id: str
    question_id: str
    question: str
    language: str
    contexts: list[Context] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_id": self.packet_id, "question_id": self.question_id,
            "question": self.question, "language": self.language,
            "contexts": [c.to_dict() for c in self.contexts],
        }


@dataclass
class AnswerRow:
    """One row of `answers.jsonl` (`answers.schema.json`, contract v0.2)."""

    question_id: str
    state: Literal["VALIDATED", "STRUCTURE_FAILED", "TRANSPORT_FAILED"]
    attempts: int
    candidate: dict[str, Any] | None = None
    errors: list[str] | None = None
    # Citation switch: {"citation_mode", "strict_errors", "dropped_citations"} of the served (last) reply. NOT part of
    # `to_dict()`: the loop's answer row admits no extra key, so the record rides on meta / work rows / raw events.
    citation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "question_id": self.question_id, "state": self.state, "attempts": self.attempts,
        }
        # The schema forbids nothing here, but the loop writes `candidate` only on VALIDATED and
        # `errors` only on failure; the evaluation step joins on those keys.
        if self.candidate is not None:
            row["candidate"] = self.candidate
        if self.errors is not None:
            row["errors"] = self.errors
        return row


# ---------------------------------------------------------------- schema validation

@lru_cache(maxsize=4)
def _schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def schema_errors(document: Any, name: str) -> list[str]:
    """Human-readable validation errors of `document` against one shipped schema."""
    validator = jsonschema.Draft202012Validator(_schema(name))
    errors = sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path))
    return [f"{'.'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors]


def validate_packet(packet: dict[str, Any]) -> None:
    if errors := schema_errors(packet, PACKET_SCHEMA):
        raise ValueError(f"answer packet violates {PACKET_SCHEMA}: {errors[0]}")


def validate_answer_row(row: dict[str, Any]) -> None:
    if errors := schema_errors(row, ANSWER_SCHEMA):
        raise ValueError(f"answer row violates {ANSWER_SCHEMA}: {errors[0]}")


# ---------------------------------------------------------------- HTTP models

class AskRequest(BaseModel):
    """`POST /v1/ask`. `lang` is required in v0: no language detector, no silent drift
    (answer-packet contract; the answer profile validates `answer_language == packet.language`)."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)   # the upper bound is `api.max_question_chars` → 413
    lang: Literal["ro", "en"]
    # The DDS bank's own id, when the caller has one (the eval hook passes it so the evaluation step can
    # join on it). Ad-hoc questions get `question_id_for(question, lang)`.
    question_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    request_id: str | None = Field(default=None, max_length=200)
    # Memory v0. Both absent = the stateless r1 request, byte for byte.
    conversation_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{8,64}$")
    profile_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9-]{1,63}$")

    @property
    def uses_memory(self) -> bool:
        return self.conversation_id is not None or self.profile_id is not None

    def resolved_question_id(self) -> str:
        return self.question_id or question_id_for(self.question, self.lang)


class Latency(BaseModel):
    """Per-stage wall-clock milliseconds; `total` covers the whole request inside the service."""

    embed: float = 0.0
    retrieve: float = 0.0
    rerank: float = 0.0
    answer: float = 0.0
    total: float = 0.0


class Filtered(BaseModel):
    """Items of the active release the read predicate withholds (release-level, not per query).

    `rights`: items the rights clause withholds — `rights_state` outside the mode's derived
    `serve_rights` and not a licensed work admitted by id (rights rule, serving-mode amendment). The number
    therefore drops when an operator switches from production to rnd, and that is the point.
    `excluded`: always 0 since the release removed the `excluded_reasons` clause from the predicate. The
    key stays so the record shape does not change under the eval hook; a non-zero value would mean
    a predicate the release rules do not allow.
    """

    rights: int = 0
    excluded: int = 0


class CitationMark(BaseModel):
    """One CITED context whose evidence basis is non-null (today: OCR-admitted, unverified)."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    chunk_id: str
    work_id: str
    evidence_basis: str


class QuoteSpan(BaseModel):
    """ceres-validate-r2/r3: where each piece of one ACCEPTED ellipsis quote sits in its passage.

    `piece_spans` are `[start, end)` offsets into `normalize(evidence_text)` under `norm_version`.
    The served `support_quote` itself is never rewritten; this record is how a reader finds the
    words the "..." stands between.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    chunk_id: str
    support_quote: str
    norm_version: str
    piece_spans: list[list[int]]


class ContextLineRecord(BaseModel):
    """The SOURCE METADATA line one context was shown with (`answer.context_lines` on).

    The loop packet schema forbids extra context keys, so the line the answerer read is recorded
    here (and in the raw request). It is metadata: a support_quote taken from it fails validation.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    line: str


class AskMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_id: str
    request_id: str | None = None
    release_attempt_id: str
    # Which LOAD of that release served the answer. `dds load --replace`
    # rewrites a release under the same id, so the id alone cannot tell two corpora apart; the store's
    # (id, manifest sha) identity can. Absent (not null) when the store has no manifest (fixtures), so
    # fixture answers' bytes are unchanged.
    release_manifest_sha256: str | None = Field(default=None, exclude_if=lambda value: value is None)
    # The served route is `route.served_route_id`, and it is null until the CLI owner
    # freezes the served-route yaml. Null is reported as `route_status: "unassigned"`.
    # `judged_route_id` is r1's id for reference only — a served answer is never labelled r1.
    route_id: str | None = None
    route_status: Literal["assigned", "unassigned"] = "unassigned"
    judged_route_id: str
    # Which serving mode produced this answer. `rnd` reads every rights state; `production`
    # reads redistributable plus the licensed works a compliance decision admitted. It is in every
    # answer because a record that does not say which mode served it cannot be audited later.
    serving_mode: Literal["rnd", "production"] = "rnd"
    read_predicate_sha256: str      # identity of the predicate this evidence came through
    model_reported: str | None      # what the endpoint said it ran; a label, never a weight proof
    lang_source: Literal["client"] = "client"
    membership_counts: dict[str, int] = Field(default_factory=dict)
    filtered: Filtered = Filtered()
    latency_ms: Latency = Latency()
    validator_revision: str = VALIDATOR_REVISION
    # The cited contexts with a non-null evidence basis, from the validated candidate's
    # citations. Empty when nothing marked was cited, and always empty for a non-VALIDATED answer
    # (no candidate, nothing was cited to anyone).
    citation_marks: list[CitationMark] = Field(default_factory=list)
    # ceres-validate-r2/r3: one entry per citation of the VALIDATED candidate whose quote passed
    # only through the ellipsis rule. Empty when every quote is one continuous substring, and always
    # empty for a non-VALIDATED answer.
    quote_spans: list[QuoteSpan] = Field(default_factory=list)
    # One entry per context that was shown a SOURCE METADATA line, in E-order. Present
    # (possibly empty) only when `answer.context_lines` is on; ABSENT from the serialized meta
    # otherwise, so a default answer's bytes are unchanged.
    context_lines: list[ContextLineRecord] | None = Field(default=None,
                                                          exclude_if=lambda value: value is None)
    # Citation switch: the mode that judged this answer, the full strict (r3) error
    # list of the served reply (what strict mode would have said), and each citation soft mode dropped, with reasons.
    citation_mode: Literal["strict", "soft"] = "strict"
    # Rerank switch: {"status": identity | ok | fallback_oom, "model", "revision", "profile_id",
    # "rerank_ms"}. fallback_oom = the reranker ran out of GPU memory and the dense order was served for this answer.
    rerank: dict[str, Any] | None = Field(default=None, exclude_if=lambda value: value is None)
    # MARS issue boost (rerank profile qwen3-0.6b-rerank-v3): the question's parsed issue [year, month|None] (null when
    # none) and how many reranked candidates got the boost. Absent on routes without the boost.
    issue_boost: dict[str, Any] | None = Field(default=None, exclude_if=lambda value: value is None)
    strict_errors: list[str] = Field(default_factory=list)
    dropped_citations: list[dict[str, Any]] = Field(default_factory=list)
    # Memory v0: {conversation_id, turn, profile_id, standalone_question, retrieval_query, rewrite_status, is_followup,
    # used_profile_fields, history_turns_used, rewrite_ms, prompt_sha256}. ABSENT for a stateless request (bytes unchanged).
    memory: dict[str, Any] | None = Field(default=None, exclude_if=lambda value: value is None)
    # Situational-split experiment: {applied, status, core_question, qualifiers, ms} when answer.situational_split ran. ABSENT otherwise.
    situational: dict[str, Any] | None = Field(default=None, exclude_if=lambda value: value is None)
    # Readable sources: one entry per context, in E-order: {evidence_id, work_id, title, authors, language, page_label}. ABSENT when off.
    sources: list[dict[str, Any]] | None = Field(default=None, exclude_if=lambda value: value is None)


class AskResponse(BaseModel):
    """The three named parts of one answer: loop packet, loop answer row, serving metadata."""

    model_config = ConfigDict(extra="forbid")

    packet: dict[str, Any]
    answer: dict[str, Any]
    meta: AskMeta


class HealthResponse(BaseModel):
    """Liveness (`api`) against readiness (everything else), the readiness contract.

    `api: "ok"` only says the process answers. A component is `ready` when it is reachable AND it
    is the thing the config expects (a reachable endpoint serving another model is `wrong_model`,
    not ready). `ready` at the top is the AND of the components the served route needs.
    """

    model_config = ConfigDict(extra="forbid")

    api: Literal["ok"] = "ok"
    ready: bool = False
    store: str = "unknown"
    model: str = "unknown"
    embedder: str = "unknown"
    reranker: str = "unknown"
    release_attempt_id: str | None = None
    detail: dict[str, str] = Field(default_factory=dict)


class InfoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    route_id: str | None                    # the served route; null until it is frozen
    route_status: Literal["assigned", "unassigned"] = "unassigned"
    judged_route_id: str                    # r1, for reference only
    serving_mode: Literal["rnd", "production"] = "rnd"   # `store` carries the derived rights
    read_predicate_sha256: str | None = None
    validator_revision: str = VALIDATOR_REVISION
    validator_guards: dict[str, Any] = Field(default_factory=dict)   # r3 guards, as run
    packet_width: int | None = None   # passages per answer packet (answer profile `max_contexts`, default 5)
    release_attempt_id: str | None = None
    release_route_id: str | None = None
    # ROUTE-STORE-GUARD: the active release's chunk/represent/embed profile shas against the
    # configured route's. "skipped" = one side does not name all three (fixture/dev); null = the
    # release could not be read. A "mismatch" release refuses /v1/ask (503 route_store_mismatch).
    route_store_check: Literal["ok", "skipped", "mismatch"] | None = None
    row_count: int | None = None
    membership_counts: dict[str, int] = Field(default_factory=dict)
    profiles: dict[str, str] = Field(default_factory=dict)      # slot → sha256 of the profile file
    prompt_template_sha256: str | None = None
    store: dict[str, Any] = Field(default_factory=dict)         # safe DSN, schema, predicate
    models: dict[str, str | None] = Field(default_factory=dict)  # as the endpoints report them
    config_sha256: str | None = None
    # The answerer's LIVE settings (llama.cpp `/props`: n_ctx, model path, build) and the check
    # against the answer profile's context_tokens, as the last `/health` saw them; null before the
    # first `/health`. Declared settings live in the profile; this is what the server runs.
    answerer_live: dict[str, Any] | None = None
    adapters: dict[str, str] = Field(default_factory=dict)      # which implementation is wired in
