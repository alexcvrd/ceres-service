"""Route r1, as pure functions: retrieve → dedup → rerank → packet → answer → validate.

Read this module top to bottom; it is the whole serving recipe. Every rule carries the profile
field or the loop line it comes from, because a rule that drifts from the judged route silently
invalidates the eval numbers (any retrieval or reranker change is a new route id, judged
before it replaces r1).

No FastAPI, no HTTP, no logging of question text: `api.py` owns the transport, this module owns
the recipe, and `store`/`models`/`llm` own the I/O. `ask()` is the only function that composes
them, and it is the function a DDS command-line client will import if it becomes a client of this package.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .contracts import (
    MAX_CONTEXTS,
    MAX_CONTEXTS_LIMIT,
    AnswerRow,
    AskMeta,
    AskRequest,
    AskResponse,
    VALIDATOR_REVISION,
    CitationMark, ContextLineRecord,
    Context,
    Filtered,
    Latency,
    Packet,
    validate_answer_row,
    validate_packet,
)
from .llm import LLM, LLMRequestRejected, LLMResponseError, LLMTransportError, build_request_body, context_line
from .models import Models
from .normalize import NORM_VERSION, normalize, token_jaccard
from .store import Item, Release, Store, StoreError, read_predicate_sha256

# ---------------------------------------------------------------- the r1 constants
# Which of these a profile can change:
#   profile-controlled, read by `Deps.build` when the profile file is configured —
#     retrieve: candidate_pool, top_k_children, min_context_tokens, near_duplicate_rule;
#     rerank:   top_k_output;   answer: model alias, decoding, maximum_attempts, prompt hash.
#   fixed in code (a change here is a new route, not a config edit) —
#     RERANK_TOP_K_INPUT, MAX_CONTEXTS (E1..E5), the dedup order (tiny → parent → near-duplicate),
#     the validator, and the request-body shape in `llm.build_request_body`.
# `04_rag/profiles/retrieve/dense-top10-v2.yaml`
CANDIDATE_POOL = 50           # candidate_pool: the pool dedup runs over
TOP_K_CHILDREN = 10           # top_k_children: what survives dedup
MIN_CONTEXT_TOKENS = 21       # min_context_tokens: a shorter chunk is never context
NEAR_DUPLICATE_RULE = {"normalization": "norm-v3", "token_jaccard_min": 0.6, "keep": "higher_rank"}
# `04_rag/profiles/rerank/qwen3-4b-rerank-v1.yaml`
RERANK_TOP_K_INPUT = 10       # top_k_input
RERANK_TOP_K_OUTPUT = 5       # top_k_output — and MAX_CONTEXTS (E1..E5) downstream

LOG = logging.getLogger("ceres.pipeline")

# Public error codes: the only failure text an answer row carries to a client. Upstream detail
# (URLs, a server's error body, exception messages) stays in the raw run record and the log
# Validator messages are NOT upstream text — they are this module's own
# fixed vocabulary, the same strings the loop writes, so the DDS evaluation step can join on them — and they
# stay in `errors` as they are.
TRANSPORT_FAILED_CODE = "transport_failed"
INVALID_RESPONSE_CODE = "invalid_response"


class PipelineError(Exception):
    """A request cannot be served. `api.py` maps it to a status code."""


# ---------------------------------------------------------------- wiring

@dataclass
class Deps:
    """Everything one request needs, resolved once at startup.

    The three adapters are protocols: the service puts `PgStore` here, the real `Models`, and
    `OpenAIChatLLM`. Nothing below this line changes when they do.
    """

    config: Any                      # ceres.config.CeresConfig
    store: Store
    models: Models
    llm: LLM
    answer_profile: dict[str, Any]
    prompt: str
    retrieve_rule: dict[str, Any] = field(default_factory=lambda: dict(NEAR_DUPLICATE_RULE))
    candidate_pool: int = CANDIDATE_POOL
    top_k_children: int = TOP_K_CHILDREN
    min_context_tokens: int = MIN_CONTEXT_TOKENS
    rerank_top_k: int = RERANK_TOP_K_OUTPUT
    rerank_top_k_input: int = RERANK_TOP_K_INPUT
    rerank_identity: bool = False    # `none-v1` (operation: identity): never call the reranker
    rerank_text: str = "payload"     # `payload` | `represented` (rerank profile's key)
    max_contexts: int = MAX_CONTEXTS  # packet width E1..En (answer profile `max_contexts`; see packet_width)
    # sha256 of every artifact as READ at startup (`{"profiles": {slot: sha}, "prompt": sha}`); /v1/info
    # reports this snapshot, not whatever the files on disk say later.
    artifacts: dict[str, Any] = field(default_factory=dict)
    memory: Any = None               # ceres.memory.MemoryRuntime when `memory.enabled`; None = stateless only
    sources: Any = None              # ceres.sources.SourceCatalog when `answer.source_titles`; None = no meta.sources

    @classmethod
    def build(cls, config: Any, store: Store, models: Models, llm: LLM) -> "Deps":
        """Load the answer profile and the prompt, and check the prompt against the profile hash.

        The answer profile carries `prompt_template_sha256`. A prompt that does not match it is a
        different route — the grounded-answer prompt v1 rewrite alone moved structural failures
        from 100 % to 0 (`grounded-qwen38-v1.yaml` comment) — so a mismatch stops startup instead
        of quietly serving another recipe.
        """
        return cls.from_files(config, store, models, llm, config.profile_paths(),
                              config.resolve(config.route.prompt_template))

    @classmethod
    def from_files(cls, config: Any, store: Store, models: Models, llm: LLM,
                   profiles: dict[str, Path], prompt_path: Path) -> "Deps":
        """`build` with the profile files and the prompt named explicitly.

        The service names them in its config (`build`); `ceres-eval` names them from a route file,
        so an eval run is the same recipe code over the route's own profiles.
        """
        answer_path = profiles.get("answer")
        if answer_path is None or not Path(answer_path).is_file():
            raise PipelineError(f"route.profiles.answer is missing or unreadable: {answer_path}")
        # Each file is read ONCE, as bytes; what is parsed and what is hashed are the same bytes.
        loaded = {slot: Path(path).read_bytes() for slot, path in profiles.items()
                  if Path(path).is_file()}
        answer_profile = yaml.safe_load(loaded["answer"].decode("utf-8"))

        prompt_path = Path(prompt_path)
        if not prompt_path.is_file():
            raise PipelineError(f"route.prompt_template is missing: {prompt_path}")
        prompt_bytes = Path(prompt_path).read_bytes()
        prompt = prompt_bytes.decode("utf-8")
        expected = answer_profile.get("prompt_template_sha256")
        actual = hashlib.sha256(prompt_bytes).hexdigest()
        if expected and expected != actual:
            raise PipelineError(
                f"{prompt_path.name} does not match the answer profile's prompt_template_sha256"
                f" ({actual[:12]}… against {expected[:12]}…) — that is a different route"
            )

        deps = cls(config=config, store=store, models=models, llm=llm,
                   answer_profile=answer_profile, prompt=prompt,
                   artifacts={"profiles": {slot: hashlib.sha256(blob).hexdigest()
                                           for slot, blob in loaded.items()},
                              "prompt": actual})
        census_root = getattr(config.paths, "census_root", None) or config.paths.loop_root
        if getattr(config.answer, "source_titles", False) and census_root:
            from .sources import SourceCatalog
            deps.sources = SourceCatalog(config.resolve(census_root))
        memory_section = getattr(config, "memory", None)
        if memory_section is not None and memory_section.enabled:
            from .memory import MemoryRuntime
            deps.memory = MemoryRuntime.from_config(memory_section, config.resolve)
            deps.artifacts["memory_prompts"] = dict(deps.memory.prompt_sha256)
        apply_hybrid(deps, loaded)   # hybrid widths + rerank tie-break; all others: route_widths
        if "retrieve" in loaded and "rerank" in loaded:
            widths = route_widths(loaded)
            deps.retrieve_rule = widths["near_duplicate_rule"]
            deps.candidate_pool = widths["candidate_pool"]
            deps.top_k_children = widths["top_k_children"]
            deps.min_context_tokens = widths["min_context_tokens"]
            deps.rerank_identity = widths["rerank_identity"]
            deps.rerank_top_k_input = widths["rerank_top_k_input"]
            deps.rerank_top_k = widths["rerank_top_k_output"]
            deps.rerank_text = profile_contract(loaded)   # token floor + rerank_text choice
        deps.max_contexts = packet_width(answer_profile, prompt)
        if deps.rerank_identity:
            deps.rerank_top_k = deps.max_contexts        # s0: the dense order fills the whole packet
        if "rerank" in loaded and deps.rerank_top_k > deps.max_contexts:
            raise PipelineError(
                f"the rerank profile returns {deps.rerank_top_k} passages but the answer packet holds {deps.max_contexts}"
                f" (answer profile max_contexts): passages {deps.max_contexts + 1}..{deps.rerank_top_k} would be dropped"
                " silently while the route id says otherwise — set max_contexts in the answer profile or lower top_k_output")
        return deps


_PROMPT_RANGE = re.compile(r"\bE1\s*(?:\.\.|…|–|-)\s*E(\d+)\b")


def packet_width(answer_profile: Mapping[str, Any] | None, prompt: str = "") -> int:
    """Passages per answer packet: the answer profile's `max_contexts`, default MAX_CONTEXTS (5).

    The width belongs to the answer profile because the prompt has to name the passages (prompts v1 and v2 say "E1..E5"; v3
    says "E1, E2, …"). The width used to be the constant 5, and a route with rerank top_k_output 8 was served
    with 5 passages under its own route id. Refuses a width outside 1..MAX_CONTEXTS_LIMIT,
    and a prompt that names a fixed range "E1..En" with n different from the width.
    """
    raw = (answer_profile or {}).get("max_contexts", MAX_CONTEXTS)
    if isinstance(raw, bool) or not isinstance(raw, int) or not 1 <= raw <= MAX_CONTEXTS_LIMIT:
        raise PipelineError(f"answer profile max_contexts must be an integer 1..{MAX_CONTEXTS_LIMIT}, got {raw!r}")
    named = {int(n) for n in _PROMPT_RANGE.findall(prompt or "")}
    if named and named != {raw}:
        raise PipelineError(f"the answer prompt names passages E1..E{max(named)} but the packet holds {raw}"
                            " (answer profile max_contexts) — use a prompt that does not fix the range")
    return raw


# ---------------------------------------------------------------- 1. retrieve

def retrieve(store: Store, qvec: Sequence[float], release: Release, *,
             serve_rights: Sequence[str], limit: int = CANDIDATE_POOL,
             admitted_licensed_works: Sequence[str] = (),
             refused_evidence_basis: str | None = None) -> list[Item]:
    """The candidate pool, already filtered by the store's read predicate.

    The rights filter lives in the SQL (store contract, read side), not here: a
    chunk that is not servable must never reach the prompt, and filtering after retrieval would
    let an internal-only passage be paraphrased into `user_answer`. The store leaves exactly three
    conditions there: the pinned release, `evidence_eligible`, and these rights. The
    rights condition two-armed (mode rights, or a licensed work admitted by id) and derives both
    arms from `store.serving_mode` — neither is written by hand in a config file.

    A fourth, mode-derived arm: production refuses `ocr_admitted_unverified` items
    (`refused_evidence_basis`); rnd serves them, and they are marked downstream.
    """
    return store.search(qvec, release.release_attempt_id, list(serve_rights), limit,
                        list(admitted_licensed_works),
                        refused_evidence_basis=refused_evidence_basis)


# ---------------------------------------------------------------- 2. dedup

def dedup(hits: Sequence[Item], texts: dict[str, str], rule: dict[str, Any] | None = None, *,
          top_k: int = TOP_K_CHILDREN, min_context_tokens: int = MIN_CONTEXT_TOKENS,
          dropped: list[dict[str, Any]] | None = None) -> list[Item]:
    """`dedup_hits` (`slot_adapter.py:1104`) over store items instead of hit dicts.

    Three rules, in this order, all from `dense-top10-v2.yaml`:
      * `min_context_tokens: 21` — a chunk with fewer whitespace tokens is dropped as "tiny";
      * `parent_deduplication: keep-best` — the first (highest-ranked) child of a parent wins and
        every later sibling is dropped, so one document cannot fill the packet;
      * `near_duplicate_rule` — norm-v3 token Jaccard ≥ 0.6 against an already-kept chunk drops
        the newcomer (`keep: higher_rank`), because near-duplicate passages waste E-slots.
    The loop's `AdapterError` for an unknown chunk becomes a `PipelineError`; the rest is the same
    code path, including the `dropped` record that names why each candidate went.
    """
    threshold = float((rule or {}).get("token_jaccard_min", 1.1))
    normalization = str((rule or {}).get("normalization", "norm-v3"))
    seen_parents: set[str] = set()
    output: list[Item] = []
    for hit in hits:
        if hit.chunk_id not in texts:
            raise PipelineError(f"retrieval candidate references an unknown child chunk: {hit.chunk_id}")
        text = texts[hit.chunk_id]
        if min_context_tokens and len(text.split()) < min_context_tokens:
            if dropped is not None:
                dropped.append({"dropped_chunk_id": hit.chunk_id, "kept_chunk_id": None,
                                "jaccard": 0.0, "reason": "tiny"})
            continue
        if hit.parent_id in seen_parents:
            continue
        duplicate_of, duplicate_jaccard = None, 0.0
        if rule is not None:
            for kept in output:
                similarity = token_jaccard(text, texts[kept.chunk_id], normalization)
                if similarity >= threshold:
                    duplicate_of, duplicate_jaccard = kept.chunk_id, similarity
                    break
        if duplicate_of is not None:
            if dropped is not None:
                dropped.append({"dropped_chunk_id": hit.chunk_id, "kept_chunk_id": duplicate_of,
                                "jaccard": round(duplicate_jaccard, 10), "reason": "near_duplicate"})
            continue
        seen_parents.add(hit.parent_id)
        hit.rank = len(output) + 1
        output.append(hit)
        if len(output) == top_k:
            break
    return output


def route_widths(loaded: dict[str, bytes]) -> dict[str, Any]:
    """The retrieve and rerank widths of a route, read from its two profiles.

    Missing retrieve keys take the LOOP's defaults (`slot_adapter.build_retrieval`), not r1's
    values: pool = `top_k_children`, no near-duplicate rule, `min_context_tokens` 0. Filling them
    with r1's numbers would serve a recipe the judged route never ran.

    Rerank: `operation: identity` (`none-v1`, route s0) means no reranker — the dense order is kept
    and the first `MAX_CONTEXTS` become E1..E5, as `build_packets` does. Any other rerank profile
    must state `top_k_input` and `top_k_output`; a missing width is refused, because the loop's
    rerank slot once ignored its profile and silently scored 10.
    """
    retrieve = yaml.safe_load(loaded["retrieve"].decode("utf-8")) or {}
    rerank_profile = yaml.safe_load(loaded["rerank"].decode("utf-8")) or {}
    if "top_k_children" not in retrieve:
        raise PipelineError("the retrieve profile does not state top_k_children")
    top_k_children = int(retrieve["top_k_children"])
    widths: dict[str, Any] = {
        "candidate_pool": int(retrieve.get("candidate_pool", top_k_children)),
        "top_k_children": top_k_children,
        "min_context_tokens": int(retrieve.get("min_context_tokens", 0)),
        "near_duplicate_rule": retrieve.get("near_duplicate_rule"),
    }
    if rerank_profile.get("operation") == "identity":
        widths.update(rerank_identity=True, rerank_top_k_input=top_k_children,
                      rerank_top_k_output=MAX_CONTEXTS)
        return widths
    for key in ("top_k_input", "top_k_output"):
        if key not in rerank_profile:
            raise PipelineError(f"the rerank profile {rerank_profile.get('profile_id')!r} does not state {key}")
    widths.update(rerank_identity=False, rerank_top_k_input=int(rerank_profile["top_k_input"]),
                  rerank_top_k_output=int(rerank_profile["top_k_output"]))
    return widths


# ---------------------------------------------------------------- 3. rerank

def rerank(models: Models, question: str, hits: Sequence[Item],
           top_k: int = RERANK_TOP_K_OUTPUT, top_k_input: int = RERANK_TOP_K_INPUT,
           identity: bool = False, rerank_text: str = "payload", tie_break: str = "dense_rank") -> list[Item]:
    """Score the first `top_k_input` hits and keep the best `top_k` (both from the rerank profile).
    `rerank_text`: the model reads `payload`, or `represented_text` (see `reranker_inputs`).
    Ties break on the model's own order first and the dense rank second, which is the loop's
    `stable_order` intent: two identical scores must not reorder between runs.
    """
    candidates = list(hits)[:top_k_input]
    if not candidates:
        return []
    if identity:
        # s0: no reranker. Keep the dense order; the packet takes the first E1..E5.
        for rank, hit in enumerate(candidates[:top_k], start=1):
            hit.rank = rank
        return candidates[:top_k]
    scores = models.rerank(question, reranker_inputs(candidates, rerank_text))
    if len(scores) != len(candidates):
        raise PipelineError(f"the reranker returned {len(scores)} scores for {len(candidates)} texts")
    for hit, score in zip(candidates, scores):
        hit.rerank_score = float(score)
    ordered = sorted(candidates, key=rerank_order_key(candidates, tie_break))   # tie-break: see rerank_order_key
    for rank, hit in enumerate(ordered[:top_k], start=1):
        hit.rank = rank
    return ordered[:top_k]


# ---------------------------------------------------------------- 4. packet

def build_packet(question: str, lang: str, question_id: str, hits: Sequence[Item],
                 max_contexts: int = MAX_CONTEXTS) -> Packet:
    """The DDS answer packet: E1..En over the hits, in order (`slot_adapter.py:3281`); n = `max_contexts` (default 5).

    `warnings` is the chunk's `excluded_reasons` as the release recorded it. No reason
    decides retrievability — the read predicate is release + `evidence_eligible` + rights — so
    every reason travels with the evidence and a judge sees what the release flagged. The store's
    reasons include those inherited from the parent chunk, so warnings can be a superset of the
    loop's packet (`tests/test_parity.py: INHERITED_WARNINGS` lists the difference exactly).
    """
    contexts: list[Context] = []
    for hit in hits:
        if len(contexts) >= max_contexts:
            break
        hit.evidence_id = f"E{len(contexts) + 1}"
        contexts.append(Context(
            evidence_id=hit.evidence_id, chunk_id=hit.chunk_id, parent_id=hit.parent_id,
            evidence_text=hit.payload, evidence_text_sha256=hit.payload_sha256,
            locator_ids=list(hit.locator_ids), lineage_ids=list(hit.lineage_ids),
            warnings=list(hit.excluded_reasons), evidence_basis=hit.evidence_basis,
        ))
    return Packet(packet_id=f"grounded-{question_id}", question_id=question_id,
                  question=question, language=lang, contexts=contexts)


# ---------------------------------------------------------------- 5. validate

def structural_answer(candidate: Any, packet: dict[str, Any]) -> list[str]:
    """The loop's frozen validator (`slot_adapter.py:1393`), closed-book branch removed.

    Ceres does not change this function. When the serving side needs a stricter rule it goes into
    `validate()` below under its own revision id, so a judged number can always be attributed.
    """
    errors: list[str] = []
    required = {"decision", "answer_language", "user_answer", "citations", "missing_information",
                "confidence", "reason_code"}
    if not isinstance(candidate, dict) or set(candidate) != required:
        return ["candidate keys do not match the profile schema"]
    if (candidate.get("answer_language") != packet["language"]
            or not isinstance(candidate.get("user_answer"), str)
            or not candidate["user_answer"].strip()):
        errors.append("answer language or user_answer is invalid")
    if not isinstance(candidate.get("citations"), list) or len(candidate["citations"]) > 2:
        errors.append("citations must be an array of at most two")
    if candidate.get("decision") not in {"ANSWER", "PARTIAL", "CLARIFY", "UNSUPPORTED"}:
        errors.append("invalid decision")
    if candidate.get("confidence") not in {"HIGH", "MEDIUM", "LOW"}:
        errors.append("invalid confidence")
    if not isinstance(candidate.get("missing_information"), list) or len(candidate["missing_information"]) > 4:
        errors.append("missing_information must be an array of at most four")
    if isinstance(candidate.get("user_answer"), str) and len(candidate["user_answer"].split()) > 120:
        errors.append("user_answer exceeds 120 whitespace tokens")
    citations = candidate.get("citations") if isinstance(candidate.get("citations"), list) else []
    missing = candidate.get("missing_information") if isinstance(candidate.get("missing_information"), list) else []
    if candidate.get("decision") == "ANSWER" and (not citations or missing):
        errors.append("ANSWER requires the profile citation rule and empty missing_information")
    if candidate.get("decision") == "PARTIAL" and (not citations or not missing):
        errors.append("PARTIAL requires the profile citation rule and missing_information")
    if candidate.get("decision") in {"CLARIFY", "UNSUPPORTED"} and not missing:
        errors.append("CLARIFY/UNSUPPORTED requires missing_information")
    if candidate.get("decision") == "UNSUPPORTED" and citations:
        errors.append("UNSUPPORTED citations must be empty")
    reason_by_decision = {
        "ANSWER": {"SUPPORTED"}, "PARTIAL": {"PARTIAL_SUPPORT"},
        "CLARIFY": {"MISSING_USER_OR_PARCEL_INFORMATION"},
        "UNSUPPORTED": {"LIVE_DATA_REQUIRED", "CURRENT_AUTHORITY_REQUIRED",
                        "CONTRACT_OR_POLICY_REQUIRED", "PREDICTION_NOT_SUPPORTED",
                        "NO_RELEVANT_SUPPORT"},
    }
    if candidate.get("reason_code") not in reason_by_decision.get(candidate.get("decision"), set()):
        errors.append("decision/reason_code mismatch")
    contexts = {value["evidence_id"]: value for value in packet.get("contexts", [])}
    for citation in citations:
        if not isinstance(citation, dict) or set(citation) != {"claim", "evidence_id", "support_quote"}:
            errors.append("citation shape is invalid")
            continue
        claim, evidence_id, quote = (citation.get("claim"), citation.get("evidence_id"),
                                     citation.get("support_quote"))
        if not isinstance(claim, str) or claim not in candidate.get("user_answer", ""):
            errors.append("citation claim is not a continuous user_answer substring")
        if evidence_id not in contexts:
            errors.append("citation evidence_id is not in the packet")
        if not isinstance(quote, str) or len(quote.split()) > 25:
            errors.append("citation support_quote is invalid or exceeds 25 words")
        elif (evidence_id in contexts
              and " ".join(quote.split()).casefold()
              not in " ".join(contexts[evidence_id]["evidence_text"].split()).casefold()):
            errors.append("citation support_quote is not a continuous evidence substring")
    return errors


# The r1 error string the ellipsis rule may lift (`structural_answer`, above). One citation that
# passes the rule lifts exactly ONE occurrence of it, so a second, genuinely misquoted citation in
# the same candidate still fails.
QUOTE_SUBSTRING_ERROR = "citation support_quote is not a continuous evidence substring"


@dataclass(frozen=True)
class QuoteGuards:
    """ceres-validate-r3: when a quote shortened with "..." or "…" still counts as a quote.

    The one place these numbers live; `/v1/info` reports `as_dict()`, and the replay that cleared
    the rule used exactly these values. A change
    to any field is a new validator revision, not a tuning knob.

    A quote that is not one continuous substring of its passage passes only when ALL hold:
      * it splits on an ellipsis marker into >= `min_pieces` non-empty pieces;
      * every piece has >= `min_words_per_piece` whitespace tokens after `norm_version`;
      * the pieces occur IN ORDER in the SAME cited passage (both sides under `norm_version`;
        case is kept — norm-v3 never folds case). Each piece is taken at its first occurrence at
        or after the end of the previous piece, as the measured replay did;
      * every omitted gap is <= `max_gap_chars` characters of the normalized passage;
      * no gap holds a digit (`\\d`) or a negation word — the words an ellipsis could hide to flip
        or rescale a claim. Negation words match as whole words, case-insensitive; `n't` matches
        as a contraction suffix ("don't", "isn't");
      * r3: no gap matches `gap_capitalized_word` — a
        capitalized word that does not start a sentence (the pattern's lookbehind looks inside the
        gap string only, exactly as measured). This is a BLUNT PROXY for "the gap drops a scope
        qualifier (place, region, named entity)": all 3 judged-contradicted r2 rescues dropped
        a place scope ("... and around the Adriatic Sea ..."). It also refuses clean rescues whose gap
        merely holds a proper noun — about half of the judged-clean ones — by design, accepted as the price. Prompt v2 (no ellipsis at the source) is the durable fix, not a
        finer guard here.
    """

    norm_version: str = NORM_VERSION
    ellipsis_markers: tuple[str, ...] = ("...", "…")
    min_pieces: int = 2
    min_words_per_piece: int = 3
    max_gap_chars: int = 200
    gap_forbids_digit: bool = True
    negation_words: tuple[str, ...] = (
        "not", "no", "never", "none", "nor", "without", "cannot",                       # EN
        "nu", "fără", "fara", "niciodată", "niciodata", "nici", "niciun", "nicio",      # RO
    )
    negation_suffixes: tuple[str, ...] = ("n't",)                                       # EN contraction
    # r3: the exact pattern measured (`write_r3_sets.py` CAPITAL); case-sensitive, searched in each gap.
    gap_capitalized_word: str = r"(?<![.!?]\s)\b[A-ZĂÂÎȘŞȚŢ][a-zăâîșşțţ]+"

    @property
    def ellipsis_pattern(self) -> re.Pattern[str]:
        return re.compile("|".join(re.escape(marker) for marker in self.ellipsis_markers))

    @property
    def negation_pattern(self) -> re.Pattern[str]:
        words = r"\b(?:" + "|".join(re.escape(word) for word in self.negation_words) + r")\b"
        suffixes = "|".join(re.escape(suffix) for suffix in self.negation_suffixes)
        return re.compile(words + ("|" + suffixes if suffixes else ""), re.IGNORECASE)

    @property
    def capitalized_word_pattern(self) -> re.Pattern[str]:
        return re.compile(self.gap_capitalized_word)

    def as_dict(self) -> dict[str, Any]:
        return {"revision": VALIDATOR_REVISION, "decision": "quote-ellipsis guard, amended",
                "norm_version": self.norm_version,
                "ellipsis_markers": list(self.ellipsis_markers),
                "min_pieces": self.min_pieces, "min_words_per_piece": self.min_words_per_piece,
                "one_passage": True, "in_order": True, "max_gap_chars": self.max_gap_chars,
                "gap_forbids_digit": self.gap_forbids_digit,
                "gap_forbids_negation": self.negation_pattern.pattern,
                "gap_forbids_capitalized_word": self.capitalized_word_pattern.pattern,
                "served_quote_rewritten": False}


QUOTE_GUARDS = QuoteGuards()
_DIGIT = re.compile(r"\d")


def ellipsis_piece_spans(quote: str, passage: str,
                         guards: QuoteGuards = QUOTE_GUARDS) -> list[list[int]] | None:
    """The `[start, end)` span of every piece in `normalize(passage)` when `quote` passes the r3
    ellipsis rule (see `QuoteGuards`), else None. Pure; the quote itself is never changed."""
    pieces = [normalize(piece, guards.norm_version) for piece in guards.ellipsis_pattern.split(quote)]
    pieces = [piece for piece in pieces if piece]
    if len(pieces) < guards.min_pieces:
        return None
    if any(len(piece.split()) < guards.min_words_per_piece for piece in pieces):
        return None
    text = normalize(passage, guards.norm_version)
    negation = guards.negation_pattern
    capitalized = guards.capitalized_word_pattern
    spans: list[list[int]] = []
    position = 0
    for index, piece in enumerate(pieces):
        found = text.find(piece, position)
        if found < 0:
            return None                         # missing, or only before the previous piece
        if index:
            gap = text[position:found]
            if len(gap) > guards.max_gap_chars:
                return None
            if guards.gap_forbids_digit and _DIGIT.search(gap):
                return None
            if negation.search(gap):
                return None
            if capitalized.search(gap):
                return None                     # r3: a proper noun (scope qualifier) was omitted
        spans.append([found, found + len(piece)])
        position = found + len(piece)
    return spans


def _r1_quote_miss(citation: Any, contexts: dict[str, dict[str, Any]]) -> bool:
    """True exactly when `structural_answer` emitted QUOTE_SUBSTRING_ERROR for this citation."""
    if not isinstance(citation, dict) or set(citation) != {"claim", "evidence_id", "support_quote"}:
        return False
    quote, evidence_id = citation.get("support_quote"), citation.get("evidence_id")
    if not isinstance(quote, str) or len(quote.split()) > 25 or evidence_id not in contexts:
        return False
    return (" ".join(quote.split()).casefold()
            not in " ".join(contexts[evidence_id]["evidence_text"].split()).casefold())


def accepted_quote_spans(candidate: Any, packet: dict[str, Any]) -> list[dict[str, Any]]:
    """`meta.quote_spans`: one record per citation whose quote passed ONLY through the ellipsis rule.

    `{evidence_id, chunk_id, support_quote, norm_version, piece_spans}`; offsets are into
    `normalize(evidence_text)`. A quote that is one continuous substring gets no record.
    """
    if not isinstance(candidate, dict) or not isinstance(candidate.get("citations"), list):
        return []
    contexts = {value["evidence_id"]: value for value in packet.get("contexts", [])}
    records: list[dict[str, Any]] = []
    for citation in candidate["citations"]:
        if not _r1_quote_miss(citation, contexts):
            continue
        context = contexts[citation["evidence_id"]]
        spans = ellipsis_piece_spans(citation["support_quote"], context["evidence_text"])
        if spans is None:
            continue
        records.append({"evidence_id": citation["evidence_id"], "chunk_id": context["chunk_id"],
                        "support_quote": citation["support_quote"],
                        "norm_version": QUOTE_GUARDS.norm_version, "piece_spans": spans})
    return records


def validate(candidate: Any, packet: dict[str, Any]) -> list[str]:
    """Serving validation revision `ceres-validate-r3` = the loop validator plus two Ceres rules.

    r1 (kept): a citation's `claim` and `support_quote` must be non-empty. The loop validator
    accepts empty strings because the empty string is a substring of everything — this was reproduced
    against `structural_answer` (the answer-packet contract), and an "answer" citing nothing must
    not be served to a customer.

    r2: a `support_quote` that is not one continuous substring of its passage passes when it
    is a faithful ellipsis quote under every guard in `QUOTE_GUARDS` (`ellipsis_piece_spans`); r3
 adds the capitalized-word gap guard to those guards and changes nothing else. Each
    such citation lifts exactly one "not a continuous evidence substring" error; every other rule,
    including the 25-word limit, is the loop's. The revision id travels in `meta.validator_revision`,
    so this change is measurable against the frozen validator instead of being a silent change.
    """
    errors = structural_answer(candidate, packet)
    if errors and errors[0] == "candidate keys do not match the profile schema":
        return errors
    citations = candidate.get("citations") if isinstance(candidate.get("citations"), list) else []
    contexts = {value["evidence_id"]: value for value in packet.get("contexts", [])}
    for citation in citations:
        if (_r1_quote_miss(citation, contexts) and QUOTE_SUBSTRING_ERROR in errors
                and ellipsis_piece_spans(citation["support_quote"],
                                         contexts[citation["evidence_id"]]["evidence_text"]) is not None):
            errors.remove(QUOTE_SUBSTRING_ERROR)
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        if not str(citation.get("claim") or "").strip():
            errors.append("ceres-validate-r1: citation claim must not be empty")
        if not str(citation.get("support_quote") or "").strip():
            errors.append("ceres-validate-r1: citation support_quote must not be empty")
    # The documented API contract: `missing_information` holds strings. The loop validator only
    # checks that it is a list of at most four, so a number, null or object in it came back VALIDATED.
    missing = candidate.get("missing_information") if isinstance(candidate, dict) else None
    if isinstance(missing, list) and not all(isinstance(entry, str) for entry in missing):
        errors.append("ceres-validate-r1: missing_information entries must be strings")
    return errors


# ---------------------------------------------------------------- 6. answer

# Kept below the validator itself (`validate()` and its helpers), which is copied elsewhere and stays untouched.
ANSWERER_REJECTED_CODE = "answerer_rejected_request"   # HTTP 400/413/422 for this request


# ---------------------------------------------------------------- premise-corrected ANSWER 
# A profile may add reason codes for decision ANSWER (loop key `extra_answer_reason_codes`, slot_adapter.structural_answer).
# Kept below the validator: with no extra code the validator IS `validate()` (r3), byte-for-byte; with the profile's
# codes, an ANSWER carrying one is judged exactly as an ANSWER/SUPPORTED, so every ANSWER rule (≥ 1 citation, empty
# missing_information, exact quotes) still applies. A code under any other decision stays a decision/reason_code mismatch.
EXTRA_ANSWER_REASON_CODES = frozenset({"PREMISE_CORRECTED"})
PREMISE_CORRECTED_REVISION = "ceres-validate-r3-pc1"


def extra_answer_reasons(profile: Mapping[str, Any] | None) -> frozenset[str]:
    """The answer profile's extra ANSWER reason codes; a code Ceres does not know is refused (never silently accepted)."""
    codes = frozenset((profile or {}).get("extra_answer_reason_codes") or ())
    unknown = codes - EXTRA_ANSWER_REASON_CODES
    if unknown:
        raise PipelineError(f"answer profile: unknown extra_answer_reason_codes {sorted(unknown)}"
                            f" (Ceres knows {sorted(EXTRA_ANSWER_REASON_CODES)})")
    return codes


def validator_revision_for(profile: Mapping[str, Any] | None) -> str:
    """r3 for every profile without extra codes; the premise revision when the profile enables them."""
    return PREMISE_CORRECTED_REVISION if extra_answer_reasons(profile) else VALIDATOR_REVISION


def validate_profiled(candidate: Any, packet: dict[str, Any], extras: frozenset[str] = frozenset()) -> list[str]:
    """`validate()` (r3), plus the profile's extra ANSWER reason codes."""
    if (extras and isinstance(candidate, dict) and candidate.get("decision") == "ANSWER"
            and candidate.get("reason_code") in extras):
        return validate({**candidate, "reason_code": "SUPPORTED"}, packet)
    return validate(candidate, packet)


# ---------------------------------------------------------------- citation switch
# Citations can be enforced strictly or softly; a switch selects the mode.
# Kept below the validator: strict IS `validate()` (r3), unchanged; soft reuses r3's own per-citation checks.
CITATION_MODES = ("strict", "soft")
# The errors that concern one citation: in soft mode they drop that citation only (each is judged alone, so the r3
# ellipsis rule lifts exactly that citation's quote miss).
CITATION_LEVEL_ERRORS = frozenset({
    "citation shape is invalid",
    "citation claim is not a continuous user_answer substring",
    "citation evidence_id is not in the packet",
    "citation support_quote is invalid or exceeds 25 words",
    QUOTE_SUBSTRING_ERROR,
    "ceres-validate-r1: citation claim must not be empty",
    "ceres-validate-r1: citation support_quote must not be empty",
})
CITATION_COUNT_ERRORS = frozenset({"citations must be an array of at most two", "UNSUPPORTED citations must be empty"})
ANSWER_RULE = "ANSWER requires the profile citation rule and empty missing_information"
PARTIAL_RULE = "PARTIAL requires the profile citation rule and missing_information"


@dataclass
class CitationVerdict:
    """How one reply is judged under a citation mode. `errors` decide the attempt; `strict_errors` is always r3's list
    for the reply as the model gave it; `candidate` is what is served (soft: only the kept citations)."""

    mode: str
    errors: list[str]
    strict_errors: list[str]
    candidate: Any
    dropped: list[dict[str, Any]]

    def record(self) -> dict[str, Any]:
        return {"citation_mode": self.mode, "strict_errors": list(self.strict_errors),
                "dropped_citations": list(self.dropped)}


def citation_mode_of(config: Any) -> str:
    mode = str(getattr(getattr(config, "answer", None), "citation_mode", "strict") or "strict")
    if mode not in CITATION_MODES:
        raise PipelineError(f"unknown answer.citation_mode {mode!r}")
    return mode


def judge_citations(candidate: Any, packet: dict[str, Any], mode: str = "strict",
                    extras: frozenset[str] = frozenset()) -> CitationVerdict:
    """strict: r3 decides, nothing changes. soft:
      1. each citation failing a citation-level rule is dropped (with its reasons);
      2. more than two passing citations: the first two are kept, the rest dropped as `over_limit`;
      3. ANSWER/PARTIAL may go out with no citation; UNSUPPORTED citations are dropped (`unsupported_has_citations`);
      4. every other rule stays hard (keys, enums, decision/reason_code, missing_information, language, user_answer);
      5. a reply failing only citation rules is accepted as it is (no retry)."""
    strict_errors = validate_profiled(candidate, packet, extras)
    if mode == "strict" or not strict_errors:
        return CitationVerdict(mode, list(strict_errors), list(strict_errors), candidate, [])
    if strict_errors[0] == "candidate keys do not match the profile schema":
        return CitationVerdict(mode, list(strict_errors), list(strict_errors), candidate, [])
    raw = candidate.get("citations")
    dropped: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        dropped.append({"index": None, "evidence_id": None, "reasons": ["citations_not_an_array"]})
        raw = []
    passing: list[tuple[int, Any]] = []
    for index, citation in enumerate(raw):
        reasons = sorted({e for e in validate_profiled({**candidate, "citations": [citation]}, packet, extras)
                          if e in CITATION_LEVEL_ERRORS})
        if reasons:
            dropped.append({"index": index, "evidence_id": citation.get("evidence_id") if isinstance(citation, dict)
                            else None, "reasons": reasons})
        else:
            passing.append((index, citation))
    if candidate.get("decision") == "UNSUPPORTED":
        dropped.extend({"index": i, "evidence_id": c.get("evidence_id"), "reasons": ["unsupported_has_citations"]}
                       for i, c in passing)
        passing = []
    if len(passing) > 2:
        dropped.extend({"index": i, "evidence_id": c.get("evidence_id"), "reasons": ["over_limit"]}
                       for i, c in passing[2:])
        passing = passing[:2]
    served = {**candidate, "citations": [c for _i, c in passing]}
    hard = [e for e in validate_profiled(served, packet, extras)
            if e not in CITATION_LEVEL_ERRORS and e not in CITATION_COUNT_ERRORS and e not in (ANSWER_RULE, PARTIAL_RULE)]
    missing = served.get("missing_information")
    if served.get("decision") == "ANSWER" and isinstance(missing, list) and missing:
        hard.append(ANSWER_RULE)                       # the missing_information half of the rule stays hard
    if served.get("decision") == "PARTIAL" and isinstance(missing, list) and not missing:
        hard.append(PARTIAL_RULE)
    dropped.sort(key=lambda d: (d["index"] is None, d["index"] if d["index"] is not None else 0))
    return CitationVerdict(mode, hard, list(strict_errors), served if not hard else candidate, dropped)


NO_EVIDENCE_ANSWER = {
    "en": "None of the available documents contains a passage that answers this question.",
    "ro": "Niciun document disponibil nu conține un pasaj care să răspundă la această întrebare.",
}


def no_evidence_candidate(packet: dict[str, Any]) -> dict[str, Any]:
    """The deterministic refusal for an empty packet (`answer.no_evidence_short_circuit`)."""
    language = packet["language"]
    text = NO_EVIDENCE_ANSWER.get(language, NO_EVIDENCE_ANSWER["en"])
    return {"decision": "UNSUPPORTED", "answer_language": language, "user_answer": text, "citations": [],
            "missing_information": [text], "confidence": "HIGH", "reason_code": "NO_RELEVANT_SUPPORT"}


def answer_enabled_short_circuit(config: Any) -> bool:
    """`answer.no_evidence_short_circuit` of a config; False for a config (or test stub) without it."""
    return bool(getattr(getattr(config, "answer", None), "no_evidence_short_circuit", False))


def answer(llm: LLM, profile: dict[str, Any], prompt: str, packet: dict[str, Any], *,
           raw_dir: Path | None = None, events: list[dict[str, Any]] | None = None,
           context_lines: Mapping[str, str] | None = None,
           no_evidence_short_circuit: bool = False, citation_mode: str = "strict",
           memory: Any = None, memory_rules: str | None = None) -> AnswerRow:
    """The two-attempt answer loop (the loop's `slot_adapter` answer call and the DDS CLI's port of it).

    Attempts 1..`maximum_attempts` with `stable_seed(question_id, attempt)`, so a retry is a
    different sample and not a different recipe. A transport failure terminates immediately —
    retrying a server that is down only makes the client wait. A 2xx reply that is not a JSON
    object is a structural failure and spends an attempt, as in the loop (`slot_adapter.py:1553`).
    Every attempt's raw event is recorded before the candidate is judged, so a structural failure
    always leaves the exact bytes the model returned.

    The returned row carries public codes only; the raw events carry the full detail.

    `context_lines` (evidence_id → SOURCE METADATA line, `answer.context_lines` on) changes the
    user message only; the validator still reads the packet, i.e. `evidence_text` alone. None is r1.
    """
    decoding = profile.get("decoding") or {}
    attempts = int(decoding.get("maximum_attempts", 2))
    question_id = packet["question_id"]
    feedback_mode = retry_feedback_mode(profile)
    extras = extra_answer_reasons(profile)                    # empty for every profile before v3 → r3 unchanged
    feedback_on = feedback_mode != "off"
    previous: tuple[Any, ...] | None = None      # (assistant content, errors, candidate) of the last judged attempt
    if no_evidence_short_circuit and not packet.get("contexts"):
        # Nothing retrieved → no answerer call; attempts 0 records that no model was asked.
        candidate = no_evidence_candidate(packet)
        errors = validate(candidate, packet)
        record = {"citation_mode": citation_mode, "strict_errors": list(errors), "dropped_citations": []}
        if errors:                                    # a bug in the refusal itself: never serve it unvalidated
            return AnswerRow(question_id=question_id, state="STRUCTURE_FAILED", attempts=0, errors=errors,
                             citation=record)
        return AnswerRow(question_id=question_id, state="VALIDATED", attempts=0, candidate=candidate, citation=record)
    errors: list[str] = ["no attempt was made"]
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
    if citation_mode not in CITATION_MODES:
        raise PipelineError(f"unknown citation_mode {citation_mode!r}")
    last_record: dict[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        attempt_verdict: CitationVerdict | None = None
        body = build_request_body(packet, profile, prompt, attempt, context_lines, memory=memory, memory_rules=memory_rules)
        if feedback_on and previous is not None:
            # The previous reply and what the validator rejected, in plain
            # words, appended after the unchanged attempt-1 messages. Attempt 1 itself is never touched.
            located = (located_quote_lines(previous[2], packet)
                       if feedback_mode == "validator_errors_located" else [])
            body["messages"] = [*body["messages"], {"role": "assistant", "content": previous[0]},
                                {"role": "user", "content": retry_feedback_message(previous[1], located)}]
        event: dict[str, Any] = {"question_id": question_id, "attempt": attempt, "request": body}
        try:
            response, raw = llm.chat(body)
        except LLMTransportError as exc:
            event["transport_error"] = str(exc)                 # private: the record keeps it all
            _record(event, events, raw_dir, question_id, attempt)
            return AnswerRow(question_id=question_id, state="TRANSPORT_FAILED", attempts=attempt,
                             errors=[TRANSPORT_FAILED_CODE])
        except LLMRequestRejected as exc:
            # The server refused THIS request (e.g. context overflow). Another
            # question may succeed, so spend the attempt like a structural failure instead of stopping.
            event["request_rejected"] = {"status": exc.status, "detail": str(exc)}   # private record
            event["structural_errors"] = [f"{ANSWERER_REJECTED_CODE}: HTTP {exc.status}"]
            errors = [ANSWERER_REJECTED_CODE]
            last_record = {"citation_mode": citation_mode, "strict_errors": list(errors), "dropped_citations": []}
            _record(event, events, raw_dir, question_id, attempt)
            continue
        except LLMResponseError as exc:
            # The server answered, badly. Keep its bytes (as text, lossless for valid UTF-8 and
            # replacement-marked otherwise) and spend the attempt, like any structural failure.
            event["response_sha256"] = hashlib.sha256(exc.raw).hexdigest()
            event["response_text"] = exc.raw.decode("utf-8", errors="replace")
            event["structural_errors"] = [f"{INVALID_RESPONSE_CODE}: {exc}"]
            errors = [INVALID_RESPONSE_CODE]
            previous = (event["response_text"], errors, None)
            last_record = {"citation_mode": citation_mode, "strict_errors": list(errors), "dropped_citations": []}
            _record(event, events, raw_dir, question_id, attempt)
            continue

        event["response_sha256"] = hashlib.sha256(raw).hexdigest()
        event["response"] = response
        content: Any = None          # the reply text, taken once; stays None when the envelope is malformed
        try:
            content = response["choices"][0]["message"]["content"]
            candidate = json.loads(content)
            attempt_verdict = judge_citations(candidate, packet, citation_mode, extras)
            candidate, errors = attempt_verdict.candidate, attempt_verdict.errors
            event["structural_errors"] = attempt_verdict.strict_errors  # always r3's list (strict: unchanged)
            if citation_mode != "strict":
                event["citation"] = attempt_verdict.record()
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            # The envelope or the content is not what a chat completion looks like. The exception
            # text may quote the model's output, so it goes to the record, not to the row.
            candidate, errors = None, [INVALID_RESPONSE_CODE]
            event["structural_errors"] = [f"{INVALID_RESPONSE_CODE}: {exc.__class__.__name__}: {exc}"]
        previous = (content if isinstance(content, str) else json.dumps(response, ensure_ascii=False), errors, candidate)
        _record(event, events, raw_dir, question_id, attempt)
        # This attempt's citation record: the verdict when the reply parsed, else the plain error list.
        last_record = (attempt_verdict.record() if attempt_verdict is not None
                       else {"citation_mode": citation_mode, "strict_errors": list(errors), "dropped_citations": []})
        if not errors:
            return AnswerRow(question_id=question_id, state="VALIDATED", attempts=attempt,
                             candidate=candidate, citation=last_record)
    return AnswerRow(question_id=question_id, state="STRUCTURE_FAILED", attempts=attempts,
                     errors=errors, citation=last_record)


# Answer-profile key `retry_feedback`, off by default.
RETRY_FEEDBACK_MODES = ("off", "validator_errors", "validator_errors_located")
# Each validator error (this module's fixed vocabulary) in plain words for the model. An error not listed
# here is passed on verbatim after "Fix:".
RETRY_FEEDBACK_RULES = {
    "citation claim is not a continuous user_answer substring":
        'Each citation "claim" must be copied exactly, character for character, from your "user_answer": one'
        ' continuous piece of it, with no "...", no omitted words and no rewording.',
    "citation support_quote is not a continuous evidence substring":
        'Each "support_quote" must be copied exactly from the passage named in its "evidence_id", as one'
        ' continuous piece: no "...", no omitted words, no rewording.',
    "citation support_quote is invalid or exceeds 25 words":
        'Each "support_quote" must be at most 25 words, copied exactly from the passage, with no "...".',
    "citations must be an array of at most two": 'Give at most two citations.',
    "decision/reason_code mismatch":
        '"reason_code" must match "decision": ANSWER → SUPPORTED; PARTIAL → PARTIAL_SUPPORT; CLARIFY →'
        ' MISSING_USER_OR_PARCEL_INFORMATION; UNSUPPORTED → one of LIVE_DATA_REQUIRED, CURRENT_AUTHORITY_REQUIRED,'
        ' CONTRACT_OR_POLICY_REQUIRED, PREDICTION_NOT_SUPPORTED, NO_RELEVANT_SUPPORT.',
    INVALID_RESPONSE_CODE: 'Return exactly one JSON object with the required keys, and nothing else.',
}


def retry_feedback_mode(profile: Mapping[str, Any]) -> str:
    """The answer profile's `retry_feedback` (absent = "off"); an unknown value is refused, never ignored."""
    mode = str(profile.get("retry_feedback") or "off")
    if mode not in RETRY_FEEDBACK_MODES:
        raise PipelineError(f"answer profile: unknown retry_feedback {mode!r} (known: {', '.join(RETRY_FEEDBACK_MODES)})")
    return mode


def located_quote_lines(candidate: Any, packet: Mapping[str, Any]) -> list[str]:
    """P3b: for each citation whose quote is not in its own passage but occurs verbatim in ANOTHER passage of the packet,
    one line naming that passage. The comparison folds whitespace and case exactly as the validator's quote check does;
    it is a diagnosis for the retry message only and never changes what the validator accepts."""
    if not isinstance(candidate, dict) or not isinstance(candidate.get("citations"), list):
        return []
    fold = lambda text: " ".join(str(text).split()).casefold()          # noqa: E731 — the validator's comparison
    passages = {c["evidence_id"]: fold(c["evidence_text"]) for c in packet.get("contexts", [])}
    lines: list[str] = []
    for citation in candidate["citations"]:
        if not isinstance(citation, dict):
            continue
        quote, cited = citation.get("support_quote"), citation.get("evidence_id")
        if not isinstance(quote, str) or not quote.strip() or cited not in passages or fold(quote) in passages[cited]:
            continue
        found = [eid for eid, text in passages.items() if eid != cited and fold(quote) in text]
        if found:
            words = quote.split()
            shown = " ".join(words[:12]) + ("…" if len(words) > 12 else "")
            lines.append(f'The quote "{shown}" occurs in {found[0]}, not in {cited}: cite {found[0]}, or quote {cited}.')
    return lines


def retry_feedback_message(errors: Sequence[str], located: Sequence[str] = ()) -> str:
    """The one user message a feedback retry adds: the previous reply's validator errors in plain words (deduplicated),
    then any P3b located-quote lines."""
    lines: list[str] = []
    for error in [*errors]:
        line = RETRY_FEEDBACK_RULES.get(error, f"Fix: {error}")
        if line not in lines:
            lines.append(line)
    lines.extend(line for line in located if line not in lines)
    return ("Your previous reply failed the format check:\n" + "\n".join(f"- {line}" for line in lines)
            + "\nReturn the complete corrected JSON object only.")


def _record(event: dict[str, Any], events: list[dict[str, Any]] | None, raw_dir: Path | None,
            question_id: str, attempt: int) -> None:
    """Keep the raw event in memory, and on disk when a runtime root is configured."""
    if events is not None:
        events.append(event)
    if raw_dir is None:
        return
    path = raw_dir / f"{question_id[:16]}-attempt-{attempt:02d}.json"
    path.write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- 7. OCR marks and the citation log

_OCR_LOG_LOCK = threading.Lock()     # worker threads append to one file; one line per write


def cited_marked_contexts(row: AnswerRow, hits: Sequence[Item]) -> list[tuple[Item, dict[str, Any]]]:
    """The contexts the VALIDATED candidate cites whose evidence basis is non-null.

    One entry per cited CONTEXT, in citation order, with the first citation that names it: a
    context cited twice is one mark and one log line. A non-VALIDATED row cites nothing (no
    candidate reached anyone), so it yields nothing.
    """
    if row.state != "VALIDATED" or not isinstance(row.candidate, dict):
        return []
    by_evidence = {hit.evidence_id: hit for hit in hits if hit.evidence_id}
    seen: set[str] = set()
    marked: list[tuple[Item, dict[str, Any]]] = []
    for citation in row.candidate.get("citations") or []:
        evidence_id = citation.get("evidence_id") if isinstance(citation, dict) else None
        hit = by_evidence.get(evidence_id)
        if hit is None or evidence_id in seen or hit.evidence_basis is None:
            continue
        seen.add(evidence_id)
        marked.append((hit, citation))
    return marked


def log_ocr_citations(path: Path | None, marked: Sequence[tuple[Item, dict[str, Any]]], *,
                      execution_id: str, question_id: str, lang: str) -> None:
    """Append one JSON line per cited OCR-admitted context (OCR backfire measurement).

    Fields: {at, execution_id, question_id, lang, chunk_id, work_id, evidence_id, locator_ids,
    support_quote, page_start, page_end}. `page_start`/`page_end` are the store's 0-based,
    inclusive page span of the chunk's source section when the release has it, else null — never
    derived from locators here.

    Fail closed: when there is nowhere to write (no runtime root) or the write fails, the answer is
    NOT served (PipelineError → 503). OCR evidence is admitted on the condition that every citation
    of it is logged; an unlogged citation would silently escape the stop rule.
    """
    if not marked:
        return
    if path is None:
        raise PipelineError("an OCR-admitted context was cited but no runtime root is configured"
                            " for the OCR citation log — refusing to serve an unlogged citation")
    at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    lines = []
    for hit, citation in marked:
        lines.append(json.dumps({
            "at": at, "execution_id": execution_id, "question_id": question_id, "lang": lang,
            "chunk_id": hit.chunk_id, "work_id": hit.work_id, "evidence_id": hit.evidence_id,
            "locator_ids": list(hit.locator_ids), "support_quote": citation.get("support_quote"),
            "page_start": hit.page_start, "page_end": hit.page_end,
        }, ensure_ascii=False, sort_keys=True))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _OCR_LOG_LOCK, open(path, "a", encoding="utf-8") as handle:
            handle.write("".join(line + "\n" for line in lines))
    except OSError as exc:
        raise PipelineError(f"the OCR citation log is not writable ({exc.__class__.__name__})"
                            " — refusing to serve an unlogged citation") from exc


# ---------------------------------------------------------------- floor + rerank_text (v3.5)
# The v3.5 store carries `items.represented_text` (the represent-v2 text
# the route embedded). A route that declares `rerank_text: represented` scores THAT text; the packet, evidence_text,
# quotes and validation stay on `payload`. There is no fallback: a candidate without the text, or a release without
# the column, refuses the request (`RerankTextUnavailable`) — scoring the payload instead would serve another route
# under this route's id.
RERANK_TEXT_UNAVAILABLE_CODE = "represented_text_unavailable"


class RerankTextUnavailable(PipelineError):
    """The route reranks on `represented_text` and the release cannot give it. `str()` is private
    (which chunks, which release); `code` is the public text."""

    code = RERANK_TEXT_UNAVAILABLE_CODE


def profile_contract(loaded: Mapping[str, bytes]) -> str:
    """One token floor + rerank_text choice over a route's loaded profile bytes (`Deps.from_files`).

    The serving/eval twin of the loop's route freeze (`ceres.route_contract`); a violation is a
    `PipelineError` at startup, never a request-time surprise. Returns the rerank text choice.
    """
    from .route_contract import RouteContractError, check_token_floor, rerank_text_choice

    def parsed(slot: str) -> dict[str, Any]:
        blob = loaded.get(slot)
        return (yaml.safe_load(blob.decode("utf-8")) or {}) if blob is not None else {}

    try:
        check_token_floor(parsed("chunk"), parsed("retrieve"), parsed("represent"))
        return rerank_text_choice(parsed("rerank"))
    except RouteContractError as exc:
        raise PipelineError(str(exc)) from None


def reranker_inputs(candidates: Sequence[Item], rerank_text: str = "payload") -> list[str]:
    """The texts the reranker scores for one question's candidates, in candidate order.

    `payload` (every route before v3.5): the chunk payload, exactly as r1. `represented`: the store's
    `represented_text`; a candidate without one refuses the whole request, naming the chunks.
    """
    if rerank_text == "payload":
        return [hit.payload for hit in candidates]
    if rerank_text != "represented":
        raise PipelineError(f"unknown rerank_text {rerank_text!r}")
    missing = [hit.chunk_id for hit in candidates if hit.represented_text is None]
    if missing:
        raise RerankTextUnavailable(
            f"{RERANK_TEXT_UNAVAILABLE_CODE}: {len(missing)} of {len(candidates)} rerank candidates have no"
            f" represented_text (NULL): {missing[:3]} — refusing; the route never falls back to payload")
    return [str(hit.represented_text) for hit in candidates]


def require_rerank_text(deps: Deps, release: Release) -> None:
    """Refuse a `represented` route on a release whose items table has no `represented_text` column."""
    if getattr(deps, "rerank_text", "payload") != "represented":
        return
    if "represented_text" not in (release.optional_columns or ()):
        raise RerankTextUnavailable(
            f"{RERANK_TEXT_UNAVAILABLE_CODE}: release {release.release_attempt_id} has no"
            " items.represented_text column — a rerank_text: represented route cannot be served from it")


# ---------------------------------------------------------------- chunk context lines

def context_lines_enabled(config: Any) -> bool:
    """`answer.context_lines` of a config; False for a config (or test stub) without the section."""
    return bool(getattr(getattr(config, "answer", None), "context_lines", False))


def packet_context_lines(hits: Sequence[Item]) -> dict[str, str]:
    """evidence_id → `context_line(hit.context)` for the hits that became E1..E5 and have a line.

    A hit without a context (v3.4 store, NULL column) or with nothing found gets no entry, and its
    passage is rendered exactly as in r1. Ordered E1..E5 (dict order = packet order).
    """
    lines: dict[str, str] = {}
    for hit in hits:
        if not hit.evidence_id:
            continue
        line = context_line(hit.context)
        if line:
            lines[hit.evidence_id] = line
    return lines


# ---------------------------------------------------------------- the composition

def _model_reported(events: Sequence[dict[str, Any]]) -> str | None:
    """What the endpoint called itself on the last answered attempt. A label, not a weight proof."""
    for event in reversed(events):
        response = event.get("response")
        # Type-check the envelope: a reply that parsed to a list or a string has no model label.
        model = response.get("model") if isinstance(response, dict) else None
        if isinstance(model, str):
            return model
    return None


# ---------------------------------------------------------------- phase functions
# `ask` composes these four; `ceres-eval` calls them one phase at a time over a whole bank, so
# an eval run and a served answer share every rule below by construction, not by copy.

# ROUTE-STORE-GUARD. `dds load --replace` can re-point a release to another route under
# the SAME release_attempt_id (e.g. a release of 8 rows re-pointed to a v3.4 route). A config still
# naming the old chunk/represent/embed profiles would then serve those chunks under the configured
# route's identity. The release row's three profile shas decide; `index` and `route_id` never do
# (the index profile is a serving-side choice, and `route_id` is the loader's label).
ROUTE_STORE_MISMATCH_CODE = "route_store_mismatch"
ROUTE_STORE_SLOTS = ("chunk", "represent", "embed")


class RouteStoreMismatch(PipelineError):
    """The active release was built with other chunk/represent/embed profiles than the configured
    route. `str()` is PRIVATE (slot and both shas, for the log); `code` is the public text."""

    code = ROUTE_STORE_MISMATCH_CODE


def _route_store_verdict(deps: Deps, release: Release) -> tuple[str, str]:
    """`(status, private detail)`, status "ok" | "skipped" | "mismatch". Pure; no cache.

    "ok" needs all three slots on BOTH sides and equal. A slot either side does not name is not
    compared; with fewer than three compared and none different the verdict is "skipped" (not
    verified), and any compared slot that differs is a mismatch regardless of the others.
    """
    configured = (deps.artifacts or {}).get("profiles") or {}
    carried = release.profile_sha256 or {}
    compared = [slot for slot in ROUTE_STORE_SLOTS if configured.get(slot) and carried.get(slot)]
    wrong = [slot for slot in compared if configured[slot] != carried[slot]]
    if wrong:
        return "mismatch", "; ".join(
            f"{slot}: release {release.release_attempt_id} has {carried[slot]},"
            f" the configured route has {configured[slot]}" for slot in wrong)
    if len(compared) == len(ROUTE_STORE_SLOTS):
        return "ok", ""
    return "skipped", f"compared {compared or 'no slot'}; the release or the config lacks the rest"


def route_store_check(deps: Deps, release: Release) -> str:
    """"ok" or "skipped"; raises `RouteStoreMismatch` when the release is not the configured route's.

    Cached per release identity (id, manifest sha), plus the release's own shas: a `--replace`
    under the same id changes the manifest sha and so re-runs the check. The configured side is
    `deps.artifacts`, fixed at startup. The mismatch is logged once per identity, not per request.
    """
    cache = deps.__dict__.setdefault("_route_store_checks", {})
    key = (release.identity, tuple(sorted((release.profile_sha256 or {}).items())))
    verdict = cache.get(key)
    if verdict is None:
        verdict = _route_store_verdict(deps, release)
        cache[key] = verdict
        if verdict[0] == "mismatch":
            LOG.error(json.dumps({"event": "route_store.mismatch", "code": ROUTE_STORE_MISMATCH_CODE,
                                  "release_attempt_id": release.release_attempt_id,
                                  "detail": verdict[1]}, ensure_ascii=False))
    status, detail = verdict
    if status == "mismatch":
        raise RouteStoreMismatch(f"{ROUTE_STORE_MISMATCH_CODE}: {detail}")
    return status


def pinned_release(deps: Deps) -> Release:
    """The active release, with the rights-withheld count when the store can give it.

    Refuses (`RouteStoreMismatch`) a release whose chunk/represent/embed profile shas differ from
    the configured route's (`route_store_check`); a release or config without them is not checked.
    """
    store_config = deps.config.store
    release = deps.store.active_release()
    route_store_check(deps, release)
    counted = getattr(deps.store, "withheld", None)
    if counted is not None:
        release = counted(release, store_config.serve_rights, store_config.admitted_works)
    return release


def apply_hybrid(deps: Deps, loaded: Mapping[str, bytes]) -> None:
    """The retrieve profile's hybrid config (None = dense only) and the rerank tie-break.

    Kept below the validator (`validate()` and its helpers), like every hybrid addition; `Deps.from_files`
    calls it in one line. `tie_break: pool_position` (rerank profile) orders equal rerank scores by
    position in the input pool; the default `dense_rank` keeps every earlier route as it was. A union
    route must use `pool_position`: its lexical-only rows carry dense_rank 0.
    """
    from .hybrid import HybridError, hybrid_config      # imported here: nothing new above the validator

    deps.hybrid, deps.rerank_tie_break = None, "dense_rank"
    deps.rerank_on_oom, deps.rerank_model, deps.issue_boost = None, None, None
    extra_answer_reasons(getattr(deps, "answer_profile", None))   # an unknown extra reason code refuses startup
    if "retrieve" not in loaded or "rerank" not in loaded:
        return
    retrieve = yaml.safe_load(loaded["retrieve"].decode("utf-8")) or {}
    rerank_profile = yaml.safe_load(loaded["rerank"].decode("utf-8")) or {}
    # `on_oom: dense_order` serves the dense order when the reranker runs out of GPU
    # memory for one request (recorded as fallback_oom); absent = an OOM fails the request, as every earlier route.
    on_oom = rerank_profile.get("on_oom")
    if on_oom not in (None, "dense_order"):
        raise PipelineError(f"the rerank profile {rerank_profile.get('profile_id')!r} states an unknown on_oom {on_oom!r}")
    deps.rerank_on_oom = on_oom
    # MARS issue boost (rerank profile qwen3-0.6b-rerank-v3): logit + beta for a candidate whose MARS issue matches the question's.
    from .issue_boost import IssueBoostError, boost_config

    try:
        deps.issue_boost = boost_config(rerank_profile) if rerank_profile.get("operation") != "identity" else None
    except IssueBoostError as exc:
        raise PipelineError(str(exc)) from None
    if rerank_profile.get("operation") != "identity":
        deps.rerank_model = {"profile_id": rerank_profile.get("profile_id"), "model": rerank_profile.get("hf_id"),
                             "revision": rerank_profile.get("revision")}
    try:
        deps.hybrid = hybrid_config(retrieve)
    except HybridError as exc:
        raise PipelineError(str(exc)) from None
    tie_break = str(rerank_profile.get("tie_break") or "dense_rank")
    if tie_break not in ("dense_rank", "pool_position"):
        raise PipelineError(f"the rerank profile {rerank_profile.get('profile_id')!r} states an unknown tie_break {tie_break!r}")
    if (deps.hybrid or {}).get("method") == "union_candidates_then_dedup" and tie_break != "pool_position" \
            and rerank_profile.get("operation") != "identity":
        raise PipelineError("a union hybrid route needs a rerank profile with tie_break: pool_position")
    deps.rerank_tie_break = tie_break


def rerank_order_key(candidates: Sequence[Item], tie_break: str = "dense_rank") -> Any:
    """Sort key for reranked hits: higher score first; ties by dense rank (every route before hybrid) or
    by position in the input pool (`pool_position`). For a dense-only pool the two agree:
    the pool is in dense order."""
    if tie_break == "pool_position":
        position = {id(hit): index for index, hit in enumerate(candidates)}
        return lambda hit: (-float(hit.rerank_score or 0.0), position[id(hit)])
    return lambda hit: (-float(hit.rerank_score or 0.0), hit.dense_rank)


def retrieve_phase(deps: Deps, qvec: Sequence[float], release: Release, *,
                   dropped: list[dict[str, Any]] | None = None, query: str | None = None,
                   trace: dict[str, Any] | None = None, language: str | None = None,
                   query_en: str | None = None, qvec_en: Sequence[float] | None = None) -> tuple[list[Item], list[Item]]:
    """Retrieve the route's `candidate_pool` under the mode's read predicate, then dedup it to
    `top_k_children`. Returns `(candidates, kept)`; `dropped` collects why candidates went.

    A hybrid route adds the lexical leg over `query` and fuses (`ceres.hybrid`);
    `trace` (eval) receives its intermediate id lists. An xl route (addendum xl-v1) also takes the question's
    `language` and, for a Romanian question, its translation `query_en` + `qvec_en`. Dense-only routes ignore all."""
    if getattr(deps, "hybrid", None):
        from .hybrid import HybridError, retrieve_hybrid

        try:
            return retrieve_hybrid(deps, qvec, query or "", release, dropped=dropped, trace=trace,
                                   language=language, query_en=query_en, qvec_en=qvec_en)
        except HybridError as exc:
            raise PipelineError(str(exc)) from None
    store_config = deps.config.store
    candidates = retrieve(deps.store, qvec, release, serve_rights=store_config.serve_rights,
                          limit=deps.candidate_pool,
                          admitted_licensed_works=store_config.admitted_works,
                          refused_evidence_basis=store_config.refused_evidence_basis)
    texts = {hit.chunk_id: hit.payload for hit in candidates}
    kept = dedup(candidates, texts, deps.retrieve_rule, top_k=deps.top_k_children,
                 min_context_tokens=deps.min_context_tokens, dropped=dropped)
    return candidates, kept


def rerank_phase(deps: Deps, question: str, kept: Sequence[Item]) -> list[Item]:
    """The route's rerank widths (or identity) over one question's deduplicated hits."""
    return rerank_with_status(deps, question, kept)[0]


def is_cuda_oom(exc: BaseException) -> bool:
    """torch's CUDA out-of-memory error, found by type name or message (pipeline never imports torch)."""
    names = {cls.__name__ for cls in type(exc).__mro__}
    return "OutOfMemoryError" in names or "CUDA out of memory" in str(exc)


def _release_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                      # noqa: BLE001 — best effort after an OOM
        pass


def rerank_with_status(deps: Deps, question: str, kept: Sequence[Item]) -> tuple[list[Item], dict[str, Any]]:
    """`rerank_phase` plus what happened: status `identity` | `ok` | `fallback_oom`, the model and rerank_ms.

    With the profile's `on_oom: dense_order`, a CUDA out-of-memory error while scoring serves the first
    `top_k_output` candidates in dense order (they are all read-predicate eligible) instead of failing the request."""
    model = getattr(deps, "rerank_model", None)
    info: dict[str, Any] = ({"status": "identity"} if deps.rerank_identity
                            else {"status": "ok", **(model or {})})
    started = time.perf_counter()
    try:
        top = rerank(deps.models, question, kept, top_k=deps.rerank_top_k,
                     top_k_input=deps.rerank_top_k_input, identity=deps.rerank_identity,
                     rerank_text=getattr(deps, "rerank_text", "payload"),
                     tie_break=getattr(deps, "rerank_tie_break", "dense_rank"))
    except Exception as exc:                               # noqa: BLE001 — only an OOM with on_oom is absorbed
        if getattr(deps, "rerank_on_oom", None) != "dense_order" or not is_cuda_oom(exc):
            raise
        _release_cuda_cache()
        top = list(kept)[: deps.rerank_top_k]
        for rank, hit in enumerate(top, start=1):
            hit.rank, hit.rerank_score = rank, None
        info["status"] = "fallback_oom"
        LOG.warning(json.dumps({"event": "rerank.fallback_oom", "model": (model or {}).get("model"),
                                "candidates": len(kept), "served": len(top)}))
    boost = getattr(deps, "issue_boost", None)
    if boost and info["status"] == "ok":
        # The reranker stamped `rerank_score` on every scored candidate (dense order); re-rank them with the boost.
        from .issue_boost import boosted_order

        scored = list(kept)[: deps.rerank_top_k_input]
        order, target, count = boosted_order(question, [hit.parent_id for hit in scored],
                                             [float(hit.rerank_score or 0.0) for hit in scored], boost["beta"], deps.rerank_top_k)
        top = [scored[i] for i in order]
        for rank, hit in enumerate(top, start=1):
            hit.rank = rank
        info["issue_boost"] = {"target": target, "boosted_count": count}
    info["rerank_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return top, info


def ask(deps: Deps, request: AskRequest) -> AskResponse:
    """One question through the whole route, with per-stage latency.

    Raises `StoreError` when there is no usable release (the API answers 503) and never invents an
    answer: a structural failure returns the answer row without `candidate`, a transport failure
    returns `TRANSPORT_FAILED`. A healthy store with no admissible evidence is a normal answer
    with an empty context list, not a failure (the admission contract).
    """
    started = time.perf_counter()
    execution_id = uuid.uuid4().hex
    question_id = request.resolved_question_id()
    # serving_mode-derived predicate inputs, the pinned release and V35's rerank-text check live in `_ask`.
    # Memory v0: only when the request names a conversation or a persona. `query` is what retrieval and the reranker see.
    if request.uses_memory:
        from .memory import MemoryError as _MemoryError
        if deps.memory is None:
            raise _MemoryError("memory is not enabled on this service (memory.enabled: false)")
        with deps.memory.conversation_lock(request.conversation_id):   # one turn of a conversation at a time
            return _ask(deps, request, started, execution_id, question_id)
    return _ask(deps, request, started, execution_id, question_id)


def _ask(deps: Deps, request: AskRequest, started: float, execution_id: str, question_id: str) -> AskResponse:
    release = pinned_release(deps)
    require_rerank_text(deps, release)
    if not request.uses_memory:
        if getattr(deps.config.answer, "situational_split", False) and deps.memory is not None:
            # Experiment: split the asker's situation off; retrieve with the core question when there is one.
            from .memory import split_situation
            ctx = split_situation(deps.memory, deps.llm, deps.answer_profile, question=request.question, lang=request.lang)
            return _answer_turn(deps, request, started, execution_id, question_id, release,
                                ctx if ctx.situational["applied"] else None, situational=ctx.situational)
        return _answer_turn(deps, request, started, execution_id, question_id, release, None)
    from .memory import finish, prepare
    mem = prepare(deps.memory, deps.llm, deps.answer_profile, question=request.question, lang=request.lang,
                  conversation_id=request.conversation_id, profile_id=request.profile_id, request_id=request.request_id)
    try:
        if mem.replay is not None:
            return mem.replay                      # the same (conversation_id, request_id) already completed: replay it
        response = _answer_turn(deps, request, started, execution_id, question_id, release, mem)
        if mem.snapshot is not None:
            deps.memory.store.remember(mem.snapshot, request.request_id, response)
        return response
    finally:
        finish(deps.memory, mem)


def _answer_turn(deps: Deps, request: AskRequest, started: float, execution_id: str, question_id: str, release: Any,
                 mem: Any, situational: dict[str, Any] | None = None) -> AskResponse:
    store_config = deps.config.store
    serve_rights = store_config.serve_rights
    admitted = store_config.admitted_works
    refused_basis = store_config.refused_evidence_basis
    query = mem.retrieval_query if mem is not None else request.question

    mark = time.perf_counter()
    qvec = deps.models.embed_query(query, request.lang)
    embed_ms = (time.perf_counter() - mark) * 1000

    mark = time.perf_counter()
    _candidates, kept = retrieve_phase(deps, qvec, release, query=query)
    retrieve_ms = (time.perf_counter() - mark) * 1000

    mark = time.perf_counter()
    top, rerank_info = rerank_with_status(deps, query, kept)
    rerank_ms = (time.perf_counter() - mark) * 1000

    packet = build_packet(request.question, request.lang, question_id, top, deps.max_contexts).to_dict()
    validate_packet(packet)
    fixed_row = None
    if mem is not None and mem.rewrite_status == "transport_failed":
        # The answerer endpoint just failed the rewrite call: do not call it again.
        fixed_row = AnswerRow(question_id=question_id, state="TRANSPORT_FAILED", attempts=0, errors=[TRANSPORT_FAILED_CODE])
    elif mem is not None and mem.unresolved_followup:
        # A follow-up the rewrite could not resolve: ask the user to restate it, no answerer call.
        from .memory import clarify_candidate
        cand = clarify_candidate(request.lang)
        errs = validate(cand, packet)
        fixed_row = AnswerRow(question_id=question_id, state="VALIDATED" if not errs else "STRUCTURE_FAILED", attempts=0,
                              candidate=cand if not errs else None, errors=errs,
                              citation={"citation_mode": citation_mode_of(deps.config), "strict_errors": list(errs),
                                        "dropped_citations": []})

    raw_dir = None
    executions = deps.config.raw_executions_dir
    if executions is not None:
        raw_dir = Path(executions) / execution_id / "raw"
    events: list[dict[str, Any]] = []
    # Only with `answer.context_lines` on; off, `lines` is None and nothing below changes.
    lines = packet_context_lines(top) if context_lines_enabled(deps.config) else None
    mark = time.perf_counter()
    row = fixed_row if fixed_row is not None else \
        answer(deps.llm, deps.answer_profile, deps.prompt, packet, raw_dir=raw_dir, events=events,
                 no_evidence_short_circuit=answer_enabled_short_circuit(deps.config),
                 citation_mode=citation_mode_of(deps.config),
                 context_lines=lines,
                 memory=mem if (mem is not None and mem.active) else None,
                 memory_rules=deps.memory.rules_for(mem) if mem is not None else None)
    answer_ms = (time.perf_counter() - mark) * 1000
    if row.state == "TRANSPORT_FAILED" and events:
        # The one place the upstream detail is logged; the row itself says `transport_failed`.
        LOG.error(json.dumps({"event": "answer.transport_failed", "execution_id": execution_id,
                              "attempt": row.attempts,
                              "detail": events[-1].get("transport_error")}, ensure_ascii=False))

    answer_row = row.to_dict()
    validate_answer_row(answer_row)
    # Mark and log every cited OCR-admitted context BEFORE the answer can leave this function.
    marked = cited_marked_contexts(row, top)
    # r2: where the pieces of each accepted ellipsis quote sit. The quote is served as written.
    quote_spans = accepted_quote_spans(row.candidate, packet) if row.state == "VALIDATED" else []
    log_ocr_citations(deps.config.ocr_citation_log, marked, execution_id=execution_id,
                      question_id=question_id, lang=request.lang)
    memory_meta = None
    if mem is not None and request.uses_memory:
        from .memory import record_turn
        if mem.rewrite_event is not None:
            if raw_dir is not None:
                raw_dir.mkdir(parents=True, exist_ok=True)   # the answer loop may not have run (rewrite transport failure)
            _record({**mem.rewrite_event, "question_id": question_id, "attempt": 0, "kind": "memory_rewrite"},
                    None, raw_dir, question_id, 0)   # attempt 00 = the rewrite call
        turn_number = (record_turn(deps.memory, mem, question=request.question, lang=request.lang,
                                   profile_id=request.profile_id, candidate=row.candidate, execution_id=execution_id)
                       if row.state != "TRANSPORT_FAILED" else None)     # a turn that never got an answer is not history
        memory_meta = {"conversation_id": request.conversation_id, "turn": turn_number, "profile_id": request.profile_id,
                       "standalone_question": mem.standalone_question, "retrieval_query": mem.retrieval_query,
                       "rewrite_status": mem.rewrite_status, "is_followup": mem.is_followup,
                       "used_profile_fields": mem.used_profile_fields, "history_turns_used": len(mem.history),
                       "rewrite_ms": mem.rewrite_ms, "prompt_sha256": dict(deps.memory.prompt_sha256)}
    route_id = deps.config.route.served_route_id
    meta = AskMeta(
        execution_id=execution_id,
        request_id=request.request_id,
        release_attempt_id=release.release_attempt_id,
        release_manifest_sha256=release.manifest_sha256 or None,
        route_id=route_id,
        route_status="assigned" if route_id else "unassigned",
        judged_route_id=deps.config.route.judged_route_id,
        serving_mode=store_config.serving_mode,
        read_predicate_sha256=read_predicate_sha256(serve_rights, store_config.db_schema,
                                                    admitted_licensed_works=admitted,
                                                    refused_evidence_basis=refused_basis),
        model_reported=_model_reported(events),
        lang_source="client",
        membership_counts=release.membership_counts,
        filtered=Filtered(rights=release.withheld_rights),
        latency_ms=Latency(
            embed=round(embed_ms, 3), retrieve=round(retrieve_ms, 3), rerank=round(rerank_ms, 3),
            answer=round(answer_ms, 3), total=round((time.perf_counter() - started) * 1000, 3),
        ),
        citation_marks=[CitationMark(evidence_id=hit.evidence_id, chunk_id=hit.chunk_id,
                                     work_id=hit.work_id, evidence_basis=hit.evidence_basis)
                        for hit, _citation in marked],
        quote_spans=quote_spans,
        context_lines=(None if lines is None else
                       [ContextLineRecord(evidence_id=evidence_id, line=line)
                        for evidence_id, line in lines.items()]),
        rerank=rerank_info,
        issue_boost=rerank_info.get("issue_boost"),
        validator_revision=validator_revision_for(deps.answer_profile),
        citation_mode=(row.citation or {}).get("citation_mode", citation_mode_of(deps.config)),
        strict_errors=(row.citation or {}).get("strict_errors", []),
        dropped_citations=(row.citation or {}).get("dropped_citations", []),
        memory=memory_meta,
        situational=situational,
        sources=deps.sources.describe(release.release_attempt_id, top) if deps.sources is not None else None,
    )
    return AskResponse(packet=packet, answer=answer_row, meta=meta)


__all__ = [
    "Deps", "PipelineError", "StoreError", "retrieve", "dedup", "rerank", "build_packet",
    "structural_answer", "validate", "QuoteGuards", "QUOTE_GUARDS", "ellipsis_piece_spans",
    "accepted_quote_spans", "answer", "ask", "cited_marked_contexts",
    "log_ocr_citations", "pinned_release", "retrieve_phase", "rerank_phase", "route_widths",
    "ROUTE_STORE_MISMATCH_CODE", "ROUTE_STORE_SLOTS", "RouteStoreMismatch", "route_store_check",
    "context_lines_enabled", "packet_context_lines",
    "RERANK_TEXT_UNAVAILABLE_CODE", "RerankTextUnavailable", "profile_contract", "reranker_inputs",
    "require_rerank_text",
]
