"""Hybrid retrieval (a lexical BM25 leg fused with the dense leg): Ceres's glue around the vendored reference module `hybrid_v1.py`.

`hybrid_v1.py` is the loop's file byte-for-byte (tokenizer, BM25, RRF, union candidates); this module only
reads the store and wires the legs together, so the lexical ranking and the fusion are the loop's by
construction and parity is tested end to end. Two methods, chosen by the retrieve profile's `fusion.method`:

* `rrf` (s0-hyb): dense top-500 and lexical top-500, each filtered by the read predicate BEFORE truncation
  → `rrf(k)` capped at `out_depth` → the standard dedup over the fused order → `top_k_children`.
* `union_candidates_then_dedup` (s1-wide-hyb): dense leg → dedup → `survivors` (today's s1-wide pool);
  lexical leg → dedup → the first `extra_n` survivors; `union_candidates` concatenates them and the
  standard dedup runs again over the concatenation (≤ `max_pool`), so a lexical row stays only with a
  new parent and no near-duplicate already kept. The pool order is the reranker's tie-break.

The BM25 index covers ALL rows of the release (statistics independent of serving mode); it is built once
per (release identity, read predicate) and kept on `Deps`, so `ask()` shares it with the eval path.
"""
from __future__ import annotations

import copy
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Sequence

from . import hybrid_v1
from .store import Item, Release

LOG = logging.getLogger("ceres.hybrid")

MODULE_PATH = Path(__file__).with_name("hybrid_v1.py")
METHODS = ("rrf", "union_candidates_then_dedup")


class HybridError(Exception):
    """A hybrid route cannot run as its profile states (the pipeline maps it to PipelineError)."""


def module_sha256() -> str:
    return hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest()


def hybrid_config(retrieve: dict[str, Any]) -> dict[str, Any] | None:
    """The hybrid widths of a retrieve profile, or None for a dense-only profile (no `fusion` key).

    Every number the code uses is read from the profile, and every constant the vendored module fixes
    (BM25 k1/b, the tokenizer, the indexed text and corpus) must be what the profile states — a profile
    that names another module sha, another text or other BM25 constants is refused, never run."""
    fusion = retrieve.get("fusion")
    if fusion is None:
        return None
    legs = retrieve.get("legs") or {}
    dense, lexical = legs.get("dense") or {}, legs.get("lexical") or {}
    method = (fusion or {}).get("method")
    if method not in METHODS:
        raise HybridError(f"retrieve profile {retrieve.get('profile_id')!r}: unknown fusion method {method!r}")
    if retrieve.get("hybrid_module_sha256") != module_sha256():
        raise HybridError(f"retrieve profile {retrieve.get('profile_id')!r} names hybrid module"
                          f" {str(retrieve.get('hybrid_module_sha256'))[:12]}…; the vendored ceres/hybrid_v1.py is"
                          f" {module_sha256()[:12]}…")
    bm25 = lexical.get("bm25") or {}
    stated = {"text": lexical.get("text"), "corpus": lexical.get("corpus"), "tokenizer": lexical.get("tokenizer"),
              "k1": bm25.get("k1"), "b": bm25.get("b"), "read_predicate": retrieve.get("read_predicate")}
    expected = {"text": "represented_text", "corpus": "all_release_rows", "tokenizer": "hybrid_v1.tokens",
                "k1": hybrid_v1.K1, "b": hybrid_v1.B, "read_predicate": "filter_first"}
    wrong = {key: stated[key] for key in expected if stated[key] != expected[key]}
    if wrong:
        raise HybridError(f"retrieve profile {retrieve.get('profile_id')!r} states {wrong}; Ceres implements {expected}")
    config: dict[str, Any] = {"method": method, "module_sha256": module_sha256(),
                              "dense_depth": int(dense["depth"]), "lexical_depth": int(lexical["depth"])}
    if method == "rrf":
        config.update(k=int(fusion["k"]), out_depth=int(fusion["out_depth"]))
    else:
        config.update(dense_survivors=int(dense["survivors"]), extra_n=int(fusion["extra_n"]),
                      max_pool=int(fusion["max_pool"]))
        if int(retrieve["top_k_children"]) != config["max_pool"]:
            raise HybridError("union route: top_k_children must equal fusion.max_pool")
    if "translation" in retrieve:
        config["translation"] = translation_config(retrieve, method)
    return config


# Addendum xl-v1: both-language legs for Romanian questions. The legs each method fuses for an
# `applies_to_language` question, exactly as the profile must state them; English questions keep the base method.
XL_RO_LEGS = {"rrf": ["dense_original", "dense_translation", "lexical_original", "lexical_translation"],
              "union_candidates_then_dedup": {"dense": "original", "lexical": "translation"}}


def translation_config(retrieve: dict[str, Any], method: str) -> dict[str, Any]:
    """The xl profile's translation input. The source file is pinned by sha256 in the profile (read and checked by the
    eval); the reranker and the answerer always read the ORIGINAL question (`rerank_query: original`)."""
    name = retrieve.get("profile_id")
    translation = retrieve.get("translation") or {}
    if translation.get("applies_to_language") != "ro":
        raise HybridError(f"retrieve profile {name!r}: translation.applies_to_language must be ro")
    if not translation.get("source") or len(str(translation.get("source_sha256") or "")) != 64:
        raise HybridError(f"retrieve profile {name!r}: translation needs source and source_sha256")
    if retrieve.get("ro_legs") != XL_RO_LEGS[method]:
        raise HybridError(f"retrieve profile {name!r}: ro_legs {retrieve.get('ro_legs')!r}; Ceres implements"
                          f" {XL_RO_LEGS[method]!r} for {method}")
    if retrieve.get("rerank_query", "original") != "original":
        raise HybridError(f"retrieve profile {name!r}: rerank_query must be original")
    return {"applies_to_language": "ro", "source": str(translation["source"]),
            "source_sha256": str(translation["source_sha256"]), "ro_legs": XL_RO_LEGS[method]}


class LexicalLeg:
    """The BM25 index over every row of one release plus the read predicate's id set."""

    def __init__(self, store: Any, release: Release, predicate: tuple[Any, ...]) -> None:
        started = time.perf_counter()
        ids, texts = store.lexical_corpus(release.release_attempt_id)
        try:
            self.index = hybrid_v1.BM25Index(ids, texts)
        except ValueError as exc:
            raise HybridError(f"release {release.release_attempt_id}: {exc}") from None
        serve_rights, admitted, refused = predicate
        self.keep = store.predicate_ids(release.release_attempt_id, list(serve_rights), list(admitted), refused)
        self.stats_sha256 = self.index.stats_sha256
        self.build_s = round(time.perf_counter() - started, 3)
        LOG.info("lexical index: release %s, %d docs, %d pass the predicate, stats %s…, %.1f s",
                 release.release_attempt_id, self.index.n, len(self.keep), self.stats_sha256[:12], self.build_s)


def _predicate(deps: Any) -> tuple[Any, ...]:
    store_config = deps.config.store
    return (tuple(store_config.serve_rights), tuple(store_config.admitted_works), store_config.refused_evidence_basis)


def lexical_leg(deps: Any, release: Release) -> LexicalLeg:
    """Built once per (release identity, predicate) and kept on `deps` (a release switch rebuilds it)."""
    predicate = _predicate(deps)
    key = (release.identity, predicate)
    cache = deps.__dict__.setdefault("_lexical_legs", {})
    if key not in cache:
        cache.clear()
        cache[key] = LexicalLeg(deps.store, release, predicate)
    return cache[key]


def _clone(items: Sequence[Item]) -> list[Item]:
    """Per-leg copies: dedup stamps `rank` on the items it keeps, and one chunk can sit in two legs.
    `copy.copy` keeps the instance attributes (`represented_text`, `context`) that `replace` would drop."""
    return [copy.copy(item) for item in items]


def retrieve_hybrid(deps: Any, qvec: Sequence[float], query: str, release: Release,
                    dropped: list[dict[str, Any]] | None = None,
                    trace: dict[str, Any] | None = None, *, language: str | None = None,
                    query_en: str | None = None, qvec_en: Sequence[float] | None = None) -> tuple[list[Item], list[Item]]:
    """`(candidates, kept)` for a hybrid route; `trace` (eval) receives every intermediate id list.

    An xl route (addendum xl-v1) adds the translation's legs for a question in `translation.applies_to_language`:
    `query_en` (the translation) and `qvec_en` (its query vector) are then required; any other question runs the
    base method unchanged."""
    from .pipeline import dedup                                       # pipeline imports this module

    config = deps.hybrid
    if not isinstance(query, str) or not query.strip():
        raise HybridError("a hybrid route needs the question text for its lexical leg")
    xl = config.get("translation") and language == config["translation"]["applies_to_language"]
    if xl and (not isinstance(query_en, str) or not query_en.strip() or qvec_en is None):
        raise HybridError(f"xl route: a {language} question needs its translation and the translation's query vector")
    rights, admitted, refused = _predicate(deps)
    leg = lexical_leg(deps, release)

    dense = deps.store.search(qvec, release.release_attempt_id, list(rights), config["dense_depth"],
                              list(admitted), refused_evidence_basis=refused)
    dense_ids = [item.chunk_id for item in dense]
    lexical_ids = leg.index.ranked(query, depth=config["lexical_depth"], keep=leg.keep)
    dense_en_ids: list[str] = []
    lexical_en_ids: list[str] = []
    if xl:
        lexical_en_ids = leg.index.ranked(query_en, depth=config["lexical_depth"], keep=leg.keep)
        if config["method"] == "rrf":
            dense_en_ids = [item.chunk_id for item in deps.store.search(
                qvec_en, release.release_attempt_id, list(rights), config["dense_depth"], list(admitted),
                refused_evidence_basis=refused)]
    by_id = {item.chunk_id: item for item in dense}
    # Rows outside the original dense leg are read by id with the predicate, scored against the ORIGINAL query vector.
    wanted = list(dict.fromkeys(lexical_ids + dense_en_ids + lexical_en_ids))
    missing = [cid for cid in wanted if cid not in by_id]
    fetched = deps.store.fetch(missing, qvec, release.release_attempt_id, list(rights), list(admitted),
                               refused_evidence_basis=refused)
    lost = [cid for cid in missing if cid not in fetched]
    if lost:
        # The lexical ids already passed `keep` (the same predicate); a row the SQL predicate then
        # refuses means the store changed under the index — never serve a half-filtered list.
        raise HybridError(f"{len(lost)} lexical candidates passed the cached predicate but not the store's: {lost[:3]}")
    by_id.update(fetched)
    texts = {cid: item.payload for cid, item in by_id.items()}
    rule, min_tokens = deps.retrieve_rule, deps.min_context_tokens

    def run_dedup(hits: list[Item], top_k: int, leg_name: str) -> list[Item]:
        """The standard dedup; its `dropped` rows are tagged with the leg (hybrid routes only)."""
        local: list[dict[str, Any]] = []
        out = dedup(hits, texts, rule, top_k=top_k, min_context_tokens=min_tokens, dropped=local)
        if dropped is not None:
            dropped.extend({**row, "leg": leg_name} for row in local)
        return out

    if trace is not None and xl:
        trace.update(dense_en_ids=dense_en_ids, lexical_en_ids=lexical_en_ids, xl=True)
    if config["method"] == "rrf":
        lists = [dense_ids, dense_en_ids, lexical_ids, lexical_en_ids] if xl else [dense_ids, lexical_ids]
        fused_ids = hybrid_v1.rrf(lists, k=config["k"],
                                  depth=max(config["dense_depth"], config["lexical_depth"]),
                                  out_depth=config["out_depth"])
        fused = _clone([by_id[cid] for cid in fused_ids])
        kept = run_dedup(fused, deps.top_k_children, "fused")
        if trace is not None:
            trace.update(stats_sha256=leg.stats_sha256, dense_ids=dense_ids, lexical_ids=lexical_ids,
                         fused_ids=fused_ids)
        return fused, kept

    dense_kept = run_dedup(_clone(dense), config["dense_survivors"], "dense")
    extra_ids = lexical_en_ids if xl else lexical_ids          # xl union: the TRANSLATION's lexical leg (variant U2)
    lexical_kept = run_dedup(_clone([by_id[cid] for cid in extra_ids]), config["extra_n"], "lexical")
    union = _clone(hybrid_v1.union_candidates(dense_kept, lexical_kept, config["extra_n"]))
    pool = run_dedup(union, config["max_pool"], "union")
    if trace is not None:
        trace.update(stats_sha256=leg.stats_sha256, dense_ids=dense_ids, lexical_ids=lexical_ids,
                     dense_survivor_ids=[item.chunk_id for item in dense_kept],
                     lexical_survivor_ids=[item.chunk_id for item in lexical_kept],
                     union_candidate_ids=[item.chunk_id for item in union])
    candidates = list(dense) + [fetched[cid] for cid in missing]
    return candidates, pool
