"""Hybrid retrieval in Ceres (a lexical BM25 leg fused with the dense leg): the hybrid layer.

How the hybrid layer can fail, and the test that catches each:
  * the vendored module drifts from the loop's reference file → test_vendored_module_is_the_loop_file
  * a profile naming another module, text, corpus or BM25 constant runs anyway → test_a_profile_that_differs_from_the_vendored_module_is_refused
  * the lexical leg truncates BEFORE the read predicate (fewer than depth, or an ineligible chunk) → test_lexical_leg_is_filter_first_and_fills_its_depth,
    test_retrieve_hybrid_filters_the_lexical_leg_before_truncating_it
  * BM25 statistics depend on the serving mode → test_index_statistics_do_not_depend_on_the_serving_mode
  * a lexical-only row the store's predicate refuses is served → test_a_row_the_store_predicate_refuses_stops_the_request
  * s1-wide-hyb: lexical extras never reach the reranker, or a dense_rank-0 row wins a tie → test_union_extras_reach_the_reranker_and_ties_go_by_pool_position
  * cross-leg near-duplicates survive the union → test_union_drops_a_cross_leg_near_duplicate
  * s0-hyb: the fused order is not the module's RRF, or dedup runs on the wrong order → test_rrf_route_dedups_the_module_fused_order
  * dense-only routes change (config or cache keys) → test_dense_routes_get_no_hybrid_config_and_keep_the_dense_rank_tie_break
  * a hybrid route runs without the question text → test_a_hybrid_route_without_question_text_is_refused
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from ceres import hybrid, hybrid_v1, pipeline
from ceres.store import FixtureStore, Item, Release

RELEASE = "rel-h"
WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi"


def row(cid, parent, text, *, eligible=True, rights="redistributable", work="w1", filler=None):
    # ≥ 21 tokens (never "tiny"); a filler of the row's own words keeps rows apart under the near-duplicate rule
    body = f"{text} {filler if filler is not None else ' '.join(f'{cid}{w}' for w in WORDS.split())}"
    return {"release_attempt_id": RELEASE, "chunk_id": cid, "parent_id": parent, "work_id": work,
            "membership": "train_eligible", "rights_state": rights, "evidence_eligible": eligible,
            "payload": body, "payload_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "represented_text": body}


def make_store(rows, vectors):
    release = Release(release_attempt_id=RELEASE, route_id="r", row_count=len(rows))
    return FixtureStore(rows, np.asarray(vectors, dtype=np.float32), release)


def make_deps(store, config, *, mode_rights=("redistributable", "internal_research_only", "licensed"), top_k=10):
    store_cfg = SimpleNamespace(serve_rights=tuple(mode_rights), admitted_works=(), refused_evidence_basis=None)
    return SimpleNamespace(config=SimpleNamespace(store=store_cfg), store=store, hybrid=config,
                           retrieve_rule=dict(pipeline.NEAR_DUPLICATE_RULE), min_context_tokens=21, top_k_children=top_k)


def profile(method="rrf", **over):
    doc = {"profile_id": "hybrid-test", "hybrid_module_sha256": hybrid.module_sha256(), "read_predicate": "filter_first",
           "legs": {"dense": {"depth": 500, "survivors": 100},
                    "lexical": {"depth": 500, "text": "represented_text", "corpus": "all_release_rows",
                                "tokenizer": "hybrid_v1.tokens", "bm25": {"k1": 1.2, "b": 0.75}}},
           "top_k_children": 10 if method == "rrf" else 150}
    doc["fusion"] = ({"method": "rrf", "k": 60, "out_depth": 500} if method == "rrf"
                     else {"method": "union_candidates_then_dedup", "extra_n": 50, "max_pool": 150})
    doc.update(over)
    return doc


def test_vendored_module_is_the_loop_file():
    """The loop's reference module and Ceres's copy are the same bytes (skip when the loop is not mounted)."""
    loop = os.environ.get("DDS_LOOP_ROOT")
    reference = Path(loop or "/nonexistent") / "loop-v0" / "04_rag" / "code" / "hybrid_v1.py"
    if not reference.is_file():
        pytest.skip("DDS_LOOP_ROOT/loop-v0/04_rag/code/hybrid_v1.py not available")
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == hybrid.module_sha256()


@pytest.mark.parametrize("change", [
    {"hybrid_module_sha256": "0" * 64},
    {"read_predicate": "after_truncation"},
    {"legs": {"dense": {"depth": 500}, "lexical": {"depth": 500, "text": "payload", "corpus": "all_release_rows",
                                                   "tokenizer": "hybrid_v1.tokens", "bm25": {"k1": 1.2, "b": 0.75}}}},
    {"legs": {"dense": {"depth": 500}, "lexical": {"depth": 500, "text": "represented_text", "corpus": "eligible_rows",
                                                   "tokenizer": "hybrid_v1.tokens", "bm25": {"k1": 1.2, "b": 0.75}}}},
    {"legs": {"dense": {"depth": 500}, "lexical": {"depth": 500, "text": "represented_text", "corpus": "all_release_rows",
                                                   "tokenizer": "hybrid_v1.tokens", "bm25": {"k1": 0.9, "b": 0.75}}}},
    {"fusion": {"method": "weighted_sum"}},
], ids=["module", "predicate", "text", "corpus", "k1", "method"])
def test_a_profile_that_differs_from_the_vendored_module_is_refused(change):
    with pytest.raises(hybrid.HybridError):
        hybrid.hybrid_config(profile(**change))


def test_lexical_leg_is_filter_first_and_fills_its_depth():
    """The best lexical match is ineligible: it must not appear, and the leg still returns `depth` kept ids."""
    rows = [row("c0", "p0", "wheat " * 9, eligible=False)] + [row(f"c{i}", f"p{i}", "wheat") for i in range(1, 6)]
    store = make_store(rows, np.eye(6))
    leg = hybrid.LexicalLeg(store, store.active_release(), (("redistributable",), (), None))
    ranked = leg.index.ranked("wheat", depth=3, keep=leg.keep)
    assert "c0" not in ranked and len(ranked) == 3
    assert leg.index.ranked("wheat", depth=3)[0] == "c0"          # without `keep` it would have been first


def test_index_statistics_do_not_depend_on_the_serving_mode():
    rows = [row("a", "p1", "maize"), row("b", "p2", "maize", rights="internal_research_only")]
    store = make_store(rows, np.eye(2))
    rnd = hybrid.LexicalLeg(store, store.active_release(), (("redistributable", "internal_research_only"), (), None))
    prod = hybrid.LexicalLeg(store, store.active_release(), (("redistributable",), (), None))
    assert rnd.stats_sha256 == prod.stats_sha256 and rnd.keep != prod.keep


def test_a_row_the_store_predicate_refuses_stops_the_request():
    """If the cached predicate set lets a lexical id through that the store's SQL refuses, nothing is served."""
    rows = [row("a", "p1", "rye"), row("b", "p2", "oats")]
    store = make_store(rows, np.eye(2))
    deps = make_deps(store, hybrid.hybrid_config(profile()))
    leg = hybrid.lexical_leg(deps, store.active_release())
    store.rows[1]["evidence_eligible"] = False          # the store changes under the cached predicate
    leg.keep = frozenset({"a", "b"})
    with pytest.raises(hybrid.HybridError, match="predicate"):
        hybrid.retrieve_hybrid(deps, [1.0, 0.0], "oats", store.active_release())


def _union_fixture():
    # dense favours d1..d3 (vector axis 0); the lexical leg finds "barley" only in x1 (a new parent) and in d1's sibling.
    rows = [row("d1", "pd1", "wheat yield"), row("d2", "pd2", "wheat price"), row("d3", "pd3", "wheat soil"),
            row("x1", "px1", "barley barley barley"), row("d1b", "pd1", "barley wheat")]
    vectors = [[1, 0], [0.9, 0.1], [0.8, 0.2], [0, 1], [0.1, 0.9]]
    return make_store(rows, vectors)


def test_union_extras_reach_the_reranker_and_ties_go_by_pool_position():
    store = _union_fixture()
    config = hybrid.hybrid_config(profile("union"))
    config.update(dense_depth=3, dense_survivors=3, extra_n=2, max_pool=5)   # x1 is outside the dense leg: fetched by id
    deps = make_deps(store, config, top_k=5)
    trace: dict = {}
    _candidates, pool = hybrid.retrieve_hybrid(deps, [1.0, 0.0], "barley", store.active_release(), trace=trace)
    ids = [item.chunk_id for item in pool]
    assert ids[:3] == ["d1", "d2", "d3"] and "x1" in ids          # the lexical-only parent reaches the pool
    assert "d1b" not in ids                                          # same parent as d1: dropped by the union dedup
    x1 = next(item for item in pool if item.chunk_id == "x1")
    assert x1.dense_rank == 0                                        # unstamped: the reason pool_position exists
    for item in pool:
        item.rerank_score = 0.5                                      # all tie
    ordered = sorted(pool, key=pipeline.rerank_order_key(pool, "pool_position"))
    assert [item.chunk_id for item in ordered] == ids               # ties keep pool order …
    assert sorted(pool, key=pipeline.rerank_order_key(pool, "dense_rank"))[0].chunk_id == "x1"   # … dense_rank would not


def test_union_drops_a_cross_leg_near_duplicate():
    # x1 is lexical-only ("harvest") and a near-duplicate of the dense survivor d1 (shared filler), from another parent
    rows = [row("d1", "pd1", "wheat yield", filler=WORDS), row("x1", "px1", "wheat yield harvest", filler=WORDS)]
    store = make_store(rows, [[1, 0], [0, 1]])
    config = hybrid.hybrid_config(profile("union"))
    config.update(dense_survivors=1, extra_n=1, max_pool=2)
    deps = make_deps(store, config, top_k=2)
    dropped: list = []
    _c, pool = hybrid.retrieve_hybrid(deps, [1.0, 0.0], "harvest", store.active_release(), dropped=dropped)
    assert [item.chunk_id for item in pool] == ["d1"]
    assert any(d["dropped_chunk_id"] == "x1" and d["reason"] == "near_duplicate" and d["leg"] == "union" for d in dropped)


def test_retrieve_hybrid_filters_the_lexical_leg_before_truncating_it():
    """Two ineligible chunks outscore every eligible one lexically; with depth 2 a filter applied AFTER truncation
    would leave the lexical leg empty. Ceres must pass the predicate into the ranking (`keep`)."""
    rows = [row("bad1", "pb1", "rye " * 6, eligible=False), row("bad2", "pb2", "rye " * 5, eligible=False),
            row("ok1", "po1", "rye"), row("ok2", "po2", "rye")]
    store = make_store(rows, np.eye(4))
    config = hybrid.hybrid_config(profile())
    config.update(lexical_depth=2)
    deps = make_deps(store, config)
    trace: dict = {}
    hybrid.retrieve_hybrid(deps, [0, 0, 1, 0], "rye", store.active_release(), trace=trace)
    assert trace["lexical_ids"] == ["ok1", "ok2"]


def test_rrf_route_dedups_the_module_fused_order():
    store = _union_fixture()
    deps = make_deps(store, hybrid.hybrid_config(profile()), top_k=10)
    trace: dict = {}
    _c, kept = hybrid.retrieve_hybrid(deps, [1.0, 0.0], "barley", store.active_release(), trace=trace)
    fused = hybrid_v1.rrf([trace["dense_ids"], trace["lexical_ids"]], k=60, depth=500, out_depth=500)
    assert trace["fused_ids"] == fused
    expected, parents = [], set()
    for cid in fused:                                  # the standard dedup over the fused order: parent keep-best
        parent = next(r["parent_id"] for r in store.rows if r["chunk_id"] == cid)
        if parent not in parents:
            parents.add(parent)
            expected.append(cid)
    assert [item.chunk_id for item in kept] == expected


def test_dense_routes_get_no_hybrid_config_and_keep_the_dense_rank_tie_break():
    loaded = {"retrieve": yaml.safe_dump({"profile_id": "dense-top10-v2", "top_k_children": 10}).encode(),
              "rerank": yaml.safe_dump({"profile_id": "qwen3-4b-rerank-v3", "top_k_input": 100, "top_k_output": 5}).encode()}
    deps = SimpleNamespace()
    pipeline.apply_hybrid(deps, loaded)
    assert deps.hybrid is None and deps.rerank_tie_break == "dense_rank"
    pool = [Item(chunk_id=f"c{i}", parent_id="p", work_id="w", membership="m", rights_state="r", payload="x",
                 payload_sha256="s", dense_rank=i + 1, rerank_score=0.1) for i in range(4)]
    assert (sorted(pool, key=pipeline.rerank_order_key(pool, "dense_rank"))
            == sorted(pool, key=pipeline.rerank_order_key(pool, "pool_position")))   # dense pools: the two agree


def test_a_union_route_with_a_dense_rank_tie_break_is_refused():
    loaded = {"retrieve": yaml.safe_dump(profile("union")).encode(),
              "rerank": yaml.safe_dump({"profile_id": "qwen3-4b-rerank-v3", "top_k_input": 150, "top_k_output": 5}).encode()}
    with pytest.raises(pipeline.PipelineError, match="pool_position"):
        pipeline.apply_hybrid(SimpleNamespace(), loaded)


def test_a_hybrid_route_without_question_text_is_refused():
    store = _union_fixture()
    deps = make_deps(store, hybrid.hybrid_config(profile()))
    with pytest.raises(hybrid.HybridError, match="question text"):
        hybrid.retrieve_hybrid(deps, [1.0, 0.0], "  ", store.active_release())


# ================================================================ addendum xl-v1 (both-language legs for Romanian questions)
# How it can fail, and the test that catches each:
#   * a Romanian question runs without its translation (silently monolingual) → test_xl_ro_question_without_translation_is_refused
#   * English questions change on an xl route → test_xl_en_question_is_exactly_the_base_route
#   * s0-xl fuses other lists than the four the spec names → test_xl_rrf_fuses_the_four_legs_with_the_module
#   * s1-xl takes its lexical extras from the original instead of the translation → test_xl_union_extras_come_from_the_translation
#   * a profile stating other legs, another language or another rerank query runs anyway → test_xl_profile_mismatch_is_refused
#   * the translation file is not the pinned one, or a question lacks a translation → test_translation_file_is_pinned_and_complete

def xl_profile(method="rrf", **over):
    doc = profile(method)
    doc["translation"] = {"applies_to_language": "ro", "source": "t.jsonl", "source_sha256": "a" * 64}
    doc["ro_legs"] = hybrid.XL_RO_LEGS[doc["fusion"]["method"]]
    doc["rerank_query"] = "original"
    doc.update(over)
    return doc


def _xl_fixture():
    # axis 0 = the RO question's direction, axis 1 = the EN translation's; "grâu" only in r1, "wheat" only in e1
    rows = [row("r1", "pr1", "grâu toamnă"), row("r2", "pr2", "porumb"), row("e1", "pe1", "wheat winter"),
            row("e2", "pe2", "maize")]
    return make_store(rows, [[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]])


def test_xl_ro_question_without_translation_is_refused():
    store = _xl_fixture()
    deps = make_deps(store, hybrid.hybrid_config(xl_profile()))
    with pytest.raises(hybrid.HybridError, match="translation"):
        hybrid.retrieve_hybrid(deps, [1.0, 0.0], "grâu", store.active_release(), language="ro")


def test_xl_en_question_is_exactly_the_base_route():
    store = _xl_fixture()
    base_t, xl_t = {}, {}
    _c, base = hybrid.retrieve_hybrid(make_deps(store, hybrid.hybrid_config(profile())), [0.0, 1.0], "wheat",
                                      store.active_release(), trace=base_t, language="en")
    _c, xl = hybrid.retrieve_hybrid(make_deps(store, hybrid.hybrid_config(xl_profile())), [0.0, 1.0], "wheat",
                                    store.active_release(), trace=xl_t, language="en", query_en="ignored",
                                    qvec_en=[1.0, 0.0])
    assert [i.chunk_id for i in xl] == [i.chunk_id for i in base] and xl_t == base_t


def test_xl_rrf_fuses_the_four_legs_with_the_module():
    store = _xl_fixture()
    deps = make_deps(store, hybrid.hybrid_config(xl_profile()))
    trace: dict = {}
    hybrid.retrieve_hybrid(deps, [1.0, 0.0], "grâu", store.active_release(), trace=trace, language="ro",
                           query_en="wheat", qvec_en=[0.0, 1.0])
    assert trace["lexical_en_ids"] == ["e1"] and trace["dense_en_ids"][0] == "e1"
    assert trace["fused_ids"] == hybrid_v1.rrf([trace["dense_ids"], trace["dense_en_ids"], trace["lexical_ids"],
                                                trace["lexical_en_ids"]], k=60, depth=500, out_depth=500)


def test_xl_union_extras_come_from_the_translation():
    store = _xl_fixture()
    config = hybrid.hybrid_config(xl_profile("union"))
    config.update(dense_depth=1, dense_survivors=1, extra_n=1, max_pool=2)   # dense leg = r1 only
    deps = make_deps(store, config, top_k=2)
    _c, pool = hybrid.retrieve_hybrid(deps, [1.0, 0.0], "porumb", store.active_release(), language="ro",
                                      query_en="wheat", qvec_en=[0.0, 1.0])
    assert [i.chunk_id for i in pool] == ["r1", "e1"]          # "porumb" (r2) would be the original's lexical extra


@pytest.mark.parametrize("change", [{"ro_legs": ["dense_original", "lexical_original"]}, {"rerank_query": "translation"},
                                    {"translation": {"applies_to_language": "hu", "source": "t", "source_sha256": "a" * 64}},
                                    {"translation": {"applies_to_language": "ro", "source": "t"}}],
                         ids=["legs", "rerank-query", "language", "no-sha"])
def test_xl_profile_mismatch_is_refused(change):
    with pytest.raises(hybrid.HybridError):
        hybrid.hybrid_config(xl_profile(**change))


def test_translation_file_is_pinned_and_complete(tmp_path):
    from ceres import eval as ceval

    body = '{"question_id": "q1", "text": "wheat"}\n'
    (tmp_path / "t.jsonl").write_text(body, encoding="utf-8")
    good = {"applies_to_language": "ro", "source": "t.jsonl", "source_sha256": hashlib.sha256(body.encode()).hexdigest()}
    ctx = SimpleNamespace(route=SimpleNamespace(loop_root=tmp_path),
                          bank=[{"question_id": "q1", "language": "ro"}, {"question_id": "q2", "language": "en"}])
    assert ceval.load_translations(ctx, good) == {"q1": "wheat"}
    with pytest.raises(ceval.Refused, match="pins"):
        ceval.load_translations(ctx, {**good, "source_sha256": "b" * 64})
    ctx.bank.append({"question_id": "q3", "language": "ro"})
    with pytest.raises(ceval.Refused, match="no translation"):
        ceval.load_translations(ctx, good)
