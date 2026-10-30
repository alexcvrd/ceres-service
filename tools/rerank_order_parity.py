#!/usr/bin/env python3
"""Rerank ORDER parity: Ceres's eval rerank rows against the loop's reranked hits, per question.

Input parity (the exact text each candidate is scored on) is a CPU test
(`tests/test_v35.py::test_ceres_sends_the_reranker_the_text_the_loop_scored_window2`). ORDER parity
needs the model, so it is checked here once the owner has run `ceres-eval rerank` on the GPU:

    .venv/bin/python tools/rerank_order_parity.py \\
        --ceres-work-dir <work dir> --route s1-wide-v35hr \\
        --loop-rerank-attempt <loop>/04_rag/runs/slice-v0/rerank/s1-wide-v35hr-rerank-attempt-0001 \\
        [--top-k 5] [--out report.json]

Per question the top-k chunk-id ORDER is compared. A difference is attributed to:
  * `filter-first` (parity rule): the two sides did not score the same
    candidates. Ceres applies the read predicate (release + evidence_eligible + rights) in SQL
    BEFORE dedup and top-k; the loop keeps ineligible chunks in its pool and drops them only when it
    builds packets. The report names the loop-only and Ceres-only candidates and how many of the
    loop's top-k Ceres never scored.
  * `score`: both sides scored exactly the same candidate set and still ordered them differently
    (model numerics: Ceres device/dtype vs the loop's cuda bf16 run, near ties). The report gives the
    first differing rank and both sides' scores for the two chunks there.
  * `missing`: the question is on one side only.
Read-only on both inputs; `--out` must lie outside the loop. Exit 0 always (it is a report, not a
gate); the summary line says how many questions matched.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def by_question(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["question_id"], []).append(row)
    return grouped


def keyed_last(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """ceres-eval phase rows by question id; a later row supersedes an earlier one (as the eval reads them)."""
    return {row["question_id"]: row for row in rows if "question_id" in row}


def loop_root_of(attempt: Path) -> Path:
    """`<loop>/04_rag/runs/slice-v0/rerank/<attempt>` → `<loop>`."""
    return Path(attempt).resolve().parents[4]


def loop_inputs(attempt: Path, hits_override: Path | None = None) -> tuple[dict[str, list[dict]], dict[str, list[dict]], int, dict]:
    """(reranked hits by question, retrieve hits by question in file order, top_k_input, manifest facts)."""
    attempt = Path(attempt)
    manifest = json.loads((attempt / "MANIFEST.json").read_text(encoding="utf-8"))
    reranked = by_question(read_jsonl(attempt / "reranked-hits.jsonl"))
    if hits_override is not None:
        hits_path = Path(hits_override)
    else:
        retrieve_manifest = loop_root_of(attempt) / manifest["input_manifest_path"]
        hits_path = retrieve_manifest.parent / "hits.jsonl"
    retrieved = by_question(read_jsonl(hits_path))
    top_k_input = 100
    profile_path = manifest.get("profile_path")
    if profile_path and (loop_root_of(attempt) / profile_path).is_file():
        import yaml

        profile = yaml.safe_load((loop_root_of(attempt) / profile_path).read_text(encoding="utf-8")) or {}
        top_k_input = int(profile.get("top_k_input") or top_k_input)
    facts = {"attempt": str(attempt), "hits": str(hits_path), "rerank_text": (manifest.get("counts") or {}).get("rerank_text"),
             "questions_sha256": (manifest.get("counts") or {}).get("questions_sha256"),
             "hardware_flags": manifest.get("hardware_flags"), "top_k_input": top_k_input}
    return reranked, retrieved, top_k_input, facts


def compare(ceres_rerank: dict[str, dict], ceres_retrieve: dict[str, dict], loop_reranked: dict[str, list[dict]],
            loop_retrieved: dict[str, list[dict]], *, top_k: int = 5, top_k_input: int = 100) -> dict[str, Any]:
    """The per-question report (pure; the CLI only reads files and prints)."""
    questions = sorted(set(ceres_rerank) | set(loop_reranked))
    rows = []
    for qid in questions:
        if qid not in ceres_rerank or qid not in loop_reranked:
            rows.append({"question_id": qid, "status": "missing",
                         "side": "loop only" if qid not in ceres_rerank else "ceres only"})
            continue
        ceres_hits = sorted(ceres_rerank[qid]["hits"], key=lambda hit: hit["rank"])[:top_k]
        loop_hits = sorted(loop_reranked[qid], key=lambda hit: hit["rank"])[:top_k]
        ceres_top = [hit["chunk_id"] for hit in ceres_hits]
        loop_top = [hit["chunk_id"] for hit in loop_hits]
        row: dict[str, Any] = {"question_id": qid, "ceres_top": ceres_top, "loop_top": loop_top}
        if ceres_top == loop_top:
            rows.append({**row, "status": "match"})
            continue
        scored = int(ceres_rerank[qid].get("scored") or top_k_input)
        ceres_scored = [item["chunk_id"] for item in (ceres_retrieve.get(qid) or {}).get("kept", [])][:scored]
        loop_scored = [hit["chunk_id"] for hit in loop_retrieved.get(qid, [])][:top_k_input]
        loop_only = [cid for cid in loop_scored if cid not in set(ceres_scored)]
        ceres_only = [cid for cid in ceres_scored if cid not in set(loop_scored)]
        row["same_top_set"] = set(ceres_top) == set(loop_top)
        if loop_only or ceres_only or not ceres_scored:
            rows.append({**row, "status": "filter-first", "loop_only_candidates": loop_only,
                         "ceres_only_candidates": ceres_only,
                         "loop_top_not_scored_by_ceres": [cid for cid in loop_top if cid in set(loop_only)],
                         # the loop keeps these in its pool and drops them only at packet time
                         "loop_top_evidence_ineligible": [hit["chunk_id"] for hit in loop_hits
                                                          if hit.get("evidence_eligible") is False],
                         "ceres_retrieve_row": bool(ceres_scored)})
            continue
        first = next(index for index, (a, b) in enumerate(zip(ceres_top, loop_top)) if a != b)
        ceres_scores = {hit["chunk_id"]: hit.get("rerank_score") for hit in ceres_rerank[qid]["hits"]}
        loop_scores = {hit["chunk_id"]: hit.get("score") for hit in loop_reranked[qid]}
        pair = [ceres_top[first], loop_top[first]]
        rows.append({**row, "status": "score", "first_difference_rank": first + 1, "pair": pair,
                     "ceres_scores": [ceres_scores.get(cid) for cid in pair],
                     "loop_scores": [loop_scores.get(cid) for cid in pair]})
    counts = Counter(row["status"] for row in rows)
    return {"top_k": top_k, "top_k_input": top_k_input, "questions": len(rows),
            "counts": dict(sorted(counts.items())), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rerank_order_parity", description=__doc__.split("\n\n")[0])
    parser.add_argument("--ceres-work-dir", required=True, type=Path)
    parser.add_argument("--route", required=True, help="route name (the work dir's routes/<route>/)")
    parser.add_argument("--loop-rerank-attempt", required=True, type=Path)
    parser.add_argument("--loop-hits", type=Path, help="override: the loop retrieve hits.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, help="write the full JSON report here (outside the loop)")
    args = parser.parse_args(argv)
    route_dir = args.ceres_work_dir / "routes" / args.route
    ceres_rerank = keyed_last(read_jsonl(route_dir / "rerank.jsonl"))
    ceres_retrieve = keyed_last(read_jsonl(route_dir / "retrieve.jsonl")) if (route_dir / "retrieve.jsonl").is_file() else {}
    loop_reranked, loop_retrieved, top_k_input, facts = loop_inputs(args.loop_rerank_attempt, args.loop_hits)
    report = compare(ceres_rerank, ceres_retrieve, loop_reranked, loop_retrieved, top_k=args.top_k,
                     top_k_input=top_k_input)
    report["loop"] = facts
    report["ceres"] = {"work_dir": str(args.ceres_work_dir.resolve()), "route": args.route}
    if args.out is not None:
        loop = loop_root_of(args.loop_rerank_attempt)
        out = args.out.resolve()
        if out == loop or loop in out.parents:
            print(f"rerank_order_parity: --out must lie outside the loop ({loop})", file=sys.stderr)
            return 2
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"questions": report["questions"], "counts": report["counts"], "top_k": args.top_k},
                     sort_keys=True))
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
