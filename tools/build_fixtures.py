#!/usr/bin/env python3
"""Build the hermetic Ceres fixture corpus + golden outputs.

The fixture lets every later Ceres stage be tested without a GPU and without
Postgres, with golden outputs taken verbatim from the DDS loop's own r1 attempt
chain, so a Ceres stage that matches the fixture matches the judged route.

Sources (all read-only):

* the local serving store (role ``ceres_ro``), schema ``ceres_serve_v0``, the
  release row whose ``active`` flag is true;
* the loop attempt chain that produced that release's embeddings, resolved from
  ``releases.manifest.attempts.embed.dir`` and then forward through each slot
  MANIFEST's ``input_manifest_path``;
* the r1 route profiles, prompt template and output schemas under the loop.

Nothing here writes outside ``--out``.

Usage
-----
    <loop>/00_control/.venv/bin/python tools/build_fixtures.py --out tests/fixtures
    ... --force          # overwrite an existing fixture directory

Exit codes: 0 ok, 2 bad inputs / broken chain, 3 assertion failure.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
import yaml

BUILDER_VERSION = "c0-fixtures-v1"

# Slot order for the r1 route id recipe.
# Source: loop-v0/04_rag/common.py:39  SLOTS = ("chunk", "represent", "embed",
# "index", "retrieve", "rerank", "answer")
SLOTS = ("chunk", "represent", "embed", "index", "retrieve", "rerank", "answer")

# Expected judged route id (r1). Asserted after recomputation.
JUDGED_ROUTE_ID = "73c2e7911d632ad5602bb72f869d5be81a920343fdca9c204fe2a7d7a6bf7315"

DEFAULT_DSN = "postgresql://ceres_ro@127.0.0.1:54329/postgres"
DEFAULT_SCHEMA = "ceres_serve_v0"

SCHEMA_FILES = (
    "rag-answer-packets.schema.json",
    "answers.schema.json",
    "hits.schema.json",
    "reranked-hits.schema.json",
)


class BuildError(RuntimeError):
    """Any condition that must stop the build."""


# --------------------------------------------------------------------------- io


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def repo_default_loop_root(here: Path) -> Path:
    """ceres-service/tools/ -> <workspace>/dds/loop-v0."""
    env = os.environ.get("DDS_LOOP_ROOT")
    if env:
        return Path(env).resolve()
    eo_root = here.parents[3]  # tools -> ceres-service -> <workspace>
    return (eo_root / "dds/loop-v0").resolve()


# ------------------------------------------------------------------ attempt chain


def manifest_of(attempt_dir: Path) -> dict[str, Any]:
    path = attempt_dir / "MANIFEST.json"
    if not path.is_file():
        raise BuildError(f"attempt has no MANIFEST.json: {attempt_dir}")
    return read_json(path)


def resolve_downstream(loop_root: Path, slot: str, want_input: Path) -> Path:
    """Find the single attempt in ``slot`` whose input_manifest_path is want_input.

    ``want_input`` is the MANIFEST.json path of the upstream attempt. The loop
    records ``input_manifest_path`` loop-relative, so compare resolved paths.
    If several attempts qualify, take the newest by directory name and record
    the ambiguity for the caller (the manifest notes it).
    """
    slot_root = loop_root / "04_rag/runs/slice-v0" / slot
    matches: list[Path] = []
    for attempt in sorted(slot_root.iterdir()):
        mpath = attempt / "MANIFEST.json"
        if not attempt.is_dir() or not mpath.is_file():
            continue
        manifest = read_json(mpath)
        declared = manifest.get("input_manifest_path")
        if not declared:
            continue
        resolved = (loop_root / declared).resolve() if not Path(declared).is_absolute() else Path(declared).resolve()
        if resolved == want_input.resolve():
            matches.append(attempt)
    if not matches:
        raise BuildError(
            f"no {slot} attempt consumes {want_input}; the chain for the active release is incomplete"
        )
    return matches[-1], [p.name for p in matches]


# ------------------------------------------------------------------------- store


def load_store(dsn: str, schema: str) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    with psycopg.connect(dsn) as conn:
        release = conn.execute(
            f"SELECT release_attempt_id, route_id, chunk_profile_sha256, represent_profile_sha256, "
            f"embed_profile_sha256, embed_attempt_dir, row_count, loaded_by, manifest "
            f"FROM {schema}.releases WHERE active"
        ).fetchall()
        if len(release) != 1:
            raise BuildError(f"expected exactly one active release, found {len(release)}")
        (
            release_id, route_id, chunk_sha, represent_sha, embed_sha,
            embed_dir, row_count, loaded_by, manifest,
        ) = release[0]
        rows = conn.execute(
            f"SELECT release_attempt_id, chunk_id, parent_id, work_id, membership, rights_state, "
            f"eligibility_state, evidence_eligible, payload, payload_sha256, locator_ids, "
            f"lineage_ids, excluded_reasons, representation_id, embedding "
            f"FROM {schema}.items WHERE release_attempt_id = %s ORDER BY chunk_id",
            (release_id,),
        ).fetchall()

    items: list[dict[str, Any]] = []
    vectors: list[list[float]] = []
    undecodable = 0
    for r in rows:
        payload = bytes(r[8])
        if sha256_bytes(payload) != r[9]:
            raise BuildError(f"payload sha256 mismatch for chunk_id {r[1]}")
        item: dict[str, Any] = {
            "release_attempt_id": r[0],
            "chunk_id": r[1],
            "parent_id": r[2],
            "work_id": r[3],
            "membership": r[4],
            "rights_state": r[5],
            "eligibility_state": r[6],
            "evidence_eligible": r[7],
            "payload_sha256": r[9],
            "locator_ids": r[10],
            "lineage_ids": r[11],
            "excluded_reasons": r[12],
            "representation_id": r[13],
        }
        try:
            item["payload"] = payload.decode("utf-8")
        except UnicodeDecodeError:
            undecodable += 1
            item["payload"] = None
            item["payload_b64"] = base64.b64encode(payload).decode("ascii")
        items.append(item)
        vectors.append(json.loads(r[14]))

    embeddings = np.asarray(vectors, dtype=np.float32)
    release_row = {
        "release_attempt_id": release_id,
        "route_id": route_id,
        "chunk_profile_sha256": chunk_sha,
        "represent_profile_sha256": represent_sha,
        "embed_profile_sha256": embed_sha,
        "embed_attempt_dir": embed_dir,
        "row_count": row_count,
        "loaded_by": loaded_by,
        "manifest": manifest,
        "payload_undecodable_rows": undecodable,
    }
    return release_row, items, embeddings


# ------------------------------------------------------------------- ranking math


def exact_top_k(
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    chunk_ids: list[str],
    parent_ids: list[str],
    question_ids: list[str],
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    """Mirror loop-v0/04_rag/slot_adapter.py:946 exact_rankings.

    L2-normalise both sides, cosine by dot product, ``np.lexsort((chunk_ids,
    -scores))`` so score descends and chunk_id ascends on a tie, score rounded
    to 10 decimals in the output only.
    """
    docs = np.asarray(doc_vectors, dtype=np.float32).copy()
    queries = np.asarray(query_vectors, dtype=np.float32).copy()
    docs /= np.clip(np.linalg.norm(docs, axis=1, keepdims=True), 1e-12, None)
    queries /= np.clip(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12, None)
    matrix = queries @ docs.T
    ids = np.asarray(chunk_ids, dtype=object)
    out: dict[str, list[dict[str, Any]]] = {}
    for qindex, question_id in enumerate(question_ids):
        scores = matrix[qindex]
        order = np.lexsort((ids, -scores))[: min(top_k, len(chunk_ids))]
        out[question_id] = [
            {
                "question_id": question_id,
                "rank": rank,
                "chunk_id": chunk_ids[int(j)],
                "parent_id": parent_ids[int(j)],
                "score": round(float(scores[int(j)]), 10),
            }
            for rank, j in enumerate(order, start=1)
        ]
    return out


# -------------------------------------------------------------------------- build


def build(out_dir: Path, loop_root: Path, dsn: str, schema: str, force: bool) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise BuildError(f"{out_dir} is not empty; pass --force to overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs: dict[str, str] = {}          # path -> sha256, every file read
    notes: list[str] = []

    def record(path: Path) -> str:
        digest = sha256_file(path)
        inputs[str(path)] = digest
        return digest

    # ---- 1. store -----------------------------------------------------------
    release, items, embeddings = load_store(dsn, schema)
    release_id = release["release_attempt_id"]
    if len(items) != release["row_count"]:
        raise BuildError(f"row_count {release['row_count']} != rows read {len(items)}")

    # ---- 2. attempt chain ---------------------------------------------------
    embed_dir = Path(release["embed_attempt_dir"]).resolve()
    if not embed_dir.is_dir():
        raise BuildError(f"embed attempt dir from the release row does not exist: {embed_dir}")
    embed_manifest_path = embed_dir / "MANIFEST.json"
    embed_manifest = read_json(embed_manifest_path)
    embed_manifest_sha = record(embed_manifest_path)
    declared_embed_sha = (release["manifest"].get("attempts", {}).get("embed", {}) or {}).get("manifest_sha256")
    if declared_embed_sha and declared_embed_sha != embed_manifest_sha:
        raise BuildError(
            f"embed MANIFEST sha256 {embed_manifest_sha} != the sha recorded in the release manifest {declared_embed_sha}"
        )

    index_dir, index_alts = resolve_downstream(loop_root, "index", embed_manifest_path)
    retrieve_dir, retrieve_alts = resolve_downstream(loop_root, "retrieve", index_dir / "MANIFEST.json")
    rerank_dir, rerank_alts = resolve_downstream(loop_root, "rerank", retrieve_dir / "MANIFEST.json")
    answer_dir, answer_alts = resolve_downstream(loop_root, "answer", rerank_dir / "MANIFEST.json")

    chain = {
        "embed": {"dir": str(embed_dir), "candidates": [embed_dir.name]},
        "index": {"dir": str(index_dir), "candidates": index_alts},
        "retrieve": {"dir": str(retrieve_dir), "candidates": retrieve_alts},
        "rerank": {"dir": str(rerank_dir), "candidates": rerank_alts},
        "answer": {"dir": str(answer_dir), "candidates": answer_alts},
    }
    manifests = {
        "embed": embed_manifest,
        "index": read_json(index_dir / "MANIFEST.json"),
        "retrieve": read_json(retrieve_dir / "MANIFEST.json"),
        "rerank": read_json(rerank_dir / "MANIFEST.json"),
        "answer": read_json(answer_dir / "MANIFEST.json"),
    }
    for slot in ("index", "retrieve", "rerank", "answer"):
        record(Path(chain[slot]["dir"]) / "MANIFEST.json")
        if manifests[slot].get("route_id") != JUDGED_ROUTE_ID:
            raise BuildError(f"{slot} attempt route_id != r1 {JUDGED_ROUTE_ID}")
        if len(chain[slot]["candidates"]) > 1:
            notes.append(
                f"{slot}: {len(chain[slot]['candidates'])} attempts consume the same input "
                f"({', '.join(chain[slot]['candidates'])}); the newest was taken"
            )
        declared = manifests[slot].get("input_manifest_sha256s") or []
        upstream = {"index": "embed", "retrieve": "index", "rerank": "retrieve", "answer": "rerank"}[slot]
        upstream_sha = sha256_file(Path(chain[upstream]["dir"]) / "MANIFEST.json")
        # A loop patch lists the upstream MANIFEST sha256 as one element (the other is
        # the questions sha256); before it the list was empty or held only the questions sha256.
        if upstream_sha in declared:
            chain[slot]["input_hash_verified"] = True
        else:
            chain[slot]["input_hash_verified"] = False
            notes.append(
                f"{slot}: input_manifest_sha256s {declared} does not contain the upstream "
                f"{upstream} MANIFEST sha256 {upstream_sha}; the chain is verified by "
                f"input_manifest_path only"
            )

    # ---- 3. questions + query vectors --------------------------------------
    questions_path = Path(manifests["retrieve"]["questions_path"])
    if not questions_path.is_file():
        raise BuildError(f"questions file from the retrieve MANIFEST is missing: {questions_path}")
    questions_sha = record(questions_path)
    if questions_sha != manifests["retrieve"]["questions_sha256"]:
        raise BuildError("questions file sha256 does not match the retrieve MANIFEST")
    questions = read_jsonl(questions_path)
    question_ids = [q["question_id"] for q in questions]

    question_rows_path = embed_dir / "question-rows.jsonl"
    query_vectors_path = embed_dir / "query-vectors.npy"
    record(question_rows_path)
    record(query_vectors_path)
    question_rows = read_jsonl(question_rows_path)
    if [r["question_id"] for r in question_rows] != question_ids:
        raise BuildError("embed question-rows.jsonl ids differ from the questions file order")
    query_vectors = np.load(query_vectors_path, allow_pickle=False).astype(np.float32)
    if query_vectors.shape != (len(questions), embeddings.shape[1]):
        raise BuildError(f"query-vectors.npy shape {query_vectors.shape} unexpected")

    # ---- 4. items.jsonl + items.npy ----------------------------------------
    write_jsonl(out_dir / "items.jsonl", items)
    np.save(out_dir / "items.npy", embeddings, allow_pickle=False)
    item_norms = np.linalg.norm(embeddings, axis=1)
    query_norms = np.linalg.norm(query_vectors, axis=1)

    write_jsonl(
        out_dir / "questions.jsonl",
        [{"question_id": q["question_id"], "language": q["language"], "text": q["text"]} for q in questions],
    )
    np.save(out_dir / "query-vectors.npy", query_vectors, allow_pickle=False)

    # ---- 5. golden: builder candidate pool + diffs --------------------------
    chunk_ids = [i["chunk_id"] for i in items]
    parent_ids = [i["parent_id"] for i in items]
    builder_pool = exact_top_k(embeddings, query_vectors, chunk_ids, parent_ids, question_ids, 50)

    # Reference A: the loop's own recomputation over the embed attempt's rows.
    # The r1 retrieve profile is dense-top10-v2, which recomputes the pool from
    # the embed vectors rather than reading the index attempt's file
    # (slot_adapter.py:1188-1193). The embed attempt holds 440 document rows;
    # the store holds one row per distinct chunk_id, so the two pools differ
    # wherever a duplicate chunk_id lands inside the top 50.
    doc_rows = read_jsonl(record_and_return(record, embed_dir / "document-rows.jsonl"))
    doc_vectors = np.load(record_and_return(record, embed_dir / "document-vectors.npy"), allow_pickle=False)
    loop_pool = exact_top_k(
        doc_vectors,
        query_vectors,
        [r["chunk_id"] for r in doc_rows],
        [r["parent_id"] for r in doc_rows],
        question_ids,
        50,
    )

    # Reference B: the index attempt's recorded pool, if it has one.
    index_pool_path = index_dir / "exact-results.jsonl"
    index_pool: dict[str, list[dict[str, Any]]] = {}
    index_pool_depth = None
    if index_pool_path.is_file():
        record(index_pool_path)
        for row in read_jsonl(index_pool_path):
            index_pool[row["question_id"]] = row["hits"]
        index_pool_depth = min(len(v) for v in index_pool.values()) if index_pool else None

    candidate_rows: list[dict[str, Any]] = []
    for question_id in question_ids:
        mine = builder_pool[question_id]
        theirs = loop_pool[question_id]
        mine_ids = [h["chunk_id"] for h in mine]
        theirs_ids = [h["chunk_id"] for h in theirs]
        dupes = sorted({c for c in theirs_ids if theirs_ids.count(c) > 1})
        idx_hits = index_pool.get(question_id, [])
        depth = len(idx_hits)
        idx_cmp = None
        if depth:
            idx_cmp = {
                "compared_depth": depth,
                "source": str(index_pool_path),
                "identical": [(h["rank"], h["chunk_id"], h["parent_id"], h["score"]) for h in mine[:depth]]
                == [(h["rank"], h["chunk_id"], h["parent_id"], h["score"]) for h in idx_hits],
                "index_profile_top_k": manifests["index"].get("counts", {}).get("top_k"),
            }
        candidate_rows.append(
            {
                "question_id": question_id,
                "source": (
                    "BUILDER-COMPUTED: exact cosine top-50 over tests/fixtures/items.npy "
                    "(the 426 active store items), L2-normalised, ties by chunk_id ascending, "
                    "score rounded to 10 dp — mirrors slot_adapter.py:946 exact_rankings"
                ),
                "candidates": mine,
                "diff_vs_loop_recomputed_pool": {
                    "source": (
                        "LOOP-DERIVED: the same computation over the embed attempt's "
                        f"document-vectors.npy ({len(doc_rows)} rows) — what the r1 retrieve "
                        "slot itself ranks over"
                    ),
                    "identical_order": mine_ids == theirs_ids,
                    "loop_rows": len(doc_rows),
                    "store_rows": len(items),
                    "duplicate_chunk_ids_inside_loop_top50": dupes,
                    "in_builder_pool_only": sorted(set(mine_ids) - set(theirs_ids)),
                    "in_loop_pool_only": sorted(set(theirs_ids) - set(mine_ids)),
                    "parent_id_mismatches": sorted(
                        (c, next(h["parent_id"] for h in mine if h["chunk_id"] == c),
                         next(h["parent_id"] for h in theirs if h["chunk_id"] == c))
                        for c in set(mine_ids) & set(theirs_ids)
                        if next(h["parent_id"] for h in mine if h["chunk_id"] == c)
                        != next(h["parent_id"] for h in theirs if h["chunk_id"] == c)
                    ),
                },
                "diff_vs_index_attempt_recorded_pool": idx_cmp,
            }
        )
    write_jsonl(out_dir / "golden/candidates-top50.jsonl", candidate_rows)

    # ---- 6. golden: verbatim loop outputs ----------------------------------
    hits = read_jsonl(record_and_return(record, retrieve_dir / "hits.jsonl"))
    hits = [h for h in hits if h["question_id"] in set(question_ids)]
    write_jsonl(out_dir / "golden/hits-top10.jsonl", hits)

    reranked = read_jsonl(record_and_return(record, rerank_dir / "reranked-hits.jsonl"))
    reranked = [h for h in reranked if h["question_id"] in set(question_ids)]
    write_jsonl(out_dir / "golden/reranked-top5.jsonl", reranked)

    packets = read_jsonl(record_and_return(record, answer_dir / "packets.jsonl"))
    packets = [p for p in packets if p["question_id"] in set(question_ids)]
    write_jsonl(out_dir / "golden/packets.jsonl", packets)

    answers = read_jsonl(record_and_return(record, answer_dir / "answers.jsonl"))
    answers = [a for a in answers if a["question_id"] in set(question_ids)]
    write_jsonl(out_dir / "golden/answers.jsonl", answers)

    raw_dir = out_dir / "golden/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_files: list[str] = []
    request_keys: set[str] = set()
    for source in sorted((answer_dir / "raw").glob("*.json")):
        if not any(source.name.startswith(q) for q in question_ids):
            continue
        record(source)
        shutil.copy2(source, raw_dir / source.name)
        (raw_dir / source.name).chmod(0o644)
        raw_files.append(source.name)
        body = read_json(source)
        request_keys |= set(body["request"])
    if not raw_files:
        raise BuildError("the answer attempt has no raw/*.json bodies for the 3 questions")
    missing = {q for q in question_ids if not any(f.startswith(q) for f in raw_files)}
    if missing:
        raise BuildError(f"the answer attempt has no raw body for question(s): {sorted(missing)}")

    # The request contract the LLM adapter must reproduce byte for byte.
    answer_request_contract = {
        "decoding": manifests["answer"].get("decoding"),
        "request_keys": sorted(request_keys),
        "canonical_json": (
            "json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, "
            "separators=(',', ':')).encode('utf-8') — loop-v0/00_control/code/keys.py:18-25. "
            "raw/*.json request_sha256 is sha256 over that encoding of `request` "
            "(slot_adapter.py:1470)"
        ),
        "response_sha256_note": (
            "response_sha256 is sha256 over the server's raw HTTP response bytes "
            "(slot_adapter.py:1475), not over the stored `response` object; it cannot be "
            "recomputed from this fixture"
        ),
        "answering_model": manifests["answer"].get("counts", {}).get("actual_models"),
        "prompt_template": "profiles/grounded_answer_v1.md",
    }

    # ---- 7. schemas ---------------------------------------------------------
    schemas_src = loop_root / "00_control/schemas"
    (out_dir / "schemas").mkdir(parents=True, exist_ok=True)
    schema_shas: dict[str, str] = {}
    for name in SCHEMA_FILES:
        source = schemas_src / name
        if not source.is_file():
            raise BuildError(f"schema missing: {source}")
        schema_shas[name] = record(source)
        shutil.copy2(source, out_dir / "schemas" / name)
        (out_dir / "schemas" / name).chmod(0o644)

    # ---- 8. profiles + prompt + route --------------------------------------
    route_src = loop_root / "04_rag/routes/r1.yaml"
    route = yaml.safe_load(route_src.read_text(encoding="utf-8"))
    record(route_src)
    (out_dir / "profiles").mkdir(parents=True, exist_ok=True)
    shutil.copy2(route_src, out_dir / "profiles/r1.route.yaml")
    (out_dir / "profiles/r1.route.yaml").chmod(0o644)

    profile_shas: dict[str, dict[str, str]] = {}
    ordered: list[str] = []
    for slot in SLOTS:
        rel = route["profiles"][slot]
        source = loop_root / rel
        if not source.is_file():
            raise BuildError(f"route profile missing: {source}")
        digest = record(source)
        ordered.append(digest)
        profile_shas[slot] = {"path": rel, "file": source.name, "sha256": digest}
        shutil.copy2(source, out_dir / "profiles" / source.name)
        (out_dir / "profiles" / source.name).chmod(0o644)

    # Route id recipe, cited: loop-v0/04_rag/run_route.py:36-49 — for each slot in
    # SLOTS, hashes.append(sha256_file(profile_path)); then
    # actual = sha256("".join(hashes).encode("ascii")).hexdigest()  (line 48).
    recomputed = hashlib.sha256("".join(ordered).encode("ascii")).hexdigest()
    if recomputed != JUDGED_ROUTE_ID:
        raise BuildError(f"recomputed route id {recomputed} != r1 {JUDGED_ROUTE_ID}")
    if route.get("route_id") != JUDGED_ROUTE_ID or release["route_id"] != JUDGED_ROUTE_ID:
        raise BuildError("route id declared by r1.yaml or by the store release row is not r1")

    prompt_src = loop_root / "04_rag/prompts/grounded_answer_v1.md"
    if not prompt_src.is_file():
        raise BuildError(f"prompt template missing: {prompt_src}")
    prompt_sha = record(prompt_src)
    shutil.copy2(prompt_src, out_dir / "profiles/grounded_answer_v1.md")
    (out_dir / "profiles/grounded_answer_v1.md").chmod(0o644)

    # ---- 9. MANIFEST --------------------------------------------------------
    outputs: dict[str, dict[str, Any]] = {}
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            rel = path.relative_to(out_dir).as_posix()
            outputs[rel] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}

    manifest = {
        "fixture_schema_version": "ceres-fixture-v1",
        "builder": "tools/build_fixtures.py",
        "builder_version": BUILDER_VERSION,
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "interpreter": sys.executable,
        "numpy_version": np.__version__,
        "loop_root": str(loop_root),
        "store": {
            "dsn": dsn,
            "schema": schema,
            "release_attempt_id": release_id,
            "route_id": release["route_id"],
            "row_count": release["row_count"],
            "loaded_by": release["loaded_by"],
            "embed_attempt_dir": release["embed_attempt_dir"],
            "release_manifest_counts": release["manifest"].get("counts"),
            "release_manifest_chain_notes": release["manifest"].get("chain_notes"),
        },
        "attempt_chain": {
            slot: {
                "dir": chain[slot]["dir"],
                "attempt": Path(chain[slot]["dir"]).name,
                "manifest_sha256": sha256_file(Path(chain[slot]["dir"]) / "MANIFEST.json"),
                "profile_id": manifests[slot].get("profile_id"),
                "profile_sha256": manifests[slot].get("profile_sha256"),
                "replayed_from_attempt": manifests[slot].get("replayed_from_attempt"),
                "input_manifest_path": manifests[slot].get("input_manifest_path"),
                "other_attempts_with_the_same_input": chain[slot]["candidates"],
                # True when the slot's input_manifest_sha256s names the upstream MANIFEST;
                # None for embed (chain root: verified against the release's load manifest instead).
                "input_hash_verified": chain[slot].get("input_hash_verified"),
            }
            for slot in ("embed", "index", "retrieve", "rerank", "answer")
        },
        "route": {
            "route_id_recomputed": recomputed,
            "route_id_expected": JUDGED_ROUTE_ID,
            "recipe": (
                "sha256(ascii concat of the seven profile file sha256s in SLOTS order) — "
                "loop-v0/04_rag/run_route.py:36-49; SLOTS at loop-v0/04_rag/common.py:39"
            ),
            "slot_order": list(SLOTS),
            "profile_sha256s": profile_shas,
            "prompt_template_sha256": prompt_sha,
        },
        "answer_request_contract": answer_request_contract,
        "questions": {
            "path": str(questions_path),
            "sha256": questions_sha,
            "count": len(questions),
            "question_ids": question_ids,
        },
        "counts": {
            "items": len(items),
            "embedding_dim": int(embeddings.shape[1]),
            "payload_rows_needing_b64": release["payload_undecodable_rows"],
            "payload_bytes_total": sum(len(i["payload"].encode("utf-8")) if i.get("payload") is not None else 0 for i in items),
            "items_l2_norm_min": float(item_norms.min()),
            "items_l2_norm_max": float(item_norms.max()),
            "items_unit_norm_atol_1e_5": bool(np.allclose(item_norms, 1.0, atol=1e-5)),
            "query_l2_norm_min": float(query_norms.min()),
            "query_l2_norm_max": float(query_norms.max()),
            "queries_unit_norm_atol_1e_5": bool(np.allclose(query_norms, 1.0, atol=1e-5)),
            "hits": len(hits),
            "reranked": len(reranked),
            "packets": len(packets),
            "answers": len(answers),
            "raw_bodies": len(raw_files),
            "index_attempt_recorded_pool_depth": index_pool_depth,
        },
        "schema_sha256s": schema_shas,
        "inputs": dict(sorted(inputs.items())),
        "notes": notes,
        "outputs": outputs,
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def record_and_return(record, path: Path) -> Path:
    record(path)
    return path


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="tests/fixtures", help="output directory (default: tests/fixtures)")
    parser.add_argument("--loop-root", default=None, help="DDS loop-v0 root (default: $DDS_LOOP_ROOT or repo-relative)")
    parser.add_argument("--store-dsn", default=os.environ.get("CERES_STORE_DSN", DEFAULT_DSN))
    parser.add_argument("--store-schema", default=os.environ.get("CERES_STORE_SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--force", action="store_true", help="overwrite a non-empty output directory")
    args = parser.parse_args(argv)

    package_root = here.parents[1]
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (package_root / out_dir).resolve()
    loop_root = Path(args.loop_root).resolve() if args.loop_root else repo_default_loop_root(here)
    if not (loop_root / "04_rag").is_dir():
        print(f"error: loop root does not look like loop-v0: {loop_root}", file=sys.stderr)
        return 2

    try:
        manifest = build(out_dir, loop_root, args.store_dsn, args.store_schema, args.force)
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except AssertionError as exc:
        print(f"assertion failed: {exc}", file=sys.stderr)
        return 3

    total = sum(o["size_bytes"] for o in manifest["outputs"].values())
    print(f"fixture written to {out_dir}")
    print(f"  release      {manifest['store']['release_attempt_id']}")
    print(f"  chain        " + " -> ".join(
        manifest["attempt_chain"][s]["attempt"] for s in ("embed", "index", "retrieve", "rerank", "answer")
    ))
    print(f"  items        {manifest['counts']['items']} x {manifest['counts']['embedding_dim']}")
    print(f"  route id     {manifest['route']['route_id_recomputed']} (recomputed, matches r1)")
    print(f"  files        {len(manifest['outputs'])}, {total} bytes ({total / 1e6:.2f} MB)")
    for note in manifest["notes"]:
        print(f"  note         {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
