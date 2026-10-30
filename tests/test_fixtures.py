"""The fixture corpus and its golden outputs are self-consistent.

Every check here runs offline: no Postgres, no GPU, no model, no network. The
only third-party imports are numpy and jsonschema, both already in the loop's
control venv, so the suite runs under either interpreter.

Run it with pytest when pytest is installed::

    python -m pytest tests/test_fixtures.py -q

The loop's control venv has no pytest, so the module also runs standalone::

    <loop>/00_control/.venv/bin/python tests/test_fixtures.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import pytest
import re
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MANIFEST = json.loads((FIXTURES / "MANIFEST.json").read_text(encoding="utf-8"))

JUDGED_ROUTE_ID = "73c2e7911d632ad5602bb72f869d5be81a920343fdca9c204fe2a7d7a6bf7315"
SLOTS = ("chunk", "represent", "embed", "index", "retrieve", "rerank", "answer")
MAX_FIXTURE_BYTES = 10 * 1024 * 1024


def _read_jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value) -> bytes:
    """loop-v0/00_control/code/keys.py:18-25 canonical_json_bytes."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


ITEMS = _read_jsonl("items.jsonl")
QUESTIONS = _read_jsonl("questions.jsonl")
HITS = _read_jsonl("golden/hits-top10.jsonl")
RERANKED = _read_jsonl("golden/reranked-top5.jsonl")
PACKETS = _read_jsonl("golden/packets.jsonl")
ANSWERS = _read_jsonl("golden/answers.jsonl")
CANDIDATES = _read_jsonl("golden/candidates-top50.jsonl")


# --------------------------------------------------------------- manifest hashes


def test_manifest_output_sha256s_hold():
    """Every file the MANIFEST lists still hashes to the recorded digest."""
    for rel, entry in MANIFEST["outputs"].items():
        path = FIXTURES / rel
        assert path.is_file(), f"missing fixture file: {rel}"
        assert _sha256(path) == entry["sha256"], f"sha256 changed: {rel}"
        assert path.stat().st_size == entry["size_bytes"], f"size changed: {rel}"


def test_manifest_lists_every_file_on_disk():
    on_disk = {
        p.relative_to(FIXTURES).as_posix()
        for p in FIXTURES.rglob("*")
        if p.is_file() and p.name not in ("MANIFEST.json", "README.md")
    }
    assert on_disk == set(MANIFEST["outputs"]), "fixture directory and MANIFEST disagree"


def test_fixture_stays_under_ten_megabytes():
    total = sum(e["size_bytes"] for e in MANIFEST["outputs"].values())
    assert total < MAX_FIXTURE_BYTES, f"fixture is {total} bytes"


# Upstream inputs that changed AFTER the fixture was built, on purpose and on record. Each entry
# names the digest the fixture recorded, the digest the file has now, and why. Any other change —
# or this file changing again — still fails the test below. The fixture itself is not
# rebuilt here (tests/fixtures/** is read-only here); the change is additive for the packet
# schema and does not touch any fixture packet (none carries `evidence_basis`).
KNOWN_INPUT_DRIFT = {
    "/path/to/dds/loop-v0/00_control/"
    "schemas/rag-answer-packets.schema.json": {
        "recorded": "46dd89c98ee8f4fca669f229e562e47da666bdabb0c75860cd0102f8b87a60ef",
        "now": "261ef189fae42284edca0cdbc54c4df929680203ea013299d2fccc60c51b1c5a",
        "why": "the engine added the optional context field `evidence_basis` (and later dropped an internal tag from one description)",
    },
}


def test_recorded_input_sha256s_still_hold_when_the_source_is_present():
    """Bug caught: an upstream input of the fixture changed and nobody recorded it.

    Inputs live outside the package; check the ones this box can still see. A change listed in
    KNOWN_INPUT_DRIFT passes only if the file has exactly the recorded new digest.
    """
    checked = 0
    for path_str, digest in MANIFEST["inputs"].items():
        path = Path(path_str)
        if not path.is_file():
            continue
        actual = _sha256(path)
        drift = KNOWN_INPUT_DRIFT.get(path_str)
        if drift is not None:
            assert drift["recorded"] == digest, f"the drift record is stale for {path}"
            assert actual == drift["now"], f"input changed again since the recorded drift: {path}"
        else:
            assert actual == digest, f"input changed since the build: {path}"
        checked += 1
    if checked == 0:
        pytest.skip("none of the recorded upstream inputs is present on this machine; nothing was checked")


# ------------------------------------------------------------------------- items


def test_item_count_is_the_active_release_row_count():
    # No literal row count: the store is reloaded by `dds load` (536 now, 426 before);
    # the fixture must equal what its MANIFEST recorded from the active release at build time.
    assert len(ITEMS) == MANIFEST["counts"]["items"]
    assert len(ITEMS) == MANIFEST["store"]["row_count"]
    assert len({i["chunk_id"] for i in ITEMS}) == len(ITEMS)
    assert {i["release_attempt_id"] for i in ITEMS} == {MANIFEST["store"]["release_attempt_id"]}


def test_items_are_sorted_by_chunk_id():
    ids = [i["chunk_id"] for i in ITEMS]
    assert ids == sorted(ids), "items.jsonl must be chunk_id ascending; items.npy shares the order"


def test_payload_sha256s_hold():
    b64_rows = 0
    for item in ITEMS:
        if item.get("payload") is None:
            payload = base64.b64decode(item["payload_b64"])
            b64_rows += 1
        else:
            payload = item["payload"].encode("utf-8")
        assert hashlib.sha256(payload).hexdigest() == item["payload_sha256"], item["chunk_id"]
    assert b64_rows == MANIFEST["counts"]["payload_rows_needing_b64"]


def test_every_item_carries_the_columns_the_store_contract_names():
    required = {
        "release_attempt_id", "chunk_id", "parent_id", "work_id", "membership", "rights_state",
        "eligibility_state", "evidence_eligible", "payload_sha256", "locator_ids", "lineage_ids",
        "excluded_reasons", "representation_id",
    }
    for item in ITEMS:
        assert required <= set(item), f"{item['chunk_id']} is missing {required - set(item)}"
        assert "embedding" not in item, "embeddings belong in items.npy, not items.jsonl"


# -------------------------------------------------------------------- embeddings


def test_item_embeddings_are_unit_norm_float32():
    vectors = np.load(FIXTURES / "items.npy", allow_pickle=False)
    assert vectors.dtype == np.float32
    assert vectors.shape == (len(ITEMS), MANIFEST["counts"]["embedding_dim"])
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"norms span [{norms.min()}, {norms.max()}]"


def test_query_vectors_match_the_questions():
    vectors = np.load(FIXTURES / "query-vectors.npy", allow_pickle=False)
    assert vectors.dtype == np.float32
    assert vectors.shape == (len(QUESTIONS), MANIFEST["counts"]["embedding_dim"])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert [q["question_id"] for q in QUESTIONS] == MANIFEST["questions"]["question_ids"]


def test_question_ids_are_the_sha256_of_their_text():
    """The loop keys a question by sha256 of its text (embed question-rows.jsonl)."""
    for question in QUESTIONS:
        assert hashlib.sha256(question["text"].encode("utf-8")).hexdigest() == question["question_id"]


# ----------------------------------------------------------------- golden shapes


def _validate(rows: list[dict], schema_name: str) -> None:
    schema = json.loads((FIXTURES / "schemas" / schema_name).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    for index, row in enumerate(rows):
        errors = sorted(validator.iter_errors(row), key=str)
        assert not errors, f"{schema_name} row {index}: {errors[0].message}"


def test_golden_hits_validate():
    _validate(HITS, "hits.schema.json")
    assert len(HITS) == 30
    per_question = {q["question_id"]: [h for h in HITS if h["question_id"] == q["question_id"]] for q in QUESTIONS}
    for question_id, rows in per_question.items():
        assert len(rows) == 10, question_id
        assert [r["rank"] for r in rows] == list(range(1, 11))
        assert len({r["parent_id"] for r in rows}) == 10, "r1 dedups by parent_id (keep-best)"


def test_golden_reranked_validate():
    _validate(RERANKED, "reranked-hits.schema.json")
    assert len(RERANKED) == 15
    for question in QUESTIONS:
        rows = [r for r in RERANKED if r["question_id"] == question["question_id"]]
        assert len(rows) == 5
        assert [r["rank"] for r in rows] == [1, 2, 3, 4, 5]


def test_golden_packets_validate():
    _validate(PACKETS, "rag-answer-packets.schema.json")
    assert len(PACKETS) == 3
    for packet in PACKETS:
        assert len(packet["contexts"]) == 5
        for context in packet["contexts"]:
            assert hashlib.sha256(context["evidence_text"].encode("utf-8")).hexdigest() == context["evidence_text_sha256"]


def test_golden_answers_validate():
    _validate(ANSWERS, "answers.schema.json")
    assert len(ANSWERS) == 3
    assert {a["question_id"] for a in ANSWERS} == {q["question_id"] for q in QUESTIONS}


# ------------------------------------------------------------- golden containment


def test_golden_hits_are_a_subset_of_the_candidate_pool():
    pools = {row["question_id"]: {c["chunk_id"] for c in row["candidates"]} for row in CANDIDATES}
    for hit in HITS:
        assert hit["chunk_id"] in pools[hit["question_id"]], (
            f"hit {hit['chunk_id']} for {hit['question_id']} is outside the top-50 candidate pool"
        )


def test_reranked_are_a_subset_of_the_hits():
    per_question = {q["question_id"]: {h["chunk_id"] for h in HITS if h["question_id"] == q["question_id"]} for q in QUESTIONS}
    for row in RERANKED:
        assert row["chunk_id"] in per_question[row["question_id"]]


def test_packet_contexts_come_from_the_reranked_five():
    per_question = {q["question_id"]: {r["chunk_id"] for r in RERANKED if r["question_id"] == q["question_id"]} for q in QUESTIONS}
    for packet in PACKETS:
        for context in packet["contexts"]:
            assert context["chunk_id"] in per_question[packet["question_id"]]


def test_candidate_pool_is_fifty_per_question_and_recomputes():
    """The builder's pool is an exact cosine ranking over items.npy — redo it here."""
    vectors = np.load(FIXTURES / "items.npy", allow_pickle=False).astype(np.float32)
    queries = np.load(FIXTURES / "query-vectors.npy", allow_pickle=False).astype(np.float32)
    vectors = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    queries = queries / np.clip(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12, None)
    matrix = queries @ vectors.T
    chunk_ids = np.asarray([i["chunk_id"] for i in ITEMS], dtype=object)
    by_question = {row["question_id"]: row for row in CANDIDATES}
    assert set(by_question) == {q["question_id"] for q in QUESTIONS}
    for qindex, question in enumerate(QUESTIONS):
        row = by_question[question["question_id"]]
        assert len(row["candidates"]) == 50
        scores = matrix[qindex]
        order = np.lexsort((chunk_ids, -scores))[:50]
        expected = [
            {
                "question_id": question["question_id"],
                "rank": rank,
                "chunk_id": ITEMS[int(j)]["chunk_id"],
                "parent_id": ITEMS[int(j)]["parent_id"],
                "score": round(float(scores[int(j)]), 10),
            }
            for rank, j in enumerate(order, start=1)
        ]
        assert row["candidates"] == expected, f"candidate pool does not recompute for {question['question_id']}"


# --------------------------------------------------------------------- route id


def test_route_id_recomputes_from_the_copied_profiles():
    """Recipe: loop-v0/04_rag/run_route.py:36-49 — sha256 of the seven profile sha256s."""
    route = FIXTURES / "profiles/r1.route.yaml"
    declared = {}
    for line in route.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        for slot in SLOTS:
            if stripped.startswith(f"{slot}:"):
                declared[slot] = stripped.split(":", 1)[1].strip()
    assert set(declared) == set(SLOTS), f"r1.route.yaml did not name all seven slots: {sorted(declared)}"

    digests = []
    for slot in SLOTS:
        name = Path(declared[slot]).name
        path = FIXTURES / "profiles" / name
        assert path.is_file(), f"profile not copied into the fixture: {name}"
        digest = _sha256(path)
        assert digest == MANIFEST["route"]["profile_sha256s"][slot]["sha256"]
        digests.append(digest)

    recomputed = hashlib.sha256("".join(digests).encode("ascii")).hexdigest()
    assert recomputed == JUDGED_ROUTE_ID
    assert recomputed == MANIFEST["route"]["route_id_recomputed"]
    assert MANIFEST["store"]["route_id"] == JUDGED_ROUTE_ID


def test_prompt_template_hash_holds():
    assert _sha256(FIXTURES / "profiles/grounded_answer_v1.md") == MANIFEST["route"]["prompt_template_sha256"]


# ------------------------------------------------------- raw LLM request bodies


def test_raw_request_bodies_hash_and_keep_the_r1_request_shape():
    contract = MANIFEST["answer_request_contract"]
    decoding = contract["decoding"]
    seen: set[str] = set()
    for path in sorted((FIXTURES / "golden/raw").glob("*.json")):
        body = json.loads(path.read_text(encoding="utf-8"))
        seen.add(body["question_id"])
        request = body["request"]
        assert hashlib.sha256(_canonical(request)).hexdigest() == body["request_sha256"], path.name
        # response_sha256 is sha256 over the server's raw HTTP bytes
        # (slot_adapter.py:1475), not over the reparsed object, so it cannot be
        # recomputed from the stored `response`.
        assert len(body["response_sha256"]) == 64
        assert body["response"]["choices"][0]["message"]["content"]
        assert sorted(request) == contract["request_keys"], path.name
        assert request["chat_template_kwargs"] == {"enable_thinking": decoding["enable_thinking"]}
        for field in ("max_tokens", "temperature", "top_p", "top_k", "presence_penalty"):
            assert request[field] == decoding[field], f"{path.name}: {field}"
        assert [m["role"] for m in request["messages"]] == ["system", "user"]
        assert isinstance(request["seed"], int)
        assert request["stream"] is False
    assert seen == {q["question_id"] for q in QUESTIONS}


def test_attempt_chain_is_the_one_the_active_release_pins():
    chain = MANIFEST["attempt_chain"]
    assert chain["embed"]["dir"] == MANIFEST["store"]["embed_attempt_dir"]
    # No literal attempt numbers: the chain is whatever the active release's load manifest pins
    # (its embed attempt) plus the downstream attempts that consumed it, one per slot, all on r1.
    for slot in ("embed", "index", "retrieve", "rerank", "answer"):
        assert re.fullmatch(r"r1-" + slot + r"-attempt-\d{4}", chain[slot]["attempt"]), slot
    assert chain["index"]["input_manifest_path"].endswith(chain["embed"]["attempt"] + "/MANIFEST.json")
    assert chain["retrieve"]["input_manifest_path"].endswith(chain["index"]["attempt"] + "/MANIFEST.json")
    assert chain["rerank"]["input_manifest_path"].endswith(chain["retrieve"]["attempt"] + "/MANIFEST.json")
    assert chain["answer"]["input_manifest_path"].endswith(chain["rerank"]["attempt"] + "/MANIFEST.json")
    assert chain["retrieve"]["profile_id"] == "dense-top10-v2"
    assert chain["rerank"]["profile_id"] == "qwen3-4b-rerank-v1"
    assert chain["answer"]["profile_id"] == "grounded-qwen38-v1"


if __name__ == "__main__":  # standalone runner: the loop control venv has no pytest
    import traceback

    tests = sorted((name, obj) for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj))
    failed = 0
    for name, func in tests:
        try:
            func()
        except Exception:  # noqa: BLE001 - a test runner reports, it does not raise
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
