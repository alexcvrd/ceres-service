"""Parity against the reference pipeline's own r1 goldens (see the parity amendment).

The question these tests answer: does the Ceres pipeline, fed the same corpus and the same query
vectors, produce what the judged route produced? Per question, with exact coverage counts from
`MANIFEST.json` (no comparison may pass on an empty set):

  1. candidate pool — our exact scan against `golden/candidates-top50.jsonl` (`computed` pool);
  2. post-dedup top-10 — against `golden/hits-top10.jsonl` (chunk ids AND ranks);
  3. packet contexts E1..E5 — against `golden/packets.jsonl`, every field, with the inherited
     warnings divergence written out per slot in `INHERITED_WARNINGS`;
  4. the answer run replayed — the request bodies captured from `pipeline.ask` itself against
     `golden/raw/*.json` (canonical-JSON `request_sha256`), and the resulting answer rows against
     `golden/answers.jsonl`, failed attempts included.

What this does NOT verify: the reranker. `FakeModels` is given the reference pipeline's own rerank scores, so the
top-5 is predetermined for the golden winners — packet plumbing is tested, the model is not.

Settings come from the amendment, with one simplification: the shipped default mode
`rnd` already serves every rights state the fixture carries, so the parity run no longer overrides
the rights list. `test_the_fixture_pool_is_fully_served_in_rnd_mode` below is what makes that safe
— it asserts the fixture holds no rights state outside the rnd set (in particular no `unknown`,
which no mode serves). Everything else is unchanged: the fixture's stored `evidence_eligible`,
fixture query vectors and golden rerank scores in `FakeModels`, golden answers in `FakeLLM`. A
mismatch fails; nothing here is loosened.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from ceres import pipeline
from ceres.config import SERVE_RIGHTS_BY_MODE
from ceres.contracts import AskRequest
from ceres.llm import FakeLLM, build_request_body, goldens_from_fixture, stable_seed
from ceres.models import FakeModels
from ceres.store import FixtureStore

from conftest import FIXTURES, read_jsonl

# No skip marker: the fixture is checked in, and a parity gate that skips when its evidence is
# missing reports green on nothing (a review required it). A missing fixture fails here instead.

# The rights list of the shipped default mode. It replaces the amendment's hand-written wide
# list: rnd covers every rights state the fixture holds, so the rights predicate is as inert here as
# the amendment intended, and the parity run now exercises the rights list the service really uses.
RND_RIGHTS = list(SERVE_RIGHTS_BY_MODE["rnd"])


def canonical_sha256(value) -> str:
    """The reference pipeline's canonical JSON digest (`keys.py:19`, `slot_adapter.py:1470` of the reference pipeline)."""
    blob = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@pytest.fixture(scope="module")
def golden():
    questions = read_jsonl(FIXTURES / "questions.jsonl")
    items = read_jsonl(FIXTURES / "items.jsonl")
    payloads = {row["chunk_id"]: row.get("payload", "") for row in items}
    reranked = read_jsonl(FIXTURES / "golden" / "reranked-top5.jsonl")
    candidates = {row["question_id"]: row for row in read_jsonl(FIXTURES / "golden" / "candidates-top50.jsonl")}
    hits: dict[str, list[dict]] = {}
    for row in read_jsonl(FIXTURES / "golden" / "hits-top10.jsonl"):
        hits.setdefault(row["question_id"], []).append(row)
    packets = {row["question_id"]: row for row in read_jsonl(FIXTURES / "golden" / "packets.jsonl")}
    answers = read_jsonl(FIXTURES / "golden" / "answers.jsonl")
    raw = {}
    for path in sorted((FIXTURES / "golden" / "raw").glob("*.json")):
        event = json.loads(path.read_text(encoding="utf-8"))
        raw.setdefault(event["question_id"], {})[int(event["attempt"])] = event
    return {
        "questions": questions, "items": items, "payloads": payloads, "reranked": reranked,
        "candidates": candidates, "hits": hits, "packets": packets, "answers": answers, "raw": raw,
        "manifest": json.loads((FIXTURES / "MANIFEST.json").read_text(encoding="utf-8")),
    }


@pytest.fixture(scope="module")
def stack(golden):
    """The amendment's stack: fixture store, fixture query vectors, golden rerank scores."""
    store = FixtureStore.from_dir(FIXTURES)
    vectors = np.load(FIXTURES / "query-vectors.npy").astype(np.float32)
    by_id = {row["question_id"]: vectors[index] for index, row in enumerate(golden["questions"])}
    by_text = {row["text"].strip(): row["question_id"] for row in golden["questions"]}
    texts = {row["question_id"]: row["text"] for row in golden["questions"]}
    scores = {(texts[row["question_id"]].strip(), golden["payloads"][row["chunk_id"]].strip()):
              float(row["score"])
              for row in golden["reranked"] if row["chunk_id"] in golden["payloads"]}
    models = FakeModels(by_id, by_text, dim=vectors.shape[1], rerank_scores=scores)
    return store, models


def question_ids(golden) -> list[str]:
    return [row["question_id"] for row in golden["questions"]]


# ---------------------------------------------------------------- 0. coverage and the rights premise

def test_the_fixture_covers_what_the_manifest_says(golden):
    """Exact, non-empty coverage from MANIFEST.json — every later comparison depends on it.

    Current fixture: 536 items, 3 questions (2 en, 1 ro), 3 packets × 5 contexts, 6 raw attempts,
    one VALIDATED golden answer. The older 426-item / 4-attempt description is history.
    """
    counts = golden["manifest"]["counts"]
    assert len(golden["items"]) == counts["items"] == 536
    assert [q["question_id"] for q in golden["questions"]] == \
        golden["manifest"]["questions"]["question_ids"]
    assert len(golden["questions"]) == golden["manifest"]["questions"]["count"] == 3
    assert len(golden["packets"]) == counts["packets"] == 3
    assert len(golden["answers"]) == counts["answers"] == 3
    assert sum(len(a) for a in golden["raw"].values()) == counts["raw_bodies"] == 6
    assert sum(len(h) for h in golden["hits"].values()) == counts["hits"] == 30
    assert len(golden["reranked"]) == counts["reranked"] == 15
    assert len(golden["candidates"]) == 3
    assert sum(row["state"] == "VALIDATED" for row in golden["answers"]) == 1



def test_the_fixture_pool_is_fully_served_in_rnd_mode(golden):
    """What licenses the parity run to drop the amendment's hand-written rights override.

    The comparison target is the reference pipeline, which had no rights predicate at all. That is only
    reproducible if the mode under test withholds nothing from this fixture. So: every row's
    `rights_state` must be inside the rnd set. A row with `unknown` (which NO mode serves) or any
    new state would silently shrink the pool and turn a rights regression into a parity failure
    nobody could read — this test names the cause instead.
    """
    states = {row.get("rights_state") for row in golden["items"]}
    assert states, "the fixture has no rows"
    assert states <= set(RND_RIGHTS), sorted(states - set(RND_RIGHTS))
    assert "unknown" not in states


# ---------------------------------------------------------------- 1. candidate pool

def test_candidate_pool_matches_the_golden_exact_scan(golden, stack):
    """Our pgvector-shaped scan (in numpy here) reproduces the reference computed top-50, ids and order."""
    store, models = stack
    release = store.active_release()
    compared = 0
    for row in golden["questions"]:
        qvec = models.embed_query(row["text"], row["language"])
        items = pipeline.retrieve(store, qvec, release, serve_rights=RND_RIGHTS, limit=50)
        expected = golden["candidates"][row["question_id"]]["candidates"]
        assert [item.chunk_id for item in items] == [c["chunk_id"] for c in expected], row["question_id"]
        for item, reference in zip(items, expected):
            assert item.score == pytest.approx(reference["score"], abs=1e-6)
            assert item.parent_id == reference["parent_id"]
        compared += len(expected)
    assert compared == 3 * 50            # three full pools: an empty pool must not pass


# ---------------------------------------------------------------- 2. post-dedup top-10

def test_dedup_top10_matches_the_golden_hits(golden, stack):
    """`dedup` reproduces the reference pipeline's `hits.jsonl` — same chunk ids at the same ranks."""
    store, models = stack
    release = store.active_release()
    compared = 0
    for row in golden["questions"]:
        qvec = models.embed_query(row["text"], row["language"])
        items = pipeline.retrieve(store, qvec, release, serve_rights=RND_RIGHTS, limit=50)
        texts = {item.chunk_id: item.payload for item in items}
        kept = pipeline.dedup(items, texts, pipeline.NEAR_DUPLICATE_RULE)
        expected = sorted(golden["hits"][row["question_id"]], key=lambda hit: hit["rank"])
        assert [(item.chunk_id, item.rank) for item in kept] == \
               [(hit["chunk_id"], hit["rank"]) for hit in expected], row["question_id"]
        compared += len(expected)
    assert compared == golden["manifest"]["counts"]["hits"] == 30


# ---------------------------------------------------------------- 3. packets

# The one intended packet divergence from the reference pipeline, written out per question and per evidence slot
# (each divergence is recorded explicitly, not as "any of two reasons"). The store's
# `excluded_reasons` is the union of a chunk's own reasons and its parent's (the store contract's
# write-side rule), while the reference pipeline's packet carried the chunk record's own reasons only — all
# empty in these goldens. So the serving packet tells a judge MORE than the reference pipeline's did. Each entry
# below is the extra set a context must carry; a slot not listed must carry none. Derived from
# `tests/fixtures/items.jsonl` and checked against it by
# `test_the_inherited_warnings_table_is_what_the_fixture_says`.
INHERITED_WARNINGS: dict[str, dict[str, list[str]]] = {
    "32d4eeba5eaef23accb8c4c68236bcf3fbfe54f841f492bd212f552e0aadd885": {
        "E1": ["table_unvalidated"], "E2": ["table_unvalidated"], "E3": ["table_unvalidated"],
        "E4": ["table_unvalidated"], "E5": ["table_unvalidated"],
    },
    "dfa10ae6aa38da81b0d5c7de0d9a6394ac4cf3262daf6196c5d2b5adb8e46049": {
        "E1": ["non_unique_locator"],
    },
    "95d315efc4ecdbea6db2b9f02e9dcb9d9e78447a5b2cc285d75e1bbad58ee3b7": {
        "E3": ["non_unique_locator"], "E5": ["table_unvalidated"],
    },
}


def test_the_inherited_warnings_table_is_what_the_fixture_says(golden):
    items = {row["chunk_id"]: row for row in golden["items"]}
    derived: dict[str, dict[str, list[str]]] = {}
    for question_id, packet in golden["packets"].items():
        for context in packet["contexts"]:
            extra = sorted(set(items[context["chunk_id"]]["excluded_reasons"])
                           - set(context["warnings"]))
            if extra:
                derived.setdefault(question_id, {})[context["evidence_id"]] = extra
    assert derived == INHERITED_WARNINGS


def test_packets_match_the_golden_packets(golden, stack, config):
    """E1..E5 equal the reference pipeline's packet contexts field by field, warnings by the table above."""
    store, models = stack
    deps = pipeline.Deps.build(config, store, models,
                               FakeLLM(goldens_from_fixture(list(golden["packets"].values()),
                                                            golden["answers"])))
    # No rights override: the shipped `rnd` default already serves the whole fixture.
    assert deps.config.store.serve_rights == tuple(RND_RIGHTS)
    compared = 0
    for row in golden["questions"]:
        request = AskRequest(question=row["text"], lang=row["language"],
                             question_id=row["question_id"])
        response = pipeline.ask(deps, request)
        expected = golden["packets"][row["question_id"]]
        for key in ("question_id", "language", "packet_id", "question"):
            assert response.packet[key] == expected[key], (row["question_id"], key)
        ours = response.packet["contexts"]
        theirs = expected["contexts"]
        assert [c["chunk_id"] for c in ours] == [c["chunk_id"] for c in theirs], row["question_id"]
        extras = INHERITED_WARNINGS.get(row["question_id"], {})
        for mine, reference in zip(ours, theirs):
            assert set(mine) == set(reference), mine["chunk_id"]
            for key in ("evidence_id", "chunk_id", "parent_id", "evidence_text",
                        "evidence_text_sha256", "locator_ids", "lineage_ids"):
                assert mine[key] == reference[key], (mine["chunk_id"], key)
            assert sorted(mine["warnings"]) == \
                sorted(set(reference["warnings"]) | set(extras.get(mine["evidence_id"], []))), \
                (row["question_id"], mine["evidence_id"])
            compared += 1
    # Exact coverage: 3 questions × 5 contexts — an empty comparison must not pass.
    assert compared == sum(len(p["contexts"]) for p in golden["packets"].values()) == 15


# ---------------------------------------------------------------- 4. the answer run, replayed

class ReplayLLM:
    """Returns the reference pipeline's recorded response for each request the pipeline actually assembles.

    The seed identifies (question, attempt) — `stable_seed` is part of the route — so the replay
    finds the golden raw event from the body it is handed, and records that body for comparison.
    Failed attempts are replayed as well as successful ones: the golden run has four structural
    failures and one validated answer across six attempts.
    """

    MODEL_ID = "replay"

    def __init__(self, raw_by_question):
        self.by_seed = {stable_seed(qid, attempt): event
                        for qid, attempts in raw_by_question.items()
                        for attempt, event in attempts.items()}
        self.bodies: list[dict] = []

    def chat(self, request_body):
        self.bodies.append(request_body)
        event = self.by_seed[request_body["seed"]]
        raw = json.dumps(event["response"], ensure_ascii=False).encode("utf-8")
        return event["response"], raw

    def health(self):
        return "ready", "replay", self.MODEL_ID


def test_the_assembled_pipeline_replays_the_golden_answer_run(golden, stack, config):
    """Request bodies captured from `pipeline.ask` equal the reference pipeline's, and so do the answer rows.

    This replaces the earlier comparison that built bodies from the golden packets: here the body is
    the one the service would really send, so a packet or prompt drift fails the digest.
    """
    store, models = stack
    llm = ReplayLLM(golden["raw"])
    deps = pipeline.Deps.build(config, store, models, llm)
    # The reference pipeline's rows, unchanged: r3 reverts the one r2 change (conftest R2_ONLY_GOLDEN_RESCUE).
    answers = {row["question_id"]: row for row in golden["answers"]}
    for row in golden["questions"]:
        response = pipeline.ask(deps, AskRequest(question=row["text"], lang=row["language"],
                                                 question_id=row["question_id"]))
        assert response.answer == answers[row["question_id"]], row["question_id"]
    # The frozen reference validator still agrees with the reference pipeline on every recorded attempt (r1 parity).
    for question_id, attempts in golden["raw"].items():
        for event in attempts.values():
            candidate = json.loads(event["response"]["choices"][0]["message"]["content"])
            assert pipeline.structural_answer(candidate, golden["packets"][question_id]) == \
                event["structural_errors"], (question_id, event["attempt"])
    # Every recorded attempt was requested exactly once (r3 skips none), and each body is the reference pipeline's
    # to the digest.
    assert sum(len(a) for a in golden["raw"].values()) == 6
    assert len(llm.bodies) == 6
    assert sorted((llm.by_seed[b["seed"]]["question_id"], llm.by_seed[b["seed"]]["attempt"]) for b in llm.bodies) == \
        sorted((qid, attempt) for qid, attempts in golden["raw"].items() for attempt in attempts)
    for body in llm.bodies:
        event = llm.by_seed[body["seed"]]
        assert canonical_sha256(body) == event["request_sha256"], (event["question_id"],
                                                                  event["attempt"])
    states = sorted(row["state"] for row in golden["answers"])
    assert states == ["STRUCTURE_FAILED", "STRUCTURE_FAILED", "VALIDATED"]


def test_request_body_matches_the_golden_raw_events(golden):
    """The body builder alone, per field, against every recorded attempt (6 of 6)."""
    import yaml

    profile = yaml.safe_load((FIXTURES / "profiles" / "grounded-qwen38-v1.yaml").read_text(encoding="utf-8"))
    prompt = (FIXTURES / "profiles" / "grounded_answer_v1.md").read_text(encoding="utf-8")
    compared = 0
    for question_id, attempts in golden["raw"].items():
        packet = golden["packets"][question_id]
        for attempt, event in sorted(attempts.items()):
            ours = build_request_body(packet, profile, prompt, attempt)
            theirs = event["request"]
            assert set(ours) == set(theirs), (question_id, attempt)
            for field in ("messages", "chat_template_kwargs", "model", "temperature", "top_p",
                          "top_k", "presence_penalty", "max_tokens", "stream", "seed"):
                assert ours[field] == theirs[field], (question_id, attempt, field)
            assert ours["seed"] == stable_seed(question_id, attempt)
            assert canonical_sha256(ours) == event["request_sha256"], (question_id, attempt)
            compared += 1
    assert compared == golden["manifest"]["counts"]["raw_bodies"] == 6


# ---------------------------------------------------------------- the recipe itself

def test_route_id_recipe_reproduces_r1(golden):
    """sha256 over the seven profile sha256s in slot order gives r1's id (MANIFEST §route)."""
    from ceres.config import ROUTE_SLOT_ORDER, compute_route_id

    profiles = {slot: FIXTURES / "profiles" / golden["manifest"]["route"]["profile_sha256s"][slot]["file"]
                for slot in ROUTE_SLOT_ORDER}
    assert compute_route_id(profiles) == golden["manifest"]["route"]["route_id_expected"]
