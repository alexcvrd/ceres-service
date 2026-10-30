"""The fake stack: fixture stores, fake models and a fake answerer, importable from the package.

Three users need the same stack, and before this module two of them reached into `smoke_test.py` for it
(`ceres-api --fake` via a `sys.path` insert — a hidden coupling):
  * `smoke_test.py` proves the whole route over it;
  * `ceres-api --fake` serves it (so `--fake` serves exactly what the smoke test proves);
  * `tests/` build on it.

Two stores:
  * `build_inline_fixture()` — six crafted items and two questions with 8-dimension vectors, so
    that each dedup and predicate rule fires exactly once and the expected outcome is readable in
    the fixture itself;
  * `build_fixture_stack()` — the `tests/fixtures/` corpus (536 real items with the loop's goldens) when
    the checkout has it, else the inline fixture.

Nothing here touches a network, a GPU or Postgres.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import question_id_for
from .llm import FakeLLM, goldens_from_fixture
from .models import FakeModels
from .store import FixtureStore, Release

# The fixture corpus lives in the source checkout. An installed package without the checkout has
# no such folder, and `build_fixture_stack` then falls back to the inline fixture below.
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

EN_QUESTION = "What sowing density does the reference give for winter wheat?"
RO_QUESTION = "Ce densitate de semănat indică referința pentru grâul de toamnă?"

# The reference passage E1 is quoted from; long enough to pass `min_context_tokens: 21`.
_WHEAT = (
    "Winter wheat is sown at a density of 450 to 550 germinable grains per square metre on well "
    "prepared seedbeds, with the lower end of the range used after a good preceding crop and the "
    "upper end used for late sowing or a rough seedbed."
)
# Near-duplicate of _WHEAT: the same sentence with a clause changed. norm-v3 token Jaccard against
# _WHEAT is above the 0.6 threshold of `dense-top10-v2.yaml`, so dedup must drop it.
_WHEAT_NEAR = (
    "Winter wheat is sown at a density of 450 to 550 germinable grains per square metre on well "
    "prepared seedbeds, with the lower end of the range used after a good preceding crop and the "
    "upper end used for very late sowing."
)
_SIBLING = (
    "The sowing depth for winter wheat is three to five centimetres, deeper on light soils that "
    "dry quickly and shallower on heavy soils that crust after rain in the autumn months."
)
_TINY = "Sowing density table follows."              # 4 tokens → dropped as "tiny"
_INTERNAL = (
    "Internal research note: the cohort trial recorded emergence rates for eight sowing densities "
    "across four sites, and the results are not cleared for redistribution to platform users."
)
_FLAGGED = (
    "Table 4.2 reports the sowing density by region, but the numeric run was extracted without "
    "its row header, so the release flagged the chunk as table_unvalidated during validation."
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# The packet schema pins `chunk_id` to 64 hex characters, so the readable labels below become
# digests. `LABEL_BY_CHUNK` turns them back into labels, which keeps test assertions legible.
CHUNK = {label: _sha(f"ceres/inline-fixture/{label}") for label in ("c1", "c2", "c3", "c4", "c5", "c6")}
LABEL_BY_CHUNK = {digest: label for label, digest in CHUNK.items()}


def _item(label: str, parent_id: str, work_id: str, text: str, *,
          rights: str = "redistributable", reasons: list[str] | None = None,
          membership: str = "train_eligible") -> dict[str, Any]:
    chunk_id = CHUNK[label]
    return {
        "release_attempt_id": "fixture-inline-0001", "chunk_id": chunk_id, "parent_id": parent_id,
        "work_id": work_id, "membership": membership, "rights_state": rights,
        "eligibility_state": "eligible", "evidence_eligible": True, "payload": text,
        "payload_sha256": _sha(text), "locator_ids": [f"loc-{chunk_id}"],
        "lineage_ids": [_sha(f"lineage-{chunk_id}")], "excluded_reasons": reasons or [],
    }


def build_inline_fixture() -> tuple[FixtureStore, FakeModels, dict[str, str]]:
    """Six items, two questions, 8-dimension vectors; one rule fires per item.

    c1 survives everything and becomes E1.
    c2 shares c1's parent            → dropped by `parent_deduplication: keep-best`.
    c3 is a near-duplicate of c1     → dropped by `near_duplicate_rule` (Jaccard ≥ 0.6).
    c4 is four tokens long           → dropped by `min_context_tokens: 21`.
    c5 is internal_research_only     → mode-dependent: served in `rnd` (the default),
                                       withheld by the rights predicate in `production`.
    c6 carries `table_unvalidated`   → retrieved and cited WITH that flag in `warnings`
                                       removed the `excluded_reasons` clause from the predicate.
    In production mode both questions get two contexts (c1, c6) and every rule is observable; in
    rnd c5 joins them as a third context.
    """
    rows = [
        _item("c1", "p1", "w1", _WHEAT),
        _item("c2", "p1", "w1", _SIBLING),
        _item("c3", "p2", "w2", _WHEAT_NEAR),
        _item("c4", "p3", "w2", _TINY),
        _item("c5", "p4", "w3", _INTERNAL, rights="internal_research_only"),
        _item("c6", "p5", "w4", _FLAGGED, reasons=["table_unvalidated"]),
    ]
    # Eight dimensions, L2-normalized. Both query vectors point at dimension 0, so c1 and its
    # near-duplicate rank first and the dedup rules are what decide the packet.
    vectors = np.array(
        [
            [1.00, 0.00, 0, 0, 0, 0, 0, 0],   # c1
            [0.80, 0.60, 0, 0, 0, 0, 0, 0],   # c2
            [0.95, 0.31, 0, 0, 0, 0, 0, 0],   # c3
            [0.90, 0.44, 0, 0, 0, 0, 0, 0],   # c4
            [0.99, 0.14, 0, 0, 0, 0, 0, 0],   # c5 (ranks second where its rights are served)
            [0.97, 0.24, 0, 0, 0, 0, 0, 0],   # c6 (flagged, still retrieved)
        ],
        dtype=np.float32,
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    release = Release(
        release_attempt_id="fixture-inline-0001", route_id="fixture-inline",
        row_count=len(rows),
        membership_counts={"train_eligible": len(rows)},
    )
    store = FixtureStore(rows, vectors, release)

    query = np.array([1.0, 0.05, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    query /= np.linalg.norm(query)
    texts = {EN_QUESTION: question_id_for(EN_QUESTION, "en"),
             RO_QUESTION: question_id_for(RO_QUESTION, "ro")}
    models = FakeModels({qid: query for qid in texts.values()}, texts, dim=vectors.shape[1])
    return store, models, texts


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def golden_rerank_scores(directory: Path) -> dict[tuple[str, str], float]:
    """The loop's reranker scores keyed by (question text, chunk text).

    Keyed by the pair because a relevance score belongs to the pair: the same chunk carries three
    different scores across the three fixture questions.
    """
    reranked = directory / "golden" / "reranked-top5.jsonl"
    items = directory / "items.jsonl"
    questions = directory / "questions.jsonl"
    if not reranked.is_file() or not items.is_file() or not questions.is_file():
        return {}
    payloads = {row["chunk_id"]: row.get("payload", "") for row in read_jsonl(items)}
    texts = {row["question_id"]: row["text"] for row in read_jsonl(questions)}
    return {(texts[row["question_id"]].strip(), payloads[row["chunk_id"]].strip()): float(row["score"])
            for row in read_jsonl(reranked)
            if row["chunk_id"] in payloads and row["question_id"] in texts}


def build_fixture_stack() -> tuple[Any, FakeModels, FakeLLM, list[tuple[str, str]]]:
    """The checked-in fixture when it is on disk, the inline one otherwise."""
    manifest = FIXTURE_DIR / "MANIFEST.json"
    if manifest.is_file() and (FIXTURE_DIR / "items.jsonl").is_file():
        store = FixtureStore.from_dir(FIXTURE_DIR)
        questions = [json.loads(line) for line in
                     (FIXTURE_DIR / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        vectors = np.load(FIXTURE_DIR / "query-vectors.npy").astype(np.float32)
        by_id = {row["question_id"]: vectors[index] for index, row in enumerate(questions)}
        by_text = {row["text"].strip(): row["question_id"] for row in questions}
        models = FakeModels(by_id, by_text, dim=vectors.shape[1],
                            rerank_scores=golden_rerank_scores(FIXTURE_DIR))
        goldens: dict[str, dict[str, Any]] = {}
        packets_path, answers_path = FIXTURE_DIR / "golden" / "packets.jsonl", FIXTURE_DIR / "golden" / "answers.jsonl"
        if packets_path.is_file() and answers_path.is_file():
            goldens = goldens_from_fixture(
                [json.loads(line) for line in packets_path.read_text(encoding="utf-8").splitlines() if line.strip()],
                [json.loads(line) for line in answers_path.read_text(encoding="utf-8").splitlines() if line.strip()],
            )
        # One EN and one RO question, as the acceptance criterion asks for — not simply the first
        # two rows, which in this bank are both English.
        asked: list[tuple[str, str]] = []
        for language in ("en", "ro"):
            row = next((q for q in questions if q.get("language") == language), None)
            if row is not None:
                asked.append((row["text"], language))
        return store, models, FakeLLM(goldens), asked

    store, models, _texts = build_inline_fixture()
    return store, models, FakeLLM(), [(EN_QUESTION, "en"), (RO_QUESTION, "ro")]


def smoke_env(runtime_root: Path) -> dict[str, str]:
    """The environment `config.example.yaml` needs, with temporary roots where it wants machine ones."""
    env = dict(os.environ)
    env.setdefault("DDS_LOOP_ROOT", str(runtime_root / "loop"))
    env.setdefault("DDS_RUNTIME_ROOT", str(runtime_root / "dds"))
    env.setdefault("DDS_GPU_LOCK", str(runtime_root / "gpu.lock"))
    env["CERES_RUNTIME_ROOT"] = str(runtime_root)
    return env


# ================================================================ fakes for `ceres-eval`

class EvalFixtureStore(FixtureStore):
    """`FixtureStore` whose active release can be switched mid-run, with a release manifest.

    The eval pins the active release in its first store phase and must refuse a later phase that
    sees another one; `switch_release` is how a test makes the store change under it. The manifest
    is the shape `dds load` writes into `releases.manifest` (product sha + the chain's profile
    sha256s), so the MANIFEST.json path that reads it is exercised without Postgres.
    """

    def __init__(self, items, vectors, release: Release,
                 manifest: dict[str, Any] | None = None) -> None:
        super().__init__(items, vectors, release)
        self.manifest = dict(manifest or {})

    @classmethod
    def from_fixture(cls, directory: Path = FIXTURE_DIR, *,
                     rows: list[dict[str, Any]] | None = None,
                     manifest: dict[str, Any] | None = None) -> "EvalFixtureStore":
        base = FixtureStore.from_dir(directory)
        items = rows if rows is not None else base.rows
        return cls(items, base.vectors, base.active_release(), manifest)

    def switch_release(self, release_attempt_id: str, manifest_sha256: str = "") -> None:
        old = self._release
        self._release = Release(release_attempt_id=release_attempt_id, route_id=old.route_id,
                                row_count=old.row_count, membership_counts=old.membership_counts,
                                manifest_sha256=manifest_sha256)

    def release_manifest(self, release_attempt_id: str) -> dict[str, Any] | None:
        return dict(self.manifest) if release_attempt_id == self._release.release_attempt_id else None


def fixture_bank() -> list[dict[str, str]]:
    """The three fixture questions as bank rows `{question_id, language, text}`."""
    return read_jsonl(FIXTURE_DIR / "questions.jsonl")


def build_eval_fake_stack() -> Any:
    """The `ceres-eval --fake --scratch` stack: the fixture store, the fixture's own query vectors
    with the loop's golden rerank scores, and a `FakeLLM` that composes a grounded answer from E1.

    No golden candidates are replayed here: a golden answer is only valid for the golden packet, and
    the eval builds packets for other routes too.
    """
    from .eval import Stack

    if not (FIXTURE_DIR / "MANIFEST.json").is_file():
        raise RuntimeError(f"the memory fixture is not present at {FIXTURE_DIR}")
    store = EvalFixtureStore.from_fixture()
    questions = fixture_bank()
    vectors = np.load(FIXTURE_DIR / "query-vectors.npy").astype(np.float32)
    models = FakeModels({row["question_id"]: vectors[i] for i, row in enumerate(questions)},
                        {row["text"].strip(): row["question_id"] for row in questions},
                        dim=vectors.shape[1], rerank_scores=golden_rerank_scores(FIXTURE_DIR))
    return Stack(store=store, models=models, llm=FakeLLM(),
                 endpoint={"base_url": "fake://answerer", "expect_model": FakeLLM.MODEL_ID,
                           "lifecycle": "fake"})
