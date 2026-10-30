"""The real embedder and reranker (`ceres/models.py`): refusals, the loop's call shapes, and parity.

How these adapters can fail (written before the tests; each test names the one it catches):
  - a weight or tokenizer file differs from the profile's sha256, or is missing, and is loaded anyway;
  - the combined reranker identity (canonical-JSON sha of the per-file rows) is computed differently
      from the loop's `keys.item_sha256`, so a correct model is refused or a wrong one accepted;
  - the platform config (max_seq_length, batch_size, instruction, limits) is read from a file that
      is not the one the profile pins;
  - the query is encoded with the wrong prompt ("Document: "), the wrong adapter task, or through
      ST's `encode_query` convenience label, or is not re-normalized to unit float32;
  - Jina is cast to the profile's declared float16 while the loop's code path runs float32, so the
      query vectors drift from the stored document vectors;
  - a reranker call mixes questions in one `predict` (bf16 scores depend on batch partners), or
      takes (question, text) pairs that allow it;
  - the rerank width is hard-coded (10) — s1-wide sends 100 candidates in one question's call;
  - the query/document token limits are swapped or not applied;
  - ties are broken differently from the loop's `stable_order`;
  - a non-finite or short score vector is passed on as if it were a ranking;
  - the profile pins cuda, torch sees none, and the adapter silently runs on the CPU;
  - an identity-rerank route (s0) reaches a reranker and gets a made-up order.

Heavy tests (marker `models`, `CERES_TEST_MODELS=1`) load the real weights on the CPU only and compare
with the loop's own fixture outputs; they write their numbers to `runtime/c6-a-models/`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from ceres.models import (
    FakeModels,
    InProcessModels,
    JinaEmbedder,
    ModelError,
    QwenReranker,
    WeightsMismatch,
    item_sha256,
    sha256_file,
    stable_order,
    truncate,
)

from conftest import FIXTURES, ROOT, read_jsonl

PROFILES = FIXTURES / "profiles"


# ================================================================ builders for fake model folders

def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_file(path)


def _jina_setup(tmp_path: Path, *, revision: str = "rev1") -> tuple[dict, Path, Path]:
    """A loop root with a Jina platform config and a model cache with two small pinned files."""
    loop_root = tmp_path / "loop"
    cache = tmp_path / "cache"
    snapshot = cache / "hub" / "models--jinaai--jina-embeddings-v5-text-nano" / "snapshots" / revision
    weights = _write(snapshot / "model.safetensors", b"weights-v1")
    tokenizer = _write(snapshot / "tokenizer.json", b"{\"tok\": 1}")
    config = {"model_cache": str(cache), "models": [{
        "hf_id": "jinaai/jina-embeddings-v5-text-nano", "revision": revision, "dimension": 768,
        "batch_size": 8, "max_seq_length": 8192, "sentence_transformers_task_override": "retrieval"}]}
    config_sha = _write(loop_root / "platform" / "embedding_config.json", json.dumps(config).encode())
    profile = {"slot": "embed", "profile_id": "jina-768-v1", "hf_id": "jinaai/jina-embeddings-v5-text-nano",
               "revision": revision, "dimension": 768,
               "query_encoding": {"prompt_name": "query", "task": "retrieval"},
               "model_weights_sha256": weights, "tokenizer_sha256": tokenizer,
               "platform_config": "platform/embedding_config.json", "platform_config_sha256": config_sha,
               "hardware_flags": {"device": "cuda", "dtype": "float16", "output_dtype": "float32"}}
    return profile, loop_root, snapshot


def _qwen_setup(tmp_path: Path, *, top_k_input: int = 10, batch_size: int = 4) -> tuple[dict, Path, Path]:
    loop_root = tmp_path / "loop"
    model_root = tmp_path / "rerankers"
    model_dir = model_root / "qwen3-reranker-4b" / "rev2"
    files = {"model-00001-of-00002.safetensors": _write(model_dir / "model-00001-of-00002.safetensors", b"shard-1"),
             "model-00002-of-00002.safetensors": _write(model_dir / "model-00002-of-00002.safetensors", b"shard-2")}
    tokenizer = _write(model_dir / "tokenizer.json", b"{\"qwen\": 1}")
    combined = item_sha256([{"path": name, "sha256": files[name]} for name in sorted(files)])
    config = {"max_query_tokens": 256, "max_document_tokens": 2048, "model_root": str(model_root),
              "rerankers": {"qwen3-reranker-4b": {
                  "hf_id": "Qwen/Qwen3-Reranker-4B", "revision": "rev2", "local_dir": "qwen3-reranker-4b/rev2",
                  "architecture": "pointwise_causal_reranker", "dtype": "bfloat16", "batch_size": batch_size,
                  "trust_remote_code": False, "instruction": "Given an agricultural user question, rank."}}}
    config_sha = _write(loop_root / "platform" / "reranker_config.json", json.dumps(config).encode())
    profile = {"slot": "rerank", "profile_id": "qwen3-4b-rerank-test", "hf_id": "Qwen/Qwen3-Reranker-4B",
               "revision": "rev2", "top_k_input": top_k_input, "top_k_output": 5,
               "model_weights_sha256": combined, "weight_files_sha256": files, "tokenizer_sha256": tokenizer,
               "platform_config": "platform/reranker_config.json", "platform_config_sha256": config_sha,
               "hardware_flags": {"device": "cuda", "dtype": "bfloat16"}}
    return profile, loop_root, model_dir


class FakeSentenceTransformer:
    """Records encode kwargs; returns an unnormalized vector per text (norm 2, not 1)."""

    def __init__(self, *, dim: int = 768, value: float | None = None) -> None:
        self.dim = dim
        self.value = value
        self.max_seq_length = None
        self.encode_calls: list[dict] = []

    def encode(self, texts, **kwargs):
        self.encode_calls.append({"texts": list(texts), **kwargs})
        rows = []
        for text in texts:
            seed = int.from_bytes(text.encode("utf-8")[:8].ljust(8, b"\0"), "big") % (2**32)
            row = np.random.default_rng(seed).normal(size=self.dim).astype(np.float32)
            row = 2.0 * row / np.linalg.norm(row)
            if self.value is not None:
                row[:] = self.value
            rows.append(row)
        return np.stack(rows)

    def encode_query(self, *args, **kwargs):          # the convenience label path
        raise AssertionError("encode_query sets ST's own task label; the pinned adapter rejects it")

    def parameters(self):
        class P:
            dtype = "torch.float32"
        return iter([P()])


class WordTokenizer:
    """Whitespace 'tokens', so truncation limits are countable in a test."""

    def __call__(self, text, add_special_tokens=False, truncation=True, max_length=None):
        ids = text.split()
        return {"input_ids": ids[:max_length] if truncation else ids}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


class FakeCrossEncoder:
    """Records every predict call; the score is the candidate's number of words (deterministic)."""

    def __init__(self, scores=None) -> None:
        self.tokenizer = WordTokenizer()
        self.predict_calls: list[dict] = []
        self.scores = scores

    def predict(self, pairs, batch_size, show_progress_bar, convert_to_numpy):
        pairs = list(pairs)
        self.predict_calls.append({"pairs": pairs, "batch_size": batch_size})
        if self.scores is not None:
            return np.asarray(self.scores(pairs), dtype=np.float32)
        return np.asarray([float(len(document.split())) for _q, document in pairs], dtype=np.float32)


def _loader_for(model, record: list | None = None):
    def loader(path, **kwargs):
        if record is not None:
            record.append({"path": path, **kwargs})
        return model
    return loader


def _refusing_loader(path, **kwargs):
    raise AssertionError("a refused model must never reach the loader")


def _reranker(tmp_path, **kwargs) -> tuple[QwenReranker, FakeCrossEncoder]:
    profile, loop_root, _ = _qwen_setup(tmp_path, **kwargs)
    fake = FakeCrossEncoder()
    reranker = QwenReranker.from_profile(profile, loop_root=loop_root, device="cpu", loader=_loader_for(fake))
    return reranker, fake


# ================================================================ identity refusals

def test_F1_a_jina_weights_file_that_differs_from_the_profile_is_refused_before_loading(tmp_path):
    profile, loop_root, snapshot = _jina_setup(tmp_path)
    (snapshot / "model.safetensors").write_bytes(b"weights-v2")
    with pytest.raises(WeightsMismatch, match="Jina weights hash mismatch"):
        JinaEmbedder.from_profile(profile, loop_root=loop_root, device="cpu", loader=_refusing_loader)


def test_F1_a_jina_tokenizer_that_differs_from_the_profile_is_refused_before_loading(tmp_path):
    profile, loop_root, snapshot = _jina_setup(tmp_path)
    (snapshot / "tokenizer.json").write_bytes(b"{}")
    with pytest.raises(WeightsMismatch, match="Jina tokenizer hash mismatch"):
        JinaEmbedder.from_profile(profile, loop_root=loop_root, device="cpu", loader=_refusing_loader)


def test_F1_a_missing_reranker_shard_is_refused_before_loading(tmp_path):
    profile, loop_root, model_dir = _qwen_setup(tmp_path)
    (model_dir / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(WeightsMismatch, match="missing declared Qwen reranker weight"):
        QwenReranker.from_profile(profile, loop_root=loop_root, device="cpu", loader=_refusing_loader)


def test_F1_a_reranker_shard_that_differs_from_the_profile_is_refused_before_loading(tmp_path):
    profile, loop_root, model_dir = _qwen_setup(tmp_path)
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"shard-1-tampered")
    with pytest.raises(WeightsMismatch, match="model-00001-of-00002.safetensors hash mismatch"):
        QwenReranker.from_profile(profile, loop_root=loop_root, device="cpu", loader=_refusing_loader)


def test_F2_a_wrong_combined_reranker_identity_is_refused_even_when_every_file_matches(tmp_path):
    profile, loop_root, _ = _qwen_setup(tmp_path)
    profile["model_weights_sha256"] = "0" * 64
    with pytest.raises(WeightsMismatch, match="combined Qwen reranker weight identity mismatch"):
        QwenReranker.from_profile(profile, loop_root=loop_root, device="cpu", loader=_refusing_loader)


@pytest.mark.parametrize("name", ["qwen3-4b-rerank-v1.yaml"])
def test_F2_the_combined_identity_recipe_reproduces_the_loop_profile_pin(name):
    """Golden: the real profile's `model_weights_sha256` is `keys.item_sha256` of its own file rows.

    A canonical-JSON drift (key order, separators, ensure_ascii) would refuse the real model at load.
    """
    profile = yaml.safe_load((PROFILES / name).read_text(encoding="utf-8"))
    rows = [{"path": key, "sha256": value} for key, value in sorted(profile["weight_files_sha256"].items())]
    assert item_sha256(rows) == profile["model_weights_sha256"]


def test_F3_a_platform_config_that_is_not_the_pinned_one_is_refused(tmp_path):
    profile, loop_root, _ = _jina_setup(tmp_path)
    config_path = loop_root / "platform" / "embedding_config.json"
    config = json.loads(config_path.read_text())
    config["models"][0]["max_seq_length"] = 512
    config_path.write_text(json.dumps(config))
    with pytest.raises(WeightsMismatch, match="platform config hash mismatch"):
        JinaEmbedder.from_profile(profile, loop_root=loop_root, device="cpu", loader=_refusing_loader)


def test_F3_the_identity_rerank_profile_cannot_build_a_reranker(tmp_path):
    none_v1 = {"slot": "rerank", "profile_id": "none-v1", "operation": "identity"}
    with pytest.raises(ModelError, match="never loads a reranker"):
        QwenReranker.from_profile(none_v1, loop_root=tmp_path, device="cpu", loader=_refusing_loader)


# ================================================================ the embedder's call shape

def test_F4_queries_use_the_query_prompt_the_retrieval_adapter_and_the_pinned_widths(tmp_path):
    profile, loop_root, snapshot = _jina_setup(tmp_path)
    fake, record = FakeSentenceTransformer(), []
    embedder = JinaEmbedder.from_profile(profile, loop_root=loop_root, device="cpu", loader=_loader_for(fake, record))
    vector = embedder.embed_query("Ce condiții meteorologice?", "ro")
    call = fake.encode_calls[0]
    assert call["prompt_name"] == "query" and call["task"] == "retrieval"
    assert call["batch_size"] == 8 and fake.max_seq_length == 8192
    assert record == [{"path": str(snapshot), "device": "cpu"}]
    assert vector.dtype == np.float32 and vector.shape == (768,)
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-6          # re-normalized, the fake returns norm 2


def test_F4_embed_queries_keeps_row_order_and_equals_the_single_path(tmp_path):
    profile, loop_root, _ = _jina_setup(tmp_path)
    fake = FakeSentenceTransformer()
    embedder = JinaEmbedder.from_profile(profile, loop_root=loop_root, device="cpu", loader=_loader_for(fake))
    rows = [{"question_id": f"q{i}", "language": "en", "text": f"question number {i}"} for i in range(5)]
    matrix = embedder.embed_queries(rows)
    assert matrix.shape == (5, 768) and matrix.dtype == np.float32
    assert len(fake.encode_calls) == 1 and fake.encode_calls[0]["texts"] == [row["text"] for row in rows]
    for index, row in enumerate(rows):
        np.testing.assert_allclose(matrix[index], embedder.embed_query(row["text"], "en"), atol=1e-7)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_F4_a_non_finite_query_vector_is_refused(tmp_path, value):
    profile, loop_root, _ = _jina_setup(tmp_path)
    embedder = JinaEmbedder.from_profile(profile, loop_root=loop_root, device="cpu",
                                         loader=_loader_for(FakeSentenceTransformer(value=value)))
    with pytest.raises(ModelError, match="invalid query embedding"):
        embedder.embed_query("q", "en")


def test_F4_a_vector_of_the_wrong_width_is_refused(tmp_path):
    profile, loop_root, _ = _jina_setup(tmp_path)
    embedder = JinaEmbedder.from_profile(profile, loop_root=loop_root, device="cpu",
                                         loader=_loader_for(FakeSentenceTransformer(dim=1024)))
    with pytest.raises(ModelError, match="invalid query embedding"):
        embedder.embed_queries([{"text": "q", "language": "en"}])


def test_F5_the_embedder_is_never_cast_to_the_profiles_declared_float16(tmp_path):
    """The loop's path loads float32; the loader gets no dtype and describe() says both."""
    profile, loop_root, _ = _jina_setup(tmp_path)
    record = []
    embedder = JinaEmbedder.from_profile(profile, loop_root=loop_root, device="cpu",
                                         loader=_loader_for(FakeSentenceTransformer(), record))
    assert "dtype" not in record[0] and "model_kwargs" not in record[0]
    info = embedder.describe()
    assert info["profile_dtype"] == "float16" and info["effective_dtype"] == "float32"
    assert info["device"] == "cpu" and info["profile_device"] == "cuda"


# ================================================================ the reranker's call shape

def test_F6_each_question_is_scored_in_its_own_predict_call(tmp_path):
    reranker, fake = _reranker(tmp_path)
    reranker.rerank("first question", ["a b", "c"])
    reranker.rerank("second question", ["d", "e f g"])
    assert len(fake.predict_calls) == 2
    for call, question in zip(fake.predict_calls, ["first question", "second question"]):
        assert {pair[0] for pair in call["pairs"]} == {question}
        assert call["batch_size"] == 4                              # the platform config's, not ST's default


@pytest.mark.parametrize("bad", [["q1", "q2"], ("q1",)])
def test_F6_a_batch_of_questions_is_refused(tmp_path, bad):
    reranker, fake = _reranker(tmp_path)
    with pytest.raises(ModelError, match="ONE question"):
        reranker.rerank(bad, ["text"])
    assert fake.predict_calls == []


def test_F6_question_text_pairs_are_refused(tmp_path):
    reranker, fake = _reranker(tmp_path)
    with pytest.raises(ModelError, match="candidate texts only"):
        reranker.rerank("q1", [("q2", "text of another question")])
    assert fake.predict_calls == []


def test_F7_a_top_k_input_of_100_scores_100_candidates_in_one_call_in_input_order(tmp_path):
    reranker, fake = _reranker(tmp_path, top_k_input=100)
    texts = [" ".join(["w"] * (index + 1)) for index in range(100)]
    scores = reranker.rerank("q", texts)
    assert reranker.top_k_input == 100 and reranker.top_k_output == 5
    assert len(fake.predict_calls) == 1 and len(fake.predict_calls[0]["pairs"]) == 100
    assert scores == [float(index + 1) for index in range(100)]      # score i belongs to text i


def test_F7_more_candidates_than_the_profile_width_are_refused(tmp_path):
    reranker, fake = _reranker(tmp_path, top_k_input=10)
    with pytest.raises(ModelError, match="exceed the profile's top_k_input 10"):
        reranker.rerank("q", ["t"] * 11)
    assert fake.predict_calls == []


def test_F8_the_query_is_cut_to_256_and_each_document_to_2048_tokens(tmp_path):
    reranker, fake = _reranker(tmp_path)
    reranker.rerank(" ".join(["q"] * 300), [" ".join(["d"] * 3000), "short doc"])
    (question, first), (_, second) = fake.predict_calls[0]["pairs"]
    assert len(question.split()) == 256
    assert len(first.split()) == 2048 and second == "short doc"


def test_F8_truncate_is_the_runtime_recipe():
    assert truncate(WordTokenizer(), "a b c d", 2) == "a b"


def test_F9_ties_break_on_the_original_rank_then_the_parent_id():
    scores = [1.0, 2.0, 2.0, 2.0]
    candidates = [{"original_rank": 1, "parent_id": "p"}, {"original_rank": 3, "parent_id": "a"},
                  {"original_rank": 2, "parent_id": "z"}, {"original_rank": 2, "parent_id": "b"}]
    assert stable_order(scores, candidates) == [3, 2, 1, 0]


def test_F10_non_finite_scores_are_refused(tmp_path):
    profile, loop_root, _ = _qwen_setup(tmp_path)
    fake = FakeCrossEncoder(scores=lambda pairs: [float("nan")] * len(pairs))
    reranker = QwenReranker.from_profile(profile, loop_root=loop_root, device="cpu", loader=_loader_for(fake))
    with pytest.raises(ModelError, match="non-finite"):
        reranker.rerank("q", ["a", "b"])


def test_F10_a_short_score_vector_is_refused(tmp_path):
    profile, loop_root, _ = _qwen_setup(tmp_path)
    fake = FakeCrossEncoder(scores=lambda pairs: [1.0] * (len(pairs) - 1))
    reranker = QwenReranker.from_profile(profile, loop_root=loop_root, device="cpu", loader=_loader_for(fake))
    with pytest.raises(ModelError, match="unexpected score shape"):
        reranker.rerank("q", ["a", "b", "c"])


def test_F10_the_cpu_reranker_loads_float32_and_the_cuda_one_the_platform_dtype(tmp_path):
    profile, loop_root, _ = _qwen_setup(tmp_path)
    record = []
    QwenReranker.from_profile(profile, loop_root=loop_root, device="cpu", loader=_loader_for(FakeCrossEncoder(), record))
    QwenReranker.from_profile(profile, loop_root=loop_root, device="cuda", cuda_available=lambda: True,
                              loader=_loader_for(FakeCrossEncoder(), record))
    assert [(call["device"], call["dtype"]) for call in record] == [("cpu", "float32"), ("cuda", "bfloat16")]
    assert record[0]["max_length"] == 256 + 2048 + 256 and record[0]["trust_remote_code"] is False


# ================================================================ device and route shape

def test_F11_a_cuda_profile_without_a_gpu_is_refused_not_moved_to_the_cpu(tmp_path):
    profile, loop_root, _ = _jina_setup(tmp_path)
    with pytest.raises(ModelError, match="torch sees no GPU"):
        JinaEmbedder.from_profile(profile, loop_root=loop_root, cuda_available=lambda: False,
                                  loader=_refusing_loader)
    qprofile, qroot, _ = _qwen_setup(tmp_path / "q")
    with pytest.raises(ModelError, match="torch sees no GPU"):
        QwenReranker.from_profile(qprofile, loop_root=qroot, cuda_available=lambda: False,
                                  loader=_refusing_loader)


def test_F12_an_identity_route_models_object_refuses_rerank():
    models = InProcessModels(embedder=FakeModels(), reranker=None)
    with pytest.raises(ModelError, match="identity-rerank route must not call rerank"):
        models.rerank("q", ["a"])
    assert models.health()["reranker"] == "absent"


def test_the_fake_batch_path_equals_its_single_path():
    """FakeModels.embed_queries (the eval's fake) must not diverge from embed_query."""
    fake = FakeModels(dim=16)
    rows = [{"text": "alpha", "language": "en"}, {"text": "beta", "language": "ro"}]
    matrix = fake.embed_queries(rows)
    assert matrix.shape == (2, 16)
    np.testing.assert_array_equal(matrix[1], fake.embed_query("beta", "ro"))


def test_importing_the_module_does_not_import_torch():
    """The base install has no torch; `ceres.models` must import there (checked in a fresh process)."""
    import subprocess
    import sys

    code = "import sys; import ceres.models; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# ================================================================ heavy: real weights, CPU only

requires_models = pytest.mark.skipif(os.environ.get("CERES_TEST_MODELS") != "1",
                                     reason="real-model tests: set CERES_TEST_MODELS=1 (CPU only)")
LOOP_ROOT = Path(os.environ.get("DDS_LOOP_ROOT")
                 or ROOT.parent.parent / "workspace" / "dds" / "archive" / "loop-v0")
if LOOP_ROOT.name != "loop-v0" and (LOOP_ROOT / "loop-v0").is_dir():
    LOOP_ROOT = LOOP_ROOT / "loop-v0"           # DDS_LOOP_ROOT may name the archive folder
RESULTS_DIR = ROOT / "runtime" / "c6-a-models"


def _record(name: str, payload: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    (RESULTS_DIR / f"{stamp}-{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, sort_keys=True))


@pytest.fixture(scope="module")
def cpu_only():
    """No test here may touch the GPU (it belongs to the answerer's lease): hide it from torch.

    Works because nothing in this process imports torch before the heavy tests (the import check
    above runs in a subprocess); if torch is already in, CUDA must at least be uninitialised after.
    """
    import sys

    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    yield
    if previous is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = previous
    torch = sys.modules.get("torch")
    if torch is not None:
        assert not torch.cuda.is_initialized(), "a CPU-only test initialised CUDA"


@pytest.mark.models
@requires_models
def test_heavy_jina_on_cpu_reproduces_the_loops_query_vectors(cpu_only):
    """Parity: CPU float32 vs the loop's recorded query vectors (recorded fixture)."""
    questions = read_jsonl(FIXTURES / "questions.jsonl")
    reference = np.load(FIXTURES / "query-vectors.npy").astype(np.float32)
    started = time.monotonic()
    embedder = JinaEmbedder.from_profile(PROFILES / "jina-768-v1.yaml", loop_root=LOOP_ROOT, device="cpu")
    ready = time.monotonic()
    matrix = embedder.embed_queries(questions)
    encoded = time.monotonic()
    singles = np.stack([embedder.embed_query(row["text"], row["language"]) for row in questions])
    delta = np.abs(matrix - reference)
    cosines = (matrix * reference).sum(axis=1)
    info = embedder.describe()
    _record("jina", {
        "questions": len(questions), "max_abs_delta": float(delta.max()), "min_cosine": float(cosines.min()),
        "batch_vs_single_max_abs_delta": float(np.abs(matrix - singles).max()),
        "load_s_incl_sha": round(ready - started, 3), "encode_s": round(encoded - ready, 3), **info})
    embedder.close()
    assert info["effective_dtype"] == "float32"
    assert float(delta.max()) < 1e-4 and float(cosines.min()) > 0.9999


# CPU float32 against the loop's GPU bfloat16: the scores differ by up to ~0.14 (measured here 0.136;
# dds-cli measured 0.119 on other inputs), and bf16 rounds many logits onto the same value (golden
# scores 2.0 and 2.0). So the CPU order may legitimately invert two candidates whose scores are within
# that noise. The test allows exactly that and nothing more: every inversion against the golden order
# must be a near-tie on the CPU (gap <= 2 x SCORE_TOL), and the golden top-5 scores must sit within
# SCORE_TOL. Exact order agreement is recorded in runtime/c6-a-models/, not hidden.
SCORE_TOL = 0.15
PROBE_DIR = Path(os.environ.get("CERES_RERANK_PROBE_DIR", "/nonexistent"))  # a recorded DDS rerank probe, if one is available


@pytest.fixture(scope="module")
def cpu_reranker(cpu_only):
    started = time.monotonic()
    reranker = QwenReranker.from_profile(PROFILES / "qwen3-4b-rerank-v1.yaml", loop_root=LOOP_ROOT, device="cpu")
    reranker.identity["load_s_incl_sha"] = round(time.monotonic() - started, 3)
    yield reranker
    reranker.close()


@pytest.mark.models
@requires_models
def test_heavy_qwen_on_cpu_matches_the_loops_bf16_top5_up_to_near_ties(cpu_reranker):
    """Parity: CPU fp32 rerank of the loop's 10 golden candidates per question vs its bf16 top 5.

    Catches a wrong prompt/instruction, a wrong truncation, a wrong pairing, or a wrong tie-break:
    any of those moves scores by far more than bf16 noise.
    """
    reranker = cpu_reranker
    questions = read_jsonl(FIXTURES / "questions.jsonl")
    payload = {row["chunk_id"]: row["payload"] for row in read_jsonl(FIXTURES / "items.jsonl")}
    hits = read_jsonl(FIXTURES / "golden" / "hits-top10.jsonl")
    golden = read_jsonl(FIXTURES / "golden" / "reranked-top5.jsonl")
    per_question, agree, max_delta, worst_inversion = [], 0, 0.0, 0.0
    for row in questions:
        candidates = sorted((hit for hit in hits if hit["question_id"] == row["question_id"]), key=lambda hit: hit["rank"])
        t0 = time.monotonic()
        scores = reranker.rerank(row["text"], [payload[hit["chunk_id"]] for hit in candidates])
        seconds = time.monotonic() - t0
        order = stable_order(scores, [{"original_rank": hit["rank"], "parent_id": hit["parent_id"]} for hit in candidates])
        top5 = [candidates[index]["chunk_id"] for index in order[:reranker.top_k_output]]
        expected_rows = sorted((g for g in golden if g["question_id"] == row["question_id"]), key=lambda g: g["rank"])
        expected = [g["chunk_id"] for g in expected_rows]
        cpu = {candidates[index]["chunk_id"]: scores[index] for index in range(len(candidates))}
        deltas = [abs(cpu[g["chunk_id"]] - g["score"]) for g in expected_rows]
        max_delta = max(max_delta, *deltas)
        # Golden order over the union: golden top 5 first (their ranks), then any CPU-only entrant.
        union = expected + [chunk for chunk in top5 if chunk not in expected]
        inversions = [(a, b, cpu[b] - cpu[a]) for i, a in enumerate(union) for b in union[i + 1:] if cpu[b] > cpu[a]]
        worst = max((gap for _a, _b, gap in inversions), default=0.0)
        worst_inversion = max(worst_inversion, worst)
        agree += top5 == expected
        per_question.append({
            "question_id": row["question_id"], "language": row["language"], "order_equal": top5 == expected,
            "set_equal": set(top5) == set(expected), "score_s": round(seconds, 2),
            "max_abs_score_delta_vs_bf16": max(deltas), "worst_inversion_gap": worst,
            "cpu_top5": [[chunk[:12], round(cpu[chunk], 4)] for chunk in top5],
            "golden_top5": [[g["chunk_id"][:12], g["score"]] for g in expected_rows]})
    _record("qwen", {"questions": len(questions), "order_agreement": f"{agree}/{len(questions)}",
                     "max_abs_score_delta_vs_bf16": max_delta, "worst_inversion_gap": worst_inversion,
                     "score_tol": SCORE_TOL, "per_question": per_question, **reranker.describe()})
    assert max_delta <= SCORE_TOL, per_question
    assert worst_inversion <= 2 * SCORE_TOL, per_question


@pytest.mark.models
@requires_models
@pytest.mark.skipif(not (PROBE_DIR / "c-fp32.json").is_file(), reason="no recorded rerank probe (set CERES_RERANK_PROBE_DIR)")
def test_heavy_qwen_port_equals_the_dds_cli_cpu_mirror_of_the_loop_runtime(cpu_reranker):
    """Port check: same inputs, same device and dtype as dds-cli's `_CpuReranker` probe,
    which runs `reranker_runtime.LocalReranker`'s code on the CPU and matched the GPU order 3/3.

    Catches any drift between this port and the loop's runtime (prompt, max_length, truncation,
    batch size, pairing): the scores must agree to float32 noise, not to bf16 noise.
    """
    cases = json.loads((PROBE_DIR / "cases.json").read_text())["cases"]
    reference = {q["question_id"]: q for q in json.loads((PROBE_DIR / "c-fp32.json").read_text())["questions"]}
    worst = 0.0
    for case in cases:
        ref = reference[case["question_id"]]
        assert [c["chunk_id"] for c in case["candidates"]] == ref["chunk_ids"]
        scores = cpu_reranker.rerank(case["text"], [c["payload"] for c in case["candidates"]])
        worst = max(worst, float(np.max(np.abs(np.asarray(scores) - np.asarray(ref["scores"])))))
    _record("qwen-port", {"cases": len(cases), "max_abs_delta_vs_ddscli_cpu_fp32": worst,
                          "reference": str(PROBE_DIR / "c-fp32.json")})
    assert worst <= 1e-3


def test_concurrent_pipelines_never_run_the_shared_models_at_the_same_time():
    """with max_inflight 2, two worker threads called the shared embedder and
    reranker at once; HF fast tokenizers raise 'Already borrowed' under that. Every model call is now serialised."""
    import threading

    class OverlapProbe:
        def __init__(self):
            self.active, self.peak, self.lock = 0, 0, threading.Lock()

        def _enter(self):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            time.sleep(0.05)
            with self.lock:
                self.active -= 1

        def embed_query(self, text, lang):
            self._enter()
            return np.zeros(4, dtype=np.float32)

        def rerank(self, question, texts):
            self._enter()
            return [0.0 for _ in texts]

    probe = OverlapProbe()
    models = InProcessModels(probe, probe)
    calls = [lambda: models.embed_query("q", "en"), lambda: models.rerank("q", ["a", "b"])] * 4
    threads = [threading.Thread(target=call) for call in calls]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert probe.peak == 1


# ================================================================ rerank switch: self-contained 0.6B profile
# How the new load path can fail, and the test that catches each:
#   * a setting the scoring depends on silently defaults → test_a_self_contained_profile_missing_a_scoring_setting_is_refused
#   * the one-file identity form opens a hole for a wrong weights file → test_a_one_file_profile_still_refuses_a_wrong_weights_file
#   * the profile's instruction / limits / batch / dtype do not reach the CrossEncoder (another scoring than the loop's)
#       → test_the_self_contained_profile_scores_with_its_own_instruction_limits_and_batch

def _qwen06_setup(tmp_path: Path, **over) -> dict:
    model_dir = tmp_path / "hub" / "qwen3-0.6b" / "snap"
    weights = _write(model_dir / "model.safetensors", b"one-file-weights")
    tokenizer = _write(model_dir / "tokenizer.json", b"{\"qwen\": 1}")
    profile = {"slot": "rerank", "profile_id": "qwen3-0.6b-rerank-test", "hf_id": "Qwen/Qwen3-Reranker-0.6B",
               "revision": "e611", "model_dir": str(model_dir), "architecture": "pointwise_causal_reranker",
               "instruction": "Given an agricultural user question, rank passages.", "max_query_tokens": 256,
               "max_document_tokens": 2048, "batch_size": 4, "top_k_input": 50, "top_k_output": 5,
               "rerank_text": "represented", "on_oom": "dense_order",
               "model_weights_sha256": weights, "weight_files_sha256": {"model.safetensors": weights},
               "tokenizer_sha256": tokenizer, "hardware_flags": {"device": "cuda", "dtype": "bfloat16"}}
    profile.update(over)
    return profile


@pytest.mark.parametrize("key", ["model_dir", "instruction", "max_query_tokens", "max_document_tokens", "batch_size",
                                 "architecture"])
def test_a_self_contained_profile_missing_a_scoring_setting_is_refused(tmp_path, key):
    profile = _qwen06_setup(tmp_path)
    del profile[key]
    with pytest.raises(ModelError, match=key):
        QwenReranker.from_profile(profile, loop_root=tmp_path, device="cpu", loader=_refusing_loader)


def test_a_one_file_profile_still_refuses_a_wrong_weights_file(tmp_path):
    profile = _qwen06_setup(tmp_path)
    (Path(profile["model_dir"]) / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(WeightsMismatch):
        QwenReranker.from_profile(profile, loop_root=tmp_path, device="cpu", loader=_refusing_loader)
    profile = _qwen06_setup(tmp_path / "b", model_weights_sha256="0" * 64)   # combined pin matches neither form
    with pytest.raises(WeightsMismatch):
        QwenReranker.from_profile(profile, loop_root=tmp_path, device="cpu", loader=_refusing_loader)


def test_the_self_contained_profile_scores_with_its_own_instruction_limits_and_batch(tmp_path):
    profile = _qwen06_setup(tmp_path)
    record, fake = [], FakeCrossEncoder()
    reranker = QwenReranker.from_profile(profile, loop_root=tmp_path, device="cpu", loader=_loader_for(fake, record))
    assert record[0]["instruction"] == profile["instruction"]
    assert record[0]["max_length"] == 256 + 2048 + 256                     # reranker_runtime: q + d + 256
    assert record[0]["path"] == profile["model_dir"] and record[0]["trust_remote_code"] is False
    reranker.rerank("q", ["a b", "c"])
    assert fake.predict_calls[0]["batch_size"] == 4
    assert reranker.describe()["platform_config"] == "profile"
