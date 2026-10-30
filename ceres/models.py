"""The embedder and the reranker behind one protocol, the real in-process adapters, and the fake.

r1 and the v33 routes pin `jina-768-v1` (query encoding `{prompt_name: query, task: retrieval}`,
768 float32, L2-normalized) and `qwen3-4b-rerank-v1`/`-v2` (top_k_input 10 or 100 → top_k_output 5).

* `JinaEmbedder` ports the loop's `slot_adapter.encode_jina` (loop-v0/04_rag/slot_adapter.py:1096):
  SentenceTransformer over the pinned snapshot, `trust_remote_code`, local files only,
  `max_seq_length`/`batch_size` from the pinned platform config, `encode(prompt_name=…,
  task="retrieval")` called directly (the convenience `encode_query` sets its own task label, which
  the pinned adapter rejects — as measured in the parity check against the loop), L2 re-normalization.
* `QwenReranker` ports `reranker_runtime.LocalReranker`'s pointwise branch and `stable_order`
  (the reranker pilot's `reranker_runtime.py`): CrossEncoder
  with the "agriculture" instruction prompt, max_length = query + document limits + 256, the
  tokenizer truncation, `predict` in `batch_size` chunks — ONE question per call (bf16 scores depend
  on their batch partners: never mix questions in one batch).
* Both verify the weight and tokenizer sha256s the profile pins BEFORE torch is imported, and refuse
  on a mismatch; both verify the platform config's sha256 against the profile.

Device and dtype. The profiles pin `device: cuda`. A CPU execution (tests only) is not
hash-neutral: it is recorded in `describe()` and must never be presented as the profile's device.
Jina's dtype is NOT the profile's declared `float16`: the loop's code path passes no dtype, and the
pinned `JinaEmbeddingsV5Model.from_pretrained` defaults to float32 (modeling_jina_embeddings_v5.py:
`dtype=kwargs.pop("dtype", torch.float32)`; `custom_st.Transformer.load` forwards no kwargs). The
loop's stored document vectors were therefore made in float32, and CPU float32 reproduces the
loop's query vectors to 1.7e-7 (fp16 would be off by 4e-4). `JinaEmbedder` follows the code path,
not the declaration, and records both. The reranker's dtype is the platform
config's (bfloat16 on cuda); on cpu it is float32 (CPU bf16 is emulated, slower and further from the GPU scores).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

LOG = logging.getLogger("ceres.models")

EMBED_DIM = 768   # `jina-768-v1.yaml: dimension`


class Models(Protocol):
    """Query embedding and reranking. Both calls are synchronous and CPU/GPU bound."""

    def embed_query(self, text: str, lang: str) -> np.ndarray:
        """One L2-normalized float32 query vector of `EMBED_DIM` dimensions."""

    def embed_queries(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """`(len(rows), EMBED_DIM)` float32, row i = the question `rows[i]["text"]` (eval batch)."""

    def rerank(self, question: str, texts: Sequence[str]) -> list[float]:
        """One score per text, higher is better, in the order the texts were given."""

    # Not part of the base protocol, but `/health` must distinguish a loaded model from a
    # missing one, and only the adapter knows. Implementations return (status, detail).
    def health(self) -> dict[str, str]:
        """`{"embedder": status, "reranker": status, "detail": ...}`."""


class FakeModels:
    """Deterministic stand-ins: fixture query vectors and order-preserving rerank scores.

    `embed_query` returns the fixture's own query vector when it knows the question, so the
    fixture store's ranking is the loop's ranking and a retrieval test means something. For an
    unknown question it returns a stable pseudo-vector derived from the text — the path runs, the
    ranking is arbitrary, and tests must not assert on it.

    `rerank` returns descending scores in input order: the reranked top 5 is then the dense top 5.
    That is the identity ordering on purpose — a test that sees a different order after rerank has
    found a bug in the pipeline, not in the model.
    """

    def __init__(
        self,
        vectors_by_question_id: dict[str, np.ndarray] | None = None,
        question_text_to_id: dict[str, str] | None = None,
        dim: int = EMBED_DIM,
        rerank_scores: dict[tuple[str, str], float] | None = None,
    ) -> None:
        self.vectors = {k: np.asarray(v, dtype=np.float32) for k, v in (vectors_by_question_id or {}).items()}
        self.by_text = dict(question_text_to_id or {})
        # The fallback vector's width must match the store it is searched against: 768 for the
        # real fixture, 8 for the hand-written one in `smoke_test.build_inline_fixture`.
        self.dim = int(dim)
        # The loop's own reranker scores from the fixture's `golden/reranked-top5.jsonl`, keyed by
        # (question text, chunk text) — a relevance score belongs to the pair, and the same chunk
        # scores differently for different questions (cd5fe51f… is in all three golden packets
        # with three scores). With them the fake reproduces r1's rerank order, which is what
        # `tests/test_parity.py` compares against the golden packets. Without them the order is
        # the dense order: honest, but not r1.
        self.rerank_scores = {(q.strip(), t.strip()): float(v)
                              for (q, t), v in (rerank_scores or {}).items()}
        self.embed_calls = 0
        self.rerank_calls = 0

    # -------------------------------------------------------- Models protocol

    def embed_query(self, text: str, lang: str) -> np.ndarray:
        self.embed_calls += 1
        question_id = self.by_text.get(text.strip())
        if question_id and question_id in self.vectors:
            return self._unit(self.vectors[question_id])
        from .contracts import question_id_for

        derived = question_id_for(text, lang)
        if derived in self.vectors:
            return self._unit(self.vectors[derived])
        seed = int.from_bytes(hashlib.sha256(text.strip().encode("utf-8")).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        return self._unit(rng.normal(size=self.dim).astype(np.float32))

    def embed_queries(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Row by row through `embed_query`, so the batch and the single path cannot disagree."""
        vectors = [self.embed_query(str(row["text"]), str(row.get("language") or row.get("lang") or ""))
                   for row in rows]
        return np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, self.dim), np.float32)

    def rerank(self, question: str, texts: Sequence[str]) -> list[float]:
        self.rerank_calls += 1
        if not self.rerank_scores:
            return [float(len(texts) - index) for index in range(len(texts))]
        # A text the goldens do not score sits below every scored one, keeping its dense order.
        floor = min(self.rerank_scores.values()) - 1.0
        key = question.strip()
        return [self.rerank_scores.get((key, text.strip()), floor - index * 1e-6)
                for index, text in enumerate(texts)]

    def health(self) -> dict[str, str]:
        return {"embedder": "fake", "reranker": "fake",
                "detail": "FakeModels: fixture query vectors, identity rerank order"}

    # -------------------------------------------------------- helpers

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector)) or 1.0
        return (vector / norm).astype(np.float32)


# ================================================================ the real in-process models
# Nothing below imports torch at module import:
# the base install (no `models` extra) must import this module and run the whole suite.

JINA_HF_ID = "jinaai/jina-embeddings-v5-text-nano"
QWEN_RERANKER_KEY = "qwen3-reranker-4b"          # the platform config's `rerankers` key
POINTWISE = "pointwise_causal_reranker"
RERANK_PROMPT_NAME = "agriculture"               # `LocalReranker.__init__`'s default_prompt_name
CPU_DTYPE = "float32"


class ModelError(RuntimeError):
    """A real model cannot be loaded or returned something unusable. Never silently degraded."""


class WeightsMismatch(ModelError):
    """A pinned file is missing or its sha256 differs from the profile. Refused before any load."""


# ---------------------------------------------------------------- hashing (loop keys.py port)

def sha256_file(path: str | Path) -> str:
    """`loop-v0/00_control/code/keys.py:32` — streamed sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def item_sha256(value: Any) -> str:
    """`keys.py:40` — sha256 of the canonical JSON encoding (`keys.py:18-25`)."""
    body = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def checked_sha(path: Path, expected: str | None, label: str) -> str:
    """`loop-v0/04_rag/common.py:205`, raising `WeightsMismatch` instead of `AdapterError(3)`."""
    if not path.is_file():
        raise WeightsMismatch(f"missing declared {label}: {path}")
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise WeightsMismatch(f"{label} hash mismatch: expected {expected}, got {actual}")
    return actual


# ---------------------------------------------------------------- profiles and platform configs

def read_profile(profile: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """A slot profile as a dict: a YAML path, or an already-parsed mapping (copied)."""
    if isinstance(profile, Mapping):
        return dict(profile)
    import yaml

    loaded = yaml.safe_load(Path(profile).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ModelError(f"{profile}: not a YAML mapping")
    return loaded


def resolve_loop_path(value: str | Path, loop_root: str | Path) -> Path:
    """`common.py:146` — a profile's relative path is relative to the loop root."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(loop_root) / path).resolve()


def read_platform_config(profile: Mapping[str, Any], loop_root: str | Path) -> tuple[Path, dict[str, Any]]:
    """The profile's `platform_config`, refused unless its sha256 is the profile's pin."""
    if not profile.get("platform_config"):
        raise ModelError(f"profile {profile.get('profile_id')!r} names no platform_config")
    path = resolve_loop_path(profile["platform_config"], loop_root)
    checked_sha(path, profile.get("platform_config_sha256"), "platform config")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ModelError(f"{path}: platform config is not a JSON object")
    return path, config


def rerank_widths(profile: Mapping[str, Any]) -> tuple[int, int]:
    """`slot_adapter.rerank_widths` candidates scored and hits kept, with the loop's checks."""
    top_k_input = int(profile.get("top_k_input", 10))
    top_k_output = int(profile.get("top_k_output", 5))
    if top_k_input < 1 or top_k_output < 1:
        raise ModelError(f"rerank profile {profile.get('profile_id')}: top_k_input and top_k_output must be >= 1")
    if top_k_output > top_k_input:
        raise ModelError(f"rerank profile {profile.get('profile_id')}: top_k_output ({top_k_output}) "
                         f"must not exceed top_k_input ({top_k_input})")
    return top_k_input, top_k_output


def resolve_device(profile: Mapping[str, Any], device: str | None,
                   cuda_available: Callable[[], bool] | None = None) -> str:
    """The profile's device unless overridden; never a silent CPU fallback.

    `cuda` when torch cannot see a GPU is refused (the loop refuses the same way); `cpu` must be
    asked for by name, and `describe()` then records it.
    """
    flags = profile.get("hardware_flags") or {}
    chosen = (device or flags.get("device") or "cuda").lower()
    if chosen not in {"cuda", "cpu"}:
        raise ModelError(f"unsupported device {chosen!r} (cuda or cpu)")
    if chosen == "cuda":
        if cuda_available is None:
            import torch

            cuda_available = torch.cuda.is_available
        if not cuda_available():
            raise ModelError("the profile pins cuda and torch sees no GPU; pass device='cpu' explicitly"
                             " (tests only — a CPU run is not the profile's execution)")
    return chosen


def _offline_env() -> None:
    """Local files only, as the loop runs them. Set before transformers/huggingface_hub import."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _unit_rows(matrix: np.ndarray, dim: int, label: str) -> np.ndarray:
    """`encode_jina`'s post-check: 2-D, `dim` wide, finite, then L2 re-normalized in float32."""
    matrix = np.array(matrix, dtype=np.float32, copy=True)
    if matrix.ndim != 2 or matrix.shape[1] != dim or not np.isfinite(matrix).all():
        raise ModelError(f"invalid {label} embedding matrix shape/content: {matrix.shape}")
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    return matrix


# ---------------------------------------------------------------- the embedder

def _load_sentence_transformer(path: str, *, device: str, **kwargs: Any) -> Any:
    _offline_env()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(path, device=device, trust_remote_code=True, local_files_only=True, **kwargs)


class JinaEmbedder:
    """`jinaai/jina-embeddings-v5-text-nano` @ the profile's revision, in process.

    Build with `JinaEmbedder.from_profile(profile, loop_root=…)`: it checks the platform config,
    the model and tokenizer sha256s, and only then loads. `embed_query` encodes one question,
    `embed_queries` a batch in the loop's `batch_size` (the loop encodes its whole question file in
    one `encode` call; the eval does the same with the bank).
    """

    def __init__(self, model: Any, *, profile: Mapping[str, Any], pinned: Mapping[str, Any],
                 snapshot: Path, device: str, identity: Mapping[str, Any]) -> None:
        self.model = model
        self.profile = dict(profile)
        self.pinned = dict(pinned)
        self.snapshot = Path(snapshot)
        self.device = device
        self.identity = dict(identity)
        self.dim = int(profile.get("dimension", EMBED_DIM))
        encoding = profile.get("query_encoding") or {}
        self.prompt_name = str(encoding.get("prompt_name", "query"))
        # `encode_jina` uses the platform config's override when present, else "retrieval".
        self.task = str(pinned.get("sentence_transformers_task_override", encoding.get("task", "retrieval")))
        self.batch_size = int(pinned["batch_size"])
        self.calls = 0

    @classmethod
    def from_profile(cls, profile: Mapping[str, Any] | str | Path, *, loop_root: str | Path,
                     model_cache: str | Path | None = None, device: str | None = None,
                     loader: Callable[..., Any] | None = None,
                     cuda_available: Callable[[], bool] | None = None) -> "JinaEmbedder":
        profile = read_profile(profile)
        if profile.get("slot") != "embed" or profile.get("hf_id") != JINA_HF_ID:
            raise ModelError(f"not a Jina embed profile: slot={profile.get('slot')!r} hf_id={profile.get('hf_id')!r}")
        config_path, config = read_platform_config(profile, loop_root)
        models = config.get("models")
        if not isinstance(models, list):
            raise ModelError("Jina platform config has no models array")
        matching = [row for row in models if row.get("revision") == profile.get("revision")]
        if len(matching) != 1:
            raise ModelError("Jina platform config does not contain the pinned revision exactly once")
        pinned = matching[0]
        if pinned.get("hf_id") != profile.get("hf_id") or int(pinned.get("dimension", -1)) != int(profile.get("dimension", -1)):
            raise ModelError("profile differs from the pinned Jina platform model")
        cache = Path(model_cache if model_cache is not None else config["model_cache"]).expanduser().resolve()
        snapshot = cache / "hub" / f"models--{profile['hf_id'].replace('/', '--')}" / "snapshots" / profile["revision"]
        started = time.monotonic()
        weights = checked_sha(snapshot / "model.safetensors", profile["model_weights_sha256"], "Jina weights")
        tokenizer = checked_sha(snapshot / "tokenizer.json", profile["tokenizer_sha256"], "Jina tokenizer")
        verify_ms = int((time.monotonic() - started) * 1000)
        chosen = resolve_device(profile, device, cuda_available)
        started = time.monotonic()
        model = (loader or _load_sentence_transformer)(str(snapshot), device=chosen)
        model.max_seq_length = int(pinned["max_seq_length"])
        load_ms = int((time.monotonic() - started) * 1000)
        identity = {"platform_config": str(config_path), "model_weights_sha256": weights,
                    "tokenizer_sha256": tokenizer, "verify_ms": verify_ms, "load_ms": load_ms}
        return cls(model, profile=profile, pinned=pinned, snapshot=snapshot, device=chosen, identity=identity)

    # -------------------------------------------------------- encoding

    def _encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        matrix = self.model.encode(
            texts,
            prompt_name=self.prompt_name,
            task=self.task,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _unit_rows(np.asarray(matrix, dtype=np.float32), self.dim, "query")

    def embed_query(self, text: str, lang: str) -> np.ndarray:
        return self._encode([text])[0]

    def embed_queries(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        texts = [str(row["text"]) for row in rows]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = self._encode(texts)
        if matrix.shape[0] != len(texts):
            raise ModelError(f"the embedder returned {matrix.shape[0]} vectors for {len(texts)} questions")
        return matrix

    # -------------------------------------------------------- reporting

    def effective_dtype(self) -> str:
        try:
            parameter = next(iter(self.model.parameters()))
        except (AttributeError, StopIteration, TypeError):
            return "unknown"
        return str(parameter.dtype).replace("torch.", "")

    def describe(self) -> dict[str, Any]:
        """What actually ran, for MANIFEST.json: device, dtypes declared vs effective, pins."""
        flags = self.profile.get("hardware_flags") or {}
        return {
            "adapter": "JinaEmbedder", "profile_id": self.profile.get("profile_id"),
            "hf_id": self.profile.get("hf_id"), "revision": self.profile.get("revision"),
            "snapshot": str(self.snapshot), "device": self.device,
            "profile_device": flags.get("device"), "profile_dtype": flags.get("dtype"),
            "effective_dtype": self.effective_dtype(), "output_dtype": "float32",
            "prompt_name": self.prompt_name, "task": self.task, "batch_size": self.batch_size,
            "max_seq_length": int(self.pinned["max_seq_length"]), **self.identity,
        }

    def close(self) -> None:
        """Drop the model (the GPU window frees the card before the reranker loads)."""
        self.model = None
        _free_cuda(self.device)


# ---------------------------------------------------------------- the reranker

def truncate(tokenizer: Any, text: str, max_tokens: int) -> str:
    """`reranker_runtime.truncate`, verbatim: tokenize without specials, cut, decode."""
    encoded = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_tokens)
    return tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)


def stable_order(scores: Sequence[float], candidates: Sequence[Mapping[str, Any]]) -> list[int]:
    """`reranker_runtime.stable_order`, verbatim: score desc, then original rank, then parent id."""
    return sorted(
        range(len(candidates)),
        key=lambda index: (-float(scores[index]), candidates[index]["original_rank"], candidates[index]["parent_id"]),
    )


def _load_cross_encoder(path: str, *, device: str, dtype: str, trust_remote_code: bool,
                        max_length: int, instruction: str) -> Any:
    """`LocalReranker.__init__`'s pointwise branch with the device and dtype as parameters."""
    _offline_env()
    import torch
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        path, device=device, trust_remote_code=trust_remote_code, local_files_only=True,
        model_kwargs={"dtype": getattr(torch, dtype)}, max_length=max_length,
        prompts={RERANK_PROMPT_NAME: instruction}, default_prompt_name=RERANK_PROMPT_NAME,
    )


def self_contained_reranker(profile: Mapping[str, Any],
                            model_root: str | Path | None = None) -> tuple[dict[str, Any], Path, int, int]:
    """(model config, model dir, query limit, document limit) from a profile that carries them itself.

    Every key the scoring depends on must be stated; nothing defaults silently except `trust_remote_code` (false, as
    the loop's pinned 4B config) and the dtype, which comes from `hardware_flags.dtype` (the profile's execution)."""
    required = ("model_dir", "instruction", "max_query_tokens", "max_document_tokens", "batch_size", "architecture",
                "hf_id", "revision")
    missing = [key for key in required if profile.get(key) in (None, "")]
    if missing:
        raise ModelError(f"rerank profile {profile.get('profile_id')!r} has no platform_config and lacks {missing}")
    flags = profile.get("hardware_flags") or {}
    model_cfg = {"hf_id": profile["hf_id"], "revision": profile["revision"], "architecture": profile["architecture"],
                 "dtype": str(flags.get("dtype") or "bfloat16"), "batch_size": int(profile["batch_size"]),
                 "trust_remote_code": bool(profile.get("trust_remote_code", False)),
                 "instruction": str(profile["instruction"])}
    model_dir = Path(model_root) if model_root is not None else Path(str(profile["model_dir"])).expanduser()
    return model_cfg, model_dir, int(profile["max_query_tokens"]), int(profile["max_document_tokens"])


class QwenReranker:
    """`Qwen/Qwen3-Reranker-4B` @ the profile's revision: one question's candidates per call.

    `rerank(question, texts)` scores ONE question's list in ONE `predict` call (split into the
    platform config's `batch_size` inside it, exactly as `LocalReranker.score`). Batches never mix
    questions: the co-residency run found bf16 scores move with their batch partners, so mixing
    questions would make a question's scores depend on which other question shared its batch.
    """

    def __init__(self, model: Any, *, profile: Mapping[str, Any], model_cfg: Mapping[str, Any],
                 model_dir: Path, device: str, dtype: str, query_limit: int, document_limit: int,
                 identity: Mapping[str, Any]) -> None:
        self.model = model
        self.tokenizer = model.tokenizer
        self.profile = dict(profile)
        self.cfg = dict(model_cfg)
        self.model_dir = Path(model_dir)
        self.device = device
        self.dtype = dtype
        self.query_limit = int(query_limit)
        self.document_limit = int(document_limit)
        self.batch_size = int(model_cfg["batch_size"])
        self.top_k_input, self.top_k_output = rerank_widths(profile)
        self.identity = dict(identity)
        self.calls = 0
        self.score_ms = 0

    @classmethod
    def from_profile(cls, profile: Mapping[str, Any] | str | Path, *, loop_root: str | Path,
                     model_root: str | Path | None = None, device: str | None = None,
                     loader: Callable[..., Any] | None = None,
                     cuda_available: Callable[[], bool] | None = None) -> "QwenReranker":
        profile = read_profile(profile)
        if profile.get("slot") != "rerank" or not profile.get("weight_files_sha256"):
            raise ModelError(f"not a model rerank profile: {profile.get('profile_id')!r}"
                             " (the identity profile none-v1 never loads a reranker)")
        rerank_widths(profile)
        if profile.get("platform_config"):
            config_path, config = read_platform_config(profile, loop_root)
            model_cfg = (config.get("rerankers") or {}).get(QWEN_RERANKER_KEY)
            if not isinstance(model_cfg, dict) or model_cfg.get("revision") != profile.get("revision"):
                raise ModelError("Qwen reranker profile differs from the pinned platform config")
            if model_cfg.get("hf_id") != profile.get("hf_id"):
                raise ModelError("Qwen reranker hf_id differs from the pinned platform config")
            model_dir = Path(model_root if model_root is not None else config["model_root"]) / model_cfg["local_dir"]
            query_limit, document_limit = int(config["max_query_tokens"]), int(config["max_document_tokens"])
            source: str = str(config_path)
        else:
            # A self-contained profile (qwen3-0.6b-rerank-v1): the model settings live in the profile
            # itself, so its sha (part of the route id) pins them; the loaded weights are still hash-checked below.
            model_cfg, model_dir, query_limit, document_limit = self_contained_reranker(profile, model_root)
            source = "profile"
        if model_cfg.get("architecture") != POINTWISE:
            raise ModelError(f"only {POINTWISE} is ported, not {model_cfg.get('architecture')!r}")
        started = time.monotonic()
        tokenizer = checked_sha(model_dir / "tokenizer.json", profile["tokenizer_sha256"], "Qwen reranker tokenizer")
        weight_rows = [{"path": name, "sha256": checked_sha(model_dir / name, expected, f"Qwen reranker weight {name}")}
                       for name, expected in sorted(profile["weight_files_sha256"].items())]
        combined = profile["model_weights_sha256"]
        # The combined identity is the item sha of the rows (multi-file models); a one-file model may name that file's
        # own sha instead (the 0.6B profile does). Every file is still checked one by one above.
        single_file = len(weight_rows) == 1 and combined == weight_rows[0]["sha256"]
        if item_sha256(weight_rows) != combined and not single_file:
            raise WeightsMismatch("combined Qwen reranker weight identity mismatch")
        verify_ms = int((time.monotonic() - started) * 1000)
        chosen = resolve_device(profile, device, cuda_available)
        dtype = str(model_cfg["dtype"]) if chosen == "cuda" else CPU_DTYPE
        started = time.monotonic()
        model = (loader or _load_cross_encoder)(
            str(model_dir), device=chosen, dtype=dtype, trust_remote_code=bool(model_cfg["trust_remote_code"]),
            max_length=query_limit + document_limit + 256, instruction=str(model_cfg["instruction"]))
        load_ms = int((time.monotonic() - started) * 1000)
        identity = {"platform_config": source, "model_weights_sha256": profile["model_weights_sha256"],
                    "tokenizer_sha256": tokenizer, "verify_ms": verify_ms, "load_ms": load_ms}
        return cls(model, profile=profile, model_cfg=model_cfg, model_dir=model_dir, device=chosen, dtype=dtype,
                   query_limit=query_limit, document_limit=document_limit, identity=identity)

    def rerank(self, question: str, texts: Sequence[str]) -> list[float]:
        """`LocalReranker.score`: one question, its candidates, one `predict` call."""
        if not isinstance(question, str):
            raise ModelError("rerank takes ONE question string per call (never a batch of questions)")
        texts = list(texts)
        if any(not isinstance(text, str) for text in texts):
            raise ModelError("rerank takes candidate texts only; (question, text) pairs would let"
                             " questions share a batch")
        if len(texts) > self.top_k_input:
            raise ModelError(f"{len(texts)} candidates exceed the profile's top_k_input {self.top_k_input}")
        if not texts:
            return []
        question = truncate(self.tokenizer, question, self.query_limit)
        documents = [truncate(self.tokenizer, text, self.document_limit) for text in texts]
        started = time.monotonic()
        scores = np.asarray(
            self.model.predict([(question, document) for document in documents],
                               batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True),
            dtype=np.float64,
        ).reshape(-1)
        self.score_ms += int((time.monotonic() - started) * 1000)
        self.calls += 1
        if scores.shape != (len(documents),):
            raise ModelError(f"unexpected score shape {scores.shape} for {len(documents)} candidates")
        if not np.isfinite(scores).all():
            raise ModelError("reranker returned non-finite scores")
        return [float(value) for value in scores]

    def describe(self) -> dict[str, Any]:
        flags = self.profile.get("hardware_flags") or {}
        return {
            "adapter": "QwenReranker", "profile_id": self.profile.get("profile_id"),
            "hf_id": self.profile.get("hf_id"), "revision": self.profile.get("revision"),
            "model_dir": str(self.model_dir), "device": self.device, "dtype": self.dtype,
            "profile_device": flags.get("device"), "profile_dtype": flags.get("dtype"),
            "top_k_input": self.top_k_input, "top_k_output": self.top_k_output,
            "batch_size": self.batch_size, "max_query_tokens": self.query_limit,
            "max_document_tokens": self.document_limit, "architecture": self.cfg.get("architecture"),
            "calls": self.calls, "score_ms": self.score_ms, **self.identity,
        }

    def close(self) -> None:
        self.model = None
        _free_cuda(self.device)


def _free_cuda(device: str) -> None:
    import gc

    gc.collect()
    if device == "cuda":
        import torch

        torch.cuda.empty_cache()


# ---------------------------------------------------------------- the Models adapter

class InProcessModels:
    """The `Models` protocol over a real embedder and, for reranking routes, a real reranker.

    `reranker=None` is the s0 shape (identity rerank): a `rerank` call is then a pipeline bug and
    is refused, never answered with a made-up order. Either side may be a stand-in with the same
    methods, so the eval can embed in one process phase and rerank in another.
    """

    def __init__(self, embedder: Any | None, reranker: Any | None = None) -> None:
        self.embedder = embedder
        self.reranker = reranker
        # `ceres-api` runs up to `max_inflight` (default 2) pipelines in
        # worker threads, and they share these model objects. HF fast tokenizers are not thread-safe
        # ("Already borrowed"), and both models sit on one GPU, so every model call is serialised here.
        self._lock = threading.Lock()

    def embed_query(self, text: str, lang: str) -> np.ndarray:
        if self.embedder is None:
            raise ModelError("no embedder loaded in this process")
        with self._lock:
            return self.embedder.embed_query(text, lang)

    def embed_queries(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if self.embedder is None:
            raise ModelError("no embedder loaded in this process")
        with self._lock:
            return self.embedder.embed_queries(rows)

    def rerank(self, question: str, texts: Sequence[str]) -> list[float]:
        if self.reranker is None:
            raise ModelError("no reranker loaded: an identity-rerank route must not call rerank")
        with self._lock:
            return self.reranker.rerank(question, texts)

    def health(self) -> dict[str, str]:
        def state(adapter: Any) -> str:
            if adapter is None:
                return "absent"
            return "loaded" if getattr(adapter, "model", None) is not None else "closed"

        parts = []
        for adapter in (self.embedder, self.reranker):
            if adapter is not None and hasattr(adapter, "describe"):
                info = adapter.describe()
                parts.append(f"{info.get('adapter')}:{info.get('profile_id')}@{info.get('device')}")
        return {"embedder": state(self.embedder), "reranker": state(self.reranker),
                "detail": ", ".join(parts) or "no models"}

    def describe(self) -> dict[str, Any]:
        return {"embedder": self.embedder.describe() if self.embedder is not None else None,
                "reranker": self.reranker.describe() if self.reranker is not None else "identity"}

    def close(self) -> None:
        """Close both models (`ceres-api` shutdown). Each side is closed even if the other fails."""
        failures = []
        for adapter in (self.reranker, self.embedder):
            close = getattr(adapter, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:                  # noqa: BLE001 — close the other side too
                    failures.append(f"{type(adapter).__name__}: {exc.__class__.__name__}")
        if failures:
            raise ModelError("close failed for " + ", ".join(failures))


# ================================================================ shared builders (api + eval)
# ONE place that turns a route's
# profiles into loaded models and the answerer client: `ceres-eval` (per phase) and `ceres-api`
# (once at startup) both call these, so the two cannot drift.

def is_identity_rerank(profile: Mapping[str, Any] | str | Path) -> bool:
    """`operation: identity` (`none-v1`, the s0 routes): the route has no reranker."""
    return read_profile(profile).get("operation") == "identity"


def model_cache_override() -> Path | None:
    """CERES_MODEL_CACHE, when set: the Hugging Face cache root (the directory that holds `hub/`) for every model of the route.
    It overrides the cache named in a pinned platform config and a profile's `model_dir`, so a checkout runs on any machine."""
    value = os.environ.get("CERES_MODEL_CACHE")
    return Path(value).expanduser() if value else None


def hf_snapshot(cache: Path, hf_id: str, revision: str) -> Path:
    """The Hugging Face cache layout: <cache>/hub/models--<org>--<name>/snapshots/<revision>."""
    return Path(cache) / "hub" / f"models--{hf_id.replace('/', '--')}" / "snapshots" / revision


def build_embedder(profile: Mapping[str, Any] | str | Path, *, loop_root: str | Path,
                   device: str | None = None) -> JinaEmbedder:
    """The route's query embedder on the profile's device (or `device`, tests only)."""
    return JinaEmbedder.from_profile(profile, loop_root=loop_root, device=device, model_cache=model_cache_override())


def build_reranker(profile: Mapping[str, Any] | str | Path, *, loop_root: str | Path,
                   device: str | None = None) -> QwenReranker | None:
    """None for an identity rerank profile; otherwise the loaded reranker, or `ModelError`.

    A reranker that cannot load (missing weights, a hash mismatch, CUDA out of memory beside the
    answerer) raises a `ModelError` that names the profile, the device and the cause. It never
    returns None for a non-identity profile: serving such a route without its reranker would be
    another route (the identity recipe) under this route's name.
    """
    loaded = read_profile(profile)
    if loaded.get("operation") == "identity":
        return None
    flags = loaded.get("hardware_flags") or {}
    where = device or flags.get("device") or "cuda"
    try:
        cache = model_cache_override()
        root = hf_snapshot(cache, loaded["hf_id"], loaded["revision"]) if cache and loaded.get("hf_id") and loaded.get("revision") else None
        return QwenReranker.from_profile(loaded, loop_root=loop_root, device=device, model_root=root)
    except Exception as exc:                              # noqa: BLE001 — re-raised with context
        raise ModelError(
            f"the route's reranker {loaded.get('profile_id')!r} could not load on {where}:"
            f" {exc.__class__.__name__}: {exc}. There is no fallback to identity rerank — free the"
            " device (or serve an identity-rerank route) and start again") from exc


def build_answerer(endpoint: Any) -> Any:
    """`OpenAIChatLLM` for a configured `endpoints.answerer` (`config.EndpointSection`)."""
    from .llm import OpenAIChatLLM

    return OpenAIChatLLM(endpoint.base_url, timeout_s=float(endpoint.request_timeout_s),
                         expect_model=endpoint.expect_model, health_url=endpoint.health_url)
