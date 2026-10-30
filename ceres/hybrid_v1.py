"""hybrid-v1 reference module: lexical (BM25) leg, fusion and union for DDS/Ceres hybrid retrieval.

Pure Python + numpy; no I/O. The loop's measurement harness and Ceres's eval path use this file byte-for-byte (Ceres vendors it, as it
vendors the served validator), so the lexical ranking and the fusion are identical by construction; parity is then tested end to end.
"""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import Counter

import numpy as np

VERSION = "hybrid-v1"
K1, B = 1.2, 0.75
RRF_K = 60
DEPTH = 500            # candidates taken from each leg before fusion / dedup
_FOLD = str.maketrans({"ş": "ș", "ţ": "ț", "Ş": "Ș", "Ţ": "Ț"})


def tokens(text: str) -> list[str]:
    """Maximal runs of Unicode letters or digits after NFC, cedilla→comma-below folding and lowercasing. No stemming, no stopwords.
    Digits are kept (doses, rates, years); punctuation splits runs, so "30-100" → ["30", "100"] and "91.5" → ["91", "5"]."""
    out: list[str] = []
    cur: list[str] = []
    for ch in unicodedata.normalize("NFC", text).translate(_FOLD).lower():
        if unicodedata.category(ch)[0] in "LN":
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


class BM25Index:
    """Okapi BM25 over a fixed document list (k1 1.2, b 0.75, idf = ln(1 + (N - df + 0.5) / (df + 0.5))), inverted index."""

    def __init__(self, ids: list[str], texts: list[str]):
        if len(ids) != len(texts) or len(set(ids)) != len(ids):
            raise ValueError("ids must be unique and aligned with texts")
        if any(not isinstance(t, str) for t in texts):
            raise ValueError("every document needs a text (missing represented_text refused)")
        self.ids = np.asarray(ids, dtype=object)
        postings: dict[str, tuple[list[int], list[int]]] = {}
        lengths = np.zeros(len(texts), dtype=np.float64)
        for i, text in enumerate(texts):
            counts = Counter(tokens(text))
            lengths[i] = sum(counts.values())
            for term, tf in counts.items():
                entry = postings.setdefault(term, ([], []))
                entry[0].append(i)
                entry[1].append(tf)
        self.n = len(texts)
        self.avgdl = float(lengths.mean()) if self.n and lengths.mean() else 1.0
        self.norm = K1 * (1.0 - B + B * lengths / self.avgdl)
        self.post = {t: (np.asarray(a, dtype=np.int64), np.asarray(f, dtype=np.float64)) for t, (a, f) in postings.items()}
        df = sorted((t, len(v[0])) for t, v in self.post.items())
        content = hashlib.sha256()
        for doc_id, text in sorted(zip(ids, texts)):          # canonical order; binds each id to its exact text
            content.update(doc_id.encode() + b"\x1f" + hashlib.sha256(text.encode()).hexdigest().encode() + b"\n")
        self.content_sha256 = content.hexdigest()
        self.stats_sha256 = hashlib.sha256(json.dumps({"version": VERSION, "n": self.n, "avgdl": round(self.avgdl, 9), "df": df,
                                                       "content_sha256": self.content_sha256},
                                                      ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float64)
        for term in dict.fromkeys(tokens(query)):
            hit = self.post.get(term)
            if hit is None:
                continue
            idx, tf = hit
            idf = math.log(1.0 + (self.n - len(idx) + 0.5) / (len(idx) + 0.5))
            s[idx] += idf * tf * (K1 + 1.0) / (tf + self.norm[idx])
        return s

    def ranked(self, query: str, depth: int = DEPTH, keep: set[str] | frozenset[str] | None = None) -> list[str]:
        """Top `depth` document ids by score, ties by id (lexsort on (-score, id)); zero-score documents are excluded. With `keep`
        (the read-predicate id set), non-kept documents are removed BEFORE truncation (filter-first), while the statistics stay
        those of the whole indexed corpus."""
        s = self.scores(query)
        order = np.lexsort((self.ids, -s))
        out: list[str] = []
        for i in order:
            if s[i] <= 0.0:
                break
            doc = str(self.ids[i])
            if keep is None or doc in keep:
                out.append(doc)
                if len(out) == depth:
                    break
        return out


def rank_by_score(ids: list[str], scores: np.ndarray, depth: int = DEPTH) -> list[str]:
    """Shared tie rule for any scored leg: (-score, id)."""
    arr = np.asarray(ids, dtype=object)
    order = np.lexsort((arr, -np.asarray(scores, dtype=np.float64)))
    return [str(arr[i]) for i in order[:depth]]


def rrf(lists: list[list[str]], k: int = RRF_K, depth: int = DEPTH, out_depth: int = DEPTH) -> list[str]:
    """Reciprocal-rank fusion of ranked id lists (each truncated to `depth`, ranks counted after the read predicate); ties by id.
    The fused output is capped at `out_depth` (500) before dedup, as in the headroom experiment."""
    fused: dict[str, float] = {}
    for ranked in lists:
        for rank, doc in enumerate(ranked[:depth], start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return [d for d, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))][:out_depth]


def union_candidates(primary: list[dict], extra: list[dict], extra_n: int) -> list[dict]:
    """Union candidates for reranked routes: all `primary` survivors in order, then the
    FIRST `extra_n` `extra` survivors in order. The caller then runs its standard dedup (tiny → parent keep-best → near-duplicate Jaccard)
    over this concatenation with top_k = len(primary) + extra_n, so a lexical row is kept only if its parent is new AND it is not a
    near-duplicate of a kept row (review D6: cross-leg near-duplicates). Position in the deduped pool is the reranker's tie-break."""
    return list(primary) + list(extra[:extra_n])
