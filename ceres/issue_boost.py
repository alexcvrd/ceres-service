"""MARS issue boost (rerank profile qwen3-0.6b-rerank-v3).

After the reranker scores the candidates: final = logit + beta for a candidate whose MARS issue (from its parent id) matches the
question's issue; then order by final, ties in dense order. A match is the same year and month, or the same year when the question
names no month (`window: same`). Non-MARS parents never match; a question with no parsed date gets no boost.

The date reader is the loop's `04_rag/analytics/mars_dates.py`, vendored byte-for-byte as `ceres/mars_dates.py` and checked against
the profile's `date_reader_sha256` (as `hybrid_v1.py` is). `boosted_order` is pure, so the parity check runs it on the loop's exact
candidates and logits.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

from . import mars_dates

DATE_READER_PATH = Path(__file__).with_name("mars_dates.py")
WINDOWS = ("same",)


class IssueBoostError(Exception):
    """The profile's issue boost cannot run as stated."""


def date_reader_sha256() -> str:
    return hashlib.sha256(DATE_READER_PATH.read_bytes()).hexdigest()


def boost_config(rerank_profile: dict[str, Any]) -> dict[str, Any] | None:
    """The profile's `issue_boost` (None when absent); refused unless window and date-reader sha are what Ceres implements."""
    boost = rerank_profile.get("issue_boost")
    if boost is None:
        return None
    name = rerank_profile.get("profile_id")
    if boost.get("window") not in WINDOWS:
        raise IssueBoostError(f"rerank profile {name!r}: issue_boost window {boost.get('window')!r}; Ceres implements {WINDOWS}")
    if boost.get("date_reader_sha256") != date_reader_sha256():
        raise IssueBoostError(f"rerank profile {name!r} names date reader {str(boost.get('date_reader_sha256'))[:12]}…; the vendored"
                              f" ceres/mars_dates.py is {date_reader_sha256()[:12]}…")
    return {"beta": float(boost["beta"]), "window": boost["window"], "date_reader_sha256": date_reader_sha256()}


def matches(target: Any, issue: Any) -> bool:
    """`window: same` — the loop's `mars_boost.match(target, issue, "same")`."""
    if not target or not issue:
        return False
    year, month = target[0], target[1]
    if month is None:
        return issue[0] == year
    return tuple(issue) == (year, month)


def boosted_order(question: str, parent_ids: Sequence[str], scores: Sequence[float], beta: float,
                  top_k: int) -> tuple[list[int], Any, int]:
    """(indices of the top_k by final score, ties in dense order; the parsed target [year, month|None] or None; how many
    candidates matched)."""
    target = mars_dates.question_issue(question)
    boosted = [matches(target, mars_dates.issue_of_parent(parent)) for parent in parent_ids]
    final = [float(score) + (beta if hit else 0.0) for score, hit in zip(scores, boosted)]
    order = sorted(range(len(final)), key=lambda i: (-final[i], i))[:top_k]
    return order, (list(target[:2]) if target else None), sum(boosted)
