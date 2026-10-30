"""norm-v3 text normalization and the token-Jaccard near-duplicate measure.

Ported verbatim (behaviour, not layout) from the DDS loop so that Ceres dedups exactly as the
judged route r1 did:

  `loop-v0/00_control/code/locator.py`  — `normalize`, `_dehyphenate`, `_PUNCT_TRANSLATION`,
      `_CEDILLA_TRANSLATION`, `NORM_VERSION`.
  `loop-v0/04_rag/slot_adapter.py:1098` — `token_jaccard`.

The loop's copy stays the source of truth; this is a serving-side copy because the service must
not import the loop at request time (no loop checkout on the AI server). `tests/test_pipeline.py`
pins the behaviour; any divergence from the loop is a route change, not a bug fix.
"""

from __future__ import annotations

import re
import unicodedata

# locator.py:31 — norm-v3 = norm-v2 plus the Romanian cedilla→comma-below fold (ş→ș, ţ→ț).
# The fold KEEPS the diacritic; it only unifies the legacy cedilla encoding of older PDF text
# layers with the comma-below letters vision OCR emits. Evidence for the fold:
# `00_control/code/checks/cedilla_fold_probe_x2.py` (unmatched text blocks 1,298 → 429).
NORM_VERSION = "norm-v3"
_SUPPORTED_NORM_VERSIONS = ("norm-v2", "norm-v3")
_CEDILLA_TRANSLATION = str.maketrans(
    {"ş": "ș", "ţ": "ț", "Ş": "Ș", "Ţ": "Ț"}
)

# locator.py:44 — typographic punctuation folded to ASCII after NFC. Diacritics are never stripped.
_PUNCT_TRANSLATION = str.maketrans(
    {
        "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
        "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
        "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
        "―": "-", "−": "-",
        "…": "...",
        " ": " ",
    }
)
_LINE_HYPHEN = re.compile(r"([^\W\d_])-\s*\r?\n\s*([^\W\d_])", re.UNICODE)


def _dehyphenate(value: str) -> str:
    """Join only letter-hyphen-linebreak-lowercase continuations (locator.py:95)."""

    def replace(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        return left + right if right.islower() else match.group(0)

    return _LINE_HYPHEN.sub(replace, value)


def normalize(value: str, version: str = NORM_VERSION) -> str:
    """Apply norm-v2 or norm-v3 without case folding or diacritic removal (locator.py:130)."""
    if version not in _SUPPORTED_NORM_VERSIONS:
        raise ValueError(f"unsupported normalization version: {version}")
    value = unicodedata.normalize("NFC", value)
    value = value.replace("­", "")  # soft hyphen
    value = _dehyphenate(value)
    value = value.translate(_PUNCT_TRANSLATION)
    if version == "norm-v3":
        value = value.translate(_CEDILLA_TRANSLATION)
    return " ".join(value.split())


def token_jaccard(left: str, right: str, normalization: str = NORM_VERSION) -> float:
    """Set Jaccard over whitespace tokens of the normalized texts (slot_adapter.py:1098).

    Whitespace tokens, not a tokenizer: the retrieve profile `dense-top10-v2` measures
    near-duplicates this way and its threshold (0.6) is calibrated against this exact measure.
    """
    left_tokens = set(normalize(left, normalization).split())
    right_tokens = set(normalize(right, normalization).split())
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
