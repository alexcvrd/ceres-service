"""Candidate validator rule r4 (measurement only). NOT used in serving.

Kept in its own module so `pipeline.py`'s served validator (lines 336–578, copied byte-for-byte by the DDS
acceptance suite) is untouched. Serving stays `ceres-validate-r3`; this module is for measurement only.
"""

from __future__ import annotations

from typing import Any

from .pipeline import validate

R4_MIN_QUOTE_WORDS, R4_MIN_QUOTE_CHARS = 3, 15
R4_CANDIDATE_REVISION = "ceres-validate-r4-candidate"


def continuous_quote_r4_errors(candidate: Any, packet: dict[str, Any]) -> list[str]:
    """Candidate rule for a future `ceres-validate-r4`. NOT used in serving.

    r1–r3 accept a continuous `support_quote` when its whitespace-collapsed, casefolded text is ANY substring of the
    passage: "5" passes against "15 kg", "NU" against "nu" inside a word. r4 would also require (a) at least
    R4_MIN_QUOTE_WORDS words or R4_MIN_QUOTE_CHARS characters and (b) a match that starts and ends at word
    boundaries. Only quotes that pass as one continuous substring are checked here; ellipsis quotes keep their own
    quote guards (at least 3 words per piece).
    """
    if not isinstance(candidate, dict) or not isinstance(candidate.get("citations"), list):
        return []
    contexts = {value["evidence_id"]: value for value in packet.get("contexts", [])}
    errors: list[str] = []
    for citation in candidate["citations"]:
        if not isinstance(citation, dict):
            continue
        quote, evidence_id = citation.get("support_quote"), citation.get("evidence_id")
        if not isinstance(quote, str) or evidence_id not in contexts:
            continue
        needle = " ".join(quote.split()).casefold()
        hay = " ".join(contexts[evidence_id]["evidence_text"].split()).casefold()
        if not needle or needle not in hay:
            continue                                   # not a continuous quote: r3's own rules apply
        exact_needle = " ".join(quote.split())
        exact_hay = " ".join(contexts[evidence_id]["evidence_text"].split())
        first_swapped = exact_needle[:1].swapcase() + exact_needle[1:]
        if exact_needle not in exact_hay and first_swapped not in exact_hay:
            # An independent review found that r3 casefolds, so "SOW 15 KG" passes against "sow 15 kg"; not verbatim.
            # Only the first letter may differ in case: a quote that starts mid-sentence is often capitalised
            # (recorded case: "Încingerea …" quoting "… încingerea …"; 4 answers, one RO question, 4 routes).
            errors.append("r4: continuous support_quote differs from the passage in letter case")
            continue
        if len(needle.split()) < R4_MIN_QUOTE_WORDS and len(needle) < R4_MIN_QUOTE_CHARS:
            errors.append("r4: continuous support_quote is shorter than 3 words and 15 characters")
            continue
        bounded = False
        start = hay.find(needle)
        while start != -1:
            end = start + len(needle)
            before_ok = start == 0 or not hay[start - 1].isalnum() or not needle[0].isalnum()
            after_ok = end == len(hay) or not hay[end].isalnum() or not needle[-1].isalnum()
            if before_ok and after_ok:
                bounded = True
                break
            start = hay.find(needle, start + 1)
        if not bounded:
            errors.append("r4: continuous support_quote does not start and end at word boundaries")
    return errors


def validate_r4_candidate(candidate: Any, packet: dict[str, Any]) -> list[str]:
    """`validate()` (r3) plus the candidate r4 rule. For measurement only; serving keeps r3."""
    return validate(candidate, packet) + continuous_quote_r4_errors(candidate, packet)
