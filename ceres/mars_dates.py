"""Date reader v2 for MARS bulletin questions, fixing failures of v1 found in review: numeric "YYYY MM" issues, an event month overriding the named issue, a comparison
range overriding the target year, Romanian months, and the verb "may".

question_issue(text) -> (year, month|None, how) or None. A date next to an issue word (bulletin, report, issue, edition, MARS, JRC,
buletin, raport) wins; then a "Month YYYY" / "YYYY MM" date; then a lone year. Years inside a range ("2016-2020", "2016 to 2020") or
after "average"/"compared"/"since"/"than" are comparison years and are skipped.
issue_of_parent(parent_id) -> (year, month) for "mars-<region>-YYYY-MM-…" parents, else None."""
from __future__ import annotations

import re

MONTHS = {}
for i, names in enumerate([("january", "jan", "ianuarie"), ("february", "feb", "februarie"), ("march", "mar", "martie"),
                           ("april", "apr", "aprilie"), ("may", "mai"), ("june", "jun", "iunie"), ("july", "jul", "iulie"),
                           ("august", "aug"), ("september", "sep", "sept", "septembrie"), ("october", "oct", "octombrie"),
                           ("november", "nov", "noiembrie"), ("december", "dec", "decembrie")], start=1):
    for n in names:
        MONTHS[n] = i
MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
YEAR = r"(?:19[89]\d|20[0-3]\d)"
ISSUE_WORDS = r"(?:bulletin|report|issue|edition|mars|jrc|buletin(?:ul)?|raport(?:ul)?)"
RANGE = re.compile(rf"\b{YEAR}\s*(?:[-–—/]|to|and|până la|si|și)\s*{YEAR}\b", re.I)
COMPARE_BEFORE = re.compile(r"(?:average|mean|compared(?: with| to)?|since|than|versus|vs\.?|relative to|media|față de|fata de)\s+(?:of\s+|the\s+|in\s+)*$", re.I)
MONTH_YEAR = re.compile(rf"\b({MONTH_RE})\.?\s+(?:of\s+)?({YEAR})\b", re.I)
YEAR_MONTH_NUM = re.compile(rf"\b({YEAR})[\s\-_/.](0[1-9]|1[0-2])\b")
YEAR_ONLY = re.compile(rf"\b({YEAR})\b")


def _month_ok(word: str, text: str, start: int) -> bool:
    """'may'/'mai'/'mar' are only months when capitalised as in the text ("May", "Mai") — the verb "may" is lower case."""
    w = text[start:start + len(word)]
    if word.lower() in ("may", "mai", "mar", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "jan", "feb", "apr"):
        return w[:1].isupper()
    return True


def _masked(text: str) -> str:
    """Blank out year ranges and comparison years so they cannot be read as the target issue."""
    t = RANGE.sub(lambda m: " " * len(m.group(0)), text)
    out = list(t)
    for m in YEAR_ONLY.finditer(t):
        if COMPARE_BEFORE.search(t[:m.start()][-40:]):
            out[m.start():m.end()] = " " * (m.end() - m.start())
    return "".join(out)


def question_issue(text: str):
    t = _masked(text)
    cands = []
    for m in MONTH_YEAR.finditer(t):
        if _month_ok(m.group(1), t, m.start(1)):
            cands.append((m.start(), int(m.group(2)), MONTHS[m.group(1).lower()], "month-year"))
    for m in YEAR_MONTH_NUM.finditer(t):
        cands.append((m.start(), int(m.group(1)), int(m.group(2)), "year-month-numeric"))
    if cands:
        near = [c for c in cands if re.search(ISSUE_WORDS, t[max(0, c[0] - 40):c[0] + 30], re.I)]
        pick = min(near or cands, key=lambda c: c[0])
        return pick[1], pick[2], pick[3] + ("+issue-word" if near else "")
    years = [(m.start(), int(m.group(1))) for m in YEAR_ONLY.finditer(t)]
    if years:
        near = [y for y in years if re.search(ISSUE_WORDS, t[max(0, y[0] - 40):y[0] + 30], re.I)]
        pick = min(near or years, key=lambda y: y[0])
        return pick[1], None, "year" + ("+issue-word" if near else "")
    return None


PARENT = re.compile(r"^mars-[a-z-]+?-(\d{4})-(\d{2})(?:-|:|$)")


def issue_of_parent(parent_id: str):
    m = PARENT.match(parent_id)
    return (int(m.group(1)), int(m.group(2))) if m else None
