"""Readable sources: title, authors and a page label for every passage of an answer.

The served store carries each item's `work_id`; the work's title lives in the active release's census (`census.jsonl`,
`preferred_title`) under the loop root. About 60 % of those titles are the PDF file name ("Paul A. Horne, Jessica Page - Integrated
Pest Management for Crops and Pastures.pdf"), so `parse_title` turns the "Authors - Title.pdf" convention into fields. The page label
is the PDF page (the section id's 0-based page index + 1), not the printed page number, and says so ("PDF p. 76").

Read-only: the census file is opened, never written. Off by default (`answer.source_titles`); when off nothing here runs.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping

_ROLE = re.compile(r"\s*\((?:auth|eds?|editors?|coord|ed\.)\.?\)\s*", re.I)
_PAGE = re.compile(r"[:-]page-(\d+)$")
_SECTION = re.compile(r"[:-]section-(\d+)$")


def parse_title(raw: str | None) -> dict[str, Any]:
    """'Authors - Title.pdf' → {title, authors}; anything else → {title: cleaned raw, authors: []}. Never invents a field."""
    text = " ".join(str(raw or "").split())
    if not text:
        return {"title": None, "authors": []}
    is_file = text.lower().endswith(".pdf")
    if is_file:
        text = text[:-4].rstrip()
    text = text.replace("_ ", ": ").replace("_", " ")
    text = " ".join(text.split())
    authors: list[str] = []
    if is_file and " - " in text:
        head, tail = text.split(" - ", 1)
        if tail.strip() and len(head) <= 160:
            head = _ROLE.sub(" ", head).strip(" ,;")
            authors = [a.strip() for a in re.split(r"\s*[;,]\s*(?=[A-Z][^,;]*\s)", head) if a.strip()] if head else []
            if len(authors) == 2 and " " not in authors[0] and ";" not in head:
                authors = [f"{authors[1]} {authors[0]}"]          # "Surname, Given Names" is one author
            text = tail.strip()
    return {"title": text or None, "authors": authors}


def page_label(section_id: str | None) -> str | None:
    """'…:page-0075' → 'PDF p. 76'; '…-section-0005' → 'section 5'; otherwise None."""
    if not section_id:
        return None
    m = _PAGE.search(section_id)
    if m:
        return f"PDF p. {int(m.group(1)) + 1}"
    m = _SECTION.search(section_id)
    return f"section {int(m.group(1))}" if m else None


class SourceCatalog:
    """work_id → parsed title, per release, loaded lazily from `<loop_root>/02_release/runs/<cohort>/<attempt>/census.jsonl`."""

    def __init__(self, loop_root: Path) -> None:
        self.loop_root = Path(loop_root)
        self._by_release: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def census_path(self, release_attempt_id: str) -> Path | None:
        if "/" not in (release_attempt_id or ""):
            return None                                    # not a loop release id (fixtures): no titles, no failure
        cohort, attempt = release_attempt_id.split("/", 1)
        return self.loop_root / "02_release" / "runs" / cohort / attempt / "census.jsonl"

    def works(self, release_attempt_id: str) -> dict[str, dict[str, Any]]:
        with self._lock:
            if release_attempt_id not in self._by_release:
                path = self.census_path(release_attempt_id)
                table: dict[str, dict[str, Any]] = {}
                if path is not None and path.is_file():
                    with open(path, encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                row = json.loads(line)
                                table[row["work_id"]] = {**parse_title(row.get("preferred_title")),
                                                         "language": _lang(row.get("language"))}
                self._by_release[release_attempt_id] = table
            return self._by_release[release_attempt_id]

    def describe(self, release_attempt_id: str, hits: Iterable[Any]) -> list[dict[str, Any]]:
        """One entry per passage, in evidence order: {evidence_id, work_id, title, authors, language, page_label}."""
        table = self.works(release_attempt_id)
        out = []
        for hit in hits:
            work = table.get(getattr(hit, "work_id", None)) or {}
            out.append({"evidence_id": hit.evidence_id, "work_id": getattr(hit, "work_id", None), "title": work.get("title"),
                        "authors": work.get("authors", []), "language": work.get("language"),
                        "page_label": page_label(getattr(hit, "parent_id", None))})
        return out


def _lang(values: Any) -> str | None:
    if isinstance(values, list):
        for v in values:
            if isinstance(v, Mapping) and v.get("source") == "declared" and v.get("value"):
                return v["value"]
        for v in values:
            if isinstance(v, Mapping) and v.get("value"):
                return v["value"]
    return None
