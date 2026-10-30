"""Readable sources: title, authors and page label per passage, from the release census.

How it can fail, and the test that catches each:
  * "Authors - Title.pdf" split wrongly (title in authors, '.pdf' left, '_ ' not restored to ':')   → test_parse_title_file_names
  * "Surname, Given Names" read as two authors                                                   → test_parse_title_file_names
  * a non-file title (MARS bulletins, Romanian courses) mangled or given invented authors         → test_parse_title_plain
  * the page label shows the 0-based index, or a section as a page                               → test_page_label
  * a missing census or a non-loop release id crashes the answer                                 → test_catalog_missing_census_is_harmless
  * passages and sources get out of order, or a work id is mismatched                             → test_catalog_describe
  * with the switch off the response changes                                                      → test_switch_off_no_sources
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from ceres import pipeline
from ceres.config import load_config
from ceres.contracts import AskRequest
from ceres.sources import SourceCatalog, page_label, parse_title
from ceres.testing import EN_QUESTION

from conftest import EXAMPLE_CONFIG


def test_parse_title_file_names():
    """Bug: the file-name convention is split wrongly."""
    assert parse_title("Paul A. Horne, Jessica Page - Integrated Pest Management for Crops and Pastures.pdf") == \
        {"title": "Integrated Pest Management for Crops and Pastures", "authors": ["Paul A. Horne", "Jessica Page"]}
    assert parse_title("KeShun Liu (auth.) - Soybeans_ Chemistry, Technology, and Utilization.pdf") == \
        {"title": "Soybeans: Chemistry, Technology, and Utilization", "authors": ["KeShun Liu"]}
    assert parse_title("Nita M. (Ed.) - Fungicides - Showcases of Integrated Plant Disease Management.pdf")["title"] == \
        "Fungicides - Showcases of Integrated Plant Disease Management"
    assert parse_title("Pritchett, William Lawrence - Nitrogen fertilization of small grain.pdf")["authors"] == ["William Lawrence Pritchett"]


def test_parse_title_plain():
    """Bug: a title that is not a file name gets invented authors or loses words."""
    assert parse_title("JRC MARS Bulletin 2021 06 vol29 no6") == {"title": "JRC MARS Bulletin 2021 06 vol29 no6", "authors": []}
    assert parse_title("Fungicides - Showcases") == {"title": "Fungicides - Showcases", "authors": []}
    assert parse_title(None) == {"title": None, "authors": []}


def test_page_label():
    """Bug: 'page-0075' shown as page 75 (it is the 0-based PDF index), or a section shown as a page."""
    assert page_label("ebooks_2026-629ffc430ea0:page-0075") == "PDF p. 76"
    assert page_label("ipm-crops-pastures-section-0005") == "section 5"
    assert page_label("x") is None and page_label(None) is None


def test_catalog_missing_census_is_harmless(tmp_path):
    """Bug: no census file (or a fixture release id without a cohort) crashes the answer."""
    cat = SourceCatalog(tmp_path)
    hit = SimpleNamespace(evidence_id="E1", work_id="w:1", parent_id="a:page-0001")
    assert cat.describe("fixture-inline-0001", [hit])[0]["title"] is None
    assert cat.describe("all/release-attempt-9999", [hit])[0]["page_label"] == "PDF p. 2"


def test_catalog_describe(tmp_path):
    """Bug: sources come back out of evidence order or attached to the wrong work."""
    d = tmp_path / "02_release" / "runs" / "all" / "release-attempt-0042"
    d.mkdir(parents=True)
    (d / "census.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"work_id": "w:a", "preferred_title": "A. Author - Book A.pdf", "language": [{"source": "declared", "value": "en"}]},
        {"work_id": "w:b", "preferred_title": "Curs de fitotehnie", "language": [{"source": "detector", "value": "ro"}]}]) + "\n")
    hits = [SimpleNamespace(evidence_id="E1", work_id="w:b", parent_id="b:section-0003"),
            SimpleNamespace(evidence_id="E2", work_id="w:a", parent_id="a:page-0009")]
    out = SourceCatalog(tmp_path).describe("all/release-attempt-0042", hits)
    assert [(o["evidence_id"], o["title"], o["page_label"], o["language"]) for o in out] == \
        [("E1", "Curs de fitotehnie", "section 3", "ro"), ("E2", "Book A", "PDF p. 10", "en")]


def test_switch_off_no_sources(deps, env, inline_stack):
    """Bug: with answer.source_titles off the response changes; with it on the answer path breaks."""
    off = pipeline.ask(deps, AskRequest(question=EN_QUESTION, lang="en"))
    assert "sources" not in json.loads(off.model_dump_json())["meta"]
    cfg = load_config(EXAMPLE_CONFIG, dict(env, CERES_ANSWER__SOURCE_TITLES="true"), require_api_key=False)
    store, models, _ = inline_stack
    from ceres.llm import FakeLLM
    on = pipeline.ask(pipeline.Deps.build(cfg, store, models, FakeLLM()), AskRequest(question=EN_QUESTION, lang="en"))
    assert [s["evidence_id"] for s in on.meta.sources] == [c["evidence_id"] for c in on.packet["contexts"]]
    assert on.packet == off.packet and on.answer == off.answer
