"""Contracts: the request model, the question-id encoding, and the two loop schemas."""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

import pytest
from pydantic import ValidationError

from ceres.contracts import (
    ANSWER_SCHEMA,
    PACKET_SCHEMA,
    PACKET_SCHEMA_SHA256,
    QUESTION_ID_DOMAIN,
    SCHEMA_DIR,
    UNIT_SEPARATOR,
    AnswerRow,
    AskRequest,
    Context,
    Packet,
    question_id_for,
    schema_errors,
    validate_answer_row,
    validate_packet,
)


# ---------------------------------------------------------------- question id

def test_question_id_encoding_is_reproducible_by_hand():
    """Anyone can recompute the id from the documented string; nothing hidden in the function."""
    question, lang = "  Ce densitate?  ", "ro"
    expected = hashlib.sha256(
        UNIT_SEPARATOR.join((QUESTION_ID_DOMAIN, lang, unicodedata.normalize("NFC", question.strip())))
        .encode("utf-8")).hexdigest()
    assert question_id_for(question, lang) == expected
    assert len(expected) == 64


def test_question_id_depends_on_the_language_and_ignores_surrounding_space():
    assert question_id_for("x", "ro") != question_id_for("x", "en")
    assert question_id_for(" x ", "en") == question_id_for("x", "en")


def test_nfc_normalization_folds_the_two_unicode_spellings():
    """`ș` written as a combining sequence and as a single code point are the same question."""
    assert question_id_for("grâu și", "ro") == question_id_for("grâu și", "ro")


# ---------------------------------------------------------------- AskRequest

def test_ask_request_requires_a_language():
    with pytest.raises(ValidationError):
        AskRequest(question="x")


def test_ask_request_rejects_an_unsupported_language():
    with pytest.raises(ValidationError):
        AskRequest(question="x", lang="fr")


def test_ask_request_rejects_an_empty_question_and_unknown_fields():
    with pytest.raises(ValidationError):
        AskRequest(question="", lang="en")
    with pytest.raises(ValidationError):
        AskRequest(question="x", lang="en", max_citations=3)


def test_a_supplied_bank_question_id_is_kept_verbatim():
    """The eval hook passes the bank's id so the DDS evaluation step can join on it."""
    bank = "a" * 64
    assert AskRequest(question="x", lang="en", question_id=bank).resolved_question_id() == bank
    with pytest.raises(ValidationError):
        AskRequest(question="x", lang="en", question_id="not-a-sha")


# ---------------------------------------------------------------- the loop schemas

def _packet() -> dict:
    text = "Winter wheat is sown at 450 to 550 germinable grains per square metre."
    return Packet(
        packet_id="grounded-" + "b" * 64, question_id="b" * 64, question="q", language="en",
        contexts=[Context(evidence_id="E1", chunk_id="c" * 64, parent_id="p1", evidence_text=text,
                          evidence_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                          locator_ids=["l1"], lineage_ids=["d" * 64], warnings=["table_unvalidated"])],
    ).to_dict()


def test_a_built_packet_validates_against_the_loop_schema():
    validate_packet(_packet())


def test_the_packet_schema_pins_the_identifier_shapes():
    broken = _packet()
    broken["question_id"] = "short"
    assert schema_errors(broken, PACKET_SCHEMA)
    extra = _packet()
    extra["contexts"][0]["unexpected"] = 1
    assert schema_errors(extra, PACKET_SCHEMA)


LOOP_PACKET_SCHEMA = Path("/path/to/dds/loop-v0/00_control/schemas/rag-answer-packets.schema.json")


def test_the_packet_schema_is_the_engine_schema_byte_for_byte():
    """the shipped copy is not the loop's schema (a context carrying `evidence_basis`
    would then fail validation, or an edited copy would accept what the judges' schema refuses)."""
    shipped = hashlib.sha256((SCHEMA_DIR / PACKET_SCHEMA).read_bytes()).hexdigest()
    assert shipped == PACKET_SCHEMA_SHA256
    if LOOP_PACKET_SCHEMA.is_file():
        assert hashlib.sha256(LOOP_PACKET_SCHEMA.read_bytes()).hexdigest() == PACKET_SCHEMA_SHA256
    marked = _packet()
    marked["contexts"][0]["evidence_basis"] = "ocr_admitted_unverified"
    validate_packet(marked)
    marked["contexts"][0]["evidence_basis"] = "ocr_something_else"
    assert schema_errors(marked, PACKET_SCHEMA)


def test_answer_rows_validate_in_all_three_states():
    candidate = {"decision": "UNSUPPORTED", "answer_language": "en", "user_answer": "No support.",
                 "citations": [], "missing_information": ["a source"], "confidence": "LOW",
                 "reason_code": "NO_RELEVANT_SUPPORT"}
    validate_answer_row(AnswerRow("e" * 64, "VALIDATED", 1, candidate=candidate).to_dict())
    validate_answer_row(AnswerRow("e" * 64, "STRUCTURE_FAILED", 2, errors=["bad"]).to_dict())
    validate_answer_row(AnswerRow("e" * 64, "TRANSPORT_FAILED", 1, errors=["down"]).to_dict())


def test_a_validated_row_without_a_candidate_is_rejected():
    """The schema's own `allOf`: VALIDATED must carry the candidate object."""
    assert schema_errors({"question_id": "e" * 64, "state": "VALIDATED", "attempts": 1}, ANSWER_SCHEMA)


def test_failed_rows_carry_no_user_answer():
    """A failure must never ship an invented answer; the row simply has no candidate."""
    row = AnswerRow("e" * 64, "STRUCTURE_FAILED", 2, errors=["bad"]).to_dict()
    assert "candidate" not in row
