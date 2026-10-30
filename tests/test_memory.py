"""Memory v0: static personas + conversation memory.

How memory can fail, and the test that catches each:
  * a stateless request changes (every frozen eval breaks)                           → test_stateless_request_is_byte_identical_*
  * memory fields are silently ignored when memory is off (the client thinks it works) → test_memory_off_refuses_memory_fields
  * an unknown persona costs a model call, or is silently ignored                     → test_unknown_profile_refused_before_any_model_call
  * a first question without a persona pays for a rewrite call                        → test_first_turn_without_persona_skips_the_rewrite
  * a follow-up is retrieved with the user's raw words ("And for maize?")             → test_followup_retrieves_with_the_rewritten_query
  * the rewrite's retrieval query leaks into the question the answer must support     → test_answer_supports_the_standalone_question
  * a broken rewrite reply fails the request                                          → test_rewrite_failure_falls_back_to_the_question
  * BACKGROUND becomes evidence (citable, in the packet)                             → test_background_is_never_evidence
  * the store grows without bound, or forgets too early                               → test_store_limits
  * the BACKGROUND block blows the context budget                                     → test_background_budget_drops_oldest_turn_first
  * a bad persona file is served                                                      → test_persona_loader_refuses_bad_files
  * the client cannot discover the personas                                           → test_profiles_endpoint
Added after a design review:
  * the general-guidance rule applies without a persona (conversation-only requests)  → test_persona_rules_only_with_a_persona
  * a conversation silently switches persona mid-way, or the switch surfaces as a 500 → test_profile_is_bound_to_the_conversation,
                                                                                        test_profile_switch_is_a_422
  * a dead endpoint is called again for the answer after the rewrite failed           → test_rewrite_transport_failure_skips_the_answer_call
  * too many conversations exhaust memory; turn numbers restart after the window      → test_store_global_cap_and_monotonic_turns
  * a rewrite with invented profile fields or a non-boolean flag is accepted          → test_parse_rewrite_rules
Added after a code review (each reproduces a probe from that review):
  * conversation locks pile up forever (one per id ever seen)                         → test_locks_do_not_grow
  * another conversation evicts (or TTL expires) one that is mid-turn: history lost   → test_pinned_conversation_survives_eviction_and_expiry
  * an expired conversation still blocks a new persona                                → test_expired_conversation_releases_its_binding
  * a client retry (same request_id) stores the turn twice and calls the model again   → test_same_request_id_is_replayed
  * a failed rewrite of a follow-up lets the answerer reinterpret history unlogged     → test_unresolved_followup_asks_to_restate
  * zero or negative limits defeat the caps                                            → test_invalid_memory_limits_refused
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ceres import pipeline
from ceres.api import create_app
from ceres.config import load_config
from ceres.contracts import AskRequest
from ceres.llm import FakeLLM, build_request_body
from ceres.memory import (BACKGROUND_HEADER, ConversationStore, MemoryError, Persona, Turn, build_background,
                          load_personas, parse_rewrite)
from ceres.testing import EN_QUESTION

from conftest import EXAMPLE_CONFIG

PERSONAS = Path(__file__).resolve().parent / "fixtures-memory" / "personas"


class ScriptedLLM(FakeLLM):
    """FakeLLM for answers; rewrite calls (system prompt starts "You rewrite") get the scripted replies in order."""

    def __init__(self, rewrites=(), **kw):
        super().__init__(**kw)
        self.rewrites = list(rewrites)
        self.rewrite_calls: list[dict] = []

    def chat(self, request_body):
        if request_body["messages"][0]["content"].startswith("You rewrite"):
            self.rewrite_calls.append(request_body)
            reply = self.rewrites.pop(0) if self.rewrites else "not json"
            content = reply if isinstance(reply, str) else json.dumps(reply)
            payload = {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}], "model": self.MODEL_ID}
            return payload, json.dumps(payload).encode()
        return super().chat(request_body)


@pytest.fixture
def mem_config(env):
    return load_config(EXAMPLE_CONFIG, dict(env, CERES_MEMORY__ENABLED="true", CERES_MEMORY__PERSONAS_DIR=str(PERSONAS)),
                       require_api_key=False)


def mem_deps(mem_config, inline_stack, rewrites=()):
    store, models, _ = inline_stack
    seen: list[str] = []
    original = models.embed_query

    def spy(text, lang):
        seen.append(text)
        return original(text, lang)

    models.embed_query = spy
    deps = pipeline.Deps.build(mem_config, store, models, ScriptedLLM(rewrites))
    deps.embedded = seen
    return deps


def rw(sq, rq=None, followup=True, used=()):
    return {"standalone_question": sq, "retrieval_query": rq or sq, "is_followup": followup, "used_profile_fields": list(used)}


# ---------------------------------------------------------------- stateless parity

def test_stateless_request_is_byte_identical_with_memory_enabled(deps, mem_config, inline_stack):
    """Bug: turning memory on changes the stateless request (prompt, messages, seed) — every frozen eval would move."""
    plain = pipeline.ask(deps, AskRequest(question=EN_QUESTION, lang="en"))
    md = mem_deps(mem_config, inline_stack)
    withmem = pipeline.ask(md, AskRequest(question=EN_QUESTION, lang="en"))
    assert md.llm.calls == deps.llm.calls and md.llm.rewrite_calls == []
    assert withmem.packet == plain.packet and withmem.answer == plain.answer
    assert "memory" not in withmem.meta.model_dump(exclude_none=True) and withmem.meta.memory is None
    assert "memory" not in json.loads(withmem.model_dump_json())["meta"]


def test_stateless_request_is_byte_identical_body():
    """Bug: memory=None no longer yields exactly the r1 body (build_request_body regression)."""
    packet = {"language": "en", "question": "q?", "question_id": "a" * 64,
              "contexts": [{"evidence_id": "E1", "evidence_text": "Some passage text."}]}
    profile = {"model_alias": "m", "decoding": {}}
    assert build_request_body(packet, profile, "SYS", 1) == build_request_body(packet, profile, "SYS", 1, None,
                                                                              memory=None, memory_rules="RULES")
    pinned = json.loads((Path(__file__).parent / "release_fixtures" / "stateless-request-bodies.json").read_text())["bodies"]
    for attempt in (1, 2):
        assert build_request_body(packet, profile, "SYS", attempt) == pinned[str(attempt)]


# ---------------------------------------------------------------- refusals

async def test_memory_off_refuses_memory_fields(client):
    """Bug: with memory off, conversation_id/profile_id are ignored and the client believes memory works."""
    r = await client.post("/v1/ask", json={"question": "q", "lang": "en", "conversation_id": "conv-00001"})
    assert r.status_code == 422 and "memory is not enabled" in r.text


async def test_unknown_profile_refused_before_any_model_call(mem_config, inline_stack):
    """Bug: an unknown persona is silently ignored, or costs a ~15 s answer before failing."""
    import httpx
    md = mem_deps(mem_config, inline_stack)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(md, mem_config)), base_url="http://t") as c:
        r = await c.post("/v1/ask", json={"question": "q", "lang": "en", "profile_id": "nobody-here"})
    assert r.status_code == 422 and "unknown profile_id" in r.text
    assert md.llm.calls == [] and md.llm.rewrite_calls == []


# ---------------------------------------------------------------- the rewrite path

def test_first_turn_without_persona_skips_the_rewrite(mem_config, inline_stack):
    """Bug: turn 1 of a conversation (no persona, no history) pays for a rewrite call it cannot use."""
    md = mem_deps(mem_config, inline_stack)
    r = pipeline.ask(md, AskRequest(question=EN_QUESTION, lang="en", conversation_id="conv-00001"))
    assert md.llm.rewrite_calls == [] and md.embedded == [EN_QUESTION]
    assert r.meta.memory["turn"] == 1 and r.meta.memory["rewrite_status"] == "skipped"


def test_followup_retrieves_with_the_rewritten_query(mem_config, inline_stack):
    """Bug: the follow-up is embedded and reranked with the user's raw words, so retrieval misses the topic."""
    md = mem_deps(mem_config, inline_stack, [rw("What sowing density is given for winter barley?",
                                                  "winter barley sowing density seeds per square metre")])
    pipeline.ask(md, AskRequest(question=EN_QUESTION, lang="en", conversation_id="conv-00001"))
    r = pipeline.ask(md, AskRequest(question="And for barley?", lang="en", conversation_id="conv-00001"))
    assert md.embedded[-1] == "winter barley sowing density seeds per square metre"
    prompt = md.llm.rewrite_calls[0]["messages"][1]["content"]
    assert EN_QUESTION in prompt and "New question: And for barley?" in prompt      # the earlier turn reached the rewrite
    assert r.meta.memory["turn"] == 2 and r.meta.memory["history_turns_used"] == 1 and r.meta.memory["rewrite_status"] == "ok"
    assert r.packet["question"] == "And for barley?"                                  # the packet keeps the user's words


def test_answer_supports_the_standalone_question(mem_config, inline_stack):
    """Bug: the persona-laden retrieval query becomes the question the answer must support (addendum 1: two outputs)."""
    md = mem_deps(mem_config, inline_stack, [rw("When should maize be sown?", "maize sowing date sandy soil southern Romania Dolj",
                                                  followup=False, used=["region", "crops"])])
    pipeline.ask(md, AskRequest(question="When should I sow?", lang="en", profile_id="dolj-maize-dry"))
    body = md.llm.calls[-1]
    user = body["messages"][1]["content"]
    assert "\nQuestion: When should maize be sown?\n" in user and "Question as asked: When should I sow?" in user
    assert "sandy soil southern Romania Dolj" not in user
    assert body["messages"][0]["content"].startswith(md.prompt) and "memory v0" in body["messages"][0]["content"]


def test_rewrite_failure_falls_back_to_the_question(mem_config, inline_stack):
    """Bug: a malformed rewrite reply fails the whole request instead of falling back to the user's question."""
    md = mem_deps(mem_config, inline_stack, ["Sure! Here is the rewrite: maize?"])
    r = pipeline.ask(md, AskRequest(question=EN_QUESTION, lang="en", profile_id="timis-wheat-irrigated"))
    assert r.meta.memory["rewrite_status"] == "fallback" and r.meta.memory["standalone_question"] == EN_QUESTION
    assert md.embedded[-1] == EN_QUESTION and r.answer["state"] in {"VALIDATED", "STRUCTURE_FAILED"}


def test_background_is_never_evidence(mem_config, inline_stack):
    """Bug: the persona or an earlier answer ends up in the packet (citable) instead of the labelled BACKGROUND block."""
    md = mem_deps(mem_config, inline_stack, [rw(EN_QUESTION)])
    r = pipeline.ask(md, AskRequest(question=EN_QUESTION, lang="en", profile_id="dolj-maize-dry"))
    summary = md.memory.personas["dolj-maize-dry"].summary
    assert all(summary not in c["evidence_text"] for c in r.packet["contexts"])
    user = md.llm.calls[-1]["messages"][1]["content"]
    assert BACKGROUND_HEADER in user and summary in user
    assert user.index(summary) < user.index("PRIVATE REFERENCE MATERIAL START")    # outside the evidence section
    forged = {"decision": "ANSWER", "answer_language": "en", "user_answer": "Dolj has sandy soils.",
              "citations": [{"claim": "Dolj has sandy soils.", "evidence_id": "E1", "support_quote": "sandy soils, drought-prone"}],
              "missing_information": [], "confidence": "HIGH", "reason_code": "SUPPORTED"}
    assert pipeline.validate(forged, r.packet), "a quote taken from BACKGROUND must fail the validator"


# ---------------------------------------------------------------- store, budget, personas

def test_store_limits():
    """Bug: a conversation grows without bound, or an idle one is never forgotten."""
    now = [0.0]
    st = ConversationStore(max_turns=3, ttl_s=10, clock=lambda: now[0])
    t = lambda i: Turn(f"q{i}", f"q{i}", "en", None, "ANSWER", "a", [], "x", 0.0)  # noqa: E731
    for i in range(5):
        st.append("c1", t(i))
    assert [x.question for x in st.get("c1")] == ["q2", "q3", "q4"]
    now[0] = 11
    assert st.get("c1") == [] and len(st) == 0


def test_background_budget_drops_oldest_turn_first():
    """Bug: a long conversation pushes the BACKGROUND block past its budget (context overflow), or drops the newest turn."""
    p = Persona("x", "Farmer in Dolj, maize and sunflower, no irrigation.", "farmer", "ro", {})
    turns = [Turn(f"question number {i}", "", "en", None, "ANSWER", "answer " * 60, [], "e", 0.0) for i in range(3)]
    text = build_background(p, turns, limit_chars=700, gist_chars=200)
    assert len(text) <= 700 and "question number 2" in text and "question number 0" not in text and p.summary in text
    assert build_background(None, [], 700, 200) == ""


def test_persona_loader_refuses_bad_files(tmp_path):
    """Bug: a persona whose id differs from its file name, or with no summary, is served."""
    (tmp_path / "a-b.yaml").write_text("id: other\nsummary: s\n")
    with pytest.raises(MemoryError):
        load_personas(tmp_path)
    (tmp_path / "a-b.yaml").write_text("id: a-b\nsummary: ''\n")
    with pytest.raises(MemoryError):
        load_personas(tmp_path)
    assert set(load_personas(PERSONAS)) == {"dolj-maize-dry", "timis-wheat-irrigated", "iasi-adjuster"}


def test_parse_rewrite_rules():
    """Bug: a rewrite without retrieval_query or with prose around the JSON is rejected (or a 2-page 'question' accepted)."""
    assert parse_rewrite('ok {"standalone_question": "A?", "is_followup": true}')["retrieval_query"] == "A?"
    with pytest.raises(ValueError):
        parse_rewrite('{"standalone_question": "A?", "is_followup": "yes"}')
    with pytest.raises(ValueError):
        parse_rewrite('{"standalone_question": "A?", "is_followup": false, "used_profile_fields": ["bank_account"]}')
    with pytest.raises(ValueError):
        parse_rewrite('{"standalone_question": "' + "x" * 700 + '"}')
    with pytest.raises(ValueError):
        parse_rewrite("no json here")


async def test_profiles_endpoint(mem_config, inline_stack, client):
    """Bug: the client cannot list personas, or the endpoint claims memory when it is off."""
    import httpx
    r = await client.get("/v1/profiles")
    assert r.json() == {"memory_enabled": False, "profiles": []}
    md = mem_deps(mem_config, inline_stack)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(md, mem_config)), base_url="http://t") as c:
        body = (await c.get("/v1/profiles")).json()
    assert body["memory_enabled"] is True and {p["id"] for p in body["profiles"]} == set(md.memory.personas)


def test_persona_rules_only_with_a_persona(mem_config, inline_stack):
    """Bug: the 'general guidance' relaxation reaches requests without a persona (it must be persona-gated)."""
    md = mem_deps(mem_config, inline_stack, [rw("What sowing density is given for winter barley?")])
    pipeline.ask(md, AskRequest(question=EN_QUESTION, lang="en", conversation_id="conv-00002"))
    pipeline.ask(md, AskRequest(question="And for barley?", lang="en", conversation_id="conv-00002"))
    system = md.llm.calls[-1]["messages"][0]["content"]
    assert "memory v0" in system and "general guidance" not in system
    md2 = mem_deps(mem_config, inline_stack, [rw(EN_QUESTION, followup=False)])
    pipeline.ask(md2, AskRequest(question=EN_QUESTION, lang="en", profile_id="dolj-maize-dry"))
    assert "general guidance" in md2.llm.calls[-1]["messages"][0]["content"]


def test_profile_is_bound_to_the_conversation(mem_config, inline_stack):
    """Bug: a second turn with another persona silently mixes two users' contexts."""
    md = mem_deps(mem_config, inline_stack, [rw(EN_QUESTION, followup=False)])
    pipeline.ask(md, AskRequest(question=EN_QUESTION, lang="en", conversation_id="conv-00003", profile_id="dolj-maize-dry"))
    with pytest.raises(MemoryError):
        pipeline.ask(md, AskRequest(question="And?", lang="en", conversation_id="conv-00003", profile_id="iasi-adjuster"))


def test_rewrite_transport_failure_skips_the_answer_call(mem_config, inline_stack):
    """Bug: after the endpoint failed the rewrite, Ceres calls it again for the answer (the client waits twice)."""
    from ceres.llm import LLMTransportError

    class Down(ScriptedLLM):
        def chat(self, request_body):
            self.calls.append(request_body)
            raise LLMTransportError("down")

    store, models, _ = inline_stack
    md = pipeline.Deps.build(mem_config, store, models, Down())
    r = pipeline.ask(md, AskRequest(question=EN_QUESTION, lang="en", profile_id="dolj-maize-dry"))
    assert r.answer["state"] == "TRANSPORT_FAILED" and len(md.llm.calls) == 1
    assert r.meta.memory["rewrite_status"] == "transport_failed"


def test_store_global_cap_and_monotonic_turns():
    """Bug: unbounded conversations exhaust memory; turn numbers restart once old turns leave the window."""
    now = [0.0]
    st = ConversationStore(max_turns=2, ttl_s=100, clock=lambda: now[0], max_conversations=2)
    t = lambda i: Turn(f"q{i}", "", "en", None, "ANSWER", "a", [], "x", 0.0)  # noqa: E731
    assert [st.append("c1", t(i)) for i in range(4)] == [1, 2, 3, 4]
    now[0] = 1; st.append("c2", t(0))
    now[0] = 2; st.append("c3", t(0))           # c1 is the least recently used → evicted
    assert st.get("c1") == [] and len(st) == 2


async def test_profile_switch_is_a_422(mem_config, inline_stack):
    """Bug: a persona switch mid-conversation reaches the client as a 500 (the MemoryError is not mapped)."""
    import httpx
    md = mem_deps(mem_config, inline_stack, [rw(EN_QUESTION, followup=False), rw(EN_QUESTION, followup=False)])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(md, mem_config)), base_url="http://t") as c:
        ok = await c.post("/v1/ask", json={"question": EN_QUESTION, "lang": "en", "conversation_id": "conv-00004",
                                           "profile_id": "dolj-maize-dry"})
        bad = await c.post("/v1/ask", json={"question": "And?", "lang": "en", "conversation_id": "conv-00004",
                                            "profile_id": "iasi-adjuster"})
    assert ok.status_code == 200 and bad.status_code == 422 and "bound to profile_id" in bad.text


def test_locks_do_not_grow(mem_config, inline_stack):
    """Bug (review #1): every conversation id leaves a lock behind; 2,000 ids → 2,000 locks."""
    md = mem_deps(mem_config, inline_stack)
    for i in range(50):
        pipeline.ask(md, AskRequest(question=EN_QUESTION, lang="en", conversation_id=f"conv-{i:05d}"))
    assert md.memory._locks == {}


def test_pinned_conversation_survives_eviction_and_expiry():
    """Bug (review #2): with room for one conversation, B's turn evicts A while A is mid-turn; A's turn comes back as turn 1."""
    now = [0.0]
    st = ConversationStore(max_turns=5, ttl_s=10, clock=lambda: now[0], max_conversations=1)
    t = lambda q: Turn(q, q, "en", None, "ANSWER", "a", [], "x", 0.0)  # noqa: E731
    st.append("A", t("a1"))
    snap = st.begin("A", None)                       # A mid-turn
    st.append("B", t("b1"))                          # needs room: A is pinned, so B overshoots instead of evicting A
    now[0] = 50                                      # past A's TTL while A is still mid-turn
    st.get("B")
    assert st.commit(snap, t("a2")) == 2
    st.end(snap)
    assert [x.question for x in st.get("A")] == ["a1", "a2"]


def test_expired_conversation_releases_its_binding():
    """Bug (review #2): a conversation past its TTL still refuses a different persona."""
    now = [0.0]
    st = ConversationStore(ttl_s=10, clock=lambda: now[0])
    st.append("A", Turn("q", "q", "en", "dolj-maize-dry", "ANSWER", "a", [], "x", 0.0))
    now[0] = 11
    snap = st.begin("A", "iasi-adjuster")            # no MemoryError: the old conversation expired
    st.end(snap)


def test_same_request_id_is_replayed(mem_config, inline_stack):
    """Bug (review #3): a client retry with the same request_id runs again and stores a second turn."""
    md = mem_deps(mem_config, inline_stack)
    req = AskRequest(question=EN_QUESTION, lang="en", conversation_id="conv-00009", request_id="r-1")
    first = pipeline.ask(md, req)
    calls = len(md.llm.calls)
    again = pipeline.ask(md, req)
    assert again is first and len(md.llm.calls) == calls and len(md.memory.store.get("conv-00009")) == 1


def test_unresolved_followup_asks_to_restate(mem_config, inline_stack):
    """Bug (review #4): after a malformed rewrite, "And for maize?" is answered from the answerer's own reading of history."""
    md = mem_deps(mem_config, inline_stack, ["not json at all"])
    pipeline.ask(md, AskRequest(question=EN_QUESTION, lang="en", conversation_id="conv-00010"))
    calls = len(md.llm.calls)
    r = pipeline.ask(md, AskRequest(question="And for maize?", lang="en", conversation_id="conv-00010"))
    assert r.answer["state"] == "VALIDATED" and r.answer["candidate"]["decision"] == "CLARIFY"
    assert len(md.llm.calls) == calls and r.meta.memory["rewrite_status"] == "fallback"


def test_invalid_memory_limits_refused(env):
    """Bug (review #5): max_turns 0 keeps every turn; max_conversations 0 crashes; negative TTL never expires."""
    from ceres.config import ConfigError
    for key, value in (("MAX_TURNS", "0"), ("MAX_CONVERSATIONS", "0"), ("TTL_S", "-1"), ("REWRITE_TIMEOUT_S", "0")):
        with pytest.raises(ConfigError, match=f"memory.{key.lower()}"):
            load_config(EXAMPLE_CONFIG, dict(env, CERES_MEMORY__ENABLED="true", CERES_MEMORY__PERSONAS_DIR=str(PERSONAS),
                                             **{f"CERES_MEMORY__{key}": value}), require_api_key=False)
    with pytest.raises(MemoryError):
        ConversationStore(max_turns=0)
