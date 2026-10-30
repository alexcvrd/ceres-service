"""Experiment: answer.situational_split (stateless questions that carry the asker's situation).
Baseline: retrieval drift in 9 of 11 refusals.

How it can fail, and the test that catches each:
  * with the switch off, anything changes (every frozen eval would move)                 → test_switch_off_is_today
  * the split runs but retrieval still uses the situated words (the drift stays)          → test_retrieval_uses_the_core_question
  * the answer is asked the core question instead of the user's (it answers less)         → test_answer_addresses_the_users_words
  * a question with no situation, or a failed split, still gets the relaxed rules         → test_no_situation_or_failed_split_is_plain
  * the persona/history rules leak into a situational-only request                        → test_retrieval_uses_the_core_question
"""
from __future__ import annotations

import json

import pytest

from ceres import pipeline
from ceres.config import load_config
from ceres.contracts import AskRequest
from ceres.testing import EN_QUESTION

from conftest import EXAMPLE_CONFIG
from test_memory import PERSONAS, ScriptedLLM

SPECIFIC = "How should I split nitrogen for irrigated winter wheat on chernozem in Timis?"
CORE = "How should nitrogen fertilizer be split for winter wheat?"


class SplitLLM(ScriptedLLM):
    def __init__(self, split_reply, **kw):
        super().__init__(**kw)
        self.split_reply, self.split_calls = split_reply, []

    def chat(self, request_body):
        if request_body["messages"][0]["content"].startswith("You prepare an agricultural question"):
            self.split_calls.append(request_body)
            content = self.split_reply if isinstance(self.split_reply, str) else json.dumps(self.split_reply)
            payload = {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}], "model": self.MODEL_ID}
            return payload, json.dumps(payload).encode()
        return super().chat(request_body)


def build(env, inline_stack, split_reply, on=True):
    cfg = load_config(EXAMPLE_CONFIG, dict(env, CERES_MEMORY__ENABLED="true", CERES_MEMORY__PERSONAS_DIR=str(PERSONAS),
                                           CERES_ANSWER__SITUATIONAL_SPLIT="true" if on else "false"), require_api_key=False)
    store, models, _ = inline_stack
    seen = []
    orig = models.embed_query
    models.embed_query = lambda text, lang: (seen.append(text), orig(text, lang))[1]
    deps = pipeline.Deps.build(cfg, store, models, SplitLLM(split_reply))
    deps.embedded = seen
    return deps


SITUATED = {"core_question": CORE, "qualifiers": ["irrigated", "chernozem", "Timis"], "has_situation": True}


def test_switch_off_is_today(env, deps, inline_stack):
    """Bug: the experiment changes requests while its switch is off."""
    plain = pipeline.ask(deps, AskRequest(question=SPECIFIC, lang="en"))
    d = build(env, inline_stack, SITUATED, on=False)
    off = pipeline.ask(d, AskRequest(question=SPECIFIC, lang="en"))
    assert d.llm.split_calls == [] and d.llm.calls == deps.llm.calls and off.packet == plain.packet
    assert "situational" not in json.loads(off.model_dump_json())["meta"]


def test_retrieval_uses_the_core_question(env, inline_stack):
    """Bug: retrieval still searches the situated words (Dolj, chernozem …) — the drift the baseline measured."""
    d = build(env, inline_stack, SITUATED)
    r = pipeline.ask(d, AskRequest(question=SPECIFIC, lang="en"))
    assert d.embedded == [CORE]
    system = d.llm.calls[-1]["messages"][0]["content"]
    assert "situational v0" in system and "memory v0" not in system and "general guidance" in system
    assert r.meta.situational["applied"] is True and r.meta.situational["qualifiers"] == ["irrigated", "chernozem", "Timis"]
    assert "memory" not in json.loads(r.model_dump_json())["meta"]


def test_answer_addresses_the_users_words(env, inline_stack):
    """Bug: the answer model is asked the easier core question instead of the user's own question."""
    d = build(env, inline_stack, SITUATED)
    r = pipeline.ask(d, AskRequest(question=SPECIFIC, lang="en"))
    user = d.llm.calls[-1]["messages"][1]["content"]
    assert f"\nQuestion: {SPECIFIC}\n" in user and CORE not in user and r.packet["question"] == SPECIFIC


@pytest.mark.parametrize("reply", [{"core_question": EN_QUESTION, "qualifiers": [], "has_situation": False}, "not json",
                                   {"core_question": CORE, "qualifiers": "Timis", "has_situation": True}])
def test_no_situation_or_failed_split_is_plain(env, deps, inline_stack, reply):
    """Bug: a question without a situation (or a malformed split) gets the relaxed answer rules or the wrong search."""
    plain = pipeline.ask(deps, AskRequest(question=SPECIFIC, lang="en"))
    d = build(env, inline_stack, reply)
    r = pipeline.ask(d, AskRequest(question=SPECIFIC, lang="en"))
    assert d.embedded == [SPECIFIC] and d.llm.calls == deps.llm.calls and r.packet == plain.packet
    assert r.meta.situational["applied"] is False
