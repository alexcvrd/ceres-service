"""The HTTP surface: authentication, docs, limits, admission, and the status-code mapping."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import httpx
import pytest

from ceres import pipeline
from ceres.api import create_app
from ceres.config import load_config
from ceres.llm import FakeLLM, LLMTransportError
from ceres.store import StoreError

from conftest import EXAMPLE_CONFIG, ROOT, FakePool, pg_store
from ceres.testing import EN_QUESTION

API_KEY = "k" * 40
HEADERS = {"X-API-Key": API_KEY}
ASK = {"question": EN_QUESTION, "lang": "en"}
SENTINEL = "UPSTREAM_DETAIL_SENTINEL http://10.9.8.7:8215/v1 secret-path"


def build(env, inline_stack, *, llm=None, store=None, auth=True, **overrides):
    """An app from the shipped config, with `CERES_*` overrides applied as an operator would."""
    values = dict(env, **overrides)
    if auth:
        values["CERES_API_KEY"] = API_KEY
    config = load_config(EXAMPLE_CONFIG, values, require_api_key=auth)
    inline_store, models, _ = inline_stack
    deps = pipeline.Deps.build(config, store or inline_store, models, llm or FakeLLM())
    return create_app(deps, config), deps


def call(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ceres.test")


# ---------------------------------------------------------------- authentication

async def test_health_is_unauthenticated(env, inline_stack):
    app, _ = build(env, inline_stack)
    async with call(app) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["api"] == "ok" and body["ready"] is True
    assert "dsn" not in str(body)


async def test_v1_requires_the_api_key(env, inline_stack):
    app, _ = build(env, inline_stack)
    async with call(app) as client:
        assert (await client.get("/v1/info")).status_code == 401
        assert (await client.post("/v1/ask", json=ASK)).status_code == 401
        assert (await client.post("/v1/ask", json=ASK, headers={"X-API-Key": "wrong"})).status_code == 401
        ok = await client.post("/v1/ask", json=ASK, headers={"X-API-Key": API_KEY})
    assert ok.status_code == 200


async def test_a_non_ascii_api_key_is_refused_with_401_not_a_500(env, inline_stack):
    """hmac.compare_digest on a non-ASCII str raised TypeError → 500 instead of 401."""
    app, _ = build(env, inline_stack)
    async with call(app) as client:
        bad = await client.post("/v1/ask", json=ASK, headers={"X-API-Key": "é".encode("latin-1")})
    assert bad.status_code == 401


async def test_no_auth_opens_v1_and_says_so(env, inline_stack):
    app, _ = build(env, inline_stack, auth=False)
    async with call(app) as client:
        assert (await client.get("/v1/info")).status_code == 200


# ---------------------------------------------------------------- docs

async def test_the_docs_are_off_by_default(env, inline_stack):
    app, _ = build(env, inline_stack)
    async with call(app) as client:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert (await client.get(path)).status_code == 404, path


async def test_the_docs_can_be_turned_on_together(env, inline_stack):
    app, _ = build(env, inline_stack, CERES_API__DOCS="true")
    async with call(app) as client:
        assert (await client.get("/openapi.json")).status_code == 200
        assert (await client.get("/docs")).status_code == 200


# ---------------------------------------------------------------- limits

async def test_a_long_question_is_413(env, inline_stack):
    app, _ = build(env, inline_stack, CERES_API__MAX_QUESTION_CHARS="50")
    async with call(app) as client:
        response = await client.post("/v1/ask", json={"question": "x" * 51, "lang": "en"},
                                     headers={"X-API-Key": API_KEY})
    assert response.status_code == 413


async def test_an_oversized_body_is_413_before_it_is_read(env, inline_stack):
    app, _ = build(env, inline_stack, CERES_API__MAX_BODY_BYTES="200")
    async with call(app) as client:
        response = await client.post("/v1/ask", json={"question": "x" * 500, "lang": "en"},
                                     headers={"X-API-Key": API_KEY})
    assert response.status_code == 413


async def test_an_unknown_field_is_422(env, inline_stack):
    app, _ = build(env, inline_stack)
    async with call(app) as client:
        response = await client.post("/v1/ask", json={**ASK, "max_citations": 3},
                                     headers={"X-API-Key": API_KEY})
    assert response.status_code == 422


# ---------------------------------------------------------------- admission

class SlowLLM(FakeLLM):
    """A FakeLLM that holds the worker long enough for a second request to queue."""

    def chat(self, request_body):
        time.sleep(0.25)
        return super().chat(request_body)


async def test_a_full_queue_is_429_not_a_long_wait(env, inline_stack):
    app, _ = build(env, inline_stack, llm=SlowLLM(),
                   CERES_API__ADMISSION__MAX_INFLIGHT="1",
                   CERES_API__ADMISSION__QUEUE_TIMEOUT_S="0.05")
    async with call(app) as client:
        headers = {"X-API-Key": API_KEY}
        first, second = await asyncio.gather(
            client.post("/v1/ask", json=ASK, headers=headers),
            client.post("/v1/ask", json=ASK, headers=headers),
        )
    codes = sorted([first.status_code, second.status_code])
    assert codes == [200, 429]


# ---------------------------------------------------------------- failure mapping

async def test_no_active_release_is_503_and_leaks_nothing(env, inline_stack):
    class Broken:
        def active_release(self):
            raise StoreError("ceres_serve_v0.releases has 0 active releases")

        def health(self):
            return "degraded", "no active release"

    app, deps = build(env, inline_stack)
    deps.store = Broken()
    async with call(app) as client:
        response = await client.post("/v1/ask", json=ASK, headers={"X-API-Key": API_KEY})
        health = await client.get("/health")
    assert response.status_code == 503
    assert response.json()["detail"] == "the serving store is not ready"
    assert "ceres_serve_v0" not in response.text
    assert health.json()["ready"] is False


async def test_a_transport_failure_is_503_with_the_answer_row(env, inline_stack):
    app, _ = build(env, inline_stack, llm=FakeLLM(transport_error=True))
    async with call(app) as client:
        response = await client.post("/v1/ask", json=ASK, headers={"X-API-Key": API_KEY})
    assert response.status_code == 503
    body = response.json()
    assert body["answer"]["state"] == "TRANSPORT_FAILED"
    assert "candidate" not in body["answer"]


async def test_a_structural_failure_is_200_without_a_user_answer(env, inline_stack):
    app, _ = build(env, inline_stack, llm=FakeLLM(failures=2))
    async with call(app) as client:
        response = await client.post("/v1/ask", json=ASK, headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]["state"] == "STRUCTURE_FAILED"
    assert "candidate" not in body["answer"]
    assert body["meta"]["validator_revision"] == "ceres-validate-r3"


# ---------------------------------------------------------------- info

async def test_info_reports_the_route_and_the_predicate(env, inline_stack):
    app, _ = build(env, inline_stack)
    async with call(app) as client:
        body = (await client.get("/v1/info", headers={"X-API-Key": API_KEY})).json()
    assert body["route_id"] is None and body["route_status"] == "unassigned"
    assert body["judged_route_id"].startswith("73c2e791")
    assert body["read_predicate_sha256"]
    # The shipped default, and the rights list derived from it.
    assert body["serving_mode"] == "rnd"
    assert body["store"]["serving_mode"] == "rnd"
    assert body["store"]["serve_rights"] == \
        ["redistributable", "internal_research_only", "licensed"]
    assert "admitted_licensed_works" not in body["store"]     # a production-only field
    assert body["store"]["predicate"] == "active+evidence_eligible+rights"
    assert body["store"]["dsn"] == "postgresql://db.invalid/veridian_rag"
    assert body["adapters"] == {"store": "FixtureStore", "models": "FakeModels", "llm": "FakeLLM"}
    assert body["profiles"]["answer"]
    assert body["prompt_template_sha256"] == \
        "7a37e08184ffd804fbc07da11296246ffe1479be74660cbb2be71a57ff7da0b4"


async def test_info_reports_the_r3_validator_and_its_guards_not_a_stale_revision(env, inline_stack):
    """An operator must see which quote rule serves; a /v1/info that still said r1/r2,
    or hid the guards (incl. r3's capitalized-word gap guard), would make a judged r3 number unattributable."""
    app, _ = build(env, inline_stack)
    async with call(app) as client:
        body = (await client.get("/v1/info", headers={"X-API-Key": API_KEY})).json()
    assert body["validator_revision"] == "ceres-validate-r3"
    guards = body["validator_guards"]
    assert guards["revision"] == "ceres-validate-r3"
    assert guards["gap_forbids_capitalized_word"] == r"(?<![.!?]\s)\b[A-ZĂÂÎȘŞȚŢ][a-zăâîșşțţ]+"
    assert (guards["min_pieces"], guards["min_words_per_piece"], guards["max_gap_chars"]) == (2, 3, 200)
    assert guards["gap_forbids_digit"] is True and "fără" in guards["gap_forbids_negation"]
    assert guards["norm_version"] == "norm-v3" and guards["served_quote_rewritten"] is False


async def test_info_reports_a_production_switch_and_its_decision(env, inline_stack):
    """An operator who can call /v1/info can see the mode, without reading the config on the box."""
    app, _ = build(env, inline_stack, CERES_STORE__SERVING_MODE="production",
                   CERES_STORE__PRODUCTION_DECISION="COMPLIANCE-EXAMPLE-1")
    async with call(app) as client:
        body = (await client.get("/v1/info", headers={"X-API-Key": API_KEY})).json()
    assert body["serving_mode"] == "production"
    assert body["store"]["serve_rights"] == ["redistributable"]
    assert body["store"]["admitted_licensed_works"] == 0      # a count, never the ids
    assert body["store"]["production_decision"] == "COMPLIANCE-EXAMPLE-1"


async def test_the_serving_mode_travels_into_every_answer(env, inline_stack):
    """`meta.serving_mode` is what an audit reads later; it must be in the answer, not only /info."""
    app, _ = build(env, inline_stack)
    async with call(app) as client:
        rnd = (await client.post("/v1/ask", json=ASK, headers={"X-API-Key": API_KEY})).json()
    served, _ = build(env, inline_stack, CERES_STORE__SERVING_MODE="production",
                      CERES_STORE__PRODUCTION_DECISION="COMPLIANCE-EXAMPLE-1")
    async with call(served) as client:
        production = (await client.post("/v1/ask", json=ASK,
                                        headers={"X-API-Key": API_KEY})).json()
    assert rnd["meta"]["serving_mode"] == "rnd"
    assert production["meta"]["serving_mode"] == "production"
    assert rnd["meta"]["read_predicate_sha256"] != production["meta"]["read_predicate_sha256"]


async def test_the_request_id_travels_into_meta(env, inline_stack):
    app, _ = build(env, inline_stack)
    async with call(app) as client:
        body = (await client.post("/v1/ask", json={**ASK, "request_id": "req-7"},
                                  headers={"X-API-Key": API_KEY})).json()
    assert body["meta"]["request_id"] == "req-7"
    assert len(body["meta"]["execution_id"]) == 32


# ---------------------------------------------------------------- regressions at the HTTP layer
# Admission, redaction, body limit, store failures, provenance snapshot.

async def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.01)


class GatedLLM(FakeLLM):
    """Blocks every call on a gate and counts how many calls are inside `chat` at once."""

    def __init__(self):
        super().__init__()
        self.gate = threading.Event()
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def chat(self, request_body):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.gate.wait(timeout=10)
            return super().chat(request_body)
        finally:
            with self.lock:
                self.active -= 1


# ---------------------------------------------------------------- admission accounting

async def test_a_cancelled_request_keeps_its_permit_until_the_worker_ends(env, inline_stack):
    """Cancelling the client must not free a permit while the thread still runs."""
    llm = GatedLLM()
    app, _ = build(env, inline_stack, llm=llm, CERES_API__ADMISSION__MAX_INFLIGHT="1",
                   CERES_API__ADMISSION__QUEUE_TIMEOUT_S="5")
    async with call(app) as http:
        first = asyncio.create_task(http.post("/v1/ask", json=ASK, headers=HEADERS))
        await wait_until(lambda: llm.active == 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(http.post("/v1/ask", json=ASK, headers=HEADERS))
        await asyncio.sleep(0.3)
        assert llm.max_active == 1, "a second worker started while the first still ran"
        llm.gate.set()
        response = await second
    assert response.status_code == 200
    assert llm.max_active == 1


async def test_the_pending_queue_is_bounded(env, inline_stack):
    """Beyond `max_pending` waiters a request is refused at once, not after the timeout."""
    llm = GatedLLM()
    app, _ = build(env, inline_stack, llm=llm, CERES_API__ADMISSION__MAX_INFLIGHT="1",
                   CERES_API__ADMISSION__MAX_PENDING="1",
                   CERES_API__ADMISSION__QUEUE_TIMEOUT_S="5")
    async with call(app) as http:
        running = asyncio.create_task(http.post("/v1/ask", json=ASK, headers=HEADERS))
        await wait_until(lambda: llm.active == 1)
        waiting = asyncio.create_task(http.post("/v1/ask", json=ASK, headers=HEADERS))
        await asyncio.sleep(0.05)
        started = time.monotonic()
        refused = await http.post("/v1/ask", json=ASK, headers=HEADERS)
        refused_after = time.monotonic() - started
        llm.gate.set()
        codes = [(await running).status_code, (await waiting).status_code]
    assert refused.status_code == 429
    assert refused_after < 1.0
    assert codes == [200, 200]


async def test_the_deadline_answers_504_and_the_worker_keeps_its_permit(env, inline_stack):
    """A deadline bounds the client's wait, never the accounting of the running work."""
    llm = GatedLLM()
    app, _ = build(env, inline_stack, llm=llm, CERES_API__ADMISSION__MAX_INFLIGHT="1",
                   CERES_API__ADMISSION__QUEUE_TIMEOUT_S="0.1",
                   CERES_API__REQUEST_DEADLINE_S="0.2")
    async with call(app) as http:
        late = await http.post("/v1/ask", json=ASK, headers=HEADERS)
        assert late.status_code == 504
        assert late.json()["detail"]["code"] == "deadline_exceeded"
        assert llm.active == 1                      # the thread is still working ...
        busy = await http.post("/v1/ask", json=ASK, headers=HEADERS)
        assert busy.status_code == 429              # ... and still holds the only permit
        llm.gate.set()
        await wait_until(lambda: llm.active == 0)
        await asyncio.sleep(0.05)
        ok = await http.post("/v1/ask", json=ASK, headers=HEADERS)
    assert ok.status_code == 200
    assert llm.max_active == 1


# ---------------------------------------------------------------- redaction

class SentinelTransportLLM(FakeLLM):
    def chat(self, request_body):
        self.calls.append(request_body)
        raise LLMTransportError(SENTINEL)


async def test_upstream_error_text_never_reaches_the_client(env, inline_stack, tmp_path):
    """The body carries a stable code; the detail stays in the raw record."""
    app, _ = build(env, inline_stack, llm=SentinelTransportLLM())
    async with call(app) as http:
        response = await http.post("/v1/ask", json=ASK, headers=HEADERS)
    assert response.status_code == 503
    assert "UPSTREAM_DETAIL_SENTINEL" not in response.text
    assert "10.9.8.7" not in response.text
    assert response.json()["answer"]["errors"] == ["transport_failed"]
    raw = [p.read_text(encoding="utf-8") for p in (tmp_path / "executions").rglob("*.json")]
    assert any("UPSTREAM_DETAIL_SENTINEL" in text for text in raw), "the private record lost it"


# ---------------------------------------------------------------- body limit

async def test_a_chunked_body_over_the_limit_is_413(env, inline_stack):
    """No Content-Length, so the limit must be counted while the body is received."""
    app, _ = build(env, inline_stack, CERES_API__MAX_BODY_BYTES="200")
    body = json.dumps(ASK).encode("utf-8")[:-1] + b" " * 400 + b"}"

    async def chunks():
        for index in range(0, len(body), 64):
            yield body[index:index + 64]

    async with call(app) as http:
        response = await http.post("/v1/ask", content=chunks(),
                                   headers={**HEADERS, "Content-Type": "application/json"})
    assert response.status_code == 413


async def test_the_body_counter_does_not_refuse_a_legal_chunked_body(env, inline_stack):
    """Bug caught: counting the body while it arrives must not refuse a body under the limit."""
    app, _ = build(env, inline_stack)
    body = json.dumps(ASK).encode("utf-8")

    async def chunks():
        for index in range(0, len(body), 16):
            yield body[index:index + 16]

    async with call(app) as http:
        response = await http.post("/v1/ask", content=chunks(),
                                   headers={**HEADERS, "Content-Type": "application/json"})
    assert response.status_code == 200


# ---------------------------------------------------------------- store failures and shutdown

async def test_a_driver_error_is_503_not_500(env, inline_stack):
    import psycopg

    store = pg_store(FakePool(raises=psycopg.OperationalError(SENTINEL)))
    app, _ = build(env, inline_stack, store=store)
    async with call(app) as http:
        response = await http.post("/v1/ask", json=ASK, headers=HEADERS)
    assert response.status_code == 503
    assert "UPSTREAM_DETAIL_SENTINEL" not in response.text


async def test_app_shutdown_does_not_leak_the_adapters_pools(env, inline_stack):
    """The store pool and the answerer client were never closed at shutdown."""
    class Closable(FakeLLM):
        closed = 0

        def close(self):
            Closable.closed += 1

    pool = FakePool()
    store = pg_store(pool)
    app, _ = build(env, inline_stack, llm=Closable(), store=store)
    async with app.router.lifespan_context(app):
        pass
    assert pool.closed is True
    assert Closable.closed == 1


# ---------------------------------------------------------------- provenance snapshot

def _copy_route_files(tmp_path: Path) -> dict[str, str]:
    answer = tmp_path / "answer.yaml"
    prompt = tmp_path / "prompt.md"
    answer.write_bytes((ROOT / "tests" / "fixtures" / "profiles" / "grounded-qwen38-v1.yaml").read_bytes())
    prompt.write_bytes((ROOT / "tests" / "fixtures" / "profiles" / "grounded_answer_v1.md").read_bytes())
    return {"CERES_ROUTE__PROFILES__ANSWER": str(answer),
            "CERES_ROUTE__PROMPT_TEMPLATE": str(prompt)}


async def test_info_reports_the_startup_snapshot_not_the_files_now(env, inline_stack, tmp_path):
    """Editing a profile after startup must not change reported provenance."""
    overrides = _copy_route_files(tmp_path)
    app, _ = build(env, inline_stack, **overrides)
    async with call(app) as http:
        before = (await http.get("/v1/info", headers=HEADERS)).json()
        with open(overrides["CERES_ROUTE__PROFILES__ANSWER"], "a", encoding="utf-8") as handle:
            handle.write("\n# edited after startup\n")
        Path(overrides["CERES_ROUTE__PROMPT_TEMPLATE"]).write_text("changed", encoding="utf-8")
        after = (await http.get("/v1/info", headers=HEADERS)).json()
    assert after["profiles"] == before["profiles"]
    assert after["prompt_template_sha256"] == before["prompt_template_sha256"]
    assert after["config_sha256"] == before["config_sha256"]


async def test_info_does_not_probe_the_model(env, inline_stack):
    class Counting(FakeLLM):
        probes = 0

        def health(self):
            Counting.probes += 1
            return super().health()

    app, _ = build(env, inline_stack, llm=Counting())
    async with call(app) as http:
        for _ in range(3):
            assert (await http.get("/v1/info", headers=HEADERS)).status_code == 200
    assert Counting.probes == 0


async def test_the_config_digest_covers_environment_overrides(env, inline_stack):
    digests = []
    for port in ("8300", "8301"):
        app, _ = build(env, inline_stack, CERES_API__PORT=port)
        async with call(app) as http:
            digests.append((await http.get("/v1/info", headers=HEADERS)).json()["config_sha256"])
    assert digests[0] != digests[1]


def test_the_fake_stack_does_not_need_smoke_test_on_sys_path():
    """`ceres-api --fake` reached into smoke_test.py through a sys.path insert."""
    from ceres.testing import build_fixture_stack, build_inline_fixture

    store, models, llm, asked = build_fixture_stack()
    assert asked and store.active_release().row_count > 0
    inline_store, _models, texts = build_inline_fixture()
    assert inline_store.active_release().row_count == 6 and len(texts) == 2


# ---------------------------------------------------------------- the answerer's LIVE context

def _props_llm(props_reply, counter: dict | None = None):
    """`OpenAIChatLLM` behind `httpx.MockTransport`: /v1/models lists qwen38, /props answers
    `props_reply` (a dict → 200 JSON, an int → that status). No model server is called."""
    from ceres.llm import OpenAIChatLLM

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "qwen38"}]})
        if request.url.path == "/props":
            if counter is not None:
                counter["props"] = counter.get("props", 0) + 1
            if isinstance(props_reply, int):
                return httpx.Response(props_reply, text="Not Found")
            return httpx.Response(200, json=props_reply)
        return httpx.Response(500)

    return OpenAIChatLLM("http://answerer.test/v1", expect_model="qwen38",
                         transport=httpx.MockTransport(handler))


async def test_a_live_context_mismatch_makes_health_not_ready_with_both_numbers(env, inline_stack):
    """The server runs n_ctx 8192 while the answer profile declares 16384 — a different route.
    Before the fix /health said ready because /v1/models listed the right id."""
    app, _ = build(env, inline_stack, llm=_props_llm({"default_generation_settings": {"n_ctx": 8192}}))
    async with call(app) as client:
        body = (await client.get("/health")).json()
    assert body["ready"] is False and body["model"] == "wrong_context"
    assert "8192" in body["detail"]["model"] and "16384" in body["detail"]["model"]


async def test_an_unknown_props_shape_keeps_health_ready_and_says_unavailable(env, inline_stack):
    """A server without a readable /props (404) is not refused — unknown is not a mismatch —
    and /v1/info records the snapshot as unavailable rather than inventing an n_ctx."""
    app, _ = build(env, inline_stack, llm=_props_llm(404))
    async with call(app) as client:
        health = (await client.get("/health")).json()
        info = (await client.get("/v1/info", headers=HEADERS)).json()
    assert health["ready"] is True and health["model"] == "ready"
    live = info["answerer_live"]
    assert live["server_props"]["available"] is False
    assert live["live_n_ctx"] is None and live["live_n_ctx_matches_profile"] is None


async def test_info_reports_the_last_health_snapshot_and_does_not_refetch_props(env, inline_stack):
    """/v1/info carries what the last /health saw (n_ctx 16384, matches), null before any
    /health, and never probes /props itself."""
    counter: dict = {}
    props = {"default_generation_settings": {"n_ctx": 16384}, "model_path": "/m/qwen.gguf"}
    app, _ = build(env, inline_stack, llm=_props_llm(props, counter))
    async with call(app) as client:
        assert (await client.get("/v1/info", headers=HEADERS)).json()["answerer_live"] is None
        health = (await client.get("/health")).json()
        for _ in range(3):
            info = (await client.get("/v1/info", headers=HEADERS)).json()
    assert health["ready"] is True
    assert counter["props"] == 1
    live = info["answerer_live"]
    assert live["live_n_ctx"] == 16384 and live["live_n_ctx_matches_profile"] is True
    assert live["profile_context_tokens"] == 16384
    assert live["server_props"]["model_path"] == "/m/qwen.gguf" and live["checked_at"]


# ---------------------------------------------------------------- ROUTE-STORE-GUARD
# A loader replace can swap a release's rows with another route's chunks under the SAME release id.
# Serving must compare the active release's chunk/represent/embed profile shas with the configured
# route's, and refuse a mismatch; `index` and `route_id` never decide.
# FixtureStore with a crafted Release; the PostgreSQL end of it is in test_store.py.

import dataclasses                                               # noqa: E402
import hashlib                                                   # noqa: E402
import logging                                                   # noqa: E402

from ceres.store import Release                                   # noqa: E402

ROUTE_PROFILE_FILES = {"chunk": "chunk-bridge-v3.1.yaml", "represent": "verbatim-v1.yaml",
                       "embed": "jina-768-v1.yaml", "index": "inmem-exact-v1.yaml"}
FIXTURE_PROFILES = ROOT / "tests" / "fixtures" / "profiles"
requires_route_profiles = pytest.mark.skipif(
    not all((FIXTURE_PROFILES / name).is_file() for name in ROUTE_PROFILE_FILES.values()),
    reason="fixture profiles are not present (tests/fixtures/profiles/)")
OTHER_SHA = "f" * 64                                              # a v3.4-style profile nobody configured


def route_shas() -> dict[str, str]:
    """sha256 of the configured route's profile files, the way a `releases` row names them."""
    return {slot: hashlib.sha256((FIXTURE_PROFILES / name).read_bytes()).hexdigest()
            for slot, name in ROUTE_PROFILE_FILES.items()}


def route_overrides(*slots: str) -> dict[str, str]:
    """`CERES_ROUTE__PROFILES__<SLOT>` for the named slots (default chunk, represent, embed)."""
    return {f"CERES_ROUTE__PROFILES__{slot.upper()}": str(FIXTURE_PROFILES / ROUTE_PROFILE_FILES[slot])
            for slot in (slots or ("chunk", "represent", "embed"))}


def replace_release(store, *, manifest: str, **shas: str) -> Release:
    """What `dds load --replace` does to the inline fixture: same id, new manifest, given shas."""
    store._release = dataclasses.replace(store._release, manifest_sha256=manifest, profile_sha256=dict(shas))
    return store._release


async def _ask_health_info(app):
    async with call(app) as client:
        ask = await client.post("/v1/ask", json=ASK, headers=HEADERS)
        health = await client.get("/health")
        info = await client.get("/v1/info", headers=HEADERS)
    return ask, health.json(), info.json()


@requires_route_profiles
@pytest.mark.parametrize("slot", ["chunk", "represent", "embed"])
async def test_a_replaced_release_built_with_another_profile_refuses_asks_and_health(
        env, inline_stack, slot, caplog):
    """Bug: a release replaced under the same id with another route's chunks was served under the
    configured route's identity. Now: 503 route_store_mismatch, /health not ready, public text only,
    and the log names the slot and both shas."""
    shas = {k: v for k, v in route_shas().items() if k != "index"}
    replace_release(inline_stack[0], manifest="m-v34", **dict(shas, **{slot: OTHER_SHA}))
    app, _ = build(env, inline_stack, **route_overrides())
    with caplog.at_level(logging.ERROR):
        ask, health, info = await _ask_health_info(app)
    assert ask.status_code == 503
    assert ask.json()["detail"]["code"] == "route_store_mismatch"
    assert OTHER_SHA not in ask.text and shas[slot] not in ask.text        # no shas to the client
    assert health["ready"] is False and health["store"] == "route_store_mismatch"
    assert health["detail"]["store"] == "route_store_mismatch"
    assert info["route_store_check"] == "mismatch"
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert f"{slot}:" in logged and OTHER_SHA in logged and shas[slot] in logged


@requires_route_profiles
async def test_a_release_matching_the_configured_route_serves_and_reports_ok(env, inline_stack):
    """Bug guarded: a guard that refuses even a matching release (e.g. compares the wrong side)."""
    shas = route_shas()
    replace_release(inline_stack[0], manifest="m-match",
                    **{slot: shas[slot] for slot in ("chunk", "represent", "embed")})
    app, _ = build(env, inline_stack, **route_overrides())
    ask, health, info = await _ask_health_info(app)
    assert ask.status_code == 200
    assert health["store"] == "ready" and health["ready"] is True
    assert info["route_store_check"] == "ok"


@requires_route_profiles
@pytest.mark.parametrize("case", ["release_without_shas", "config_without_profiles",
                                  "config_with_two_of_three"])
async def test_a_missing_side_skips_the_check_and_reports_skipped(env, inline_stack, case):
    """Bug guarded: a fixture/dev release (no shas) or a config without the three profiles must not
    be refused, and must not be reported as verified ("ok")."""
    shas = route_shas()
    overrides = route_overrides()
    if case == "config_without_profiles":
        overrides = {}
    if case == "config_with_two_of_three":
        overrides = route_overrides("chunk", "represent")
    if case != "release_without_shas":
        replace_release(inline_stack[0], manifest="m-dev",
                        **{slot: shas[slot] for slot in ("chunk", "represent", "embed")})
    app, _ = build(env, inline_stack, **overrides)
    ask, health, info = await _ask_health_info(app)
    assert ask.status_code == 200
    assert health["store"] == "ready"
    assert info["route_store_check"] == "skipped"


@requires_route_profiles
async def test_an_index_or_route_id_difference_alone_never_refuses(env, inline_stack):
    """Bug guarded: comparing `index` (a serving-side choice) or the loader's `route_id` label would
    refuse a correct release. The config names an index profile the release
    does not; the release carries another index sha and an unrelated route_id."""
    shas = route_shas()
    store = inline_stack[0]
    replace_release(store, manifest="m-index", index=OTHER_SHA,
                    **{slot: shas[slot] for slot in ("chunk", "represent", "embed")})
    store._release = dataclasses.replace(store._release, route_id="some-other-route-label")
    app, _ = build(env, inline_stack, **route_overrides("chunk", "represent", "embed", "index"))
    ask, health, info = await _ask_health_info(app)
    assert ask.status_code == 200
    assert health["ready"] is True
    assert info["route_store_check"] == "ok"


@requires_route_profiles
async def test_a_replace_under_the_same_id_re_runs_the_check(env, inline_stack):
    """Bug: a verdict cached by release id alone would keep serving after `--replace` re-pointed the
    same id to another route (and keep refusing after it was put back). Cache key = identity."""
    shas = {slot: route_shas()[slot] for slot in ("chunk", "represent", "embed")}
    store = inline_stack[0]
    replace_release(store, manifest="m-v33", **shas)
    app, _ = build(env, inline_stack, **route_overrides())
    async with call(app) as client:
        first = await client.post("/v1/ask", json=ASK, headers=HEADERS)
        assert (await client.get("/health")).json()["ready"] is True
        replace_release(store, manifest="m-v34", **dict(shas, chunk=OTHER_SHA))
        second = await client.post("/v1/ask", json=ASK, headers=HEADERS)
        health = (await client.get("/health")).json()
        replace_release(store, manifest="m-v33-again", **shas)
        third = await client.post("/v1/ask", json=ASK, headers=HEADERS)
    assert store._release.release_attempt_id == "fixture-inline-0001"
    assert first.status_code == 200
    assert second.status_code == 503 and second.json()["detail"]["code"] == "route_store_mismatch"
    assert health["ready"] is False and health["store"] == "route_store_mismatch"
    assert third.status_code == 200


def test_the_withheld_copy_keeps_the_release_profile_shas(inline_stack):
    """Bug: `withheld` rebuilds the Release; dropping `profile_sha256` there would turn the check
    into "skipped" for any consumer that checks the counted copy (eval, meta)."""
    store = inline_stack[0]
    release = replace_release(store, manifest="m", chunk="a" * 64, represent="b" * 64, embed="c" * 64)
    assert store.withheld(release, ["redistributable"]).profile_sha256 == release.profile_sha256
    pg = pg_store(FakePool())
    active = pg.active_release()
    assert active.profile_sha256 == {"chunk": "a" * 64, "represent": "b" * 64, "embed": "c" * 64}
    assert pg.withheld(active, ["redistributable"]).profile_sha256 == active.profile_sha256


# ================================================================ REAL-API (route construction and the route-store guard)
# How the real start can fail, and which test catches it:
#   R1 an identity-rerank route (s0) loads a reranker anyway (4B weights beside the answerer).
#   R2 a reranking route starts WITHOUT its reranker (silently serves identity under its name).
#   R3 a reranker that cannot load (CUDA OOM) is swallowed, or leaves the embedder on the GPU.
#   R4 models are built per request instead of once at startup, or never closed at shutdown.
#   R5 the configured profiles are not the served route's (config runs one recipe, labels another).
#   R6 the route-store guard refuses s0-v34 over the s1-wide-v34 store (same stored stages), or
#      accepts a store built with another chunk stage.
# The fakes replace the model CLASSES (`JinaEmbedder.from_profile`, `QwenReranker.from_profile`,
# `PgStore.from_config`, `ceres.llm.OpenAIChatLLM`): `api.build_deps` runs unchanged, no weights,
# no GPU, no network.

import os                                                          # noqa: E402

import yaml                                                        # noqa: E402

from ceres import api, models as models_module                     # noqa: E402
from ceres.config import compute_route_id                          # noqa: E402

LOOP = Path(os.environ.get("DDS_LOOP_ROOT")
            or ROOT.parent.parent / "path" / "to" / "reference-pipeline" / "loop-v0")
if LOOP.name != "loop-v0" and (LOOP / "loop-v0").is_dir():
    LOOP = LOOP / "loop-v0"
ROUTES = LOOP / "04_rag" / "routes"
requires_loop_routes = pytest.mark.skipif(
    not all((ROUTES / f"{name}.yaml").is_file() for name in ("s0-v34", "s1-wide-v34")),
    reason=f"the reference pipeline's v3.4 route files are not on this box ({ROUTES})")
SLOTS = ("chunk", "represent", "embed", "index", "retrieve", "rerank", "answer")


def _route_doc(path: Path) -> tuple[str, dict[str, Path]]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parents[2]
    return doc["route_id"], {slot: root / rel for slot, rel in doc["profiles"].items()}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _serve_overrides(route_file: Path, profiles: dict[str, Path], loop_root: Path) -> dict[str, str]:
    values = {f"CERES_ROUTE__PROFILES__{slot.upper()}": str(path) for slot, path in profiles.items()}
    values["CERES_ROUTE__SERVED_ROUTE_FILE"] = str(route_file)
    values["CERES_PATHS__LOOP_ROOT"] = str(loop_root)
    return values


@requires_loop_routes
async def test_a_serve_route_sharing_the_stored_stages_is_not_refused(env, inline_stack):
    """R6: s0-v34 served from the store s1-wide-v34 loaded (same chunk/represent/embed) must
    serve and report `ok` — the release's route_id is the loader's label and never decides."""
    s0_id, s0_profiles = _route_doc(ROUTES / "s0-v34.yaml")
    wide_id, wide_profiles = _route_doc(ROUTES / "s1-wide-v34.yaml")
    assert s0_id != wide_id
    store = inline_stack[0]
    replace_release(store, manifest="m-s1-wide-v34",
                    **{slot: _sha(wide_profiles[slot]) for slot in ("chunk", "represent", "embed")})
    store._release = dataclasses.replace(store._release, route_id=wide_id)
    overrides = _serve_overrides(ROUTES / "s0-v34.yaml", s0_profiles, LOOP)
    app, deps = build(env, inline_stack, **overrides)
    assert api.served_route_check(deps.config).route_id == s0_id     # the real-start check passes
    assert deps.rerank_identity is True                             # s0: no reranker
    ask, health, info = await _ask_health_info(app)
    assert ask.status_code == 200, ask.text
    assert health["store"] == "ready" and health["ready"] is True
    assert info["route_store_check"] == "ok"
    assert info["route_id"] == s0_id and info["release_route_id"] == wide_id


@requires_loop_routes
async def test_a_serve_route_with_a_different_chunk_stage_is_refused(env, inline_stack):
    """R6: the same configuration over a release built with the v3.3 chunk profile is another
    store; asks are 503 route_store_mismatch and /health is not ready."""
    s0_id, s0_profiles = _route_doc(ROUTES / "s0-v34.yaml")
    wide_id, wide_profiles = _route_doc(ROUTES / "s1-wide-v34.yaml")
    v33_chunk = LOOP / "04_rag" / "profiles" / "chunk" / "chunk-bridge-v3.3.yaml"
    if not v33_chunk.is_file():
        pytest.skip(f"the v3.3 chunk profile is not on this box ({v33_chunk})")
    store = inline_stack[0]
    shas = {slot: _sha(wide_profiles[slot]) for slot in ("represent", "embed")}
    replace_release(store, manifest="m-v33-chunk", chunk=_sha(v33_chunk), **shas)
    store._release = dataclasses.replace(store._release, route_id=wide_id)
    app, _deps = build(env, inline_stack, **_serve_overrides(ROUTES / "s0-v34.yaml", s0_profiles, LOOP))
    ask, health, info = await _ask_health_info(app)
    assert ask.status_code == 503 and ask.json()["detail"]["code"] == "route_store_mismatch"
    assert health["ready"] is False and health["store"] == "route_store_mismatch"
    assert info["route_store_check"] == "mismatch"


# ---------------------------------------------------------------- the real build path, with fakes

IDENTITY_RERANK = {"profile_schema_version": "dds-loop-rag-slot-profile-v1", "slot": "rerank",
                   "profile_id": "none-v1", "operation": "identity",
                   "hardware_flags": {"device": "cpu", "deterministic": True}}


def _mini_loop(tmp_path: Path, *, identity: bool, name: str = "t-serve") -> tuple[Path, Path, dict[str, Path]]:
    """A pipeline folder with seven profiles and a frozen route declaration (id recomputed)."""
    loop = tmp_path / "loop"
    profiles_dir = loop / "04_rag" / "profiles"
    profiles_dir.mkdir(parents=True)
    sources = {"chunk": "chunk-bridge-v3.1.yaml", "represent": "verbatim-v1.yaml",
               "embed": "jina-768-v1.yaml", "index": "inmem-exact-v1.yaml",
               "retrieve": "dense-top10-v2.yaml", "rerank": "qwen3-4b-rerank-v1.yaml",
               "answer": "grounded-qwen38-v1.yaml"}
    profiles: dict[str, Path] = {}
    for slot, file in sources.items():
        target = profiles_dir / f"{slot}.yaml"
        if slot == "rerank" and identity:
            target.write_text(yaml.safe_dump(IDENTITY_RERANK), encoding="utf-8")
        else:
            target.write_bytes((FIXTURE_PROFILES / file).read_bytes())
        if slot == "index":          # r1's seven profiles would be r1's id, which is refused for serving
            target.write_bytes(target.read_bytes() + b"\n# test serve route, not r1\n")
        profiles[slot] = target
    routes = loop / "04_rag" / "routes"
    routes.mkdir(parents=True)
    route_file = routes / f"{name}.yaml"
    route_file.write_text(yaml.safe_dump({
        "route_name": name, "route_id": compute_route_id(profiles),
        "profiles": {slot: f"04_rag/profiles/{slot}.yaml" for slot in SLOTS}}), encoding="utf-8")
    return loop, route_file, profiles


class _Built:
    """What the fake classes saw: one entry per construction, plus closes and calls."""

    def __init__(self) -> None:
        self.embedders: list[dict] = []
        self.rerankers: list[dict] = []
        self.stores = 0
        self.llms = 0
        self.closed: list[str] = []
        self.embed_calls = 0
        self.rerank_calls = 0


def _install_fakes(monkeypatch, inline_stack, *, reranker_error: BaseException | None = None) -> _Built:
    built = _Built()
    store, inline_models, _ = inline_stack

    class FakeEmbedder:
        def __init__(self) -> None:
            self.model = object()

        def embed_query(self, text, lang):
            built.embed_calls += 1
            return inline_models.embed_query(text, lang)

        def describe(self):
            return {"adapter": "FakeEmbedder", "profile_id": "jina-768-v1", "device": "fake"}

        def close(self):
            self.model = None
            built.closed.append("embedder")

    class FakeReranker:
        def __init__(self) -> None:
            self.model = object()

        def rerank(self, question, texts):
            built.rerank_calls += 1
            return inline_models.rerank(question, texts)

        def describe(self):
            return {"adapter": "FakeReranker", "profile_id": "qwen3-4b-rerank-v1", "device": "fake"}

        def close(self):
            self.model = None
            built.closed.append("reranker")

    def embedder_from_profile(cls, profile, *, loop_root, device=None, **_):
        built.embedders.append({"profile": str(profile), "loop_root": str(loop_root), "device": device})
        return FakeEmbedder()

    def reranker_from_profile(cls, profile, *, loop_root, device=None, **_):
        built.rerankers.append({"profile": profile.get("profile_id"), "device": device})
        if reranker_error is not None:
            raise reranker_error
        return FakeReranker()

    def store_from_config(cls, section, **_):
        built.stores += 1
        original_close = getattr(store, "close", None)

        def close():
            built.closed.append("store")
            if callable(original_close):
                original_close()
        store.close = close
        return store

    class FakeAnswerer(FakeLLM):
        def __init__(self, base_url, **_):
            super().__init__()
            built.llms += 1

        def close(self):
            built.closed.append("llm")

    from ceres import llm as llm_module
    from ceres.store import PgStore

    monkeypatch.setattr(models_module.JinaEmbedder, "from_profile", classmethod(embedder_from_profile))
    monkeypatch.setattr(models_module.QwenReranker, "from_profile", classmethod(reranker_from_profile))
    monkeypatch.setattr(PgStore, "from_config", classmethod(store_from_config))
    monkeypatch.setattr(llm_module, "OpenAIChatLLM", FakeAnswerer)
    return built


def _real_config(env, tmp_path, *, identity: bool, **extra):
    loop, route_file, profiles = _mini_loop(tmp_path, identity=identity)
    values = dict(env, CERES_API_KEY=API_KEY, **_serve_overrides(route_file, profiles, loop), **extra)
    return load_config(EXAMPLE_CONFIG, values, require_api_key=True), loop


@requires_route_profiles
def test_an_identity_rerank_route_builds_no_reranker(env, inline_stack, tmp_path, monkeypatch):
    """R1: s0-shaped route — the reranker class is never constructed, /health says `absent`."""
    built = _install_fakes(monkeypatch, inline_stack)
    config, loop = _real_config(env, tmp_path, identity=True)
    deps = api.build_deps(config)
    assert built.rerankers == [] and deps.models.reranker is None and deps.rerank_identity is True
    assert len(built.embedders) == 1 and built.embedders[0]["device"] is None      # profile's device
    assert built.embedders[0]["loop_root"] == str(loop)
    assert deps.models.health()["reranker"] == "absent"


@requires_route_profiles
def test_a_reranking_route_builds_its_reranker_on_the_profiles_device(env, inline_stack, tmp_path,
                                                                     monkeypatch):
    """R2: a non-identity rerank profile constructs exactly one reranker, device not overridden."""
    built = _install_fakes(monkeypatch, inline_stack)
    config, _loop = _real_config(env, tmp_path, identity=False)
    deps = api.build_deps(config)
    assert built.rerankers == [{"profile": "qwen3-4b-rerank-v1", "device": None}]
    assert deps.models.reranker is not None and deps.rerank_identity is False


@requires_route_profiles
def test_a_reranker_that_cannot_load_fails_startup_and_frees_the_embedder(env, inline_stack, tmp_path,
                                                                          monkeypatch):
    """R3: CUDA OOM beside the answerer stops startup with the reason; no identity fallback, and
    the already-loaded embedder and store are released."""
    oom = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    built = _install_fakes(monkeypatch, inline_stack, reranker_error=oom)
    config, _loop = _real_config(env, tmp_path, identity=False)
    with pytest.raises(SystemExit) as refused:
        api.build_deps(config)
    message = str(refused.value)
    assert "reranker" in message and "CUDA out of memory" in message
    assert "no fallback to identity" in message
    assert "embedder" in built.closed and "store" in built.closed
    assert built.llms == 0                                           # nothing further was built


@requires_route_profiles
async def test_models_are_built_once_across_requests_and_closed_at_shutdown(env, inline_stack, tmp_path,
                                                                           monkeypatch):
    """R4: three asks reuse the startup models (one construction each); shutdown closes all four."""
    built = _install_fakes(monkeypatch, inline_stack)
    config, _loop = _real_config(env, tmp_path, identity=False)
    app = create_app(api.build_deps(config), config)
    async with app.router.lifespan_context(app):
        async with call(app) as http:
            for _ in range(3):
                response = await http.post("/v1/ask", json=ASK, headers=HEADERS)
                assert response.status_code == 200, response.text
            info = (await http.get("/v1/info", headers=HEADERS)).json()
    assert (len(built.embedders), len(built.rerankers), built.stores, built.llms) == (1, 1, 1, 1)
    assert built.embed_calls == 3 and built.rerank_calls == 3
    assert sorted(built.closed) == ["embedder", "llm", "reranker", "store"]
    assert info["adapters"]["models"] == "InProcessModels"
    assert info["route_id"] == compute_route_id(config.profile_paths())


@requires_route_profiles
@pytest.mark.parametrize("case", ["other_rerank", "no_served_route_file"])
def test_a_config_that_is_not_the_served_routes_recipe_refuses_to_start(env, inline_stack, tmp_path,
                                                                       monkeypatch, case):
    """R5: the configured profiles decide what runs, the declaration what it is called; a config
    whose rerank profile is not the route's (or that names no route) must not start — and must
    refuse before any model loads."""
    built = _install_fakes(monkeypatch, inline_stack)
    config, _loop = _real_config(env, tmp_path, identity=True)
    if case == "other_rerank":
        config.route.profiles["rerank"] = str(FIXTURE_PROFILES / "qwen3-4b-rerank-v1.yaml")
        expected = "rerank"
    else:
        config.route.served_route_file = None
        expected = "served_route_file"
    with pytest.raises(SystemExit, match=expected):
        api.build_deps(config)
    assert built.embedders == [] and built.rerankers == [] and built.stores == 0


# ---------------------------------------------------------------- opt-in: the live stack

@pytest.mark.skipif(os.environ.get("CERES_TEST_LIVE_SERVING") != "1",
                    reason="live serving: set CERES_TEST_LIVE_SERVING=1 (loads the models on the GPU,"
                           " reads the real store, calls the answerer)")
async def test_live_serving_answers_one_english_question():
    """live_serving: `config.yaml` as `ceres-api` would start it (real store, models, answerer);
    one EN question must come back VALIDATED or STRUCTURE_FAILED, never 5xx."""
    config = load_config(Path(os.environ.get("CERES_CONFIG") or ROOT / "config.yaml"),
                         require_api_key=False)
    app = create_app(api.build_deps(config), config)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ceres.test",
                                     timeout=config.api.request_deadline_s + 30) as http:
            health = (await http.get("/health")).json()
            assert health["ready"] is True, health
            response = await http.post("/v1/ask", json={
                "question": "When should winter wheat be sown in southern Romania?", "lang": "en"})
    assert response.status_code == 200, response.text
    assert response.json()["answer"]["state"] in ("VALIDATED", "STRUCTURE_FAILED")


# ---------------------------------------------------------------- startup warm-up
# The bug: "the first live query after start hits a cold cache and times out" (exact scan over ~330k rows, 5 s statement
# timeout). How the fix can fail, and the test that catches each:
#   * no warm-up runs, or it runs with the served 5 s timeout → test_the_first_live_query_after_start_does_not_meet_a_cold_cache
#   * /v1/ask is served (and times out) while the cache is still cold → same test (503 "warming" until it ends)
#   * a failed warm-up keeps the service down → test_a_failed_warm_up_does_not_block_the_service
#   * a release built for another route is warmed (or served) → test_the_warm_up_runs_after_the_route_store_guard
#   * the long timeout leaks to every request (SET instead of SET LOCAL) → test_the_long_timeout_is_set_for_the_warm_up_query_only

class WarmStore:
    """The inline fixture store, whose warm_up records its call and waits for `release` (a cold scan in progress)."""

    def __init__(self, inner, fail=None):
        self.inner, self.fail = inner, fail
        self.calls, self.release = [], threading.Event()

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def warm_up(self, release_attempt_id, serve_rights, admitted=(), refused=None, *, timeout_ms):
        self.calls.append((release_attempt_id, timeout_ms))
        self.release.wait(5)
        if self.fail:
            raise self.fail
        return {"rows": 3, "timeout_ms": timeout_ms}


def warm_app(env, inline_stack, store):
    values = dict(env, CERES_API_KEY=API_KEY)
    config = load_config(EXAMPLE_CONFIG, values, require_api_key=True)
    _s, models, _ = inline_stack
    deps = pipeline.Deps.build(config, store, models, FakeLLM())
    return create_app(deps, config, warm_up=True)


async def test_the_first_live_query_after_start_does_not_meet_a_cold_cache(env, inline_stack):
    from ceres.api import WARM_UP_TIMEOUT_MS

    store = WarmStore(inline_stack[0])
    app = warm_app(env, inline_stack, store)
    async with app.router.lifespan_context(app):                       # startup starts the warm-up thread
        async with call(app) as client:
            for _ in range(100):
                if store.calls:
                    break
                await asyncio.sleep(0.01)
            assert store.calls == [(store.active_release().release_attempt_id, WARM_UP_TIMEOUT_MS)]
            assert WARM_UP_TIMEOUT_MS >= 300_000                           # not the served 5 s
            health = (await client.get("/health")).json()
            assert health["store"] == "warming" and health["ready"] is False
            refused = await client.post("/v1/ask", json=ASK, headers=HEADERS)
            assert refused.status_code == 503 and refused.json()["detail"]["code"] == "warming"
            store.release.set()
            for _ in range(100):
                if app.state.warm_up.state != "warming":
                    break
                await asyncio.sleep(0.01)
            assert app.state.warm_up.state == "done"
            assert (await client.get("/health")).json()["ready"] is True
            assert (await client.post("/v1/ask", json=ASK, headers=HEADERS)).status_code == 200


async def test_a_failed_warm_up_does_not_block_the_service(env, inline_stack):
    store = WarmStore(inline_stack[0], fail=StoreError("warm-up: QueryCanceled"))
    store.release.set()
    app = warm_app(env, inline_stack, store)
    app.state.warm_up.run()                                            # what the startup thread does
    assert app.state.warm_up.state == "failed"
    async with call(app) as client:
        assert (await client.get("/health")).json()["ready"] is True
        assert (await client.post("/v1/ask", json=ASK, headers=HEADERS)).status_code == 200


async def test_the_warm_up_runs_after_the_route_store_guard(env, inline_stack, monkeypatch):
    store = WarmStore(inline_stack[0])
    store.release.set()
    app = warm_app(env, inline_stack, store)

    def mismatch(_deps, _release):
        raise pipeline.RouteStoreMismatch("route_store_mismatch: chunk differs")

    monkeypatch.setattr(pipeline, "route_store_check", mismatch)
    app.state.warm_up.run()
    assert store.calls == [] and app.state.warm_up.state == "failed"


def test_the_long_timeout_is_set_for_the_warm_up_query_only():
    """`SET LOCAL` (this transaction) — a session-level SET would leave 300 s on a pooled connection for later requests."""
    pool = FakePool()
    store = pg_store(pool)
    store.warm_up("rel-1", ["redistributable"], timeout_ms=300_000)
    sets = [sql for sql in pool.executed if "statement_timeout" in sql]
    assert sets == ["SET LOCAL statement_timeout = 300000"]


def test_the_first_withheld_count_on_a_cold_cache_does_not_time_out():
    """Seen in a parity run: `store withheld count failed: QueryCanceled` (cold aggregate over ~330k rows > 5 s). The
    once-per-release count runs with the long timeout, for its own transaction only, and is cached."""
    pool = FakePool()
    store = pg_store(pool)
    release = store.active_release()
    store.withheld(release, ["redistributable"])
    store.withheld(release, ["redistributable"])
    assert [sql for sql in pool.executed if "statement_timeout" in sql] == ["SET LOCAL statement_timeout = 300000"]
    assert sum("count(*)" in sql for sql in pool.executed) == 1


async def test_the_warm_up_fills_the_withheld_count_before_serving(env, inline_stack):
    store = WarmStore(inline_stack[0])
    store.release.set()
    counted = []
    store.withheld = lambda *a, **k: counted.append(a) or inline_stack[0].withheld(*a, **k)
    app = warm_app(env, inline_stack, store)
    app.state.warm_up.run()
    assert app.state.warm_up.state == "done" and len(counted) == 1


# ---------------------------------------------------------------- fix (a): warm-up until fast
# The bug (seen in a live check): one 54 s cold warm-up scan "finished", /health said ready, and the next four
# searches still passed the 5 s request timeout (503 QueryCanceled). How the fix can fail, and the test that catches each:
#   * one try marks a cold store ready → test_a_cold_store_whose_first_search_is_slow_is_not_ready_after_one_try
#   * a store that never gets fast is reported ready → test_a_store_that_never_gets_fast_is_never_reported_ready
#   * a fast store pays for extra tries → test_a_fast_store_is_warmed_in_one_try

class TimedStore:
    """The inline fixture store whose warm_up takes the next duration from `durations_s` (a cold scan, then warmer)."""

    def __init__(self, inner, durations_s):
        self.inner, self.durations = inner, list(durations_s)
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def warm_up(self, release_attempt_id, serve_rights, admitted=(), refused=None, *, timeout_ms):
        self.calls += 1
        time.sleep(self.durations[min(self.calls, len(self.durations)) - 1])
        return {"rows": 3, "timeout_ms": timeout_ms}


def _fast_threshold(monkeypatch, ms=50):
    import ceres.api as api_module
    monkeypatch.setattr(api_module, "WARM_UP_FAST_MS", ms)


async def test_a_cold_store_whose_first_search_is_slow_is_not_ready_after_one_try(env, inline_stack, monkeypatch):
    _fast_threshold(monkeypatch)
    store = TimedStore(inline_stack[0], [0.12, 0.08, 0.0])       # cold, still cold, then fast (threshold 50 ms)
    app = warm_app(env, inline_stack, store)
    app.state.warm_up.run()
    assert store.calls == 3 and app.state.warm_up.state == "done"
    assert app.state.warm_up.tries_ms[0] >= 50 and app.state.warm_up.tries_ms[-1] < 50
    async with call(app) as client:
        assert (await client.get("/health")).json()["ready"] is True


async def test_a_store_that_never_gets_fast_is_never_reported_ready(env, inline_stack, monkeypatch):
    import ceres.api as api_module
    _fast_threshold(monkeypatch)
    store = TimedStore(inline_stack[0], [0.06])                   # every try over the threshold
    app = warm_app(env, inline_stack, store)
    app.state.warm_up.run()
    assert store.calls == api_module.WARM_UP_TRIES and app.state.warm_up.state == "slow"
    async with call(app) as client:
        health = (await client.get("/health")).json()
    assert health["ready"] is False and health["store"] == "slow_store" and "tries" in health["detail"]["store"]


async def test_a_fast_store_is_warmed_in_one_try(env, inline_stack, monkeypatch):
    _fast_threshold(monkeypatch)
    store = TimedStore(inline_stack[0], [0.0])
    app = warm_app(env, inline_stack, store)
    app.state.warm_up.run()
    assert store.calls == 1 and app.state.warm_up.state == "done"
