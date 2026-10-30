"""The HTTP surface: `GET /health`, `GET /v1/info`, `POST /v1/ask`.

Security baseline (the service's security contract): the API key comes from the environment and is compared
in constant time; `/health` is the only unauthenticated route; the interactive docs are off unless
the config asks for them; bodies (counted as they arrive) and questions are bounded; a client sees
stable public error codes, never upstream error text; admission is bounded in both running and
waiting requests, and a permit is held by the running work until it ends, because one inference
worker cannot absorb a queue.

Nothing in this module decides anything about the route — that is `pipeline.py`. This module maps
outcomes to status codes and writes one structured log line per request.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hmac
import json
import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from . import __version__, pipeline
from .config import CeresConfig, ConfigError, load_config
from .contracts import AskRequest, AskResponse, HealthResponse, InfoResponse
from .llm import context_check, live_server_props
from .memory import MemoryError as MemoryRefused
from .store import StoreError, read_predicate_sha256

LOG = logging.getLogger("ceres.api")


def _log_line(event: str, **fields: Any) -> str:
    """One structured JSON log line. The question text never appears here (DEBUG only)."""
    return json.dumps({"event": event, **fields}, ensure_ascii=False, default=str)


def _cause(exc: BaseException) -> str:
    """The private detail of a wrapped failure, for the log only — never for a response body."""
    cause = exc.__cause__
    return f"{exc} <- {cause.__class__.__name__}: {cause}" if cause is not None else str(exc)


# ---------------------------------------------------------------- admission

class Busy(Exception):
    """Admission refused the request: the queue is full, or the wait ran out."""


class Admission:
    """Bounded admission whose permits follow the WORK, not the waiting client.

    `max_inflight` pipeline runs at once; at most `max_pending` requests wait for one, each for at
    most `queue_timeout_s`; anything beyond is refused at once. The pipeline is synchronous and
    GPU bound, so it runs in a worker thread — and a Python thread cannot be interrupted. That is
    why the permit is released by the worker thread itself when the pipeline returns (via
    `call_soon_threadsafe`), and never by the request handler: a client that disconnects, or a
    request that hits its deadline, stops waiting, but its thread keeps running and keeps its
    permit until it ends. Previously the handler released it in a `finally`, and a cancelled
    request let a second worker onto a one-worker GPU.
    """

    def __init__(self, max_inflight: int, max_pending: int, queue_timeout_s: float) -> None:
        self.max_inflight = int(max_inflight)
        self.max_pending = int(max_pending)
        self.queue_timeout_s = float(queue_timeout_s)
        self._slots = asyncio.Semaphore(self.max_inflight)
        self.pending = 0
        # Every started run, until it ends — so shutdown can wait for work nobody awaits any more.
        self.running: set[concurrent.futures.Future] = set()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_inflight, thread_name_prefix="ceres-worker")

    async def acquire(self, timeout_s: float) -> None:
        """Wait for a permit, or raise `Busy`. Refuses at once when the waiting room is full."""
        if self._slots.locked() and self.pending >= self.max_pending:
            raise Busy("queue_full")
        self.pending += 1
        try:
            await asyncio.wait_for(self._slots.acquire(), timeout=min(self.queue_timeout_s, timeout_s))
        except (asyncio.TimeoutError, TimeoutError):
            raise Busy("queue_timeout") from None
        finally:
            self.pending -= 1

    def start(self, fn: Callable[..., Any], *args: Any) -> "asyncio.Future[Any]":
        """Run `fn(*args)` in a worker thread that holds the permit until `fn` returns.

        Call only after `acquire()` succeeded. Returns an asyncio future for the result; awaiting
        it may be abandoned (deadline, disconnect) without affecting the run or its permit.
        """
        loop = asyncio.get_running_loop()
        slots = self._slots

        def worker() -> Any:
            try:
                return fn(*args)
            finally:
                try:
                    loop.call_soon_threadsafe(slots.release)
                except RuntimeError:          # the loop is closed: the process is exiting
                    pass

        future = self._executor.submit(worker)
        self.running.add(future)
        future.add_done_callback(self.running.discard)
        result = asyncio.wrap_future(future)
        # If nobody awaits the result any more, retrieve its exception anyway so asyncio does not
        # log "exception was never retrieved"; the run itself already logged what matters.
        result.add_done_callback(lambda f: f.cancelled() or f.exception())
        return result

    async def drain(self, timeout_s: float) -> None:
        """At shutdown: wait (bounded) for runs still in flight, then stop the worker pool."""
        pending = [asyncio.wrap_future(f) for f in list(self.running)]
        if pending:
            LOG.info(_log_line("shutdown.draining", runs=len(pending)))
            await asyncio.wait(pending, timeout=timeout_s)
        self._executor.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------- body limit

class BodyLimit:
    """Pure ASGI middleware: refuse a request body larger than `max_bytes` with 413.

    Two checks. A declared `Content-Length` over the limit is refused before anything is read.
    A body without one (chunked transfer) is counted while its `http.request` messages arrive,
    and refused as soon as the count passes the limit — previously only the header was checked,
    so a chunked body of any size reached the JSON parser. An accepted body
    is at most `max_bytes`, buffered, and replayed to the app as one message.
    """

    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = int(max_bytes)

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._too_large(scope, receive, send)
            return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return                                    # the client left; nobody to answer
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                await self._too_large(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay() -> dict:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()                        # later calls: disconnect detection

        await self.app(scope, replay, send)

    @staticmethod
    async def _too_large(scope: dict, receive: Callable, send: Callable) -> None:
        response = JSONResponse(status_code=413, content={"detail": "request body too large"})
        await response(scope, receive, send)


# ---------------------------------------------------------------- the app

# Startup warm-up : one exact search over the active release before
# /v1/ask is served, with this statement timeout for that ONE query (every request keeps `store.statement_timeout_ms`).
WARM_UP_TIMEOUT_MS = 300_000
WARMING_CODE = "warming"


# One warm-up scan did not warm a cold 331k-row store (a 54 s warm-up was followed by four
# 5 s request timeouts). The warm-up repeats until one search runs under WARM_UP_FAST_MS, at most WARM_UP_TRIES times.
WARM_UP_FAST_MS = 2_000
WARM_UP_TRIES = 5
SLOW_STORE_CODE = "slow_store"


class WarmUp:
    """The startup warm-up and its state: "off" (not requested), "warming", "done", "slow" or "failed".

    It runs AFTER the route-store guard (a release built for another route is not warmed) and on a daemon thread, so
    /health can say "warming" meanwhile. Each try is one served-shape exact search (long timeout, this query only); the
    warm-up ends when a try runs under WARM_UP_FAST_MS. If WARM_UP_TRIES never get there, the state is "slow": /health is
    not ready and names the timings — a slow store is never reported ready. An error (not slowness) is logged as a warning
    and does not block the service: the first query then simply meets the cold cache, as before."""

    def __init__(self, deps: pipeline.Deps, enabled: bool) -> None:
        self.deps = deps
        self.state = "warming" if enabled else "off"
        self.duration_ms: float | None = None
        self.tries_ms: list[float] = []
        self.detail: str | None = None

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.run, name="ceres-warm-up", daemon=True)
        thread.start()
        return thread

    def run(self) -> None:
        started = time.perf_counter()
        store_config = self.deps.config.store
        release_id = None
        try:
            # The guard, then the release-level withheld count (cached; long timeout): what the first request pins.
            release = pipeline.pinned_release(self.deps)
            release_id = release.release_attempt_id
            warm = getattr(self.deps.store, "warm_up", None)
            if not callable(warm):
                self.state, self.detail = "done", "the store has no warm_up"
                return
            for attempt in range(1, WARM_UP_TRIES + 1):
                mark = time.perf_counter()
                result = warm(release_id, store_config.serve_rights, store_config.admitted_works,
                              store_config.refused_evidence_basis, timeout_ms=WARM_UP_TIMEOUT_MS)
                took = round((time.perf_counter() - mark) * 1000, 1)
                self.tries_ms.append(took)
                LOG.info(_log_line("startup.warm_up_try", release=release_id, attempt=attempt, duration_ms=took,
                                   fast_ms=WARM_UP_FAST_MS, **{k: v for k, v in result.items() if k != "timeout_ms"}))
                if took < WARM_UP_FAST_MS:
                    break
            self.duration_ms = round((time.perf_counter() - started) * 1000, 1)
            fast = self.tries_ms[-1] < WARM_UP_FAST_MS
            self.state = "done" if fast else "slow"
            self.detail = f"tries {self.tries_ms} ms"
            log = LOG.info if fast else LOG.warning
            log(_log_line("startup.warm_up" if fast else "startup.warm_up_slow", release=release_id,
                          duration_ms=self.duration_ms, tries_ms=self.tries_ms, timeout_ms=WARM_UP_TIMEOUT_MS))
        except Exception as exc:                                    # noqa: BLE001 — an error never blocks the service
            self.duration_ms = round((time.perf_counter() - started) * 1000, 1)
            self.detail = exc.__class__.__name__
            self.state = "failed"
            LOG.warning(_log_line("startup.warm_up_failed", release=release_id, duration_ms=self.duration_ms,
                                  tries_ms=self.tries_ms, detail=_cause(exc)))


def create_app(deps: pipeline.Deps, config: CeresConfig | None = None, *, warm_up: bool = False) -> FastAPI:
    """Build the ASGI app around one resolved `Deps`. `warm_up` (the real service) runs `WarmUp` at startup."""
    config = config or deps.config
    warming = WarmUp(deps, warm_up)
    docs_on = bool(config.api.docs)
    admission = Admission(config.api.admission.max_inflight, config.api.admission.max_pending,
                          config.api.admission.queue_timeout_s)

    # Provenance is a snapshot taken NOW, from what `Deps.build` actually loaded, plus the digest
    # of the effective configuration. `/v1/info` reports it without touching the disk or the
    # model, so what it says is what the running pipeline uses.
    snapshot = {
        "profiles": dict(deps.artifacts.get("profiles", {})),
        "prompt_template_sha256": deps.artifacts.get("prompt"),
        "config_sha256": config.config_sha256(),
    }

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if warming.state == "warming":
            warming.start()
        yield
        # Shutdown: let runs in flight finish (bounded by the request deadline), then release the
        # adapters' resources — the store's connection pool, the answerer's HTTP client.
        await admission.drain(config.api.request_deadline_s)
        for name, adapter in (("store", deps.store), ("models", deps.models), ("llm", deps.llm)):
            close = getattr(adapter, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:                  # noqa: BLE001 — shutdown goes on
                    LOG.warning(_log_line("shutdown.close_failed", adapter=name,
                                          detail=exc.__class__.__name__))

    app = FastAPI(
        title="Ceres", version=__version__, lifespan=lifespan,
        # Off by default: an unauthenticated schema endpoint tells an attacker the shape of
        # everything. `api.docs: true` turns all three on together.
        docs_url="/docs" if docs_on else None,
        redoc_url="/redoc" if docs_on else None,
        openapi_url="/openapi.json" if docs_on else None,
    )
    app.state.deps = deps
    app.state.config = config
    app.state.warm_up = warming
    app.state.admission = admission
    # The answerer's live settings as the LAST /health saw them. `/v1/info`
    # reports this snapshot and never re-fetches it, so /v1/info stays a no-probe route.
    app.state.answerer_live = None
    app.add_middleware(BodyLimit, max_bytes=config.api.max_body_bytes)

    def require_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
        """Constant-time API-key check. `--no-auth` (smoke tests only) skips it."""
        if not config.auth_required:
            return
        expected = config.api_key or ""
        # Compare bytes: compare_digest raises TypeError on non-ASCII str, which turned a bad key into a 500
        if not expected or not x_api_key or not hmac.compare_digest(x_api_key.encode("utf-8"),
                                                                    expected.encode("utf-8")):
            raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")

    # ------------------------------------------------------------ health (unauthenticated)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Liveness against readiness. The ONE route that probes the store and the model."""
        store_status, store_detail = _safe(deps.store.health, ("down", "unavailable"))
        model_status, model_detail, _model_id = _safe(deps.llm.health, ("down", "unavailable", None))
        # Readiness of the model also needs its LIVE context to be the profile's: reachable AND
        # `/v1/models` lists expect_model AND (live n_ctx unknown OR == context_tokens). A known
        # mismatch is a different route → not ready, with both numbers.
        props = live_server_props(deps.llm) if model_status == "ready" else None
        check = context_check(props, deps.answer_profile)
        if check["live_n_ctx_matches_profile"] is False:
            model_status = "wrong_context"
            model_detail = (f"live n_ctx {check['live_n_ctx']} != answer profile context_tokens"
                            f" {check['profile_context_tokens']}")
        app.state.answerer_live = {"checked_at": _utc_now(), "model": model_status,
                                   "server_props": props, **check}
        model_health = getattr(deps.models, "health", None)
        models_state = model_health() if callable(model_health) else {}
        release_id = None
        try:
            release = deps.store.active_release()
            release_id = release.release_attempt_id
        except Exception:                                    # noqa: BLE001 — health never raises
            release = None
        # ROUTE-STORE-GUARD: a reachable store whose active release was built for another route is
        # not ready. Public code only here (unauthenticated route); slot and shas go to the log,
        # once per release identity (`pipeline.route_store_check` caches the verdict).
        if store_status == "ready" and release is not None:
            try:
                pipeline.route_store_check(deps, release)
            except pipeline.RouteStoreMismatch:
                store_status = store_detail = pipeline.ROUTE_STORE_MISMATCH_CODE
        if store_status == "ready" and warming.state == "warming":
            store_status = store_detail = WARMING_CODE        # until the startup warm-up ends (done, slow or failed)
        elif store_status == "ready" and warming.state == "slow":
            store_status = SLOW_STORE_CODE                    # never ready on a store the warm-up could not get fast
            store_detail = f"warm-up never under {WARM_UP_FAST_MS} ms in {WARM_UP_TRIES} tries: {warming.tries_ms} ms"
        embedder = models_state.get("embedder", "unknown")
        reranker = models_state.get("reranker", "unknown")
        ready = (store_status == "ready" and model_status == "ready"
                 and embedder not in ("down", "unknown") and reranker not in ("down", "unknown"))
        return HealthResponse(
            api="ok", ready=ready, store=store_status, model=model_status,
            embedder=embedder, reranker=reranker, release_attempt_id=release_id,
            # Detail is a short reason, never a DSN, a URL with credentials or an upstream body.
            detail={k: v for k, v in
                    {"store": store_detail, "model": model_detail,
                     "models": models_state.get("detail", "")}.items() if v},
        )

    # ------------------------------------------------------------ info

    @app.get("/v1/info", response_model=InfoResponse, dependencies=[Depends(require_key)])
    def info() -> InfoResponse:
        """What this process serves. Artifacts and config from the startup snapshot; the release
        is live state and is read per call (one row), because it can move under the service."""
        release = None
        try:
            release = deps.store.active_release()
        except Exception:                                    # noqa: BLE001 — info must still answer
            release = None
        route_store = None
        if release is not None:
            try:
                route_store = pipeline.route_store_check(deps, release)
            except pipeline.RouteStoreMismatch:
                route_store = "mismatch"
        route_id = config.route.served_route_id
        # The mode is the key an operator sets; `serve_rights` is derived from it. Reporting
        # both makes a mode switch visible to anyone who can call /v1/info, without them having to
        # read the config file on the box.
        store_info: dict[str, Any] = {
            "dsn": config.safe_dsn, "schema": config.store.db_schema,
            "serving_mode": config.store.serving_mode,
            "serve_rights": list(config.store.serve_rights),
            "predicate": "active+evidence_eligible+rights",
            # The evidence basis this mode refuses (production: OCR-admitted), or null.
            "refused_evidence_basis": config.store.refused_evidence_basis,
        }
        if config.store.serving_mode == "production":
            store_info["admitted_licensed_works"] = len(config.store.admitted_works)
            store_info["production_decision"] = config.store.production_decision
        answerer = config.endpoints.get("answerer")
        return InfoResponse(
            version=__version__,
            # Null until the served-route yaml is frozen. An unassigned route must be
            # visible as unassigned — a served answer is never labelled r1.
            route_id=route_id,
            route_status="assigned" if route_id else "unassigned",
            judged_route_id=config.route.judged_route_id,
            serving_mode=config.store.serving_mode,
            read_predicate_sha256=read_predicate_sha256(
                config.store.serve_rights, config.store.db_schema,
                admitted_licensed_works=config.store.admitted_works,
                refused_evidence_basis=config.store.refused_evidence_basis),
            release_attempt_id=release.release_attempt_id if release else None,
            release_route_id=release.route_id if release else None,
            route_store_check=route_store,
            row_count=release.row_count if release else None,
            membership_counts=release.membership_counts if release else {},
            profiles=snapshot["profiles"],
            prompt_template_sha256=snapshot["prompt_template_sha256"],
            store=store_info,
            # As configured, not as probed: `/health` asks the endpoint what it serves.
            models={"answerer_alias": deps.answer_profile.get("model_alias"),
                    "answerer_expected": answerer.expect_model if answerer else None},
            config_sha256=snapshot["config_sha256"],
            # ceres-validate-r3: the ellipsis-quote guards this process validates with.
            validator_guards=pipeline.QUOTE_GUARDS.as_dict(),
            packet_width=deps.max_contexts,
            # The snapshot the last /health took; null before the first /health. Not re-fetched.
            answerer_live=app.state.answerer_live,
            adapters={"store": type(deps.store).__name__, "models": type(deps.models).__name__,
                      "llm": type(deps.llm).__name__},
        )

    # ------------------------------------------------------------ profiles (memory v0)

    @app.get("/v1/profiles", dependencies=[Depends(require_key)])
    def profiles() -> dict[str, Any]:
        """The static personas (memory v0): id, role, language, summary. Empty list when memory is off."""
        if deps.memory is None:
            return {"memory_enabled": False, "profiles": []}
        return {"memory_enabled": True, "store": "in-memory (lost on restart)",
                "profiles": [{"id": p.id, "role": p.role, "language": p.language, "summary": p.summary}
                             for p in deps.memory.personas.values()]}

    # ------------------------------------------------------------ ask

    @app.post("/v1/ask", response_model=AskResponse, dependencies=[Depends(require_key)])
    async def ask(request: AskRequest) -> AskResponse:
        if len(request.question) > config.api.max_question_chars:
            raise HTTPException(status_code=413,
                                detail=f"question exceeds {config.api.max_question_chars} characters")
        LOG.debug(_log_line("ask.question", lang=request.lang, question=request.question))
        if request.uses_memory:          # memory v0: refuse before any model call or admission permit
            if deps.memory is None:
                raise HTTPException(status_code=422, detail="memory is not enabled on this service (memory.enabled: false)")
            if request.profile_id is not None and request.profile_id not in deps.memory.personas:
                raise HTTPException(status_code=422, detail=f"unknown profile_id {request.profile_id!r}; "
                                                            f"known: {sorted(deps.memory.personas)}")
        if warming.state == "warming":
            raise HTTPException(status_code=503, detail={
                "code": WARMING_CODE, "message": "the store cache is warming after start; retry shortly"})
        started = time.perf_counter()
        deadline_s = config.api.request_deadline_s
        try:
            await admission.acquire(deadline_s)
        except Busy as refusal:
            LOG.warning(_log_line("ask.rejected", reason=str(refusal),
                                  waited_ms=round((time.perf_counter() - started) * 1000, 1)))
            raise HTTPException(status_code=429, detail="the service is busy; retry shortly") from None

        # From here the worker thread owns the permit. This handler only waits for the result;
        # if the client disconnects, the await is cancelled and the run finishes on its own.
        run = admission.start(pipeline.ask, deps, request)
        remaining = max(0.0, deadline_s - (time.perf_counter() - started))
        try:
            response = await asyncio.wait_for(asyncio.shield(run), timeout=remaining)
        except (asyncio.TimeoutError, TimeoutError):
            LOG.error(_log_line("ask.deadline", deadline_s=deadline_s,
                                note="the run continues and keeps its permit until it ends"))
            raise HTTPException(status_code=504, detail={
                "code": "deadline_exceeded",
                "message": "no answer within the request deadline"}) from None
        except StoreError as exc:
            LOG.error(_log_line("ask.failed", reason="store", detail=_cause(exc)))
            raise HTTPException(status_code=503, detail="the serving store is not ready") from None
        except MemoryRefused as exc:     # memory v0: e.g. a persona switch mid-conversation
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except pipeline.RouteStoreMismatch as exc:
            # The active release is another route's chunks. Public code only; slot + shas are logged.
            LOG.error(_log_line("ask.failed", reason=pipeline.ROUTE_STORE_MISMATCH_CODE,
                                detail=_cause(exc)))
            raise HTTPException(status_code=503, detail={
                "code": pipeline.ROUTE_STORE_MISMATCH_CODE,
                "message": "the active release was not built for the configured route"}) from None
        except pipeline.RerankTextUnavailable as exc:
            # a rerank_text: represented route and a release/candidate without that text.
            LOG.error(_log_line("ask.failed", reason=pipeline.RERANK_TEXT_UNAVAILABLE_CODE,
                                detail=_cause(exc)))
            raise HTTPException(status_code=503, detail={
                "code": pipeline.RERANK_TEXT_UNAVAILABLE_CODE,
                "message": "the release cannot give the text this route reranks on"}) from None
        except pipeline.PipelineError as exc:
            LOG.error(_log_line("ask.failed", reason="pipeline", detail=_cause(exc)))
            raise HTTPException(status_code=503, detail="the service cannot serve this route") from None

        state = response.answer["state"]
        LOG.info(_log_line(
            "ask.answered", execution_id=response.meta.execution_id,
            request_id=response.meta.request_id, question_id=response.packet["question_id"],
            lang=response.packet["language"], state=state,
            attempts=response.answer["attempts"], contexts=len(response.packet["contexts"]),
            release=response.meta.release_attempt_id, model=response.meta.model_reported,
            latency_ms=response.meta.latency_ms.model_dump(),
        ))
        if state == "TRANSPORT_FAILED":
            # The row says `transport_failed` and nothing more (`pipeline.answer`); the upstream
            # text is in the raw run record and in the `answer.transport_failed` log line.
            return JSONResponse(status_code=503, content=json.loads(response.model_dump_json()))
        # VALIDATED and STRUCTURE_FAILED are both 200: the answer row says which, and a
        # STRUCTURE_FAILED row carries no `candidate`, so there is no invented user answer.
        return response

    return app


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe(call: Callable[[], Any], fallback: Any) -> Any:
    """Health probes never raise into a response; an unexpected failure is reported as down."""
    try:
        return call()
    except Exception as exc:                                  # noqa: BLE001
        LOG.warning(_log_line("health.probe_failed", detail=exc.__class__.__name__))
        return fallback


# ---------------------------------------------------------------- `ceres-api`

class StartupRefused(SystemExit):
    """`ceres-api` refuses to start; the message says why. A `SystemExit`, so the process exits
    with the message on stderr before uvicorn binds the port."""


def served_route_check(config: CeresConfig) -> Any:
    """The real stack serves a NAMED route, and the configured profiles must be that route's.

    `route.served_route_file` names the served identity. Its seven profiles (recomputed
    route id checked by `eval.load_route`) must be byte-identical to `route.profiles`: the config
    decides what runs (widths, reranker, prompt pin), the declaration decides what the answers are
    labelled, and the two must be the same recipe. Returns the loaded `eval.Route`.
    """
    from .eval import EvalError, load_route
    from .models import sha256_file

    if not config.route.served_route_file:
        raise StartupRefused(
            "route.served_route_file is not set: the real stack serves a named route, so name"
            " its frozen declaration (e.g. <loop>/04_rag/routes/s0hyb150-p3-v35h-r0013.yaml). `--fake` needs none.")
    try:
        route = load_route(config.resolve(config.route.served_route_file))
    except EvalError as exc:
        raise StartupRefused(f"route.served_route_file: {exc}") from None
    configured = config.profile_paths()
    wrong = []
    for slot, expected in route.profile_sha256.items():
        path = configured.get(slot)
        if path is None or not path.is_file():
            wrong.append(f"{slot}: not configured (the route names {route.profiles[slot].name})")
        elif sha256_file(path) != expected:
            wrong.append(f"{slot}: configured {path.name} is not the route's {route.profiles[slot].name}")
    if wrong:
        raise StartupRefused(f"route.profiles are not the profiles of {route.name}"
                             f" ({route.route_id[:12]}…): " + "; ".join(wrong))
    return route


def build_deps(config: CeresConfig, *, fake: bool = False) -> pipeline.Deps:
    """Wire the adapters: the fixture stack (`--fake`) or the real one.

    Real: `PgStore` from `config.store`, the route's Jina embedder, its reranker (none for an
    identity-rerank route such as s0), and `OpenAIChatLLM` for `endpoints.answerer`, built by the
    same builders `ceres-eval` uses (`ceres.models.build_*`). The models load ONCE, here, before
    uvicorn binds the port; the lifespan closes them at shutdown. Any failure refuses startup
    with its reason — a reranker that cannot load never degrades the route to identity rerank.
    """
    if fake:
        # One definition of the fixture stack (`ceres/testing.py`), and it is the one the smoke
        # test proves. `--fake` therefore serves exactly what `smoke_test.py` exercises.
        from .testing import build_fixture_stack

        store, models, llm, _questions = build_fixture_stack()
        return pipeline.Deps.build(config, store, models, llm)

    endpoint = config.endpoints.get("answerer")
    if endpoint is None:
        raise StartupRefused("endpoints.answerer is not configured")
    route = served_route_check(config)
    if not config.paths.loop_root:
        raise StartupRefused("paths.loop_root is not set: the models' platform configs live in the loop")
    loop_root = config.resolve(config.paths.loop_root)
    profiles = config.profile_paths()

    from .models import InProcessModels, build_answerer, build_embedder, build_reranker
    from .store import PgStore

    opened: list[Any] = []

    def refuse(what: str, exc: BaseException) -> StartupRefused:
        for adapter in reversed(opened):                  # release what did load (GPU memory)
            try:
                adapter.close()
            except Exception:                             # noqa: BLE001 — refusing anyway
                pass
        LOG.error(_log_line("startup.refused", stage=what, detail=f"{exc.__class__.__name__}: {exc}"))
        return StartupRefused(f"ceres-api cannot start: {what}: {exc}")

    started = time.perf_counter()
    try:
        store = PgStore.from_config(config.store)
    except Exception as exc:                              # noqa: BLE001
        raise refuse("the store", exc) from exc
    opened.append(store)
    try:
        embedder = build_embedder(profiles["embed"], loop_root=loop_root)
    except Exception as exc:                              # noqa: BLE001
        raise refuse("the embedder did not load", exc) from exc
    opened.append(embedder)
    try:
        reranker = build_reranker(profiles["rerank"], loop_root=loop_root)
    except Exception as exc:                              # noqa: BLE001
        raise refuse("the reranker did not load", exc) from exc
    if reranker is not None:
        opened.append(reranker)
    llm = build_answerer(endpoint)
    opened.append(llm)
    try:
        deps = pipeline.Deps.build(config, store, InProcessModels(embedder, reranker), llm)
    except pipeline.PipelineError as exc:
        raise refuse("the route", exc) from exc
    if deps.rerank_identity != (reranker is None):        # both read the same rerank profile
        raise refuse("the route", RuntimeError(
            f"rerank_identity={deps.rerank_identity} but a reranker was "
            f"{'not ' if reranker is None else ''}loaded"))
    LOG.info(_log_line("startup.models_loaded", route=route.name, route_id=route.route_id,
                       embedder=getattr(embedder, "describe", dict)().get("device"),
                       reranker="identity" if reranker is None
                       else getattr(reranker, "describe", dict)().get("device"),
                       load_ms=round((time.perf_counter() - started) * 1000)))
    return deps


def main(argv: list[str] | None = None) -> int:
    """`ceres-api` — start uvicorn on the configured host and port."""
    parser = argparse.ArgumentParser(prog="ceres-api", description="Serve the Ceres API")
    parser.add_argument("--config", type=str, default=None, help="config file (default: $CERES_CONFIG)")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--fake", action="store_true",
                        help="fixture store and fake models; without it: PgStore + the served route's models")
    parser.add_argument("--no-auth", action="store_true",
                        help="serve /v1/* without an API key — smoke tests only")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(message)s", stream=sys.stderr)
    try:
        from pathlib import Path

        config = load_config(Path(args.config) if args.config else None,
                             require_api_key=not args.no_auth)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    app = create_app(build_deps(config, fake=args.fake), config, warm_up=not args.fake)

    import uvicorn

    uvicorn.run(app, host=args.host or config.api.host, port=args.port or config.api.port,
                log_level=args.log_level)
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
