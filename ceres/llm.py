"""The answerer: the r1 request body, one OpenAI-compatible client, and the fake used for tests and smoke runs.

The request body is the load-bearing part of this module. It is the body
`slot_adapter.call_answer_endpoint` (`04_rag/slot_adapter.py:1452`) sends, field for field, with
the same decoding values from the answer profile and the same `stable_seed(question_id, attempt)`.
The DDS CLI has a port of the same thing (`_request_body`). If Ceres changes any field
here, it is no longer serving the judged route — `tests/test_pipeline.py` asserts the shape.

In service, `OpenAIChatLLM` replaces `FakeLLM` and talks to the answer server (Qwen3.8). When the shared
DDS transport library lands, swapping is this one file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import date, datetime, timezone
from typing import Any, Mapping, Protocol, Sequence

LOG = logging.getLogger("ceres.llm")

DEFAULT_DECODING = {
    "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5,
    "max_tokens": 700, "maximum_attempts": 2, "request_timeout_seconds": 300,
    "enable_thinking": False,
}


class LLMTransportError(Exception):
    """The answerer did not answer (network error, timeout, non-2xx status).

    It terminates the attempt loop (the admission contract). The message may carry upstream detail — a
    URL, a server's error text — so it goes to the private run record and the log only; the
    client sees the public code `transport_failed`.
    """


class LLMRequestRejected(Exception):
    """The answerer answered, but refused THIS request as malformed or too large (HTTP 400, 413, 422 —
    e.g. llama.cpp's context overflow). Not a transport failure: the server is up and another question
    may succeed, so the loop spends the attempt and the question ends STRUCTURE_FAILED with
    `answerer_rejected_request` instead of stopping the run.
    Other 4xx (401/403/404/405/429) mean a configuration or capacity problem and stay transport failures.
    """

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = int(status)


# HTTP statuses that reject one request, not the endpoint (see LLMRequestRejected).
REQUEST_REJECTED_STATUSES = frozenset({400, 413, 422})


class LLMResponseError(Exception):
    """The answerer answered HTTP 2xx, but the body is not a JSON object.

    Not a transport failure: the server is up and produced a sample, just a bad one. The loop
    (`slot_adapter.py:1553`) counts it as a structural failure and spends the next attempt on it,
    so Ceres does too. `raw` keeps the exact bytes for the run record.
    """

    def __init__(self, message: str, raw: bytes) -> None:
        super().__init__(message)
        self.raw = raw


class LLM(Protocol):
    """One chat call. The raw bytes come back with the parsed JSON so the run record can keep
    exactly what the server sent, before anything judges it."""

    def chat(self, request_body: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
        """Post one chat completion and return `(envelope, raw_bytes)`.

        Raise `LLMTransportError` if the server did not answer, and `LLMResponseError` if it
        answered with something that is not a JSON object.
        """

    def health(self) -> tuple[str, str, str | None]:
        """`(status, detail, model_reported)`; status is "ready", "wrong_model" or "down"."""


# ---------------------------------------------------------------- chunk context (optional per-chunk context)
# The store's optional `items.context` (the DDS chunk-context row, `dds-chunk-context-v1`) rendered as
# ONE line of metadata for the answerer. Behind `answer.context_lines` (default off). The field names
# are the ROWS' names (`loop-v0/04_rag/context/v35/.../chunk-context.jsonl`, rule ctx-v35.2), not
# the context request's (`issued_date`, `place_coverage`, …): a name this function does not
# read renders nothing, it is never mapped by guess.

SOURCE_METADATA_LABEL = "SOURCE METADATA:"
# English month names, fixed here rather than taken from the locale: the line is metadata the
# answerer reads, not user text, and a locale must not change the prompt bytes.
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December")
_MONTHS_SHORT = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
# `doc_type` is a controlled value of the context rule, not a {value, status} field. Only a type
# listed here names its source; any other type (or none) gives no "Source:" part.
CONTEXT_SOURCE_LABELS = {"mars_series": "JRC MARS Bulletin"}
_ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_ISO_MONTH = re.compile(r"^(\d{4})-(\d{2})$")


def _found(context: dict[str, Any], key: str) -> Any:
    """The value of one {value, status, basis} field when its status is exactly "found", else None."""
    entry = context.get(key)
    if isinstance(entry, dict) and entry.get("status") == "found":
        return entry.get("value")
    return None


def _iso_date(value: Any) -> date | None:
    """`YYYY-MM-DD` as a real calendar date; anything else is None (never repaired)."""
    match = _ISO_DATE.match(value) if isinstance(value, str) else None
    if match is None:
        return None
    try:
        return date(int(match[1]), int(match[2]), int(match[3]))
    except ValueError:
        return None


def _issued(value: Any) -> str | None:
    """A found issue date as text: `YYYY-MM-DD` → "22 July 2024"; `YYYY-MM` → "April 2017".

    Month precision is what the rule found for some older bulletins (76 of 6,527 rows of
    release-0008, e.g. "2017-04"); it is rendered at that precision, never given an invented day.
    """
    day = _iso_date(value)
    if day is not None:
        return f"{day.day} {_MONTHS[day.month - 1]} {day.year}"
    match = _ISO_MONTH.match(value) if isinstance(value, str) else None
    if match is not None and 1 <= int(match[2]) <= 12:
        return f"{_MONTHS[int(match[2]) - 1]} {match[1]}"
    return None


def _one_line(value: Any) -> str | None:
    """A non-empty string with its whitespace collapsed (the line must stay one line), else None."""
    text = " ".join(value.split()) if isinstance(value, str) else ""
    return text or None


def _period(start: date, end: date) -> str:
    day = lambda d: f"{d.day} {_MONTHS_SHORT[d.month - 1]}"          # noqa: E731
    if start == end:
        return f"{day(start)} {start.year}"
    if start.year == end.year:
        return f"{day(start)}\u2013{day(end)} {end.year}"
    return f"{day(start)} {start.year}\u2013{day(end)} {end.year}"


def context_line(context: Any) -> str | None:
    """One compact metadata line from a chunk-context value, or None. Pure.

    Renders ONLY fields whose status is "found" and whose value is well formed:
      `issue_date` ("issued 22 July 2024", or "issued April 2017" at month precision), `reporting_period` {start, end} ("reporting period
      1 Jun–13 Jul 2024"), `place` ("place: Poland") and, for books, `heading_path`
      ("section: A › B"). `not_found`, `ambiguous`, `extraction_failed`, `not_applicable` and any
      malformed value are omitted, never guessed. The "Source:" part comes from `doc_type` via
      `CONTEXT_SOURCE_LABELS`; it does not count as a found field, so a context with nothing found
      gives None. `place.basis.coverage` is not rendered; `document_scope` renders only as the place fallback
    when no place was found, labelled with its scope .
    """
    if not isinstance(context, dict):
        return None
    facts: list[str] = []
    issued = _issued(_found(context, "issue_date"))
    if issued is not None:
        facts.append(f"issued {issued}")
    period = _found(context, "reporting_period")
    if isinstance(period, dict):
        start, end = _iso_date(period.get("start")), _iso_date(period.get("end"))
        if start is not None and end is not None and start <= end:
            facts.append(f"reporting period {_period(start, end)}")
    place = _one_line(_found(context, "place"))
    if place is not None:
        facts.append(f"place: {place}")
    else:
        # When no place was found, a FOUND
        # document_scope (a regional bulletin: Ukraine, Russia, Türkiye, …) may stand in, labelled with its scope
        # so it is never mistaken for a page-level place.
        scope_value = _one_line(_found(context, "document_scope"))
        if scope_value is not None:
            # A truthy non-dict basis raised AttributeError; malformed → default label.
            document_scope = context.get("document_scope")
            basis = document_scope.get("basis") if isinstance(document_scope, dict) else None
            scope = (basis.get("scope") if isinstance(basis, dict) else None) or "document"
            facts.append(f"place: {scope_value} ({_one_line(scope) or 'document'} scope)")
    heading = _found(context, "heading_path")
    if isinstance(heading, list) and heading:
        steps = [_one_line(step) for step in heading]
        if all(steps):
            facts.append("section: " + " \u203a ".join(steps))
    if not facts:
        return None
    label = CONTEXT_SOURCE_LABELS.get(context.get("doc_type")) if isinstance(context.get("doc_type"), str) else None
    return " \u00b7 ".join(([f"Source: {label}"] if label else []) + facts)


# ---------------------------------------------------------------- the r1 request body

def stable_seed(question_id: str, attempt: int) -> int:
    """The loop's seed (`slot_adapter.py:1377`), including its domain string.

    The literal `slice-answer-qwen38-v1` is part of the identity: change it and the same question
    draws different samples than the judged run did, so the eval hook stops being comparable.
    """
    digest = hashlib.sha256(f"slice-answer-qwen38-v1\0{question_id}\0{attempt}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def answer_user_prompt(packet: dict[str, Any],
                       context_lines: Mapping[str, str] | None = None, *,
                       memory: Any = None) -> str:
    """The user message of the grounded-answer call (`slot_adapter.py:1382`).

    The loop's `closed_book` branch is dropped: closed book is a loop ablation, and a service that
    can be asked to answer without evidence is a service that can be made to answer without
    evidence. Sections are joined by a blank line, exactly as the loop joins them, because the
    prompt hash is part of the route identity.

    `context_lines` (evidence_id → `context_line(...)`) is given only when
    `answer.context_lines` is on. A context with a line gets one extra section,
    `SOURCE METADATA: <line>`, between `[E#]` and `REFERENCE TEXT START`: outside the quoted
    passage, so the validator (which reads `evidence_text` only) can never accept a quote of it.
    None or an empty mapping yields exactly the r1 message, byte for byte.
    """
    lines = context_lines or {}
    if memory is not None and getattr(memory, "active", False):
        # Memory v0: BACKGROUND (persona + earlier turns, NOT evidence) before the evidence, and the resolved question.
        # `memory is None` (every stateless request) leaves this function's output byte-identical to r1.
        sections = [f"Required answer language: {packet['language']}"]
        if memory.background:
            sections.append(memory.background)
        sections.append(f"Question as asked: {packet['question']}")
        sections.append(f"Question: {memory.standalone_question}")
    else:
        sections = [f"Required answer language: {packet['language']}", f"Question: {packet['question']}"]
    sections.append("PRIVATE REFERENCE MATERIAL START")
    for context in packet["contexts"]:
        sections.append(f"[{context['evidence_id']}]")
        line = lines.get(context["evidence_id"])
        if line:
            sections.append(f"{SOURCE_METADATA_LABEL} {line}")
        sections.extend(["REFERENCE TEXT START", context["evidence_text"], "REFERENCE TEXT END"])
    sections.append("PRIVATE REFERENCE MATERIAL END")
    sections.append("Return only the required JSON object.")
    return "\n\n".join(sections)


def build_request_body(packet: dict[str, Any], profile: dict[str, Any], prompt: str,
                       attempt: int, context_lines: Mapping[str, str] | None = None, *,
                       memory: Any = None, memory_rules: str | None = None) -> dict[str, Any]:
    """The exact chat body of route r1 (the loop's `slot_adapter` and the DDS CLI's port of it).

    `context_lines` only changes the user message (`answer_user_prompt`); None is the r1 body."""
    decoding = {**DEFAULT_DECODING, **(profile.get("decoding") or {})}
    return {
        "model": profile["model_alias"],
        "messages": [{"role": "system", "content": (prompt + "\n\n" + memory_rules
                                                    if memory is not None and getattr(memory, "active", False) and memory_rules
                                                    else prompt)},
                     {"role": "user", "content": answer_user_prompt(packet, context_lines, memory=memory)}],
        "temperature": decoding["temperature"],
        "top_p": decoding["top_p"],
        "top_k": decoding["top_k"],
        "presence_penalty": decoding["presence_penalty"],
        "max_tokens": decoding["max_tokens"],
        "seed": stable_seed(packet["question_id"], attempt),
        "stream": False,
        # Qwen3.x templates think by default and exhaust max_tokens before emitting the JSON; the
        # profile pins `enable_thinking: false` and it must reach the server as a template kwarg,
        # not as a decoding field. (slot_adapter.py:1466)
        "chat_template_kwargs": {"enable_thinking": bool(decoding.get("enable_thinking", False))},
    }


# ---------------------------------------------------------------- the real client

class OpenAIChatLLM:
    """httpx client for an OpenAI-compatible `/chat/completions`.

    NOT exercised by the unit tests: no model server runs in the test environment. `lifecycle:
    external` means this class never starts or stops the endpoint — on a development machine it is the DDS
    loop's Qwen3.8 on :8215, and starting a second copy would take the GPU from the loop.
    """

    def __init__(self, base_url: str, *, timeout_s: float = 300.0, expect_model: str | None = None,
                 health_url: str | None = None, transport: Any = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.expect_model = expect_model
        self.health_url = health_url
        # An `httpx` transport, for tests only (`httpx.MockTransport`): the failure classes below
        # are testable without a model server. None = a real network transport.
        self._transport = transport
        self._client = None

    def _http(self):
        import httpx

        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s, transport=self._transport)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def chat(self, request_body: dict[str, Any], *, timeout_s: float | None = None) -> tuple[dict[str, Any], bytes]:
        import httpx

        url = f"{self.base_url}/chat/completions"
        # Two failure classes, kept apart because they mean different things to the attempt loop.
        # 1. HTTP/network: the server did not answer → transport failure, stop.
        try:
            response = self._http().post(url, json=request_body, headers={"Content-Type": "application/json"},
                                         **({"timeout": timeout_s} if timeout_s is not None else {}))
            response.raise_for_status()
            raw = response.content
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in REQUEST_REJECTED_STATUSES:
                raise LLMRequestRejected(f"HTTP {status}: {exc}", status) from None
            raise LLMTransportError(f"{exc.__class__.__name__}: {exc}") from None
        except httpx.HTTPError as exc:
            # Full detail for the private record; `pipeline.answer` shows the client a code only.
            raise LLMTransportError(f"{exc.__class__.__name__}: {exc}") from None
        # 2. A 2xx whose body is not a JSON object: a bad sample → structural failure, retry.
        try:
            envelope = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMResponseError(f"the answerer returned non-JSON: {exc.__class__.__name__}",
                                   raw) from None
        if not isinstance(envelope, dict):
            raise LLMResponseError(
                f"the answerer returned JSON {type(envelope).__name__}, not an object", raw)
        return envelope, raw

    def health(self) -> tuple[str, str, str | None]:
        """Readiness = reachable AND serving `expect_model` (the readiness contract); when `health_url`
        is configured it must also answer with a 2xx status.

        `GET /v1/models` is a label, not a weight proof: the id is what the server was told to
        call itself. The deployment record is what proves the weights.
        """
        import httpx

        try:
            response = self._http().get(f"{self.base_url}/models", timeout=min(self.timeout_s, 10.0))
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return "down", exc.__class__.__name__, None
        ids = [str(row.get("id")) for row in (payload.get("data") or []) if isinstance(row, dict)]
        reported = ids[0] if ids else None
        if self.expect_model and self.expect_model not in ids:
            return "wrong_model", f"expected {self.expect_model}, found {ids or 'nothing'}", reported
        if self.health_url:
            try:
                probe = self._http().get(self.health_url, timeout=min(self.timeout_s, 10.0))
            except httpx.HTTPError as exc:
                return "down", f"health_url: {exc.__class__.__name__}", reported
            if not 200 <= probe.status_code < 300:
                return "down", f"health_url answered HTTP {probe.status_code}", reported
        return "ready", f"serving {reported}", reported

    def server_props(self) -> dict[str, Any]:
        """The answer server's LIVE settings from llama.cpp's `GET /props` (an earlier defect).

        The declared start command says what the server was told; `/props` says what it runs. A
        server started with an 8192 context while the profile declared 16384 was exactly that
        gap, so the context size is read here.
        Never raises: an unreachable server, a non-2xx status (a server that has no `/props`), a
        non-JSON body or an unknown shape all come back as `{"available": False, "reason": ...}`,
        because a missing snapshot must not stop an answer that would otherwise be served.
        """
        fetched_at = _utc_now()
        url = f"{server_root(self.base_url)}/props"
        try:
            response = self._http().get(url, timeout=min(self.timeout_s, 10.0))
        except Exception as exc:                              # noqa: BLE001 — a probe never raises
            return props_unavailable(f"GET /props failed: {exc.__class__.__name__}", fetched_at)
        if not 200 <= response.status_code < 300:
            return props_unavailable(f"GET /props answered HTTP {response.status_code}", fetched_at)
        try:
            payload = response.json()
        except ValueError:
            return props_unavailable("GET /props returned a non-JSON body", fetched_at)
        return parse_server_props(payload, fetched_at)


# ---------------------------------------------------------------- live server settings

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def server_root(base_url: str) -> str:
    """The server root of an OpenAI-compatible base URL: `http://h:8215/v1` → `http://h:8215`."""
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


def props_unavailable(reason: str, fetched_at: str | None = None) -> dict[str, Any]:
    return {"available": False, "reason": reason, "fetched_at": fetched_at or _utc_now()}


def _positive_int(value: Any) -> int | None:
    """An int > 0, and not a bool (`True` is an int in Python, and n_ctx 1 is not a context)."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def parse_server_props(payload: Any, fetched_at: str | None = None) -> dict[str, Any]:
    """`{available, n_ctx, model_path?, model_alias?, build_info?, fetched_at}` from a `/props` body.

    `n_ctx` is `default_generation_settings.n_ctx`, else a top-level `n_ctx` (the task's order).
    Note: llama.cpp reports the PER-SLOT context in `default_generation_settings`; with
    `--parallel N` it is the total divided by N. The loop's answerer runs one slot.
    A body without a usable n_ctx is an unknown shape: `available: False`, never a guess.
    """
    fetched_at = fetched_at or _utc_now()
    if not isinstance(payload, dict):
        return props_unavailable(f"/props is JSON {type(payload).__name__}, not an object", fetched_at)
    settings = payload.get("default_generation_settings")
    n_ctx = _positive_int(settings.get("n_ctx")) if isinstance(settings, dict) else None
    if n_ctx is None:
        n_ctx = _positive_int(payload.get("n_ctx"))
    if n_ctx is None:
        return props_unavailable("/props has no positive integer n_ctx (default_generation_settings"
                                 " or top level): unknown server shape", fetched_at)
    props: dict[str, Any] = {"available": True, "n_ctx": n_ctx}
    for key in ("model_path", "model_alias", "build_info"):
        if isinstance(payload.get(key), str):
            props[key] = payload[key]
    props["fetched_at"] = fetched_at
    return props


def live_server_props(llm: Any) -> dict[str, Any]:
    """`llm.server_props()` when the client has it, else unavailable. Never raises."""
    getter = getattr(llm, "server_props", None)
    if not callable(getter):
        return props_unavailable(f"{type(llm).__name__} reports no server props")
    try:
        value = getter()
    except Exception as exc:                                  # noqa: BLE001 — a probe never raises
        return props_unavailable(f"server_props failed: {exc.__class__.__name__}")
    if not isinstance(value, dict):
        return props_unavailable(f"server_props returned {type(value).__name__}, not a dict")
    return value


def context_check(props: dict[str, Any] | None, profile: dict[str, Any] | None) -> dict[str, Any]:
    """The live n_ctx against the answer profile's `hardware_flags.context_tokens`.

    `live_n_ctx_matches_profile` is True/False only when BOTH numbers are known, else None. A
    False is a different route: the callers refuse on it.
    """
    flags = (profile or {}).get("hardware_flags") or {}
    declared = _positive_int(flags.get("context_tokens")) if isinstance(flags, dict) else None
    live = _positive_int((props or {}).get("n_ctx")) if (props or {}).get("available") else None
    return {"live_n_ctx": live, "profile_context_tokens": declared,
            "live_n_ctx_matches_profile": None if live is None or declared is None else live == declared}


# ---------------------------------------------------------------- the fake (tests, smoke)

_REFERENCE = re.compile(r"REFERENCE TEXT START\n\n(.*?)\n\nREFERENCE TEXT END", re.DOTALL)
_LANGUAGE = re.compile(r"^Required answer language: (\w{2})$", re.MULTILINE)
_QUESTION = re.compile(r"^Question: (.*)$", re.MULTILINE)


class FakeLLM:
    """A model-free answerer that returns a genuinely VALIDATED grounded answer.

    It reads the prompt it was given — the same prompt the real model reads — and quotes E1 back.
    That makes the fake answer pass the real validator (`pipeline.validate`) for real reasons:
    the claim is a substring of `user_answer`, the support quote is a continuous quote of E1, the
    language matches the packet. A canned golden answer (the fixture's `golden/answers.jsonl`) wins when
    the question is one of the fixture's, so the smoke path uses what the loop actually produced.

    Test switches: `failures` makes the first N attempts return a structurally invalid candidate
    (to exercise the two-attempt loop); `transport_error` makes every call raise.
    """

    MODEL_ID = "fake-answerer"

    def __init__(self, goldens_by_question: dict[str, dict[str, Any]] | None = None, *,
                 failures: int = 0, transport_error: bool = False, quote_words: int = 12) -> None:
        self.goldens = dict(goldens_by_question or {})
        self.failures = int(failures)
        self.transport_error = bool(transport_error)
        self.quote_words = int(quote_words)
        self.calls: list[dict[str, Any]] = []

    def chat(self, request_body: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
        self.calls.append(request_body)
        if self.transport_error:
            raise LLMTransportError("FakeLLM: the answerer is configured to be unreachable")
        user = request_body["messages"][1]["content"]
        if len(self.calls) <= self.failures:
            candidate: dict[str, Any] = {"decision": "ANSWER"}   # missing keys → STRUCTURE_FAILED
        else:
            candidate = self.goldens.get(self._question(user)) or self._compose(user)
        content = json.dumps(candidate, ensure_ascii=False)
        payload = {
            "id": "fake-completion", "object": "chat.completion", "model": self.MODEL_ID,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
        }
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return payload, raw

    def health(self) -> tuple[str, str, str | None]:
        return "ready", "FakeLLM: no model server", self.MODEL_ID

    # -------------------------------------------------------- composition

    @staticmethod
    def _question(user_prompt: str) -> str:
        match = _QUESTION.search(user_prompt)
        return match.group(1).strip() if match else ""

    def _compose(self, user_prompt: str) -> dict[str, Any]:
        language = (_LANGUAGE.search(user_prompt) or [None, "en"])[1]
        references = _REFERENCE.findall(user_prompt)
        if not references:
            # No evidence reached the model: the honest grounded answer is UNSUPPORTED with no
            # citation, which is what the prompt contract demands (grounded_answer_v1.md).
            return {
                "decision": "UNSUPPORTED", "answer_language": language,
                "user_answer": "The private reference material does not cover this question.",
                "citations": [], "missing_information": ["relevant reference passages"],
                "confidence": "LOW", "reason_code": "NO_RELEVANT_SUPPORT",
            }
        quote = " ".join(references[0].split()[: self.quote_words])
        answer = f"Reference E1 states: {quote}"
        return {
            "decision": "ANSWER", "answer_language": language, "user_answer": answer,
            "citations": [{"claim": quote, "evidence_id": "E1", "support_quote": quote}],
            "missing_information": [], "confidence": "MEDIUM", "reason_code": "SUPPORTED",
        }


def goldens_from_fixture(packets: Sequence[dict[str, Any]],
                         answers: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Key the fixture's golden candidates by question text, which is what `FakeLLM` can see in the prompt."""
    by_id = {row["question_id"]: row.get("candidate") for row in answers
             if row.get("state") == "VALIDATED" and row.get("candidate")}
    return {packet["question"].strip(): by_id[packet["question_id"]]
            for packet in packets if packet["question_id"] in by_id}
