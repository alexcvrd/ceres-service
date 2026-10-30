"""Memory v0: static personas + conversation memory.

The rule this module exists to keep: memory is CONTEXT, never evidence. Personas and earlier turns reach the answerer only inside a
labelled BACKGROUND block in the user message; the packet (what the validator checks quotes against) never contains them.

A request with neither `conversation_id` nor `profile_id` never touches this module: the stateless path is the r1 route byte for byte.
"""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

PROMPTS = Path(__file__).resolve().parent / "memory_prompts"
REWRITE_PROMPT = PROMPTS / "rewrite_v0.md"
ANSWER_RULES_PROMPT = PROMPTS / "answer_memory_v0.md"          # with any memory (history and/or persona)
PERSONA_RULES_PROMPT = PROMPTS / "answer_persona_v0.md"        # added only when a persona is resolved
SPLIT_PROMPT = PROMPTS / "split_v2.md"   # ADVICE vs REPORT; v1 (first-person) was too narrow, v0 split report questions (checked for regressions)
SITUATIONAL_RULES_PROMPT = PROMPTS / "answer_situational_v0.md"
PROFILE_FIELDS = frozenset({"region", "crops", "soil", "irrigation", "role", "farm"})
PERSONA_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
BACKGROUND_HEADER = ("BACKGROUND — about the user and the conversation. NOT EVIDENCE: never cite it, never state a fact or a "
                     "number from it as true; only the reference passages are evidence.")
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


class MemoryError(Exception):
    """A memory request the service refuses (unknown persona, memory disabled): HTTP 422."""


# ---------------------------------------------------------------- personas

@dataclass(frozen=True)
class Persona:
    id: str
    summary: str
    role: str
    language: str | None
    data: dict[str, Any]


def load_personas(directory: Path) -> dict[str, Persona]:
    """Every `<id>.yaml` in `directory`; the file name must equal its `id`. Bad files stop startup."""
    out: dict[str, Persona] = {}
    if not directory.is_dir():
        raise MemoryError(f"memory.personas_dir is not a directory: {directory}")
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        pid = data.get("id")
        if pid != path.stem or not isinstance(pid, str) or not PERSONA_ID_RE.match(pid):
            raise MemoryError(f"persona file {path.name}: id must equal the file name and match {PERSONA_ID_RE.pattern}")
        summary = " ".join(str(data.get("summary") or "").split())
        if not summary:
            raise MemoryError(f"persona {pid}: summary is empty")
        out[pid] = Persona(id=pid, summary=summary, role=str(data.get("role") or "user"),
                           language=data.get("language"), data=data)
    return out


# ---------------------------------------------------------------- conversations

@dataclass
class Turn:
    question: str
    standalone_question: str
    lang: str
    profile_id: str | None
    decision: str | None
    answer_gist: str
    cited: list[str]
    execution_id: str
    at: float


@dataclass
class _Conversation:
    turns: list[Turn] = field(default_factory=list)
    touched: float = 0.0
    count: int = 0                         # turns ever appended (monotonic; survives the max_turns window)
    profile_id: str | None = None          # bound at the first turn
    pins: int = 0                          # turns in progress: a pinned conversation is never evicted or expired
    replies: dict[str, Any] = field(default_factory=dict)   # request_id → completed response (replay, bounded)


@dataclass
class Snapshot:
    """What one turn sees, taken under the store lock at the start of the turn."""
    conversation_id: str
    history: list[Turn]
    count: int
    replay: Any = None                     # the completed response for this request_id, when it ran before


class ConversationStore:
    """v0: in the Ceres process, lost on restart. A turn is a transaction: `begin` (expire stale, check the persona binding,
    snapshot history, pin) → `commit` (append) → `end` (unpin, always). A pinned conversation is never evicted or expired, so a
    turn in progress cannot lose its history. At most `max_turns` turns per conversation, `max_conversations` conversations (the
    least recently used unpinned one is evicted), idle `ttl_s` → forgotten. A completed (conversation_id, request_id) is replayed,
    not re-run (a client retry after a timeout must not store the turn twice)."""

    REPLIES_PER_CONVERSATION = 20

    def __init__(self, max_turns: int = 20, ttl_s: float = 7200.0, clock: Any = time.monotonic,
                 max_conversations: int = 1000) -> None:
        if max_turns < 1 or max_conversations < 1 or ttl_s <= 0:
            raise MemoryError("memory store limits must be positive (max_turns, max_conversations, ttl_s)")
        self.max_turns, self.ttl_s, self.clock = int(max_turns), float(ttl_s), clock
        self.max_conversations = int(max_conversations)
        self._data: dict[str, _Conversation] = {}
        self._lock = threading.Lock()

    def _expire(self, now: float) -> None:
        for cid in [c for c, conv in self._data.items() if conv.pins == 0 and now - conv.touched > self.ttl_s]:
            del self._data[cid]

    def _make_room(self) -> None:
        while len(self._data) >= self.max_conversations:
            idle = [c for c, conv in self._data.items() if conv.pins == 0]
            if not idle:
                return                                  # every conversation is mid-turn: allow a temporary overshoot
            del self._data[min(idle, key=lambda c: self._data[c].touched)]

    def begin(self, conversation_id: str, profile_id: str | None, request_id: str | None = None) -> Snapshot:
        with self._lock:
            now = self.clock()
            self._expire(now)
            conv = self._data.get(conversation_id)
            if conv is not None and conv.count and conv.profile_id != profile_id:
                raise MemoryError(f"conversation {conversation_id} is bound to profile_id {conv.profile_id!r}; "
                                  "start a new conversation to change it")
            if conv is not None and request_id and request_id in conv.replies:
                return Snapshot(conversation_id, list(conv.turns), conv.count, replay=conv.replies[request_id])
            if conv is None:
                self._make_room()
                conv = self._data[conversation_id] = _Conversation(profile_id=profile_id, touched=now)
            conv.pins += 1
            return Snapshot(conversation_id, list(conv.turns), conv.count)

    def commit(self, snapshot: Snapshot, turn: Turn, request_id: str | None = None, reply: Any = None) -> int:
        """Appends the turn; returns its 1-based number (monotonic). Remembers the reply under request_id for replay."""
        with self._lock:
            conv = self._data[snapshot.conversation_id]          # pinned since begin: cannot have been evicted
            conv.turns.append(turn)
            del conv.turns[:-self.max_turns]
            conv.count += 1
            conv.touched = self.clock()
            if request_id and reply is not None:
                conv.replies[request_id] = reply
                while len(conv.replies) > self.REPLIES_PER_CONVERSATION:
                    conv.replies.pop(next(iter(conv.replies)))
            return conv.count

    def remember(self, snapshot: Snapshot, request_id: str | None, reply: Any) -> None:
        """Keeps the completed response under request_id, so a retry of the same request is replayed, not re-run."""
        if not request_id:
            return
        with self._lock:
            conv = self._data.get(snapshot.conversation_id)
            if conv is not None:
                conv.replies[request_id] = reply
                while len(conv.replies) > self.REPLIES_PER_CONVERSATION:
                    conv.replies.pop(next(iter(conv.replies)))

    def end(self, snapshot: Snapshot) -> None:
        if snapshot.replay is not None:
            return                                              # a replay never pinned
        with self._lock:
            conv = self._data.get(snapshot.conversation_id)
            if conv is not None:
                conv.pins -= 1
                conv.touched = self.clock()
                if conv.count == 0 and conv.pins == 0:
                    del self._data[snapshot.conversation_id]    # a failed first turn leaves nothing behind

    # read-only helpers (tests, /v1/info)
    def get(self, conversation_id: str) -> list[Turn]:
        with self._lock:
            self._expire(self.clock())
            conv = self._data.get(conversation_id)
            return list(conv.turns) if conv else []

    def append(self, conversation_id: str, turn: Turn) -> int:
        """begin + commit + end in one call (no request id)."""
        snap = self.begin(conversation_id, turn.profile_id)
        try:
            return self.commit(snap, turn)
        finally:
            self.end(snap)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


# ---------------------------------------------------------------- runtime (one per service)

@dataclass
class MemoryRuntime:
    personas: dict[str, Persona]
    store: ConversationStore
    rewrite_prompt: str
    answer_rules: str
    persona_rules: str = ""
    split_prompt: str = ""
    situational_rules: str = ""
    history_turns: int = 3
    rewrite_max_tokens: int = 300
    rewrite_timeout_s: float = 45.0
    gist_chars: int = 220
    background_chars: int = 1800          # ≈ 450 tokens (persona ≤150 + turns ≤300)
    prompt_sha256: dict[str, str] = field(default_factory=dict)
    _locks: dict[str, list[Any]] = field(default_factory=dict)       # conversation_id → [lock, holders + waiters]
    _locks_guard: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def from_config(cls, section: Any, resolve: Any) -> "MemoryRuntime":
        rewrite = REWRITE_PROMPT.read_bytes()
        rules = ANSWER_RULES_PROMPT.read_bytes()
        persona_rules = PERSONA_RULES_PROMPT.read_bytes()
        split, situational = SPLIT_PROMPT.read_bytes(), SITUATIONAL_RULES_PROMPT.read_bytes()
        return cls(personas=load_personas(resolve(section.personas_dir)),
                   store=ConversationStore(section.max_turns, section.ttl_s, max_conversations=section.max_conversations),
                   rewrite_prompt=rewrite.decode("utf-8"), answer_rules=rules.decode("utf-8"),
                   persona_rules=persona_rules.decode("utf-8"),
                   split_prompt=split.decode("utf-8"), situational_rules=situational.decode("utf-8"),
                   history_turns=section.history_turns, rewrite_max_tokens=section.rewrite_max_tokens,
                   rewrite_timeout_s=section.rewrite_timeout_s,
                   prompt_sha256={"rewrite": hashlib.sha256(rewrite).hexdigest(),
                                  "answer_rules": hashlib.sha256(rules).hexdigest(),
                                  "persona_rules": hashlib.sha256(persona_rules).hexdigest(),
                                  "split": hashlib.sha256(split).hexdigest(),
                                  "situational_rules": hashlib.sha256(situational).hexdigest()})

    @contextlib.contextmanager
    def conversation_lock(self, conversation_id: str | None):
        """One turn of a conversation at a time: read → rewrite → answer → append is atomic per conversation."""
        if not conversation_id:
            yield
            return
        with self._locks_guard:                    # reference-counted: the entry lives while someone holds or waits
            entry = self._locks.setdefault(conversation_id, [threading.Lock(), 0])
            entry[1] += 1
        try:
            with entry[0]:
                yield
        finally:
            with self._locks_guard:
                entry[1] -= 1
                if entry[1] == 0 and self._locks.get(conversation_id) is entry:
                    del self._locks[conversation_id]

    def rules_for(self, ctx: "MemoryContext") -> str:
        """The system-prompt addition: history rules always; the general-guidance rule only with a resolved persona.
        A situational-only request (no persona, no history) gets the situational rules alone."""
        if ctx.situational and ctx.situational.get("applied") and not (ctx.persona or ctx.history):
            return self.situational_rules
        return self.answer_rules + ("\n" + self.persona_rules if ctx.persona is not None and self.persona_rules else "")

    def persona(self, profile_id: str | None) -> Persona | None:
        if profile_id is None:
            return None
        if profile_id not in self.personas:
            raise MemoryError(f"unknown profile_id {profile_id!r}; known: {sorted(self.personas)}")
        return self.personas[profile_id]


# ---------------------------------------------------------------- the per-request memory context

@dataclass
class MemoryContext:
    conversation_id: str | None
    persona: Persona | None
    history: list[Turn]
    standalone_question: str
    retrieval_query: str
    is_followup: bool = False
    used_profile_fields: list[str] = field(default_factory=list)
    rewrite_status: str = "skipped"       # skipped | ok | fallback
    rewrite_ms: float = 0.0
    rewrite_event: dict[str, Any] | None = None
    background: str = ""
    snapshot: Any = None                  # the store transaction of this turn (None without a conversation)
    replay: Any = None                    # a completed response for the same (conversation_id, request_id)

    @property
    def unresolved_followup(self) -> bool:
        """The rewrite failed while there is history: the follow-up cannot be resolved, so Ceres asks the user to restate it
        instead of letting the answerer reinterpret the history on its own."""
        return self.rewrite_status == "fallback" and bool(self.history)

    situational: dict[str, Any] | None = None   # {core_question, qualifiers, status, ms} when the split ran

    @property
    def active(self) -> bool:
        """Whether the answer call gets the memory rules and the BACKGROUND block (or the situational rules)."""
        return bool(self.persona or self.history or (self.situational and self.situational.get("applied")))


def recent(history: Sequence[Turn], n: int) -> list[Turn]:
    return list(history[-n:]) if n > 0 else []


def _gist(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return (cut[: cut.rfind(" ")] if " " in cut else cut) + " …"


def build_background(persona: Persona | None, turns: Sequence[Turn], limit_chars: int, gist_chars: int) -> str:
    """The BACKGROUND block (persona + recent turns). Over budget: drop the oldest turn first, then shorten the persona."""
    def render(p_text: str | None, ts: Sequence[Turn]) -> str:
        lines = [BACKGROUND_HEADER]
        if p_text:
            lines.append(f"About the user: {p_text}")
        if ts:
            lines.append("Earlier in this conversation:")
            for i, t in enumerate(ts, 1):
                outcome = "answered" if t.decision == "ANSWER" else (t.decision or "no answer").lower()
                lines.append(f"{i}. User asked: {t.question}  → Ceres ({outcome}): {_gist(t.answer_gist, gist_chars)}")
        return "\n".join(lines)

    p_text = persona.summary if persona else None
    ts = list(turns)
    text = render(p_text, ts)
    while len(text) > limit_chars and ts:
        ts = ts[1:]
        text = render(p_text, ts)
    if len(text) > limit_chars and p_text:
        text = render(_gist(p_text, max(60, limit_chars - len(BACKGROUND_HEADER) - 40)), ts)
    return text if (p_text or ts) else ""


# ---------------------------------------------------------------- rewrite (the seed and rewrite rules)

def rewrite_seed(conversation_id: str | None, turn_number: int, question: str) -> int:
    digest = hashlib.sha256(f"ceres-memory-rewrite-v0\0{conversation_id or '-'}\0{turn_number}\0{question}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def rewrite_request_body(runtime: MemoryRuntime, profile: Mapping[str, Any], question: str, lang: str,
                         persona: Persona | None, history: Sequence[Turn], seed: int) -> dict[str, Any]:
    lines = [f"Question language: {lang}"]
    if persona:
        lines.append(f"About the user: {persona.summary}")
    if history:
        lines.append("Earlier turns (oldest first):")
        for i, t in enumerate(history, 1):
            lines.append(f"{i}. {t.question}  → {_gist(t.answer_gist, runtime.gist_chars)}")
    lines.append(f"New question: {question}")
    lines.append("Return only the required JSON object.")
    return {
        "model": profile["model_alias"],
        "messages": [{"role": "system", "content": runtime.rewrite_prompt}, {"role": "user", "content": "\n".join(lines)}],
        "temperature": 0.0, "top_p": 1.0, "max_tokens": runtime.rewrite_max_tokens, "seed": seed, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def parse_rewrite(content: str) -> dict[str, Any]:
    """The rewrite JSON, checked; raises ValueError on anything else (the caller falls back to the original question)."""
    match = _JSON_OBJECT.search(content or "")
    if not match:
        raise ValueError("no JSON object")
    data = json.loads(match.group(0))
    sq, rq = data.get("standalone_question"), data.get("retrieval_query")
    if not isinstance(sq, str) or not sq.strip() or len(sq) > 600:
        raise ValueError("standalone_question missing or too long")
    if not isinstance(rq, str) or not rq.strip() or len(rq) > 800:
        rq = sq
    followup = data.get("is_followup")
    if not isinstance(followup, bool):
        raise ValueError("is_followup must be true or false")
    used = data.get("used_profile_fields") or []
    if not isinstance(used, list) or any(u not in PROFILE_FIELDS for u in used):
        raise ValueError(f"used_profile_fields must be a list of {sorted(PROFILE_FIELDS)}")
    return {"standalone_question": " ".join(sq.split()), "retrieval_query": " ".join(rq.split()),
            "is_followup": followup, "used_profile_fields": list(dict.fromkeys(used))}


def prepare(runtime: MemoryRuntime, llm: Any, profile: Mapping[str, Any], *, question: str, lang: str,
            conversation_id: str | None, profile_id: str | None, request_id: str | None = None) -> MemoryContext:
    """Persona + history → stand-alone question, retrieval query and BACKGROUND block. Never raises on a model failure (falls back
    to the original question); raises MemoryError for an unknown persona."""
    persona = runtime.persona(profile_id)
    snapshot = runtime.store.begin(conversation_id, profile_id, request_id) if conversation_id else None
    if snapshot is not None and snapshot.replay is not None:
        return MemoryContext(conversation_id=conversation_id, persona=persona, history=[], standalone_question=question,
                             retrieval_query=question, snapshot=snapshot, replay=snapshot.replay)
    history = snapshot.history if snapshot else []
    window = recent(history, runtime.history_turns)
    ctx = MemoryContext(conversation_id=conversation_id, persona=persona, history=window,
                        standalone_question=question, retrieval_query=question, snapshot=snapshot)
    if not (persona or window):
        return ctx                                            # turn 1 without a persona: nothing to rewrite
    seed = rewrite_seed(conversation_id, (snapshot.count if snapshot else 0) + 1, question)
    body = rewrite_request_body(runtime, profile, question, lang, persona, window, seed)
    started = time.perf_counter()
    event: dict[str, Any] = {"request": body}
    from .llm import LLMTransportError
    try:
        timeout_kw = ({"timeout_s": runtime.rewrite_timeout_s}
                      if "timeout_s" in inspect.signature(llm.chat).parameters else {})
        response, _raw = llm.chat(body, **timeout_kw)
        content = response["choices"][0]["message"]["content"]
        if (response["choices"][0].get("finish_reason") == "length"):
            raise ValueError("rewrite truncated (max_tokens)")
        event["content"] = content
        parsed = parse_rewrite(content)
        ctx.standalone_question = parsed["standalone_question"]
        ctx.retrieval_query = parsed["retrieval_query"]
        ctx.is_followup = parsed["is_followup"]
        ctx.used_profile_fields = parsed["used_profile_fields"]
        ctx.rewrite_status = "ok"
    except LLMTransportError as exc:
        # The endpoint is down: calling it again for the answer only makes the client wait.
        event["error"] = f"{type(exc).__name__}: {exc}"
        ctx.rewrite_status = "transport_failed"
    except Exception as exc:   # noqa: BLE001 — any other rewrite failure falls back; the answer path continues
        event["error"] = f"{type(exc).__name__}: {exc}"
        ctx.rewrite_status = "fallback"
    ctx.rewrite_ms = round((time.perf_counter() - started) * 1000, 3)
    ctx.rewrite_event = event
    ctx.background = build_background(persona, window, runtime.background_chars, runtime.gist_chars)
    return ctx


def finish(runtime: MemoryRuntime, ctx: MemoryContext) -> None:
    """Ends the store transaction of this turn (always called, also on failure)."""
    if ctx.snapshot is not None:
        runtime.store.end(ctx.snapshot)


def clarify_candidate(lang: str) -> dict[str, Any]:
    """The fixed answer for a follow-up the rewrite could not resolve: ask the user to restate it (no model call)."""
    text = ("Nu am putut lega întrebarea de conversația anterioară. Vă rog să o formulați complet."
            if lang == "ro" else "I could not link this question to the earlier conversation. Please ask it in full.")
    return {"decision": "CLARIFY", "answer_language": lang, "user_answer": text, "citations": [],
            "missing_information": ["the question in full" if lang != "ro" else "întrebarea formulată complet"],
            "confidence": "LOW", "reason_code": "MISSING_USER_OR_PARCEL_INFORMATION"}


def record_turn(runtime: MemoryRuntime, ctx: MemoryContext, *, question: str, lang: str, profile_id: str | None,
                candidate: Mapping[str, Any] | None, execution_id: str, request_id: str | None = None,
                reply: Any = None) -> int | None:
    """Stores the turn when the request named a conversation; returns the turn number (None without a conversation)."""
    if not ctx.conversation_id or ctx.snapshot is None:
        return None
    cand = candidate or {}
    turn = Turn(question=question, standalone_question=ctx.standalone_question, lang=lang, profile_id=profile_id,
                decision=cand.get("decision"), answer_gist=str(cand.get("user_answer") or ""),
                cited=[c.get("evidence_id") for c in cand.get("citations") or [] if isinstance(c, Mapping)],
                execution_id=execution_id, at=time.time())
    return runtime.store.commit(ctx.snapshot, turn, request_id, reply)


# ---------------------------------------------------------------- situational-split experiment: situational split (stateless requests)

def parse_split(content: str) -> dict[str, Any]:
    match = _JSON_OBJECT.search(content or "")
    if not match:
        raise ValueError("no JSON object")
    data = json.loads(match.group(0))
    core, quals, has = data.get("core_question"), data.get("qualifiers"), data.get("has_situation")
    if not isinstance(core, str) or not core.strip() or len(core) > 600:
        raise ValueError("core_question missing or too long")
    if not isinstance(quals, list) or not all(isinstance(q, str) for q in quals) or not isinstance(has, bool):
        raise ValueError("qualifiers must be a list of strings and has_situation a boolean")
    return {"core_question": " ".join(core.split()), "qualifiers": [" ".join(q.split()) for q in quals][:8], "has_situation": has}


def split_situation(runtime: MemoryRuntime, llm: Any, profile: Mapping[str, Any], *, question: str, lang: str) -> MemoryContext:
    """A stateless question with the asker's situation in it: search with the core question, answer the user's words under the
    situational rules. No situation found, or any failure → the plain stateless path (ctx not active)."""
    from .llm import LLMTransportError
    ctx = MemoryContext(conversation_id=None, persona=None, history=[], standalone_question=question, retrieval_query=question)
    body = {"model": profile["model_alias"],
            "messages": [{"role": "system", "content": runtime.split_prompt},
                         {"role": "user", "content": f"Question language: {lang}\nQuestion: {question}\nReturn only the required JSON object."}],
            "temperature": 0.0, "top_p": 1.0, "max_tokens": runtime.rewrite_max_tokens,
            "seed": rewrite_seed("split", 1, question), "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    started = time.perf_counter()
    status, parsed = "fallback", None
    try:
        timeout_kw = {"timeout_s": runtime.rewrite_timeout_s} if "timeout_s" in inspect.signature(llm.chat).parameters else {}
        response, _raw = llm.chat(body, **timeout_kw)
        if response["choices"][0].get("finish_reason") == "length":
            raise ValueError("split truncated")
        parsed = parse_split(response["choices"][0]["message"]["content"])
        status = "ok"
    except LLMTransportError:
        status = "transport_failed"
    except Exception:   # noqa: BLE001 — a failed split falls back to the plain path
        status = "fallback"
    applied = bool(parsed and parsed["has_situation"] and parsed["qualifiers"])
    if applied:
        ctx.retrieval_query = parsed["core_question"]
    ctx.rewrite_status = "skipped"
    ctx.situational = {"applied": applied, "status": status, "core_question": parsed["core_question"] if parsed else None,
                       "qualifiers": parsed["qualifiers"] if parsed else [], "ms": round((time.perf_counter() - started) * 1000, 3)}
    return ctx
