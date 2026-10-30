"""`ceres-eval`: run a question bank through one route with Ceres's own pipeline, one phase at a time.

Phases embed, retrieve, rerank, answer, validate, manifest (or all of them with `chain`); each is resumable and pinned
to one bank and one release. A real run writes only into `--answers-dir`, otherwise `--scratch` is required.
"""

from __future__ import annotations

from collections import Counter

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import yaml

from . import __version__
from .config import ROUTE_SLOT_ORDER, compute_route_id
from .contracts import (
    ANSWER_SCHEMA,
    PACKET_SCHEMA,
    VALIDATOR_REVISION,
    schema_errors,
    validate_answer_row,
    validate_packet,
)
from .route_contract import RouteContractError, check_route_contract, rerank_text_choice
from .store import IDENTIFIER, OCR_ADMITTED_UNVERIFIED, Item, StoreError, read_predicate_sha256

EXIT_OK, EXIT_INVALID, EXIT_REFUSED, EXIT_TRANSPORT = 0, 1, 2, 75
PHASES = ("embed", "retrieve", "rerank", "answer", "validate", "manifest")
MANIFEST_SCHEMA = "ceres-eval-manifest-v1"
STATE_SCHEMA = "ceres-eval-state-v1"
SET_NAME_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"
# Stage-5 answer contract a `--set` MANIFEST asserts: prompt v1, validator r3, no context lines.
ASSERTED_PROMPT = {"name": "grounded_answer_v1.md",
                   "sha256": "7a37e08184ffd804fbc07da11296246ffe1479be74660cbb2be71a57ff7da0b4"}
ASSERTED_VALIDATOR_REVISION = "ceres-validate-r3"
# Keys a pin carries only when --schema or --release was given (a flagless pin is unchanged).
PIN_MODE_KEYS = ("schema", "pinned_explicitly")
QUESTION_ID_PATTERN = "^[0-9a-f]{64}$"
# sha256 of `loop-v0/00_control/schemas/answers.schema.json`, copied byte for byte into
# `ceres/schemas/answers.schema.json`; `tests/test_eval.py` pins the copy.
ANSWERS_SCHEMA_SHA256 = "c03023c081ca2db91fa69a7468f7d0330f61656c702cd5bb7e8ce62f26dfa354"


# ---------------------------------------------------------------- errors

class EvalError(Exception):
    exit_code = EXIT_REFUSED


class Refused(EvalError):
    """A guard, a pin or an input check says no. Nothing was written for the refused step."""


class Invalid(EvalError):
    exit_code = EXIT_INVALID


class TransportStopped(EvalError):
    """The answerer did not answer. The phase stopped; its state on disk is resumable."""

    exit_code = EXIT_TRANSPORT


# ---------------------------------------------------------------- small I/O helpers (loop forms)

def canonical_bytes(value: Any) -> bytes:
    """The loop's canonical JSON (`00_control/code/keys.py:18`)."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def item_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_line(value: Any) -> str:
    """One JSONL row as the loop writes it (`04_rag/common.py:json_bytes`, compact, sorted)."""
    return canonical_bytes(value).decode("utf-8") + "\n"


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    """Pretty JSON, sorted keys — the loop's raw-event form (`common.atomic_json`)."""
    body = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    atomic_write(path, body.encode("utf-8"))


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write(path, "".join(json_line(row) for row in rows).encode("utf-8"))


def append_row(path: Path, row: dict[str, Any]) -> None:
    """Append one phase row and fsync it: a row on disk is a row that is done."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json_line(row))
        handle.flush()
        os.fsync(handle.fileno())


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file; a torn LAST line (a crash mid-append) is dropped, any other bad line is
    corruption and refused."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if number == len(lines):
                break
            raise Refused(f"{path}: line {number} is not JSON — the phase file is corrupt")
        if not isinstance(row, dict):
            raise Refused(f"{path}: line {number} is not an object")
        rows.append(row)
    return rows


def keyed_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Phase rows by question_id; a later row for the same id supersedes an earlier one."""
    return {row["question_id"]: row for row in read_rows(path) if "question_id" in row}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def is_inside(path: Path, root: Path) -> bool:
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


# ---------------------------------------------------------------- route and bank

@dataclass
class Route:
    """A frozen route declaration, its seven profiles, and the recomputed id."""

    file: Path
    name: str
    route_id: str
    loop_root: Path
    profiles: dict[str, Path]
    profile_sha256: dict[str, str]
    file_sha256: str

    def load(self, slot: str) -> dict[str, Any]:
        return yaml.safe_load(self.profiles[slot].read_text(encoding="utf-8")) or {}

    @property
    def identity(self) -> bool:
        return self.load("rerank").get("operation") == "identity"

    @property
    def rerank_text(self) -> str:
        """The text the rerank profile scores (`payload` unless it declares `represented`)."""
        return rerank_text_choice(self.load("rerank"))

    @property
    def prompt_path(self) -> Path:
        template = self.load("answer").get("prompt_template")
        if not template:
            raise Refused(f"{self.profiles['answer']}: the answer profile names no prompt_template")
        path = Path(template)
        return path if path.is_absolute() else self.loop_root / path


def infer_loop_root(route_file: Path) -> Path:
    """`<loop>/04_rag/routes/<name>.yaml` → `<loop>`."""
    return route_file.resolve().parents[2]


def load_route(route_arg: str | Path, loop_root: Path | None = None) -> Route:
    """Read a route file (a path, or a name under `<loop_root>/04_rag/routes/`) and check its id.

    The id is recomputed from the seven profiles (`route_id_recipe`, `config.compute_route_id`);
    a declaration whose id does not match what its profiles hash to is refused — the eval would
    otherwise label one recipe's answers with another recipe's id.
    """
    path = Path(route_arg).expanduser()
    if not path.is_file():
        if loop_root is None:
            raise Refused(f"route file not found: {route_arg} (give a path, or --loop-root for a name)")
        path = Path(loop_root) / "04_rag" / "routes" / f"{route_arg}.yaml"
        if not path.is_file():
            raise Refused(f"route file not found: {path}")
    path = path.resolve()
    root = Path(loop_root).resolve() if loop_root else infer_loop_root(path)
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise Refused(f"{path}: not valid YAML ({exc.__class__.__name__})") from None
    if not isinstance(doc, dict) or not isinstance(doc.get("profiles"), dict):
        raise Refused(f"{path}: not a route declaration (no `profiles` mapping)")
    declared = str(doc.get("route_id") or "")
    profiles: dict[str, Path] = {}
    for slot in ROUTE_SLOT_ORDER:
        value = doc["profiles"].get(slot)
        if not value:
            raise Refused(f"{path}: route names no `{slot}` profile")
        profile = Path(value)
        profile = profile if profile.is_absolute() else root / profile
        if not profile.is_file():
            raise Refused(f"{path}: the {slot} profile is missing: {profile}")
        profiles[slot] = profile
    recomputed = compute_route_id(profiles)
    if recomputed != declared:
        raise Refused(f"{path}: route_id {declared[:12] or '<none>'}… does not match its seven profiles"
                      f" ({str(recomputed)[:12]}…) — refusing to label answers with the wrong route")
    # Route freeze, as the loop's `run_route.load_route` does it (after the id check):
    # one token floor, rerank_text agreement. `ceres-api` serves through this loader too.
    try:
        check_route_contract(doc, {slot: yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                                   for slot, p in profiles.items()})
    except (RouteContractError, yaml.YAMLError) as exc:
        raise Refused(f"{path}: {exc}") from None
    return Route(file=path, name=str(doc.get("route_name") or path.stem), route_id=declared,
                 loop_root=root, profiles=profiles,
                 profile_sha256={slot: sha256_file(p) for slot, p in profiles.items()},
                 file_sha256=sha256_file(path))


def load_bank(path: Path) -> tuple[list[dict[str, str]], str]:
    """The bank's rows `{question_id, language, text}` in file order, and the file's sha256.

    Ids must be unique 64-hex strings (the answers schema's pattern) and every row needs text and a
    two-letter language; the eval never mints or rewrites an id.
    """
    path = Path(path)
    if not path.is_file():
        raise Refused(f"bank not found: {path}")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    import re

    pattern = re.compile(QUESTION_ID_PATTERN)
    for number, raw in enumerate(read_rows(path), start=1):
        question_id, text, language = raw.get("question_id"), raw.get("text"), raw.get("language")
        if not isinstance(question_id, str) or not pattern.match(question_id):
            raise Refused(f"{path}: row {number}: question_id must be 64 lowercase hex characters")
        if question_id in seen:
            raise Refused(f"{path}: row {number}: duplicate question_id {question_id[:12]}…")
        if not isinstance(text, str) or not text.strip():
            raise Refused(f"{path}: row {number}: no question text")
        if not isinstance(language, str) or len(language) != 2:
            raise Refused(f"{path}: row {number}: language must be a two-letter code")
        seen.add(question_id)
        rows.append({"question_id": question_id, "language": language, "text": text})
    if not rows:
        raise Refused(f"{path}: the bank is empty")
    return rows, sha256_file(path)


# ---------------------------------------------------------------- the stack (store, models, answerer)

class Stack:
    """The three adapters, built lazily: the embed/rerank phases never open an answerer and the
    answer phase never loads a model. Tests pass instances; the CLI passes factories.

    `models_factory(phase)` builds the models ONE phase needs (the embedder for `embed`, the
    reranker for `rerank`), so the GPU window holds one model at a time; `release(phase)` closes
    what that phase loaded.
    """

    def __init__(self, *, store: Any = None, models: Any = None, llm: Any = None,
                 store_factory: Callable[[], Any] | None = None,
                 models_factory: Callable[[str], Any] | None = None,
                 llm_factory: Callable[[], Any] | None = None,
                 endpoint: dict[str, Any] | None = None) -> None:
        self._store, self._models, self._llm = store, models, llm
        self._store_factory, self._models_factory, self._llm_factory = (
            store_factory, models_factory, llm_factory)
        self._built_models: dict[str, Any] = {}
        self.endpoint = dict(endpoint or {})

    @property
    def store(self) -> Any:
        if self._store is None:
            if self._store_factory is None:
                raise Refused("no store is configured")
            self._store = self._store_factory()
        return self._store

    @property
    def llm(self) -> Any:
        if self._llm is None:
            if self._llm_factory is None:
                raise Refused("no answerer is configured")
            self._llm = self._llm_factory()
        return self._llm

    def models_for(self, phase: str) -> Any:
        if self._models is not None:
            return self._models
        if phase not in self._built_models:
            if self._models_factory is None:
                raise Refused("no models are configured")
            self._built_models[phase] = self._models_factory(phase)
        return self._built_models[phase]

    def release(self, phase: str) -> None:
        """Close the models a phase built (frees the GPU before the next model loads)."""
        models = self._built_models.pop(phase, None)
        for part in ("embedder", "reranker"):
            adapter = getattr(models, part, None)
            if adapter is not None and callable(getattr(adapter, "close", None)):
                adapter.close()


def describe_models(models: Any, part: str) -> dict[str, Any]:
    """What the models adapter says about its embedder or reranker (device, dtype, …)."""
    describe = getattr(models, "describe", None)
    if callable(describe):
        value = describe()
        if isinstance(value, dict) and isinstance(value.get(part), dict):
            return dict(value[part])
    health = getattr(models, "health", None)
    status = health().get("embedder" if part == "embedder" else "reranker") if callable(health) else None
    return {"adapter": type(models).__name__, "status": status, "device": None, "dtype": None}


def release_manifest(store: Any, release_attempt_id: str) -> dict[str, Any] | None:
    """The `releases.manifest` document of a release, when the store can read it.

    A store may offer `release_manifest(id)` itself; for `PgStore` the manifest is one read-only
    SELECT on the table it already reads. None when the store cannot say (fixture stores).
    """
    getter = getattr(store, "release_manifest", None)
    if callable(getter):
        return getter(release_attempt_id)
    pool, schema = getattr(store, "_pool", None), getattr(store, "db_schema", None)
    if pool is None or not schema:
        return None
    try:
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT manifest FROM {schema}.releases WHERE release_attempt_id = %s",
                        (release_attempt_id,))
            row = cur.fetchone()
    except Exception as exc:                                  # driver errors: name the class only
        raise StoreError(f"release manifest lookup failed: {exc.__class__.__name__}") from exc
    if row is None:
        return None
    value = row[0]
    return value if isinstance(value, dict) else json.loads(value or "{}")


# ---------------------------------------------------------------- the run context

@dataclass
class EvalContext:
    route: Route
    bank_path: Path
    bank: list[dict[str, str]]
    bank_sha256: str
    work_dir: Path
    config: Any
    stack: Stack
    out: Path | None = None
    scratch: bool = False
    answerer_cmd_file: Path | None = None
    notes_file: Path | None = None
    force: set[str] = field(default_factory=set)
    log: Callable[[str], None] = lambda message: print(message, file=sys.stderr)
    # `--release`: pin this release_attempt_id (active or not) instead of the active release.
    release_id: str | None = None
    # `--schema` as given (the config passed in already reads it; see `with_schema`). Recorded in the pin.
    schema: str | None = None
    # `--set NAME`: the question set's name; the answers folder becomes `<route>-<set>`.
    set_name: str | None = None
    # `--answers-dir`: the one folder inside the loop a non-scratch run may write its route folder into.
    answers_dir: Path | None = None
    # A release that does not name a chunk/represent/embed profile sha is
    # refused by default (unknown is not a match). Only fixture stores, which carry no profile shas, and an
    # operator's explicit `--allow-unknown-release-profiles` switch it off.
    allow_unknown_release_profiles: bool = False
    # Citation switch: the eval judges with `--citation-mode` (default strict = r3), NEVER with the served
    # config's `answer.citation_mode`, so a judged measurement cannot change mode because serving did.
    citation_mode: str = "strict"

    # -------------------------------------------------------- paths
    @property
    def state_path(self) -> Path:
        return self.work_dir / "RUN-STATE.json"

    @property
    def embed_dir(self) -> Path:
        return self.work_dir / "embed"

    @property
    def route_dir(self) -> Path:
        return self.work_dir / "routes" / self.route.name

    def phase_file(self, phase: str) -> Path:
        return self.embed_dir / "embed.jsonl" if phase == "embed" else self.route_dir / f"{phase}.jsonl"

    # -------------------------------------------------------- deps
    def deps(self, *, store: bool = False, models: str | None = None, llm: bool = False) -> Any:
        """A `pipeline.Deps` over the ROUTE's profiles and prompt (not the service config's)."""
        from .pipeline import Deps

        return Deps.from_files(self.config, self.stack.store if store else None,
                               self.stack.models_for(models) if models else None,
                               self.stack.llm if llm else None,
                               dict(self.route.profiles), self.route.prompt_path)

    def bank_ids(self) -> list[str]:
        return [row["question_id"] for row in self.bank]


# ---------------------------------------------------------------- pins: bank, release, route

def _read_state(ctx: EvalContext) -> dict[str, Any]:
    if ctx.state_path.is_file():
        return json.loads(ctx.state_path.read_text(encoding="utf-8"))
    return {"schema_version": STATE_SCHEMA, "created_at": utc_now(), "bank": None, "release": None,
            "routes": {}}


def pin_bank_and_route(ctx: EvalContext) -> dict[str, Any]:
    """Record the bank once and each route once; a different bank or a changed route is refused.

    With `--set` the set name is pinned beside the bank (key `set`, absent for a flagless run); a
    later phase with another set, or without one, is refused (one work dir, one question set)."""
    state = _read_state(ctx)
    bank = {"path": str(ctx.bank_path.resolve()), "sha256": ctx.bank_sha256, "rows": len(ctx.bank)}
    if state["bank"] is None:
        state["bank"] = bank
        if ctx.set_name is not None:
            state["set"] = ctx.set_name
    elif state.get("set") != ctx.set_name:
        raise Refused(f"this work dir is pinned to question set {state.get('set') or '<none>'};"
                      f" this run asks for {ctx.set_name or '<none>'} — use a new --work-dir per set")
    elif state["bank"]["sha256"] != ctx.bank_sha256:
        raise Refused(f"this work dir is pinned to bank {state['bank']['sha256'][:12]}…"
                      f" ({state['bank']['path']}); {ctx.bank_path} is {ctx.bank_sha256[:12]}…"
                      " — use a new --work-dir for another bank")
    pinned = state["routes"].get(ctx.route.name)
    if pinned is None:
        state["routes"][ctx.route.name] = {"route_id": ctx.route.route_id,
                                           "file": str(ctx.route.file),
                                           "profile_sha256": ctx.route.profile_sha256}
    elif pinned["route_id"] != ctx.route.route_id:
        raise Refused(f"route {ctx.route.name} is pinned here as {pinned['route_id'][:12]}…;"
                      f" {ctx.route.file} declares {ctx.route.route_id[:12]}…")
    atomic_json(ctx.state_path, state)
    return state


def with_schema(config: Any, schema: str) -> Any:
    """The config with `store.schema` replaced for one eval run (`--schema`); the same plain-identifier
    rule as `PgStore` (schema names are not quoted). The effective config (and its sha) says so."""
    if not IDENTIFIER.match(schema or ""):
        raise Refused(f"--schema is not a plain identifier: {schema!r}")
    return config.model_copy(update={"store": config.store.model_copy(update={"db_schema": schema})})


def _pin_mode(ctx: EvalContext, deps: Any) -> dict[str, Any]:
    """What a pin records about HOW it pinned: nothing without --schema/--release (the flagless state
    is unchanged), else the schema read and whether the release was named explicitly."""
    if ctx.release_id is None and ctx.schema is None:
        return {}
    return {"schema": deps.config.store.db_schema, "pinned_explicitly": ctx.release_id is not None}


def current_release(ctx: EvalContext, deps: Any) -> Any:
    """The release this run reads now: the store's active release, or with `--release` the named one
    (active or not). A named release the schema does not hold is refused; nothing is pinned."""
    if ctx.release_id is None:
        return deps.store.active_release()
    getter = getattr(deps.store, "release", None)
    if not callable(getter):
        raise Refused(f"--release needs a store that reads a named release ({type(deps.store).__name__} does not)")
    release = getter(ctx.release_id)
    if release is None:
        raise Refused(f"release {ctx.release_id} does not exist in schema {deps.config.store.db_schema}"
                      " — nothing was pinned or read")
    return release


def pin_release(ctx: EvalContext, deps: Any) -> dict[str, Any]:
    """The run's release, recorded once per work dir; any later phase that sees another release (another
    id, or the same id re-loaded with another manifest) is refused. Active by default; with `--release`
    the named release, recorded with its schema and `pinned_explicitly: true`."""
    state = _read_state(ctx)
    pinned = state.get("release")
    mode = _pin_mode(ctx, deps)
    if pinned is not None:
        pinned_mode = {key: pinned[key] for key in PIN_MODE_KEYS if key in pinned}
        if pinned_mode != mode:
            raise Refused(f"this work dir pinned its release with {pinned_mode or 'no --schema/--release'};"
                          f" this run asks for {mode or 'no --schema/--release'} — pass the same flags,"
                          " or start a new --work-dir")
        if ctx.release_id is not None and pinned["release_attempt_id"] != ctx.release_id:
            raise Refused(f"this work dir is pinned to release {pinned['release_attempt_id']};"
                          f" --release {ctx.release_id} names another — start a new --work-dir")
    # The identity comes from the store's own release lookup — (id, manifest sha) — not from a
    # derived Release (a store's `withheld` copy need not carry the manifest sha).
    release = current_release(ctx, deps)
    seen = {"release_attempt_id": release.release_attempt_id,
            "manifest_sha256": release.manifest_sha256}
    if pinned is None:
        manifest = release_manifest(deps.store, release.release_attempt_id) or {}
        state["release"] = {**seen, "product_manifest_sha256": manifest.get("product_manifest_sha256"),
                            "route_id": release.route_id, "row_count": release.row_count,
                            "membership_counts": release.membership_counts,
                            # The releases row's own *_profile_sha256 columns first (what the
                            # serving guard reads), the manifest's attempts only where the row has none.
                            "profile_sha256": {slot: ((release.profile_sha256 or {}).get(slot)
                                                      or ((manifest.get("attempts") or {}).get(slot) or {})
                                                      .get("profile_sha256"))
                                               for slot in ("chunk", "represent", "embed")},
                            **mode,
                            "pinned_at": utc_now()}
        atomic_json(ctx.state_path, state)
        return state["release"]
    if (pinned["release_attempt_id"], pinned["manifest_sha256"]) != (seen["release_attempt_id"],
                                                                        seen["manifest_sha256"]):
        which = "pinned" if ctx.release_id is not None else "active"
        raise Refused(f"the {which} release changed between phases: pinned"
                      f" {pinned['release_attempt_id']} ({pinned['manifest_sha256'][:12]}…), {which} now"
                      f" {seen['release_attempt_id']} ({seen['manifest_sha256'][:12]}…)"
                      " — answers must come from one release; start a new --work-dir")
    # A same-manifest replace can still change the row's profile
    # columns; compare every slot the row names now with what was pinned.
    changed = [slot for slot, sha in (release.profile_sha256 or {}).items()
               if sha and (pinned.get("profile_sha256") or {}).get(slot) not in (None, sha)]
    if changed:
        which = "pinned" if ctx.release_id is not None else "active"
        raise Refused(f"the {which} release's {', '.join(changed)} profile sha changed between phases under the same"
                      " id and manifest — answers must come from one release; start a new --work-dir")
    return pinned


def live_release(ctx: EvalContext, deps: Any) -> Any:
    """The Release the retrieve phase searches: `pipeline.pinned_release` (the active one) without
    `--release`; with it, the named release plus the same rights-withheld count."""
    from .pipeline import pinned_release

    if ctx.release_id is None:
        return pinned_release(deps)
    store_config = deps.config.store
    release = current_release(ctx, deps)
    counted = getattr(deps.store, "withheld", None)
    if counted is not None:
        release = counted(release, store_config.serve_rights, store_config.admitted_works)
    return release


def check_route_against_release(ctx: EvalContext, release: dict[str, Any]) -> None:
    """The store's chunk/represent/embed profiles must be the route's (matched by sha256, the way
    Ceres matches routes to releases). A release that does not name them is recorded as unknown."""
    named = {slot: sha for slot, sha in (release.get("profile_sha256") or {}).items() if sha}
    wrong = [slot for slot, sha in named.items() if sha != ctx.route.profile_sha256[slot]]
    if wrong:
        raise Refused(f"release {release['release_attempt_id']} was built with other {', '.join(wrong)}"
                      f" profile(s) than route {ctx.route.name} — its chunks are not this route's")
    unknown = [slot for slot in ("chunk", "represent", "embed") if slot not in named]
    if unknown and not ctx.allow_unknown_release_profiles:
        raise Refused(f"release {release['release_attempt_id']} does not name its {', '.join(unknown)} profile"
                      f" sha(s), so it cannot be shown to be route {ctx.route.name}'s store — unknown is not a"
                      " match (pass --allow-unknown-release-profiles only for a store known to lack them)")
    if unknown:
        ctx.log(f"warning: release {release['release_attempt_id']} names no {', '.join(unknown)} profile sha;"
                " allowed by --allow-unknown-release-profiles")


# ---------------------------------------------------------------- phase bookkeeping

def _record_phase(ctx: EvalContext, phase: str, started: str, **counts: Any) -> None:
    target = ctx.embed_dir / "phases.jsonl" if phase == "embed" else ctx.route_dir / "phases.jsonl"
    append_row(target, {"phase": phase, "route": ctx.route.name, "started_at": started,
                        "finished_at": utc_now(), **counts})


def _pending(ctx: EvalContext, phase: str, inputs: dict[str, str]) -> tuple[dict[str, dict], list[str]]:
    """(done rows, ids to compute). `--force-phase` drops the phase's rows first."""
    path = ctx.phase_file(phase)
    if phase in ctx.force and path.exists():
        path.rename(path.with_name(f"{path.name}.forced-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"))
        ctx.force.discard(phase)
    done = keyed_rows(path)
    todo = [qid for qid in ctx.bank_ids()
            if done.get(qid, {}).get("input_sha256") != inputs[qid]]
    return done, todo


def _require_upstream(ctx: EvalContext, phase: str, upstream: str) -> dict[str, dict]:
    rows = keyed_rows(ctx.phase_file(upstream))
    missing = [qid for qid in ctx.bank_ids() if qid not in rows]
    if missing:
        raise Refused(f"{phase}: the {upstream} phase is not complete for route {ctx.route.name}"
                      f" ({len(missing)} of {len(ctx.bank)} questions missing) — run it first")
    return rows


def item_record(item: Item, *, represented: bool = False) -> dict[str, Any]:
    """A hit as a phase-row record. `represented` (a `rerank_text: represented` route) adds the
    store's `represented_text`; every other route's record is `asdict(item)`, unchanged."""
    record = asdict(item)
    if represented:
        record["represented_text"] = item.represented_text
    return record


def item_from_record(record: dict[str, Any]) -> Item:
    record = dict(record)
    represented_text = record.pop("represented_text", None)
    item = Item(**record)
    item.represented_text = represented_text
    return item


# ---------------------------------------------------------------- 1. embed

def route_translation(ctx: EvalContext) -> dict[str, Any] | None:
    """The xl route's translation input (the xl route), read from its pinned retrieve profile; None otherwise."""
    from .hybrid import HybridError, hybrid_config

    path = ctx.route.profiles.get("retrieve")
    if path is None or not Path(path).is_file():
        return None
    try:
        config = hybrid_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    except HybridError as exc:
        raise Refused(str(exc)) from None
    return (config or {}).get("translation")


def load_translations(ctx: EvalContext, translation: dict[str, Any]) -> dict[str, str]:
    """question_id → translated text for every bank question in `applies_to_language`. The file is the profile's
    `source` (under the loop root) and must hash to its `source_sha256`; a question without a translation is refused."""
    path = ctx.route.loop_root / translation["source"]
    if not path.is_file():
        raise Refused(f"xl route: the translation file is missing: {path}")
    actual = sha256_file(path)
    if actual != translation["source_sha256"]:
        raise Refused(f"xl route: {path.name} is {actual[:12]}…, the profile pins {translation['source_sha256'][:12]}…")
    texts = {row["question_id"]: str(row.get("text") or "") for row in read_rows(path)}
    wanted = [row["question_id"] for row in ctx.bank if row["language"] == translation["applies_to_language"]]
    missing = [qid for qid in wanted if not texts.get(qid, "").strip()]
    if missing:
        raise Refused(f"xl route: {len(missing)} {translation['applies_to_language']} questions have no translation"
                      f" in {path.name}: {[q[:12] for q in missing[:3]]}")
    return {qid: texts[qid] for qid in wanted}


TRANSLATION_VECTORS = "translation.jsonl"      # in the embed dir, next to the question vectors (xl routes only)


def embed_translations(ctx: EvalContext, embed_sha: str) -> dict[str, Any] | None:
    """xl routes: the translation's query vector per Romanian question, with the same embedder and prompt as the
    questions, cached in `embed/translation.jsonl` by (text, embed profile, source file sha). None for other routes."""
    translation = route_translation(ctx)
    if translation is None:
        return None
    texts = load_translations(ctx, translation)
    inputs = {qid: item_sha256({"text": text, "embed_profile_sha256": embed_sha,
                                "source_sha256": translation["source_sha256"]}) for qid, text in texts.items()}
    path = ctx.embed_dir / TRANSLATION_VECTORS
    done = keyed_rows(path)
    todo = [qid for qid in texts if done.get(qid, {}).get("input_sha256") != inputs[qid]]
    info = None
    if todo:
        models = ctx.stack.models_for("embed")
        vectors = np.asarray(models.embed_queries([{"question_id": q, "text": texts[q], "language": "en"} for q in todo]),
                             dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(todo) or not np.all(np.isfinite(vectors)):
            raise Refused(f"embed_queries returned an unusable matrix {vectors.shape} for {len(todo)} translations")
        info = describe_models(models, "embedder")
        ctx.stack.release("embed")
        for qid, vector in zip(todo, vectors):
            append_row(path, {"question_id": qid, "input_sha256": inputs[qid], "embed_profile_sha256": embed_sha,
                              "source_sha256": translation["source_sha256"], "text_sha256": item_sha256(texts[qid]),
                              "vector": [float(v) for v in vector],
                              "vector_sha256": hashlib.sha256(vector.astype(np.float32).tobytes()).hexdigest()})
    return {"source": translation["source"], "source_sha256": translation["source_sha256"], "questions": len(texts),
            "computed": len(todo), "cached": len(texts) - len(todo), **({"embedder": info} if info else {})}


def phase_embed(ctx: EvalContext) -> dict[str, Any]:
    started = utc_now()
    pin_bank_and_route(ctx)
    embed_sha = ctx.route.profile_sha256["embed"]
    inputs = {row["question_id"]: item_sha256({"text": row["text"], "language": row["language"],
                                               "embed_profile_sha256": embed_sha})
              for row in ctx.bank}
    done, todo = _pending(ctx, "embed", inputs)
    info: dict[str, Any] | None = None
    if todo:
        models = ctx.stack.models_for("embed")
        rows = [row for row in ctx.bank if row["question_id"] in set(todo)]
        vectors = np.asarray(models.embed_queries(rows), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(rows):
            raise Refused(f"embed_queries returned shape {vectors.shape} for {len(rows)} questions")
        if not np.all(np.isfinite(vectors)):
            raise Refused("embed_queries returned non-finite values")
        info = describe_models(models, "embedder")
        ctx.stack.release("embed")
        for row, vector in zip(rows, vectors):
            record = {"question_id": row["question_id"], "input_sha256": inputs[row["question_id"]],
                      "embed_profile_sha256": embed_sha, "vector": [float(v) for v in vector],
                      "vector_sha256": hashlib.sha256(vector.astype(np.float32).tobytes()).hexdigest()}
            append_row(ctx.phase_file("embed"), record)
            done[row["question_id"]] = record
    # The loop's artefacts, in bank order: query-vectors.npy + question-rows.jsonl.
    matrix = np.asarray([done[qid]["vector"] for qid in ctx.bank_ids()], dtype=np.float32)
    ctx.embed_dir.mkdir(parents=True, exist_ok=True)
    temporary = ctx.embed_dir / f".query-vectors.npy.tmp-{os.getpid()}"
    with temporary.open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
    os.replace(temporary, ctx.embed_dir / "query-vectors.npy")
    atomic_jsonl(ctx.embed_dir / "question-rows.jsonl",
                 [{"row": index, "question_id": row["question_id"], "language": row["language"]}
                  for index, row in enumerate(ctx.bank)])
    state_path = ctx.embed_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    if info is not None:
        state = {"embed_profile_sha256": embed_sha, "embedder": info, "computed_at": utc_now()}
    state.update(questions=len(ctx.bank), matrix_shape=list(matrix.shape),
                 query_vectors_sha256=sha256_file(ctx.embed_dir / "query-vectors.npy"))
    translated = embed_translations(ctx, embed_sha)
    if translated is not None:
        state["translation"] = translated
    atomic_json(state_path, state)
    _record_phase(ctx, "embed", started, computed=len(todo), cached=len(ctx.bank) - len(todo),
                  **({"translation_computed": translated["computed"], "translation_cached": translated["cached"]}
                     if translated is not None else {}))
    return {"computed": len(todo), "cached": len(ctx.bank) - len(todo)}


# ---------------------------------------------------------------- 2. retrieve

def phase_retrieve(ctx: EvalContext) -> dict[str, Any]:
    started = utc_now()
    pin_bank_and_route(ctx)
    deps = ctx.deps(store=True)
    release = pin_release(ctx, deps)
    check_route_against_release(ctx, release)
    embedded = _require_upstream(ctx, "retrieve", "embed")
    embed_sha = ctx.route.profile_sha256["embed"]
    stale = [qid for qid in ctx.bank_ids() if embedded[qid].get("embed_profile_sha256") != embed_sha]
    if stale:
        raise Refused(f"retrieve: {len(stale)} query vectors were made with another embed profile"
                      f" than route {ctx.route.name}'s — rerun embed with this route")
    store_config = deps.config.store
    predicate = read_predicate_sha256(store_config.serve_rights, store_config.db_schema,
                                      admitted_licensed_works=store_config.admitted_works,
                                      refused_evidence_basis=store_config.refused_evidence_basis)
    widths = {"candidate_pool": deps.candidate_pool, "top_k_children": deps.top_k_children,
              "min_context_tokens": deps.min_context_tokens, "rule": deps.retrieve_rule}
    represented = deps.rerank_text == "represented"
    extra = {"rerank_text": "represented"} if represented else {}      # payload routes: inputs unchanged
    hybrid = getattr(deps, "hybrid", None)
    question_text = {row["question_id"]: row["text"] for row in ctx.bank}
    language = {row["question_id"]: row["language"] for row in ctx.bank}
    translation = (hybrid or {}).get("translation")
    texts_en: dict[str, str] = {}
    vectors_en: dict[str, dict[str, Any]] = {}
    if translation:
        texts_en = load_translations(ctx, translation)
        vectors_en = keyed_rows(ctx.embed_dir / TRANSLATION_VECTORS)
        stale_en = [qid for qid in texts_en if vectors_en.get(qid, {}).get("embed_profile_sha256") != embed_sha
                    or vectors_en[qid].get("source_sha256") != translation["source_sha256"]]
        if stale_en:
            raise Refused(f"retrieve: {len(stale_en)} translations have no query vector for this route — run embed with it")
    inputs = {qid: item_sha256({"vector_sha256": embedded[qid]["vector_sha256"],
                                "release": [release["release_attempt_id"], release["manifest_sha256"]],
                                "read_predicate_sha256": predicate, "widths": widths, **extra,
                                # hybrid routes: the lexical leg reads the question text and the vendored module
                                **({"hybrid": hybrid, "query_sha256": item_sha256(question_text[qid])}
                                   if hybrid else {}),
                                # addendum xl-v1: the translation and its vector join only for translated questions
                                **({"translation_sha256": item_sha256(texts_en[qid]),
                                    "translation_vector_sha256": vectors_en[qid]["vector_sha256"]}
                                   if qid in texts_en else {})})
              for qid in ctx.bank_ids()}
    done, todo = _pending(ctx, "retrieve", inputs)
    from .pipeline import RerankTextUnavailable, require_rerank_text, retrieve_phase

    live = live_release(ctx, deps) if todo else None
    if live is not None:
        try:
            require_rerank_text(deps, live)          # no column → refused before any row is written
        except RerankTextUnavailable as exc:
            raise Refused(f"retrieve: {exc}") from None
    for qid in todo:
        dropped: list[dict[str, Any]] = []
        vector = np.asarray(embedded[qid]["vector"], dtype=np.float32)
        trace: dict[str, Any] = {}
        candidates, kept = retrieve_phase(
            deps, vector, live, dropped=dropped, query=question_text[qid], trace=trace if hybrid else None,
            language=language[qid], query_en=texts_en.get(qid),
            qvec_en=(np.asarray(vectors_en[qid]["vector"], dtype=np.float32) if qid in texts_en else None))
        append_row(ctx.phase_file("retrieve"), {
            "question_id": qid, "input_sha256": inputs[qid],
            "release_attempt_id": live.release_attempt_id, "candidate_count": len(candidates),
            "kept": [item_record(item, represented=represented) for item in kept], "dropped": dropped,
            **({"hybrid": trace} if hybrid else {})})
    _record_phase(ctx, "retrieve", started, computed=len(todo), cached=len(ctx.bank) - len(todo),
                  read_predicate_sha256=predicate, widths=widths)
    return {"computed": len(todo), "cached": len(ctx.bank) - len(todo)}


# ---------------------------------------------------------------- 3. rerank

def phase_rerank(ctx: EvalContext) -> dict[str, Any]:
    started = utc_now()
    pin_bank_and_route(ctx)
    identity = ctx.route.identity
    deps = ctx.deps(store=True)
    pin_release(ctx, deps)
    retrieved = _require_upstream(ctx, "rerank", "retrieve")
    widths = {"identity": deps.rerank_identity, "top_k_input": deps.rerank_top_k_input,
              "top_k_output": deps.rerank_top_k}
    if getattr(deps, "rerank_tie_break", "dense_rank") != "dense_rank":
        widths["tie_break"] = deps.rerank_tie_break     # hybrid tie-break; every earlier route's inputs unchanged
    represented = deps.rerank_text == "represented"
    if represented:
        widths["rerank_text"] = "represented"           # payload routes: widths (and inputs) unchanged
    inputs = {qid: item_sha256({"retrieve_row_sha256": item_sha256(retrieved[qid]), "widths": widths,
                                "rerank_profile_sha256": ctx.route.profile_sha256["rerank"]})
              for qid in ctx.bank_ids()}
    done, todo = _pending(ctx, "rerank", inputs)
    from .pipeline import RerankTextUnavailable, rerank_with_status

    text = {row["question_id"]: row["text"] for row in ctx.bank}
    if todo and not identity:
        deps.models = ctx.stack.models_for("rerank")     # loaded only when there is work for it
    for qid in todo:
        kept = [item_from_record(record) for record in retrieved[qid]["kept"]]
        # One question per call: `Models.rerank` scores ONE question's candidates (bf16 scores
        # depend on batch partners — measured).
        try:
            top, rerank_info = rerank_with_status(deps, text[qid], kept)
        except RerankTextUnavailable as exc:
            ctx.stack.release("rerank")
            raise Refused(f"rerank: question {qid[:12]}…: {exc}") from None
        scored = 0 if identity else min(len(kept), deps.rerank_top_k_input)
        append_row(ctx.phase_file("rerank"), {"question_id": qid, "input_sha256": inputs[qid],
                                              "scored": scored,
                                              "hits": [item_record(item, represented=represented)
                                                       for item in top],
                                              # a profile with on_oom records each row's status
                                              **({"rerank_status": rerank_info["status"]}
                                                 if getattr(deps, "rerank_on_oom", None) else {}),
                                              **({"issue_boost": rerank_info["issue_boost"]}
                                                 if "issue_boost" in rerank_info else {}),
                                              # hybrid parity: every scored pool row, in pool order
                                              **({"pool_scores": [[item.chunk_id, item.rerank_score]
                                                                  for item in kept[:scored]]}
                                                 if "tie_break" in widths and not identity else {})})
    reranker = ({"operation": "identity", "device": None, "dtype": None} if identity
                else (describe_models(deps.models, "reranker") if todo else None))
    ctx.stack.release("rerank")
    state_path = ctx.route_dir / "rerank-state.json"
    if reranker is not None or not state_path.is_file():
        atomic_json(state_path, {"reranker": reranker, "widths": widths, "computed_at": utc_now()})
    _record_phase(ctx, "rerank", started, computed=len(todo), cached=len(ctx.bank) - len(todo),
                  widths=widths)
    return {"computed": len(todo), "cached": len(ctx.bank) - len(todo)}


# ---------------------------------------------------------------- 4. answer

def guard_out(ctx: EvalContext) -> Path:
    """`--out` rules.

    A real run writes only to `<answers dir>/<route name>/`, where the answers dir is `--answers-dir`
    (inside the loop, already existing; this tool never creates it). With `--set` the folder is
    `<route>-<set>` and `--release` must pin the release. A scratch run may write anywhere EXCEPT inside
    the loop. The work dir never lies in the loop.
    """
    if ctx.out is None:
        raise Refused("this phase needs --out")
    out = ctx.out.resolve()
    loop = ctx.route.loop_root
    if ctx.scratch:
        if is_inside(out, loop):
            raise Refused(f"--scratch output must lie outside the loop ({loop}): {out}")
        return out
    folder = answers_folder_name(ctx)
    if ctx.answers_dir is None:
        raise Refused(f"a real run writes only to <--answers-dir>/{folder}; no --answers-dir was given"
                      f" (got {out}) — pass --answers-dir, or --scratch for a place outside the loop")
    answers = (ctx.answers_dir if ctx.answers_dir.is_absolute() else loop / ctx.answers_dir).resolve()
    if not is_inside(answers, loop) or answers == loop.resolve():
        raise Refused(f"--answers-dir must be a folder inside the loop ({loop}): {answers}")
    if out.parent != answers:
        raise Refused(f"a real run writes only to {answers / folder}; got {out} — pass --scratch for any"
                      " other place outside the loop")
    if out.name != folder:
        raise Refused(f"a real run writes only to {answers / folder} (one folder per route"
                      f"{' and set' if ctx.set_name else ''}, named after it); got {out}")
    if not answers.is_dir():
        raise Refused(f"a real run writes only into an existing run folder: {answers} does not exist"
                      " (create it first; this tool does not)")
    if ctx.set_name is not None and ctx.release_id is None:
        raise Refused(f"a real run with --set into {answers} pins its release with --release; none was given")
    return out


def answers_folder_name(ctx: EvalContext) -> str:
    """The answers folder of a run: the route name, or `<route>-<set>` with `--set`."""
    return ctx.route.name if ctx.set_name is None else f"{ctx.route.name}-{ctx.set_name}"


def check_set_name(name: str | None) -> str | None:
    """`--set` is a plain lowercase name (it becomes part of a folder name inside the loop)."""
    import re

    if name is not None and not re.match(SET_NAME_PATTERN, name):
        raise Refused(f"--set must match {SET_NAME_PATTERN}: {name!r}")
    return name


def stage5_assertions(ctx: EvalContext, deps: Any, recorded: dict[str, Any]) -> dict[str, Any]:
    """The answer contract a `--set` run asserts in MANIFEST.json; Invalid when broken.

    prompt v1 (the route's prompt file name and sha), validator `ceres-validate-r3` as RECORDED by the
    answer phase, `answer.context_lines` off (ceres-eval never renders context lines; a config that
    turns them on would make the manifest claim a prompt the flag says was not used)."""
    from .pipeline import context_lines_enabled

    prompt = {"name": ctx.route.prompt_path.name, "sha256": deps.artifacts["prompt"]}
    lines_on = context_lines_enabled(deps.config)
    broken = []
    if prompt != ASSERTED_PROMPT:
        broken.append(f"prompt is {prompt['name']} ({prompt['sha256'][:12]}…), not prompt v1")
    if recorded.get("revision") != ASSERTED_VALIDATOR_REVISION:
        broken.append(f"validator recorded {recorded.get('revision')!r}, not {ASSERTED_VALIDATOR_REVISION}")
    if lines_on:
        broken.append("answer.context_lines is on")
    if broken:
        raise Invalid("refusing to write MANIFEST.json for a --set run: " + "; ".join(broken))
    return {"prompt": {**prompt, "revision": "v1"}, "validator_revision": ASSERTED_VALIDATOR_REVISION,
            "answer_context_lines": False,
            "release_pinned_explicitly": ctx.release_id is not None,
            "schema": deps.config.store.db_schema}


def guard_work_dir(work_dir: Path, loop_root: Path) -> None:
    if is_inside(work_dir, loop_root):
        raise Refused(f"--work-dir must lie outside the loop ({loop_root}): {work_dir}")


def _raw_name(question_id: str, attempt: int) -> str:
    """The loop's raw event name (`slot_adapter.py:1728`): full question id, two-digit attempt."""
    return f"{question_id}-attempt-{attempt:02d}.json"


def _loop_event(event: dict[str, Any]) -> dict[str, Any]:
    """The pipeline's raw event plus the loop's `request_sha256` (canonical JSON of the body)."""
    value = dict(event)
    value["request_sha256"] = item_sha256(event["request"])
    return value


def answerer_endpoint(ctx: EvalContext, answer_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Readiness before the first call: reachable AND serving the expected model id AND running the
    answer profile's context size.

    The live settings come from the server itself (`server_props`, llama.cpp `/props`), because
    the declared start command is what the server was told, not what it runs (an earlier defect: 8192 declared,
    16384 live). A known live n_ctx that differs from the profile's
    `hardware_flags.context_tokens` is a different route: refused, with
    both numbers, before any chat call. An unknown `/props` shape is recorded as unavailable and
    does not stop the phase.
    """
    from .llm import context_check, live_server_props
    from .pipeline import QUOTE_GUARDS

    status, detail, reported = ctx.stack.llm.health()
    info = {**ctx.stack.endpoint, "health": status, "models_id": reported, "checked_at": utc_now()}
    if status == "down":
        raise TransportStopped(f"the answerer is not reachable ({detail}); nothing was sent — rerun"
                               " this phase once it is up")
    if status != "ready":
        raise Refused(f"the answerer is up but not ready: {status} ({detail})")
    props = live_server_props(ctx.stack.llm)
    check = context_check(props, answer_profile)
    info.update({"server_props": props, **check})
    if check["live_n_ctx_matches_profile"] is False:
        raise Refused(f"the answerer runs n_ctx {check['live_n_ctx']} but the answer profile declares"
                      f" context_tokens {check['profile_context_tokens']}: a different context is a"
                      " different route; nothing was sent")
    # The validator this run judges with, recorded once per route (MANIFEST reports THIS, never the
    # code's constant at manifest time — an earlier r1→r3 mislabel).
    from .pipeline import validator_revision_for

    info["validator"] = {"revision": validator_revision_for(answer_profile), "guards": QUOTE_GUARDS.as_dict()}
    return info


def answer_phase_inputs(ctx: EvalContext, reranked: dict[str, Any], prompt_sha256: str,
                        no_evidence_short_circuit: bool) -> dict[str, str]:
    """Per-question identity of an answer-phase result (what `_pending` compares on resume).

    `no_evidence_short_circuit` joins only when it is on: toggling it
    must not reuse answers made under the other setting, and every run with it off keeps its old identity.
    """
    from .pipeline import validator_revision_for

    profile = yaml.safe_load(Path(ctx.route.profiles["answer"]).read_text(encoding="utf-8")) or {}
    fields = {"answer_profile_sha256": ctx.route.profile_sha256["answer"], "prompt_sha256": prompt_sha256,
              # r3 for every profile without extra ANSWER codes, so every earlier run keeps its identity
              "validator_revision": validator_revision_for(profile)}
    if no_evidence_short_circuit:
        fields["no_evidence_short_circuit"] = True
    if getattr(ctx, "citation_mode", "strict") != "strict":
        fields["citation_mode"] = ctx.citation_mode     # soft answers never reuse strict ones (strict identity unchanged)
    return {qid: item_sha256({"rerank_row_sha256": item_sha256(reranked[qid]), **fields}) for qid in ctx.bank_ids()}


def phase_answer(ctx: EvalContext) -> dict[str, Any]:
    started = utc_now()
    out = guard_out(ctx)
    pin_bank_and_route(ctx)
    deps = ctx.deps(store=True, llm=True)
    pin_release(ctx, deps)
    reranked = _require_upstream(ctx, "answer", "rerank")
    if ctx.set_name is not None:
        from .pipeline import context_lines_enabled

        if context_lines_enabled(deps.config):
            raise Refused("a --set run answers with prompt v1 and no context lines; this config turns"
                          " answer.context_lines on (ceres-eval does not render them) — nothing was sent")
    from .pipeline import (answer, answer_enabled_short_circuit, build_packet, cited_marked_contexts,
                           log_ocr_citations, validator_revision_for)
    inputs = answer_phase_inputs(ctx, reranked, deps.artifacts["prompt"],
                                 answer_enabled_short_circuit(deps.config))
    done, todo = _pending(ctx, "answer", inputs)

    raw_dir = out / "raw"
    if todo:
        endpoint = answerer_endpoint(ctx, deps.answer_profile)
        atomic_json(ctx.route_dir / "answer-endpoint.json", endpoint)
    bank = {row["question_id"]: row for row in ctx.bank}
    for qid in todo:
        question = bank[qid]
        hits = [item_from_record(record) for record in reranked[qid]["hits"]]
        packet = build_packet(question["text"], question["language"], qid, hits, deps.max_contexts).to_dict()
        validate_packet(packet)
        events: list[dict[str, Any]] = []
        row = answer(ctx.stack.llm, deps.answer_profile, deps.prompt, packet, events=events,
                     no_evidence_short_circuit=answer_enabled_short_circuit(deps.config),
                     citation_mode=ctx.citation_mode)
        if row.state == "TRANSPORT_FAILED":
            # No row is written: the question is simply not done. The event goes to the work dir
            # (it has the upstream detail); `raw/` keeps only events of questions that finished.
            append_row(ctx.route_dir / "transport-failures.jsonl",
                       {"question_id": qid, "at": utc_now(), "attempt": row.attempts,
                        "event": _loop_event(events[-1]) if events else None})
            raise TransportStopped(f"the answerer failed on question {qid[:12]}… (attempt"
                                   f" {row.attempts}); {len(done)} of {len(ctx.bank)} are done — rerun"
                                   " this phase once the answerer is back")
        answer_row = row.to_dict()
        validate_answer_row(answer_row)
        # Raw events first, then the row: a row on disk always has its raw events beside it.
        for stale in raw_dir.glob(f"{qid}-attempt-*.json"):
            stale.unlink()
        for event in events:
            atomic_json(raw_dir / _raw_name(qid, int(event["attempt"])), _loop_event(event))
        marked = cited_marked_contexts(row, hits)
        log_ocr_citations(ctx.route_dir / "ocr-citations.jsonl" if marked else None, marked,
                          execution_id=f"ceres-eval:{ctx.route.name}", question_id=qid,
                          lang=question["language"])
        record = {"question_id": qid, "input_sha256": inputs[qid], "packet": packet,
                  "answer": answer_row,
                  "model_reported": next((e["response"].get("model") for e in reversed(events)
                                          if isinstance(e.get("response"), dict)), None),
                  # The loop's answer row admits no extra key, so the revision rides on the work row.
                  "validator_revision": validator_revision_for(deps.answer_profile),
                  **({"citation": row.citation} if ctx.citation_mode != "strict" else {}),
                  "answered_at": utc_now()}
        append_row(ctx.phase_file("answer"), record)
        done[qid] = record
    # Every question is answered: the two loop files, in bank order, written whole.
    atomic_jsonl(out / "packets.jsonl", [done[qid]["packet"] for qid in ctx.bank_ids()])
    atomic_jsonl(out / "answers.jsonl", [done[qid]["answer"] for qid in ctx.bank_ids()])
    _record_phase(ctx, "answer", started, computed=len(todo), cached=len(ctx.bank) - len(todo),
                  validator_revision=VALIDATOR_REVISION)
    return {"computed": len(todo), "cached": len(ctx.bank) - len(todo)}


# ---------------------------------------------------------------- 5. validate

NOT_RECORDED = "not recorded by this run"


def _validate_with_guards(guards: Any) -> Callable[[Any, dict[str, Any]], list[str]]:
    """`pipeline.validate` with other ellipsis guards: how an earlier revision is re-checked.

    A copy of `pipeline.validate`'s body with `guards` passed through (that function is frozen by
    an external acceptance suite and takes no guards argument). `tests/test_eval.py` pins that
    this copy with `QUOTE_GUARDS` equals `pipeline.validate`.
    """
    from .pipeline import (QUOTE_SUBSTRING_ERROR, _r1_quote_miss, ellipsis_piece_spans,
                           structural_answer)

    def validate(candidate: Any, packet: dict[str, Any]) -> list[str]:
        errors = structural_answer(candidate, packet)
        if errors and errors[0] == "candidate keys do not match the profile schema":
            return errors
        citations = candidate.get("citations") if isinstance(candidate.get("citations"), list) else []
        contexts = {value["evidence_id"]: value for value in packet.get("contexts", [])}
        if guards is not None:
            for citation in citations:
                if (_r1_quote_miss(citation, contexts) and QUOTE_SUBSTRING_ERROR in errors
                        and ellipsis_piece_spans(citation["support_quote"],
                                                 contexts[citation["evidence_id"]]["evidence_text"],
                                                 guards) is not None):
                    errors.remove(QUOTE_SUBSTRING_ERROR)
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            if not str(citation.get("claim") or "").strip():
                errors.append("ceres-validate-r1: citation claim must not be empty")
            if not str(citation.get("support_quote") or "").strip():
                errors.append("ceres-validate-r1: citation support_quote must not be empty")
        return errors

    return validate


def _historical_validators() -> dict[str, Callable[[Any, dict[str, Any]], list[str]]]:
    """Earlier serving revisions this code can still re-check with (the current one is
    `pipeline.validate`). r1 = the loop's `structural_answer` + non-empty claim/quote, no ellipsis
    rule; r2 = r1 + the ellipsis rule without r3's capitalized-word gap guard (`(?!)` never matches).
    When the code moves past r3, r3 (`QUOTE_GUARDS` as it is now) must be added here —
    `test_every_shipped_validator_revision_can_still_be_rechecked` fails until it is."""
    from dataclasses import replace

    from .pipeline import QUOTE_GUARDS

    return {"ceres-validate-r1": _validate_with_guards(None),
            "ceres-validate-r2": _validate_with_guards(replace(QUOTE_GUARDS, gap_capitalized_word=r"(?!)"))}


def validator_for(recorded: str | None) -> tuple[Callable[[Any, dict[str, Any]], list[str]], str, str | None]:
    """`(validate, revision_used, note)` for answers produced under `recorded`.

    The recorded rules when this code has them; else the current rules, with a note naming both
    revisions (a legacy run that recorded nothing is named as such).
    """
    from .pipeline import validate as current

    if recorded is not None and recorded == VALIDATOR_REVISION:
        return current, VALIDATOR_REVISION, None
    historical = _historical_validators()
    if recorded in historical:
        return historical[recorded], recorded, None
    if recorded is None:
        return current, VALIDATOR_REVISION, (f"the answer run recorded no validator revision; validated"
                                             f" with the current rules ({VALIDATOR_REVISION})")
    return current, VALIDATOR_REVISION, (f"the answers were produced under {recorded}, whose rules this"
                                         f" code does not carry; validated with the current rules"
                                         f" ({VALIDATOR_REVISION})")


def recorded_validator(ctx: EvalContext) -> dict[str, Any]:
    """The validator revision (and guards) the answer phase ACTUALLY used for this route's rows.

    Source: the per-row `validator_revision` of the work-dir answer records for every bank id; the
    guard snapshot comes from `answer-endpoint.json` when it names the same revision. A row without
    it (a run from before the snapshot was recorded) makes the whole run "not recorded" — never the code's value.
    """
    records = keyed_rows(ctx.phase_file("answer"))
    revisions = {(records.get(qid) or {}).get("validator_revision") for qid in ctx.bank_ids()}
    if None in revisions or not revisions:
        return {"revision": None, "guards": None, "status": NOT_RECORDED}
    if len(revisions) > 1:
        return {"revision": None, "guards": None,
                "status": f"mixed across answer rows: {sorted(revisions)}"}
    revision = revisions.pop()
    endpoint_path = ctx.route_dir / "answer-endpoint.json"
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8")) if endpoint_path.is_file() else {}
    snapshot = endpoint.get("validator") or {}
    guards = snapshot.get("guards") if snapshot.get("revision") == revision else None
    return {"revision": revision, "guards": guards, "status": "recorded by the answer phase"}


def validate_outputs(out: Path, bank: Sequence[dict[str, str]], *,
                     validator: Callable[[Any, dict[str, Any]], list[str]] | None = None) -> list[str]:
    """Every check the acceptance gate names, as a list of findings (empty = pass).

    `validator` re-checks VALIDATED candidates; default = the current serving rules. The phases
    pass the rules the answers were produced under (`validator_for(recorded_validator(ctx))`).
    """
    from .pipeline import validate as current_validate

    validate_candidate = validator or current_validate

    findings: list[str] = []
    packets_path, answers_path = out / "packets.jsonl", out / "answers.jsonl"
    for path in (packets_path, answers_path):
        if not path.is_file():
            findings.append(f"{path.name} is missing")
    if findings:
        return findings
    packets, answers = read_rows(packets_path), read_rows(answers_path)
    bank_by_id = {row["question_id"]: row for row in bank}
    bank_ids = set(bank_by_id)

    def ids_check(name: str, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        ids = [row.get("question_id") for row in rows]
        duplicates = sorted({qid for qid in ids if ids.count(qid) > 1})
        if duplicates:
            findings.append(f"{name}: duplicated question ids: {[str(d)[:12] for d in duplicates]}")
        missing, extra = bank_ids - set(ids), set(ids) - bank_ids
        if missing:
            findings.append(f"{name}: {len(missing)} bank ids missing: {sorted(m[:12] for m in missing)}")
        if extra:
            findings.append(f"{name}: {len(extra)} ids not in the bank: {sorted(str(e)[:12] for e in extra)}")
        if len(rows) != len(bank):
            findings.append(f"{name}: {len(rows)} rows for a bank of {len(bank)}")
        return {row.get("question_id"): row for row in rows}

    packet_by_id = ids_check("packets.jsonl", packets)
    answer_by_id = ids_check("answers.jsonl", answers)
    for packet in packets:
        qid = packet.get("question_id")
        for error in schema_errors(packet, PACKET_SCHEMA):
            findings.append(f"packet {str(qid)[:12]}: {error}")
        source = bank_by_id.get(qid)
        if source is not None:
            if packet.get("question") != source["text"]:
                findings.append(f"packet {qid[:12]}: question text differs from the bank")
            if packet.get("language") != source["language"]:
                findings.append(f"packet {qid[:12]}: language differs from the bank")
            if packet.get("packet_id") != f"grounded-{qid}":
                findings.append(f"packet {qid[:12]}: packet_id is not grounded-<question_id>")
    raw_dir = out / "raw"
    expected_raw: set[str] = set()
    for row in answers:
        qid = row.get("question_id")
        for error in schema_errors(row, ANSWER_SCHEMA):
            findings.append(f"answer {str(qid)[:12]}: {error}")
        if row.get("state") == "TRANSPORT_FAILED":
            findings.append(f"answer {str(qid)[:12]}: TRANSPORT_FAILED rows are never written by a"
                            " complete run")
        packet = packet_by_id.get(qid)
        if row.get("state") == "VALIDATED" and packet is not None:
            errors = validate_candidate(row.get("candidate"), packet)
            if errors:
                findings.append(f"answer {qid[:12]}: VALIDATED but the validator says {errors[0]}")
        attempts = row.get("attempts") if isinstance(row.get("attempts"), int) else 0
        for attempt in range(1, attempts + 1):
            name = _raw_name(str(qid), attempt)
            expected_raw.add(name)
            if not (raw_dir / name).is_file():
                findings.append(f"raw/{name} is missing")
    present = {path.name for path in raw_dir.glob("*.json")} if raw_dir.is_dir() else set()
    if present - expected_raw:
        findings.append(f"raw/ holds {len(present - expected_raw)} events of no answer row")
    if set(answer_by_id) != set(packet_by_id):
        findings.append("packets.jsonl and answers.jsonl do not hold the same question ids")
    return findings


def phase_validate(ctx: EvalContext) -> dict[str, Any]:
    if ctx.out is None:
        raise Refused("validate needs --out")
    recorded = recorded_validator(ctx)
    validator, used, note = validator_for(recorded["revision"])
    if note:
        ctx.log(f"note: {note}")
    findings = validate_outputs(ctx.out.resolve(), ctx.bank, validator=validator)
    if findings:
        for finding in findings[:50]:
            ctx.log(f"invalid: {finding}")
        raise Invalid(f"{len(findings)} validation finding(s) in {ctx.out} (validated with {used})")
    return {"findings": 0, "questions": len(ctx.bank), "validator_revision_recorded": recorded["revision"],
            "validated_with": used, **({"note": note} if note else {})}


# ---------------------------------------------------------------- 6. manifest

def ocr_citations(packets: Sequence[dict[str, Any]], answers: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Counts from the files as written: answers citing an OCR-admitted context, etc."""
    by_id = {packet["question_id"]: packet for packet in packets}
    answers_citing = cited_contexts = packets_with = 0
    for packet in packets:
        if any(c.get("evidence_basis") == OCR_ADMITTED_UNVERIFIED for c in packet["contexts"]):
            packets_with += 1
    for row in answers:
        if row.get("state") != "VALIDATED":
            continue
        contexts = {c["evidence_id"]: c for c in by_id[row["question_id"]]["contexts"]}
        cited = {c.get("evidence_id") for c in (row["candidate"].get("citations") or [])
                 if isinstance(c, dict)}
        marked = [eid for eid in cited
                  if (contexts.get(eid) or {}).get("evidence_basis") == OCR_ADMITTED_UNVERIFIED]
        cited_contexts += len(marked)
        answers_citing += bool(marked)
    return {"answers_citing_ocr_admitted_unverified": answers_citing,
            "cited_ocr_admitted_unverified_contexts": cited_contexts,
            "packets_with_ocr_admitted_unverified_contexts": packets_with}


def phase_manifest(ctx: EvalContext) -> dict[str, Any]:
    from .llm import context_check

    out = guard_out(ctx)
    state = pin_bank_and_route(ctx)
    deps = ctx.deps(store=True)
    release = pin_release(ctx, deps)
    recorded = recorded_validator(ctx)
    validator, validated_with, validator_note = validator_for(recorded["revision"])
    findings = validate_outputs(out, ctx.bank, validator=validator)
    if findings:
        raise Invalid(f"refusing to write MANIFEST.json: validate reports {len(findings)} finding(s),"
                      f" first: {findings[0]}")
    asserted = stage5_assertions(ctx, deps, recorded) if ctx.set_name is not None else None
    if ctx.answerer_cmd_file is None and not ctx.scratch:
        raise Refused("a real run's manifest needs --answerer-cmd-file (the answerer's start command)")
    packets, answers = read_rows(out / "packets.jsonl"), read_rows(out / "answers.jsonl")
    states = {name: sum(row["state"] == name for row in answers)
              for name in ("VALIDATED", "STRUCTURE_FAILED", "TRANSPORT_FAILED")}
    final_errors: dict[str, int] = {}
    for row in answers:
        for error in row.get("errors") or []:
            final_errors[error] = final_errors.get(error, 0) + 1
    attempt_errors: dict[str, int] = {}
    models_reported: set[str] = set()
    raw_files = sorted((out / "raw").glob("*.json"))
    for path in raw_files:
        event = json.loads(path.read_text(encoding="utf-8"))
        for error in event.get("structural_errors") or []:
            attempt_errors[error] = attempt_errors.get(error, 0) + 1
        model = (event.get("response") or {}).get("model") if isinstance(event.get("response"), dict) else None
        if isinstance(model, str):
            models_reported.add(model)
    embed_state_path = ctx.embed_dir / "state.json"
    embed_state = json.loads(embed_state_path.read_text(encoding="utf-8")) if embed_state_path.is_file() else {}
    rerank_state_path = ctx.route_dir / "rerank-state.json"
    rerank_state = json.loads(rerank_state_path.read_text(encoding="utf-8")) if rerank_state_path.is_file() else {}
    endpoint_path = ctx.route_dir / "answer-endpoint.json"
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8")) if endpoint_path.is_file() else {}
    # Recomputed from the recorded snapshot and the route's answer profile; a run from before the
    # snapshot was recorded has none, and every live_* field is then null.
    live_props = endpoint.get("server_props")
    live = context_check(live_props, deps.answer_profile)
    command = None
    if ctx.answerer_cmd_file is not None:
        if not ctx.answerer_cmd_file.is_file():
            raise Refused(f"--answerer-cmd-file not found: {ctx.answerer_cmd_file}")
        blob = ctx.answerer_cmd_file.read_bytes()
        command = {"file": str(ctx.answerer_cmd_file.resolve()), "sha256": hashlib.sha256(blob).hexdigest(),
                   "text": blob.decode("utf-8", errors="replace")}
    phases: list[dict[str, Any]] = read_rows(ctx.embed_dir / "phases.jsonl") + read_rows(ctx.route_dir / "phases.jsonl")
    store_config = deps.config.store
    reranker = rerank_state.get("reranker") or {}
    document = {
        "schema_version": MANIFEST_SCHEMA,
        "scratch": ctx.scratch,
        "route": {
            "name": ctx.route.name, "file": str(ctx.route.file), "file_sha256": ctx.route.file_sha256,
            "route_id": ctx.route.route_id, "route_id_recomputed": compute_route_id(ctx.route.profiles),
            "profiles": {slot: {"path": str(ctx.route.profiles[slot]),
                                "sha256": ctx.route.profile_sha256[slot]} for slot in ROUTE_SLOT_ORDER},
            "prompt": {"path": str(ctx.route.prompt_path), "sha256": deps.artifacts["prompt"]},
        },
        "release": {key: release.get(key) for key in ("release_attempt_id", "manifest_sha256",
                                                     "product_manifest_sha256", "route_id",
                                                     "row_count", "profile_sha256", "pinned_at",
                                                     # only when --schema/--release pinned it
                                                     *(k for k in PIN_MODE_KEYS if k in release))},
        "store": {"serving_mode": store_config.serving_mode, "schema": store_config.db_schema,
                  # Parity ruling: the read predicate is part of the served
                  # identity and runs in SQL BEFORE dedup and top-k, unlike the loop, which drops
                  # ineligible chunks only when it builds packets. Stated so the scorecard label is exact.
                  "read_predicate_applied": "before dedup and top-k (SQL WHERE in PgStore.search)",
                  "read_predicate_sha256": read_predicate_sha256(
                      store_config.serve_rights, store_config.db_schema,
                      admitted_licensed_works=store_config.admitted_works,
                      refused_evidence_basis=store_config.refused_evidence_basis)},
        "query_embed": {"device": (embed_state.get("embedder") or {}).get("device"),
                        "dtype": ((embed_state.get("embedder") or {}).get("dtype")
                                  or (embed_state.get("embedder") or {}).get("effective_dtype")),
                        "embedder": embed_state.get("embedder"),
                        "embed_profile_sha256": embed_state.get("embed_profile_sha256"),
                        "query_vectors_sha256": embed_state.get("query_vectors_sha256")},
        "reranker": ("identity" if ctx.route.identity else
                     {"device": reranker.get("device"), "dtype": reranker.get("dtype"),
                      "detail": reranker, "widths": rerank_state.get("widths")}),
        # Machine-readable contract: the TOOL-written live reading of
        # the answer server, under one stable key. Mandatory for answer sets written by ceres >= 0.2.0; a hand-recorded
        # value never goes here — it goes to top-level `attested` (a WARN, never a PASS).
        "live_server": {"n_ctx": live["live_n_ctx"],
                        "models": [endpoint.get("models_id")] if endpoint.get("models_id") else [],
                        "read_at": (live_props or {}).get("fetched_at") or endpoint.get("checked_at"),
                        "props_available": bool(live_props) and (live_props or {}).get("available", True) is not False,
                        "profile_context_tokens": live["profile_context_tokens"],
                        "matches_profile": live["live_n_ctx_matches_profile"],
                        "tool": f"ceres-eval {__version__}"},
        "attested": [],
        **({"translation": embed_state["translation"]} if embed_state.get("translation") else {}),
        "answer": {"endpoint": endpoint.get("base_url"), "models_id": endpoint.get("models_id"),
                   "endpoint_checked_at": endpoint.get("checked_at"),
                   "models_reported_in_raw": sorted(models_reported),
                   "answerer_start_command": command,
                   "citation_mode": ctx.citation_mode,
                   "live_props": live_props,
                   "live_n_ctx": live["live_n_ctx"],
                   "profile_context_tokens": live["profile_context_tokens"],
                   "live_n_ctx_matches_profile": live["live_n_ctx_matches_profile"],
                   # What the answer phase recorded, never the code's constant now.
                   "validator_revision": recorded["revision"],
                   "validator_revision_status": recorded["status"],
                   "validator_guards": recorded["guards"],
                   "validated_with": {"revision": validated_with, "note": validator_note},
                   "answer_profile_sha256": ctx.route.profile_sha256["answer"]},
        "counts": {"questions": len(answers), **states,
                   "structural_error_histogram": {"final_rows": dict(sorted(final_errors.items())),
                                                  "all_attempts": dict(sorted(attempt_errors.items()))},
                   "raw_events": len(raw_files),
                   # How many answers needed the second attempt under
                   # the serving validator (`validator_revision` above) — the one the routes are judged with.
                   "answers_by_attempts": {str(k): v for k, v in sorted(Counter(
                       int(row.get("attempts") or 0) for row in answers).items())},
                   "validated_on_second_attempt": sum(1 for row in answers if row.get("state") == "VALIDATED"
                                                      and int(row.get("attempts") or 0) == 2),
                   **ocr_citations(packets, answers)},
        "bank": {"path": state["bank"]["path"], "sha256": state["bank"]["sha256"],
                 "rows": state["bank"]["rows"]},
        "files": {"packets.jsonl": sha256_file(out / "packets.jsonl"),
                  "answers.jsonl": sha256_file(out / "answers.jsonl")},
        "ceres": {"version": __version__, "config_sha256": deps.config.config_sha256()},
        "timestamps": {"work_dir_created_at": state.get("created_at"), "phases": phases,
                       "manifest_written_at": utc_now()},
        "notes": _read_notes(ctx.notes_file),
        "validate": "passed",
    }
    # Keys added only when they say something, so a flagless payload-route manifest is unchanged.
    if ctx.route.rerank_text != "payload":
        document["route"]["rerank_text"] = ctx.route.rerank_text
    if ctx.set_name is not None:
        document["set"] = {"name": ctx.set_name, "folder": out.name, "pinned": state.get("set")}
        document["asserted"] = asserted
    existing = out / "MANIFEST.json"
    if existing.is_file():
        try:
            previous = json.loads(existing.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        if isinstance(previous, dict) and previous.get("hand_corrected"):
            # A MANIFEST an owner corrected by hand (e.g. the r1 sets: the writer once stamped the
            # current validator onto older answers) carries facts this run cannot re-derive; overwriting it would
            # silently undo the correction. Move it aside explicitly if a rewrite is really intended.
            raise Refused(f"{existing} is hand-corrected ({previous['hand_corrected']}); move it aside to rewrite it")
    atomic_json(out / "MANIFEST.json", document)
    return {"counts": document["counts"]}


PHASE_FUNCTIONS: dict[str, Callable[[EvalContext], dict[str, Any]]] = {
    "embed": phase_embed, "retrieve": phase_retrieve, "rerank": phase_rerank,
    "answer": phase_answer, "validate": phase_validate, "manifest": phase_manifest,
}


def run_phase(phase: str, ctx: EvalContext) -> dict[str, Any]:
    guard_work_dir(ctx.work_dir, ctx.route.loop_root)
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    return PHASE_FUNCTIONS[phase](ctx)


def run_chain(ctx: EvalContext, phases: Sequence[str] = PHASES) -> dict[str, dict[str, Any]]:
    return {phase: run_phase(phase, ctx) for phase in phases}


# ---------------------------------------------------------------- CLI wiring

def _real_stack(config: Any, route: Route, device: str | None) -> Stack:
    """PgStore from the config, the answerer endpoint from the config, and the in-process models
    of `ceres.models` — each built only when a phase asks for it, and only the
    one model that phase needs."""
    # The builders are shared with `ceres-api` (`ceres.models.build_*`).
    from .models import InProcessModels, build_answerer, build_embedder, build_reranker
    from .store import PgStore

    endpoint = config.endpoints.get("answerer")
    if endpoint is None:
        raise Refused("the config has no `endpoints.answerer`")

    def models_factory(phase: str) -> Any:
        if phase == "embed":
            return InProcessModels(build_embedder(route.profiles["embed"], loop_root=route.loop_root,
                                                  device=device))
        if phase == "rerank":
            return InProcessModels(None, build_reranker(route.profiles["rerank"],
                                                        loop_root=route.loop_root, device=device))
        raise Refused(f"the {phase} phase loads no model")

    return Stack(store_factory=lambda: PgStore.from_config(config.store),
                 models_factory=models_factory,
                 llm_factory=lambda: build_answerer(endpoint),
                 endpoint={"base_url": endpoint.base_url, "expect_model": endpoint.expect_model,
                           "lifecycle": endpoint.lifecycle})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ceres-eval", description="Answer a DDS question bank through a route, phase by phase.")
    parser.add_argument("phase", choices=(*PHASES, "chain"))
    parser.add_argument("--route", required=True, help="route file, or a route name under <loop>/04_rag/routes/")
    parser.add_argument("--bank", required=True, type=Path, help="questions JSONL (question_id, language, text)")
    parser.add_argument("--work-dir", required=True, type=Path, help="phase caches (outside the loop)")
    parser.add_argument("--out", type=Path, help="answers folder: packets.jsonl, answers.jsonl, raw/, MANIFEST.json")
    parser.add_argument("--config", type=Path, help="Ceres config (store DSN, answerer endpoint)")
    parser.add_argument("--loop-root", type=Path, help="default: the config's paths.loop_root, else from the route file")
    parser.add_argument("--answers-dir", type=Path,
                        help="the existing folder inside the loop a non-scratch run writes <route>/ into"
                             " (relative to the loop root, or absolute); without it a run needs --scratch")
    parser.add_argument("--scratch", action="store_true",
                        help="allow --out outside the allowed loop answers folders (never inside the loop)")
    parser.add_argument("--schema", help="store schema for this run (overrides the config's store.schema)")
    parser.add_argument("--release", dest="release_id",
                        help="pin this release_attempt_id, active or not, instead of the active release")
    parser.add_argument("--allow-unknown-release-profiles", action="store_true",
                        help="accept a release whose row and manifest name no chunk/represent/embed profile sha"
                             " (refused by default)")
    parser.add_argument("--citation-mode", choices=("strict", "soft"), default="strict",
                        help="answer phase: strict (r3, default) or soft (soft); independent of the config")
    parser.add_argument("--set", dest="set_name",
                        help="question-set name: answers go to <route>-<set>, pinned in RUN-STATE, recorded in MANIFEST")
    parser.add_argument("--fake", action="store_true",
                        help="fixture store + fake models + fake answerer (needs --scratch)")
    parser.add_argument("--force-phase", action="append", default=[], choices=PHASES[:4],
                        help="recompute this phase's rows (repeatable)")
    parser.add_argument("--answerer-cmd-file", type=Path, help="the answerer's start command, for MANIFEST")
    parser.add_argument("--notes-file", type=Path,
                        help="JSON list of free-text notes copied into MANIFEST (e.g. run-owner rulings)")
    parser.add_argument("--device", help="model device override (cpu for tests; default: the profile's)")
    return parser


def _read_notes(path: Path | None) -> list[str]:
    """Notes for MANIFEST: a JSON list of strings, or [] without a file. A malformed file is refused
    rather than dropped, because a missing ruling note would mislabel the judged route."""
    if path is None:
        return []
    if not path.is_file():
        raise Refused(f"--notes-file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise Refused(f"--notes-file must hold a JSON list of strings: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # `--no-auth` is irrelevant here (the eval serves no HTTP); keep its warning out of the output.
    logging.getLogger("ceres.config").setLevel(logging.ERROR)
    try:
        from .config import load_config

        if args.fake and not args.scratch:
            raise Refused("--fake writes fake answers and is allowed only with --scratch")
        if args.fake:
            from .testing import build_eval_fake_stack, smoke_env

            example = Path(__file__).resolve().parent.parent / "config.example.yaml"
            config = load_config(args.config or example,
                                 smoke_env(args.work_dir.resolve() / "fake-runtime"),
                                 require_api_key=False)
            stack = build_eval_fake_stack()
        else:
            config = load_config(args.config, require_api_key=False)
        if args.schema is not None:
            config = with_schema(config, args.schema)
        if args.release_id is not None and not args.release_id.strip():
            raise Refused("--release names no release")
        check_set_name(args.set_name)
        # The fake config's loop root is a placeholder directory: a fake run takes the loop root
        # from --loop-root or from the route file itself.
        config_root = None if args.fake else config.paths.loop_root
        loop_root = args.loop_root or (Path(config_root) if config_root else None)
        route = load_route(args.route, loop_root)
        if not args.fake:
            stack = _real_stack(config, route, args.device)
        bank, bank_sha = load_bank(args.bank)
        ctx = EvalContext(route=route, bank_path=args.bank, bank=bank, bank_sha256=bank_sha,
                          work_dir=args.work_dir.resolve(), config=config, stack=stack,
                          out=args.out, scratch=args.scratch, answers_dir=args.answers_dir,
                          answerer_cmd_file=args.answerer_cmd_file, notes_file=args.notes_file,
                          force=set(args.force_phase), release_id=args.release_id,
                          schema=args.schema, set_name=args.set_name, citation_mode=args.citation_mode,
                          allow_unknown_release_profiles=bool(args.allow_unknown_release_profiles or args.fake))
        phases = PHASES if args.phase == "chain" else (args.phase,)
        for phase in phases:
            result = run_phase(phase, ctx)
            print(json.dumps({"phase": phase, "route": route.name, **result}, ensure_ascii=False,
                             sort_keys=True))
        return EXIT_OK
    except EvalError as exc:
        print(f"ceres-eval: {exc}", file=sys.stderr)
        return exc.exit_code
    except StoreError as exc:
        print(f"ceres-eval: store: {exc}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
