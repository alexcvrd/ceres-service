"""Discovery, placeholder resolution, secret refusal, env overrides and schema validation.

Ported from the DDS operator CLI's config module: `resolve_placeholders` (the ${UPPER} env / ${lower} derived two-pass), `scan_secrets`
with `_SECRET_TOKENS` / `_SECRET_VALUES` / `_is_secret_key`, and `DSN_PATTERN`. Keeping the same
rules matters operationally: one operator edits both config files, and a value the CLI refuses must
not be accepted by the service.

Two deliberate differences from the CLI:
  * validation is pydantic, not a JSON-Schema file, because the API layer already carries pydantic;
  * a key whose last token is `env`, `file` or `path` names a *location* of a secret, not a secret
    (`api.api_key_env: CERES_API_KEY` must load). The CLI has no such key today.

The API key never lives in the file: it is read from the environment variable `api.api_key_env`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

LOG = logging.getLogger("ceres.config")

SCHEMA_VERSION = "ceres-config-v1"
PACKAGE_DIR = Path(__file__).resolve().parent.parent

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_DERIVED_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
DERIVED_KEYS = ("loop_root", "runtime_root")
# CLI `dds/config.py:25` — a password-less URI, so libpq reads ~/.pgpass and no secret is in reach
# of a log line, a process listing or this file.
DSN_PATTERN = re.compile(r"^postgresql://[^:@/]+@[^/?]+/[^?]+$")

# CLI `dds/config.py:34` — token-boundary key rule: `tokenizer` must load, `api_key` must not.
_SECRET_TOKENS = frozenset(
    {"password", "passwd", "pwd", "secret", "token", "apikey", "authorization", "bearer",
     "credential", "credentials"}
)
# Keys that name where a secret lives rather than carrying one.
_LOCATION_SUFFIXES = ("env", "file", "path")
# CLI `dds/config.py:38` — credential prefixes only at a token boundary, so a path such as
# /tmp/task-configuration stays harmless.
_B = r"(?<![A-Za-z0-9_-])"
_SECRET_VALUES = (
    re.compile(_B + r"sk-[A-Za-z0-9]{8,}"),
    re.compile(_B + r"ghp_"),
    re.compile(_B + r"AKIA[0-9A-Z]{12,}"),
    re.compile(_B + r"xox[abp]-"),
    re.compile(r"://[^/@:]+:[^/@]+@"),
    re.compile(r"(?i)password\s*="),
)

# Set by `load_config`, never by the file.
RESERVED_KEYS = ("source_path", "api_key", "auth_required")

# r1.route.yaml: `route_id_recipe: sha256(concatenated seven profile sha256 values in slot order)`.
ROUTE_SLOT_ORDER = ("chunk", "represent", "embed", "index", "retrieve", "rerank", "answer")

# Serving is a MODE, and the rights list is derived
# from it. `rnd` is the shipped default — every archived work carries an evidence-based rights
# declaration, and R&D reads all of them. `production` narrows to redistributable plus the licensed
# works a compliance decision admitted one by one. `serve_rights` is never written in a config file:
# an operator who could widen the list with an editor is an operator who can publish a work the
# company has no right to publish.
SERVE_RIGHTS_BY_MODE: dict[str, tuple[str, ...]] = {
    "rnd": ("redistributable", "internal_research_only", "licensed"),
    "production": ("redistributable",),
}
SERVING_MODES = tuple(SERVE_RIGHTS_BY_MODE)

# OCR-admitted evidence is served in R&D and marked;
# production refuses it by a read-predicate arm. Derived from the mode like the rights list, never
# a config key of its own. The value is `ceres.store.OCR_ADMITTED_UNVERIFIED` (kept literal here so
# the config module does not import the driver-facing store module).
REFUSED_EVIDENCE_BASIS_BY_MODE: dict[str, str | None] = {
    "rnd": None,
    "production": "ocr_admitted_unverified",
}


class ConfigError(Exception):
    """The configuration is missing, malformed, unsafe, or fails the schema."""


# ---------------------------------------------------------------- secret refusal (CLI port)

def _is_secret_key(key: str) -> bool:
    """True if any token of the key, or the adjacent pair (api, key), names a credential."""
    tokens = [t.lower() for t in re.split(r"[_\-.]|(?<=[a-z0-9])(?=[A-Z])", str(key)) if t]
    if tokens and tokens[-1] in _LOCATION_SUFFIXES:
        return False
    pairs = zip(tokens, tokens[1:])
    return bool(_SECRET_TOKENS.intersection(tokens)) or ("api", "key") in pairs


def scan_secrets(obj: Any, _path: str = "") -> list[str]:
    """Return the config paths that look like secrets. Paths only — never the values."""
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            child = f"{_path}.{k}".lstrip(".")
            if _is_secret_key(k) and isinstance(v, str) and v.strip():
                hits.append(child)
            hits.extend(scan_secrets(v, child))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(scan_secrets(v, f"{_path}[{i}]"))
    elif isinstance(obj, str):
        if any(p.search(obj) for p in _SECRET_VALUES):
            hits.append(_path or "<root>")
    return sorted(dict.fromkeys(hits))


def _reject_secrets(source: Path, doc: Any, stage: str) -> None:
    """Raise if anything in `doc` looks like a secret; the message names paths, never values."""
    hits = scan_secrets(doc)
    if hits:
        raise ConfigError(
            f"{source}: secret-looking value {stage} at: {', '.join(hits)}"
            " — keep secrets out of config (libpq reads ~/.pgpass, the API key comes from the environment)"
        )


# ---------------------------------------------------------------- placeholders (CLI port)

def resolve_placeholders(
    obj: Any,
    env: Mapping[str, str],
    _path: str = "",
    *,
    klass: re.Pattern[str] = _ENV_NAME,
    kind: str = "environment variable",
) -> Any:
    """Substitute the ${NAME} of one class (default ${UPPER} = environment) from `env`.

    Names of other classes are left for a later pass; a name of this class that `env` does not
    define is a ConfigError naming it — an unresolved path must never reach the store or the disk.
    """
    kw = {"klass": klass, "kind": kind}
    if isinstance(obj, dict):
        return {k: resolve_placeholders(v, env, f"{_path}.{k}".lstrip("."), **kw) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_placeholders(v, env, f"{_path}[{i}]", **kw) for i, v in enumerate(obj)]
    if not isinstance(obj, str):
        return obj
    mine = [m.group(1) for m in _PLACEHOLDER.finditer(obj) if klass.match(m.group(1))]
    missing = [n for n in mine if n not in env]
    if missing:
        where = _path or "<root>"
        names = ", ".join(f"${{{n}}}" for n in dict.fromkeys(missing))
        raise ConfigError(f"{where}: unresolved placeholder(s) {names} — no such {kind}")
    return _PLACEHOLDER.sub(lambda m: env[m.group(1)] if klass.match(m.group(1)) else m.group(0), obj)


# ---------------------------------------------------------------- CERES_<SECTION>__<KEY>

def apply_env_overrides(doc: dict[str, Any], env: Mapping[str, str]) -> list[str]:
    """Apply `CERES_<SECTION>__<KEY>[__<KEY>]` scalar overrides in place; return the paths touched.

    Only names containing the double underscore are overrides, so `CERES_CONFIG`,
    `CERES_API_KEY` and `CERES_RUNTIME_ROOT` keep their own meaning. The value is parsed as YAML
    scalar, so `CERES_API__DOCS=true` is a boolean and `CERES_API__PORT=8301` an integer.
    """
    touched: list[str] = []
    for name in sorted(env):
        if not name.startswith("CERES_") or "__" not in name:
            continue
        parts = [p.lower() for p in name[len("CERES_"):].split("__") if p]
        if len(parts) < 2:
            continue
        cursor: Any = doc
        for part in parts[:-1]:
            if not isinstance(cursor, dict):
                raise ConfigError(f"{name}: {'.'.join(parts)} is not a mapping in the config")
            cursor = cursor.setdefault(part, {})
        if not isinstance(cursor, dict):
            raise ConfigError(f"{name}: {'.'.join(parts)} is not a mapping in the config")
        value = yaml.safe_load(env[name])
        if isinstance(value, (dict, list)):
            raise ConfigError(f"{name}: environment overrides carry scalars only")
        cursor[parts[-1]] = value
        touched.append(".".join(parts))
    return touched


# ---------------------------------------------------------------- the schema

class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Admission(_Section):
    """Bounded admission: one inference worker, a short queue, then 429 (the admission contract).

    Two bounds, because a timeout alone bounds how long a request waits but not how many wait:
    `max_inflight` requests run, at most `max_pending` more wait up to `queue_timeout_s`, and any
    request beyond that is refused at once.
    """

    max_inflight: int = Field(default=2, ge=1)
    max_pending: int = Field(default=8, ge=0)
    queue_timeout_s: float = Field(default=30.0, gt=0)


class ApiSection(_Section):
    host: str = "127.0.0.1"
    port: int = Field(default=8300, ge=1, le=65535)
    api_key_env: str = "CERES_API_KEY"
    docs: bool = False               # /docs, /redoc, /openapi.json off unless this is true
    max_question_chars: int = Field(default=2000, ge=1)
    max_body_bytes: int = Field(default=64 * 1024, ge=1)
    admission: Admission = Admission()
    # The longest a client waits for one answer, admission wait included. At the deadline the
    # client gets 504; the worker thread cannot be interrupted, so it runs to its end and keeps its
    # admission permit until then — capacity is never handed out twice.
    # 360 s = the answerer's 300 s request timeout plus embedding, rerank and one queue wait.
    request_deadline_s: float = Field(default=360.0, gt=0)


class StoreSection(_Section):
    """The DDS serving store. Ceres reads; it never creates, alters or drops."""

    dsn: str
    db_schema: str = Field(default="ceres_serve_v0", alias="schema")
    # The one rights key. `serve_rights` is a derived property below, never a config field —
    # a config that still sets it is refused by `load_config` with a message naming the removed key.
    serving_mode: Literal["rnd", "production"] = "rnd"
    # Work ids admitted individually in production mode, under `production_decision`. Meaningless
    # in rnd (where `licensed` is served anyway), so a non-empty list there is refused: it would
    # read as a narrowing that is not happening.
    admitted_licensed_works: tuple[str, ...] = ()
    # The id of the recorded compliance decision that authorised production serving. Required in production: the mode switch must be traceable to a recorded decision,
    # not to a config edit.
    production_decision: str | None = None
    # `evidence_exclude_reasons` was removed: the read predicate is
    # `active release AND evidence_eligible AND rights_state = ANY(serve_rights)` and nothing
    # else. `excluded_reasons` still travels into the packet's `warnings`, so a judge still sees
    # what the release flagged — it simply no longer decides what is retrievable.
    statement_timeout_ms: int = Field(default=5000, ge=1)
    pool_max_size: int = Field(default=4, ge=1)
    # How long a request waits for a pooled connection (pool exhausted or server unreachable)
    # before it is a store failure → 503; and libpq's own connect timeout. psycopg_pool's default
    # wait is 30 s, which is longer than a client should wait to learn the store is down.
    pool_timeout_s: float = Field(default=10.0, gt=0)
    connect_timeout_s: int = Field(default=5, ge=1)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _mode_is_coherent(self) -> "StoreSection":
        if self.serving_mode == "production" and not (self.production_decision or "").strip():
            raise ValueError(
                "store.production_decision is required when store.serving_mode is `production`"
                " — name the recorded compliance decision id"
            )
        if self.serving_mode == "rnd" and self.admitted_licensed_works:
            raise ValueError(
                "store.admitted_licensed_works is only meaningful in `production` mode; rnd serves"
                " every licensed work already"
            )
        return self

    @property
    def serve_rights(self) -> tuple[str, ...]:
        """The rights states this mode serves. Derived — it must not be set directly."""
        return SERVE_RIGHTS_BY_MODE[self.serving_mode]

    @property
    def refused_evidence_basis(self) -> str | None:
        """The evidence basis this mode refuses: production refuses OCR-admitted items."""
        return REFUSED_EVIDENCE_BASIS_BY_MODE[self.serving_mode]

    @property
    def admitted_works(self) -> tuple[str, ...]:
        """The individually admitted licensed work ids, sorted; empty outside production."""
        if self.serving_mode != "production":
            return ()
        return tuple(sorted(self.admitted_licensed_works))


class EndpointSection(_Section):
    """One model endpoint. Fields adopted verbatim from the CLI (the store contract's config fields)."""

    protocol: str
    base_url: str
    health_url: str | None = None
    lifecycle: Literal["managed", "external"] = "external"
    resource: str = "gpu"
    start_argv: list[str] = Field(default_factory=list)
    stop_argv: list[str] = Field(default_factory=list)
    startup_timeout_s: int = 180
    request_timeout_s: int = 300
    expect_model: str | None = None   # the id `GET /v1/models` must list before /health says ready


class RouteSection(_Section):
    """Which route this service serves, and the profile files that define it."""

    # The served route's identity comes from ONE place: the frozen served-route declaration the
    # CLI owner writes (`served_route_file`, a `*.route.yaml` with a `route_id`). `load_config`
    # fills `served_route_id` from it; writing the id by hand is refused, and so is a declaration
    # that names r1 — a served answer is never labelled r1. With no declaration configured
    # the id stays null and every answer says `route_status: unassigned`.
    served_route_file: str | None = None
    served_route_id: str | None = None      # set by `load_config` only
    judged_route_id: str
    profiles: dict[str, str] = Field(default_factory=dict)
    prompt_template: str


class AnswerSection(_Section):
    """How the answer prompt is rendered (chunk-context lines).

    `context_lines`: put each context's checked chunk-context line (`llm.context_line`) before its
    passage as `SOURCE METADATA: …`. Default false = the r1 user message, byte for byte. Turning it
    on changes what the answerer reads, so it is a measured change (a candidate route), not a
    tuning knob; the served prompt stays v1 until a new route is judged.
    """

    context_lines: bool = False
    # when retrieval yields no evidence, answer a deterministic
    # UNSUPPORTED/NO_RELEVANT_SUPPORT without calling the answerer (no model-memory text can then carry a
    # VALIDATED label). It changes answers, so it is a measured candidate: default false, never on silently.
    no_evidence_short_circuit: bool = False
    # Citation switch. `strict` = ceres-validate-r3, unchanged. `soft` = a citation that
    # fails a citation-level rule is dropped and the answer kept; every other rule stays hard. One line switches back.
    citation_mode: Literal["strict", "soft"] = "strict"
    # Situational split (experimental): a stateless question that carries the asker's situation is split; retrieval
    # uses the core question and the answer gets the situational rules. Needs memory.enabled (the prompts live there). Default off =
    # today's request, byte for byte.
    situational_split: bool = False
    # Readable sources: meta.sources gives each passage's title, authors and page label,
    # read from the active release's census under paths.loop_root. Default off = the response is unchanged.
    source_titles: bool = False


class MemorySection(_Section):
    """Memory v0. Off by default: with `enabled: false` a
    request carrying `conversation_id` or `profile_id` is refused (422) and every request is the stateless r1 request."""

    enabled: bool = False
    personas_dir: str = "personas"
    max_turns: int = Field(default=20, ge=1)
    ttl_s: float = Field(default=7200.0, gt=0)
    history_turns: int = Field(default=3, ge=0)
    rewrite_max_tokens: int = Field(default=300, ge=32)
    rewrite_timeout_s: float = Field(default=45.0, gt=0)
    max_conversations: int = Field(default=1000, ge=1)


class PathsSection(_Section):
    loop_root: str | None = None
    census_root: str | None = None   # where the DDS release census lives (readable sources); default: loop_root
    runtime_root: str | None = None


class CeresConfig(BaseModel):
    """A validated configuration. `api_key`, `auth_required` and `source_path` come from the
    process, never from the file."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["ceres-config-v1"]
    api: ApiSection = ApiSection()
    store: StoreSection
    endpoints: dict[str, EndpointSection] = Field(default_factory=dict)
    resources: dict[str, dict[str, Any]] = Field(default_factory=dict)
    route: RouteSection
    paths: PathsSection = PathsSection()
    answer: AnswerSection = AnswerSection()
    memory: MemorySection = MemorySection()

    source_path: Path
    api_key: str | None = None
    auth_required: bool = True

    # -------------------------------------------------------- derived helpers

    def resolve(self, value: str | Path) -> Path:
        """A configured path, absolute: relative entries resolve against the config file's folder."""
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.source_path.parent / path).resolve()

    @property
    def runtime_root(self) -> Path | None:
        return self.resolve(self.paths.runtime_root) if self.paths.runtime_root else None

    @property
    def ocr_citation_log(self) -> Path | None:
        """Every answer citing an OCR-admitted context appends a line here. None without a
        runtime root — and then such an answer is refused, not served unlogged (`pipeline.ask`)."""
        root = self.runtime_root
        return root / "logs" / "ocr-citations.jsonl" if root else None

    @property
    def raw_executions_dir(self) -> Path | None:
        """Where the raw OpenAI events of an execution go, when a runtime root is configured."""
        root = self.runtime_root
        return root / "executions" if root else None

    @property
    def safe_dsn(self) -> str:
        """The DSN reduced to scheme, host, port and database — no user information (CLI port)."""
        from urllib.parse import urlsplit

        u = urlsplit(self.store.dsn)
        port = f":{u.port}" if u.port else ""
        return f"{u.scheme}://{u.hostname or ''}{port}{u.path}"

    def profile_paths(self) -> dict[str, Path]:
        return {slot: self.resolve(path) for slot, path in self.route.profiles.items()}

    def config_sha256(self) -> str:
        """Digest of the EFFECTIVE configuration: after placeholders and `CERES_*` overrides.

        A digest of the file alone would call two boxes identical while one of them runs with
        `CERES_STORE__SERVING_MODE=production`. The API key and the process-only fields are left
        out (the key is a secret; `source_path` is where the file happens to live). Computed once
        at startup by `api.create_app` and reported from that snapshot.
        """
        import json

        excluded = {"api_key", "source_path", "auth_required"}
        # The `answer` section joins the digest only when it differs from its defaults,
        # so every configuration that does not set it keeps the digest it had before the section
        # existed; `answer.context_lines: true` changes it, as any effective change must.
        if self.answer == AnswerSection():
            excluded.add("answer")
        effective = self.model_dump(mode="json", by_alias=True, exclude=excluded)
        canonical = json.dumps(effective, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_route_id(profiles: Mapping[str, Path]) -> str | None:
    """sha256 over the seven profile sha256 hex strings in slot order (`r1.route.yaml` recipe).

    Parity tests only (`test_route_id_recipe_reproduces_r1`). The service never derives its own
    served id from this: identity comes from the served-route declaration (`load_served_route_id`).
    None when a slot profile is missing or unreadable.
    """
    digests: list[str] = []
    for slot in ROUTE_SLOT_ORDER:
        path = profiles.get(slot)
        if path is None or not Path(path).is_file():
            return None
        digests.append(hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return hashlib.sha256("".join(digests).encode("utf-8")).hexdigest()


_ROUTE_ID = re.compile(r"^[0-9a-f]{64}$")


def load_served_route_id(route_file: Path, judged_route_id: str) -> str:
    """The `route_id` of a served-route declaration, refused if it is malformed or r1's.

    The declaration is the CLI-owned frozen `*.route.yaml` (same shape as `r1.route.yaml`). Only
    `route_id` is read: checking it against the release and the effective profiles belongs to the
    serving gate, which this loader does not replace.
    """
    if not route_file.is_file():
        raise ConfigError(f"route.served_route_file not found: {route_file}")
    try:
        doc = yaml.safe_load(route_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{route_file}: not valid YAML: {exc.__class__.__name__}") from None
    route_id = doc.get("route_id") if isinstance(doc, dict) else None
    if not isinstance(route_id, str) or not _ROUTE_ID.match(route_id):
        raise ConfigError(f"{route_file}: `route_id` must be 64 lowercase hex characters")
    if route_id == judged_route_id:
        raise ConfigError(
            f"{route_file}: declares the judged route r1 ({route_id[:12]}…) as the served route."
            " Ceres never serves as r1 — the served route is its own frozen declaration."
        )
    return route_id


# ---------------------------------------------------------------- loading

def _discover(path: Path | None, env: Mapping[str, str]) -> Path:
    """Discovery order: explicit path > $CERES_CONFIG > <package folder>/config.yaml."""
    for candidate, origin in (
        (path, "--config"),
        (env.get("CERES_CONFIG"), "$CERES_CONFIG"),
        (PACKAGE_DIR / "config.yaml", "package default"),
    ):
        if candidate:
            p = Path(candidate).expanduser()
            if not p.is_file():
                raise ConfigError(f"config file not found ({origin}): {p}")
            return p
    raise ConfigError("no config file given")


def load_config(
    path: Path | None = None,
    env: Mapping[str, str] | None = None,
    *,
    require_api_key: bool = True,
) -> CeresConfig:
    """Load, resolve, secret-scan, override and validate; raise ConfigError on any problem.

    `require_api_key=False` is the smoke-test escape hatch (`--no-auth`). It is an explicit flag,
    it is logged at WARNING, and `/v1/*` then serves without authentication — never a default.
    """
    env = os.environ if env is None else env
    source = _discover(path, env)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{source}: not valid YAML: {exc.__class__.__name__}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: top level must be a mapping")
    for key in RESERVED_KEYS:
        if key in raw:
            raise ConfigError(f"{source}: `{key}` is set by the process, not by the file")

    _reject_secrets(source, raw, "as written")
    resolved = resolve_placeholders(raw, env)
    _reject_secrets(source, resolved, "after placeholder resolution")

    # The ${lower} class is derived from the config itself, so one root edit moves every path.
    roots = resolved.get("paths") or {}
    derived = {k: str(roots[k]) for k in DERIVED_KEYS if roots.get(k)}
    if derived:
        resolved = resolve_placeholders(resolved, derived, klass=_DERIVED_NAME, kind="derived config value")
        _reject_secrets(source, resolved, "after derived resolution")

    touched = apply_env_overrides(resolved, env)
    if touched:
        LOG.info("config overridden from the environment at: %s", ", ".join(touched))
    _reject_secrets(source, resolved, "after environment overrides")

    store_doc = resolved.get("store") or {}
    # Refuse the old key by name. `extra="forbid"` would already reject it, but "extra inputs
    # are not permitted" does not tell an operator that the rights list moved — and a config whose
    # rights list is silently ignored is exactly the failure this key exists to prevent.
    if isinstance(store_doc, dict) and "serve_rights" in store_doc:
        raise ConfigError(
            f"{source}: `store.serve_rights` is not a configuration key."
            " The rights list is derived from `store.serving_mode`: `rnd` serves"
            f" {list(SERVE_RIGHTS_BY_MODE['rnd'])}, `production` serves"
            f" {list(SERVE_RIGHTS_BY_MODE['production'])} plus `store.admitted_licensed_works`"
            " under a recorded `store.production_decision`. Remove the key."
        )

    dsn = store_doc.get("dsn") if isinstance(store_doc, dict) else None
    if isinstance(dsn, str) and not DSN_PATTERN.match(dsn):
        raise ConfigError(
            f"{source}: store.dsn must look like postgresql://user@host[:port]/dbname"
            " — no password, no query string (libpq reads ~/.pgpass)"
        )

    api_key_env = (resolved.get("api") or {}).get("api_key_env", "CERES_API_KEY")
    api_key = env.get(api_key_env) or None
    if require_api_key and not api_key:
        raise ConfigError(
            f"the API key environment variable {api_key_env} is empty or unset"
            " — export it, or start with --no-auth (smoke tests only)"
        )
    if not require_api_key:
        LOG.warning("authentication disabled by an explicit --no-auth flag; /v1/* is open")

    try:
        config = CeresConfig(**resolved, source_path=source, api_key=api_key,
                             auth_required=require_api_key)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "<root>"
        raise ConfigError(f"{source}: schema violation at {where}: {first['msg']}") from None

    if config.route.served_route_id is not None:
        raise ConfigError(
            f"{source}: `route.served_route_id` is not written by hand — name the frozen"
            " served-route declaration in `route.served_route_file` instead"
        )
    if config.route.served_route_file:
        config.route.served_route_id = load_served_route_id(
            config.resolve(config.route.served_route_file), config.route.judged_route_id)
    return config
