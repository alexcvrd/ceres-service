"""The serving store: one protocol, the real pgvector reader, and the fixture reader.

Contract: the store contract. Ceres reads the
schema `dds load` writes and never creates, alters or drops anything in it. The read predicate in
`PgStore.search` is the contract's read-side statement, unchanged — a different predicate is a
contract version, not a config edit.

The `Item` record is modelled on the operator CLI's `Hit` record, minus the CLI's
`citable`/`flags` projection: the CLI keeps non-citable rows so an operator can see what was
withheld; the service filters them in SQL so a non-servable chunk never reaches the prompt
(internal-only works are never served).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol, Sequence

import numpy as np

LOG = logging.getLogger("ceres.store")

IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")   # the CLI loader's pattern — schema names are not quoted
CANDIDATE_POOL = 50                             # r1 `dense-top10-v2.yaml: candidate_pool`
RELEASE_COUNT_TIMEOUT_MS = 300_000              # the once-per-release withheld count (cached); requests keep the served timeout

# OCR-admitted evidence. `items.evidence_basis` is an
# OPTIONAL nullable column (contract v0.2 once `dds load` writes it; absent on release-0006/0007),
# and so are `items.page_start integer` / `items.page_end integer`: the 0-based, INCLUSIVE PDF page
# span of the chunk's source section, both NULL for legacy sections with an unknown page base
# (replacing the earlier announced `page_indexes int[]`). The only basis
# value the contract knows is the one below; the packet schema
# enumerates exactly it. Any other non-null value is a contract change nobody announced, and the
# store refuses it rather than serve it (it would pass the production arm, which only names this one).
OCR_ADMITTED_UNVERIFIED = "ocr_admitted_unverified"
KNOWN_EVIDENCE_BASES = frozenset({OCR_ADMITTED_UNVERIFIED})
# Optional `items` columns → the SQL placeholder selected when a release's table lacks them.
# Detection tolerates any subset of them being present (each is added by its own ADD COLUMN IF NOT EXISTS).
OPTIONAL_ITEM_COLUMNS = {"evidence_basis": "NULL::text", "page_start": "NULL::integer",
                         "page_end": "NULL::integer"}
# Chunk context: `items.context`,
# an OPTIONAL nullable jsonb column holding the chunk-context row of a v3.5 route (NULL/absent on
# v3.4). Detected together with the OCR/page columns (any subset) and reported in
# `Release.optional_columns` only when present, so a v3.4 release reports exactly what it did.
# Metadata for question time: never part of `payload`/evidence_text, never citable.
CONTEXT_ITEM_COLUMNS = {"context": "NULL::jsonb"}
# Represented text: `items.represented_text`, an OPTIONAL nullable column
# holding the represent-v2 text the route embedded (header + cleaned payload). NEVER citable: it is
# read only as the reranker's input when a route declares `rerank_text: represented` (v3.5);
# packets, evidence_text, quotes and validation stay on `payload`. `dds load` may write it as text or
# as bytea (NUL-safe); both are accepted (`REPRESENTED_TEXT_TYPES`), decoded like `payload`, and the
# detected type is kept in `PgStore.column_types`. Detected with the columns above (any subset).
REPRESENTED_TEXT_COLUMNS = {"represented_text": "NULL::text"}
REPRESENTED_TEXT_TYPES = frozenset({"text", "bytea", "character varying"})
DETECTED_ITEM_COLUMNS = {**OPTIONAL_ITEM_COLUMNS, **CONTEXT_ITEM_COLUMNS, **REPRESENTED_TEXT_COLUMNS}


class StoreError(Exception):
    """The store is unusable or inconsistent. The message never carries the DSN.

    Driver failures are wrapped into this at the adapter boundary (`_driver_errors`), so the API
    has ONE exception to map to 503. The driver's own exception stays reachable as `__cause__`
    for the log; its text — which can name hosts and users — never becomes the message.
    """


@dataclass
class Item:
    """One retrieved chunk. Mutable: the pipeline stamps `rank` and `evidence_id` on it."""

    chunk_id: str
    parent_id: str
    work_id: str
    membership: str
    rights_state: str
    payload: str
    payload_sha256: str
    locator_ids: list[str] = field(default_factory=list)
    lineage_ids: list[str] = field(default_factory=list)
    excluded_reasons: list[str] = field(default_factory=list)
    score: float = 0.0              # cosine similarity, 1 - (embedding <=> q)
    dense_rank: int = 0
    rank: int = 0
    rerank_score: float | None = None
    evidence_id: str | None = None
    # NULL for ordinary evidence; `ocr_admitted_unverified` for admitted, unconfirmed scan OCR.
    evidence_basis: str | None = None
    # The 0-based, inclusive PDF page span of the chunk's source section;
    # both None when the store does not provide them. Never guessed from locators here.
    page_start: int | None = None
    page_end: int | None = None
    # The chunk-context object from the optional `items.context` column (None when the
    # release has no such column or the row holds NULL). Deliberately a plain instance attribute and
    # NOT a dataclass field: `ceres-eval` stores hits with `dataclasses.asdict`, and a new field
    # would add `"context": null` to every rerank record of a v3.4 run and so change the answer
    # phase's input sha (its cache key). Never part of `payload`, never in `record()`.
    context = None      # type: dict[str, Any] | None
    # The optional `items.represented_text` value (None when the release has no such column
    # or the row holds NULL). Same reasoning as `context`: an instance attribute, NOT a dataclass
    # field, so `asdict` (ceres-eval's row records and their cache keys) is unchanged for every route
    # that reranks on payload. Never citable, never in `record()`, never in a packet.
    represented_text = None      # type: str | None

    def record(self) -> dict[str, Any]:
        """The identity of the hit, with no payload — safe for logs and run records."""
        return {
            "rank": self.rank, "dense_rank": self.dense_rank, "chunk_id": self.chunk_id,
            "parent_id": self.parent_id, "work_id": self.work_id, "membership": self.membership,
            "rights_state": self.rights_state, "score": self.score,
            "rerank_score": self.rerank_score, "evidence_id": self.evidence_id,
            "evidence_basis": self.evidence_basis,
        }


@dataclass(frozen=True)
class Release:
    """The one active release, pinned per request and reported in `meta`."""

    release_attempt_id: str
    route_id: str
    row_count: int
    membership_counts: dict[str, int] = field(default_factory=dict)
    # Items of this release the rights predicate withholds. Release-level (the same for every
    # query), so the store computes it once per (release, predicate) instead of per request.
    withheld_rights: int = 0
    # sha256 of the release's manifest as read. `dds load --replace` rewrites an attempt's rows
    # under the SAME release_attempt_id (a store-contract write-side rule), so the id alone is
    # not a cache key: (release_attempt_id, manifest_sha256) is. Empty for fixture releases.
    manifest_sha256: str = ""
    # Which OPTIONAL `items` columns (`DETECTED_ITEM_COLUMNS`: the three OCR/page columns, plus `context`) this release's table has, detected once per release identity. Any subset; an
    # absent column reads NULL.
    optional_columns: tuple[str, ...] = ()
    # The `releases` row's `chunk_profile_sha256` / `represent_profile_sha256` /
    # `embed_profile_sha256`, as {"chunk", "represent", "embed"} → sha256 of the profile file the
    # release was built with (ROUTE-STORE-GUARD). `pipeline.pinned_release` compares
    # them with the configured route's profiles, so a `--replace` that re-points the release to
    # another route under the same id cannot be served under the configured identity. Part of the
    # release object only: `identity` below is unchanged, because the manifest sha already changes
    # on a replace. Empty for fixture releases (no check, reported as "skipped").
    profile_sha256: dict[str, str] = field(default_factory=dict)

    @property
    def identity(self) -> tuple[str, str]:
        """What a per-release cache is keyed by (see `manifest_sha256`)."""
        return (self.release_attempt_id, self.manifest_sha256)


class Store(Protocol):
    """What the pipeline needs from a store. The service swaps `FixtureStore` for `PgStore`."""

    def active_release(self) -> Release:
        """The single active release. Zero or more than one is an error, never a fallback."""

    def search(
        self,
        qvec: Sequence[float],
        release_attempt_id: str,
        serve_rights: Sequence[str],
        limit: int = CANDIDATE_POOL,
        admitted_licensed_works: Sequence[str] = (),
        refused_evidence_basis: str | None = None,
    ) -> list[Item]:
        """Dense top-`limit` under the serve predicate, cosine order with chunk_id as tie-break.

        `refused_evidence_basis` is the production arm: items whose basis equals it are not
        served (`evidence_basis IS DISTINCT FROM …`). None (rnd) adds no arm."""

    def health(self) -> tuple[str, str]:
        """`(status, detail)`; status is "ready", "degraded" or "down"."""

    # Optional, and used only for `meta.filtered`: how many items of the release the rights
    # predicate withholds. A store that cannot answer it cheaply may omit the method.
    # def withheld(self, release: Release, serve_rights: Sequence[str],
    #              admitted_licensed_works: Sequence[str] = ()) -> Release: ...


# ---------------------------------------------------------------- helpers shared by both stores

def serves(rights_state: Any, work_id: Any, serve_rights: Sequence[str],
           admitted_licensed_works: Sequence[str] = ()) -> bool:
    """The rights half of the read predicate, in Python.

    `rights_state = ANY(serve_rights) OR (rights_state = 'licensed' AND work_id = ANY(admitted))`.
    The second arm is the production-mode admission list: a licensed work the compliance decision
    named individually. In rnd the list is empty and the arm is dead, exactly as the SQL is.
    """
    if rights_state in set(serve_rights):
        return True
    return rights_state == "licensed" and work_id in set(admitted_licensed_works)

def _json_list(value: Any) -> list[str]:
    """A jsonb column as a list of strings, whether the driver decoded it or handed back text.

    Ported from the DDS CLI's `ask` module.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return [value]
    return [str(item) for item in value] if isinstance(value, list) else []


def read_predicate_sha256(serve_rights: Sequence[str], db_schema: str = "ceres_serve_v0",
                          predicate: str = "active+evidence_eligible+rights",
                          contract: str = "v0.1",
                          admitted_licensed_works: Sequence[str] = (),
                          refused_evidence_basis: str | None = None) -> str:
    """The identity of the read predicate a request was served under (the predicate digest).

    sha256 over the canonical JSON `{"admitted_licensed_works":[…],"contract":…,"predicate":…,
    "schema":…,"serve_rights":[…]}` with sorted keys and no spaces — the loop's canonical JSON form
    (`00_control/code/keys.py:19`). It travels in `meta`, so a served answer carries proof of the
    rights list and the predicate that produced its evidence, and a widened `serve_rights` is
    visible in the record even if nobody read the config.

    The derived rights list changes with `store.serving_mode`, so the digest does too — an
    rnd answer and a production answer are never confusable in the record. The admitted work ids
    are sorted before hashing, so the digest is a property of the admitted SET, not of the order an
    operator happened to type it in.

    Production refuses `ocr_admitted_unverified` items, so its predicate gained an arm and its
    digest gains the key `refused_evidence_basis`. The key is added ONLY when an arm exists: rnd's
    predicate did not change (rnd serves OCR items), so rnd's digest stays the one earlier answers
    carry. The digest is a property of the configuration, not of the release: on a release without
    the column every basis is NULL and the arm is a no-op, so the predicate means the same thing.
    """
    value = {"predicate": predicate, "serve_rights": list(serve_rights),
             "admitted_licensed_works": sorted(admitted_licensed_works),
             "schema": db_schema, "contract": contract}
    if refused_evidence_basis is not None:
        value["refused_evidence_basis"] = refused_evidence_basis
    canonical = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def decode_payload(payload: Any, payload_sha256: str, chunk_id: str) -> str:
    """Verify sha256 over the stored bytes, then decode strict UTF-8 (store contract, read side).

    A mismatch fails the request closed: the service must never cite text it cannot prove is the
    text DDS loaded. The DDS CLI does the same before building a packet. Bytes that hash
    correctly but are not UTF-8 are a corrupt load, and fail closed the same way.
    """
    if hasattr(payload, "tobytes"):
        payload = payload.tobytes()
    if isinstance(payload, (bytes, bytearray)):
        digest = hashlib.sha256(bytes(payload)).hexdigest()
        if digest != payload_sha256:
            raise StoreError(f"chunk {chunk_id}: payload does not match payload_sha256 — refusing to cite it")
        try:
            return bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise StoreError(f"chunk {chunk_id}: payload is not valid UTF-8 — refusing to cite it") from exc
    text = str(payload)
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != payload_sha256:
        raise StoreError(f"chunk {chunk_id}: payload does not match payload_sha256 — refusing to cite it")
    return text


def optional_fields(evidence_basis: Any, page_start: Any, page_end: Any,
                    chunk_id: str) -> tuple[str | None, int | None, int | None]:
    """Check the three optional OCR/page values of one row; refuse what the contract does not know.

    A basis outside `KNOWN_EVIDENCE_BASES` is refused (see there). The page span is either both
    NULL (unknown page base) or two non-negative integers with `page_start <= page_end` (0-based,
    inclusive). One NULL, a negative page, an inverted span or a non-integer is a load the service
    cannot cite a page from, and is refused rather than logged.
    """
    if evidence_basis is not None and evidence_basis not in KNOWN_EVIDENCE_BASES:
        raise StoreError(f"chunk {chunk_id}: unknown evidence_basis {evidence_basis!r} — refusing to serve it")
    if page_start is None and page_end is None:
        return evidence_basis, None, None
    if page_start is None or page_end is None:
        raise StoreError(f"chunk {chunk_id}: exactly one of page_start/page_end is NULL"
                         f" (page_start={page_start!r}, page_end={page_end!r})")
    for name, page in (("page_start", page_start), ("page_end", page_end)):
        if not isinstance(page, int) or isinstance(page, bool):
            raise StoreError(f"chunk {chunk_id}: {name} is not an integer ({type(page).__name__})")
        if page < 0:
            raise StoreError(f"chunk {chunk_id}: {name} is negative ({page})")
    if page_start > page_end:
        raise StoreError(f"chunk {chunk_id}: page_start {page_start} > page_end {page_end}")
    return evidence_basis, int(page_start), int(page_end)


def decode_represented_text(value: Any, chunk_id: str) -> str | None:
    """The `items.represented_text` value of one row as text, or None for SQL NULL.

    bytea (bytes / memoryview) is decoded as strict UTF-8, exactly like `payload`, so a NUL inside the
    text survives (Postgres `text` cannot hold one; bytea can). A text column arrives as str. Bytes
    that are not UTF-8 are a corrupt load and fail closed (StoreError), as a corrupt payload does:
    the reranker must never score a guessed text. There is no sha to verify: the contract carries
    none for this column, and the text is never quoted to anyone.
    """
    if value is None:
        return None
    if hasattr(value, "tobytes"):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise StoreError(f"chunk {chunk_id}: represented_text is not valid UTF-8 — refusing to rerank on it") from exc
    if isinstance(value, str):
        return value
    raise StoreError(f"chunk {chunk_id}: represented_text is neither text nor bytea ({type(value).__name__})")


def context_value(value: Any, chunk_id: str) -> dict[str, Any] | None:
    """The `items.context` value of one row as a dict, or None.

    jsonb arrives decoded (psycopg) or as text (a fixture, another driver). The column is metadata
    and never citable, so a value that is not a JSON object does NOT fail the request (unlike a bad
    payload or a page span): it reads as None — no line is rendered — and is logged by chunk id.
    """
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = None
    if isinstance(value, dict):
        return value
    LOG.warning(json.dumps({"event": "store.context_not_an_object", "chunk_id": chunk_id}))
    return None


def release_cache_identity(release: "Release") -> tuple[Any, ...]:
    """What a per-release cache is keyed by: (id, manifest sha) plus the row's route id and profile shas.

    An audit fix widened `_release_from_row`'s key, but the
    withheld-count cache still used (id, manifest sha) alone, so a same-manifest replace could serve a stale count.
    """
    return (*release.identity, release.route_id, tuple(sorted((release.profile_sha256 or {}).items())))


def _manifest_sha256(manifest: dict[str, Any]) -> str:
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                           default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@contextmanager
def _driver_errors(operation: str) -> Iterator[None]:
    """Translate expected driver failures into `StoreError` at the adapter boundary.

    Covered: every psycopg error (connection, query, statement timeout; `psycopg_pool.PoolTimeout`
    and `PoolClosed` are psycopg errors too), socket-level `OSError`/`TimeoutError`, and a
    `UnicodeDecodeError` from a text column. The message names the operation and the exception
    class only; the original is chained as `__cause__` for the private log.
    """
    import psycopg

    try:
        yield
    except StoreError:
        raise
    except (psycopg.Error, OSError, TimeoutError, UnicodeDecodeError) as exc:
        raise StoreError(f"store {operation} failed: {exc.__class__.__name__}") from exc


def _register_vector_types(conn: Any) -> None:
    """Pool `configure=` hook: teach one new connection the pgvector type, once.

    Per connection, not per query — `register_vector` looks the type up in the catalog, and doing
    that on every search was a round trip per request.
    """
    from pgvector.psycopg import register_vector

    register_vector(conn)


# ---------------------------------------------------------------- pgvector

class PgStore:
    """Read-only pgvector client for schema `ceres_serve_v0`.

    Its error mapping, caching and pool wiring are tested with a fake pool; it also runs against a
    disposable PostgreSQL 16.2 + pgvector 0.6.2 (`tests/test_store.py`, pgserver) and, opt-in,
    against a live serving database as role `ceres_reader` (`tests/test_store.py` section 4, marker `live_store`).

    Everything the class does is read-only and parameterized: the schema name is the one value
    that cannot be a parameter, so it is checked against `IDENTIFIER` before it reaches a
    statement, exactly as the DDS loader and CLI do.
    """

    def __init__(
        self,
        dsn: str,
        db_schema: str = "ceres_serve_v0",
        *,
        statement_timeout_ms: int = 5000,
        pool_max_size: int = 4,
        open_pool: bool = True,
        pool_timeout_s: float = 10.0,
        connect_timeout_s: int = 5,
    ) -> None:
        if not IDENTIFIER.match(db_schema):
            raise StoreError(f"store.schema is not a plain identifier: {db_schema!r}")
        self.db_schema = db_schema
        self.statement_timeout_ms = int(statement_timeout_ms)
        self._release_cache: dict[tuple[str, str], Release] = {}
        # (release identity, read_predicate_sha256) → withheld count. Presence, not truthiness,
        # decides a hit: zero is the common rnd answer and must be cached too.
        self._withheld_cache: dict[tuple[tuple[str, str], str], int] = {}
        # release_attempt_id → optional columns, refreshed whenever `active_release` sees a new
        # release identity (a `--replace` can add the column under the same id).
        self._columns: dict[str, tuple[str, ...]] = {}
        self.column_detections = 0                  # how often information_schema was asked
        # column name → information_schema data_type of the detected optional columns (the items
        # table is shared by every release of the schema, so this is schema-level, latest detection).
        self.column_types: dict[str, str] = {}
        from psycopg.rows import tuple_row          # imported here: no driver import at module load
        from psycopg_pool import ConnectionPool

        self._pool = ConnectionPool(
            conninfo=dsn,
            min_size=1,
            max_size=int(pool_max_size),
            open=open_pool,
            # How long a request waits for a free connection (pool exhausted, or the server
            # unreachable) before it becomes a StoreError → 503. psycopg_pool's default is 30 s.
            timeout=float(pool_timeout_s),
            # Read-only and bounded, per connection: a slow query must fail the request, not hold
            # the single inference worker. `default_transaction_read_only` is the belt to the
            # `ceres_reader` role's braces (INSERT is already denied).
            configure=_register_vector_types,
            kwargs={
                "row_factory": tuple_row,
                "connect_timeout": int(connect_timeout_s),
                "options": f"-c statement_timeout={int(statement_timeout_ms)} "
                           f"-c default_transaction_read_only=on",
            },
        )

    @classmethod
    def from_config(cls, store: Any, **overrides: Any) -> "PgStore":
        """A reader built from a validated `StoreSection` (`config.store`)."""
        kwargs = dict(statement_timeout_ms=store.statement_timeout_ms,
                      pool_max_size=store.pool_max_size, pool_timeout_s=store.pool_timeout_s,
                      connect_timeout_s=store.connect_timeout_s)
        kwargs.update(overrides)
        return cls(store.dsn, store.db_schema, **kwargs)

    def close(self) -> None:
        self._pool.close()

    # -------------------------------------------------------- release

    def release_manifest(self, release_attempt_id: str) -> dict[str, Any] | None:
        """`releases.manifest` of one release, or None when the release does not exist.

        `ceres-eval` pins the manifest's `product_manifest_sha256` and profile hashes for a run, so
        a reload under the same release id is detected. Read-only, parameterized; driver errors
        become StoreError like every other read here.
        """
        with _driver_errors("release manifest lookup"):
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute(f"SELECT manifest FROM {self.db_schema}.releases WHERE release_attempt_id = %s",
                            (release_attempt_id,))
                row = cur.fetchone()
        if row is None:
            return None
        value = row[0]
        if isinstance(value, (bytes, str)):
            value = json.loads(value or "{}")
        if not isinstance(value, dict):
            raise StoreError(f"release {release_attempt_id}: manifest is not a JSON object")
        return value

    def active_release(self) -> Release:
        """The single active release; zero or two is a refusal (store contract, read side).

        Queried on every request (one indexed row): the release can be switched or replaced under
        a running service, and the request must pin whatever is active now.
        """
        with _driver_errors("release lookup"):
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    f"SELECT {self.RELEASE_COLUMNS}"
                    f" FROM {self.db_schema}.releases WHERE active"
                )
                rows = cur.fetchall()
            if len(rows) != 1:
                raise StoreError(
                    f"{self.db_schema}.releases has {len(rows)} active releases, expected exactly one"
                )
        return self._release_from_row(rows[0])

    def release(self, release_attempt_id: str) -> Release | None:
        """ONE named release, active or not; None when the schema has no such release.

        `ceres-eval --release` only an eval may measure a staged release
        before it is activated. Serving never calls this — `/v1/ask`, `/health` and `/v1/info` read
        `active_release()` alone, so a request can never be answered from an inactive release.
        Same Release shape, manifest sha and optional-column detection as `active_release`.
        """
        with _driver_errors("release lookup"):
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    f"SELECT {self.RELEASE_COLUMNS}"
                    f" FROM {self.db_schema}.releases WHERE release_attempt_id = %s",
                    (release_attempt_id,),
                )
                rows = cur.fetchall()
        if not rows:
            return None
        return self._release_from_row(rows[0])

    # What `active_release` and `release(id)` read from a `releases` row, in `_release_from_row` order.
    # The three profile shas are NOT NULL in the contract DDL (`dds/load.py` `ddl`).
    RELEASE_COLUMNS = ("release_attempt_id, route_id, row_count, manifest,"
                       " chunk_profile_sha256, represent_profile_sha256, embed_profile_sha256")

    def _release_from_row(self, row: Sequence[Any]) -> Release:
        """A `releases` row → Release, cached per (id, manifest sha); a new identity re-detects the
        optional columns once. Shared by `active_release` and `release`."""
        (release_attempt_id, route_id, row_count, manifest,
         chunk_sha, represent_sha, embed_sha) = row
        manifest = manifest if isinstance(manifest, dict) else json.loads(manifest or "{}")
        # The cache key also carries the row's route id and profile-sha
        # columns, so a replace that changes them without changing the manifest bytes is re-read.
        key = (release_attempt_id, _manifest_sha256(manifest), route_id, chunk_sha, represent_sha, embed_sha)
        cached = self._release_cache.get(key)
        if cached is not None:
            return cached
        counts = {str(k): int(v) for k, v in (manifest.get("membership_counts") or {}).items()}
        # A new release identity: (re)detect the optional columns for it, once.
        columns = self._detect_optional_columns()
        self._columns[release_attempt_id] = columns
        release = Release(
            release_attempt_id=release_attempt_id, route_id=route_id, row_count=int(row_count),
            membership_counts=counts, manifest_sha256=key[1], optional_columns=columns,
            profile_sha256={slot: str(sha) for slot, sha in
                            (("chunk", chunk_sha), ("represent", represent_sha), ("embed", embed_sha))
                            if sha},
        )
        self._release_cache[key] = release
        return release

    def _detect_optional_columns(self) -> tuple[str, ...]:
        """Which of `DETECTED_ITEM_COLUMNS` (the three OCR/page columns + `context`) the items table has now.

        Called once per release identity (`active_release`), not per request. `information_schema`
        lists only columns the role may read, which is exactly the set a SELECT may name.
        """
        self.column_detections += 1
        with _driver_errors("column detection"):
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT column_name, data_type FROM information_schema.columns"
                    " WHERE table_schema = %(schema)s AND table_name = 'items'"
                    " AND column_name = ANY(%(names)s::text[])",
                    {"schema": self.db_schema, "names": list(DETECTED_ITEM_COLUMNS)},
                )
                types = {row[0]: (row[1] if len(row) > 1 else None) for row in cur.fetchall()}
        # `represented_text` is used only when its type is one `decode_represented_text`
        # reads; any other type is left unselected (reads NULL), so a route that reranks on it is
        # refused by name instead of scoring a mis-decoded value. Logged, never silent.
        kind = types.get("represented_text")
        if "represented_text" in types and kind is not None and kind not in REPRESENTED_TEXT_TYPES:
            LOG.warning(json.dumps({"event": "store.represented_text_unsupported_type", "data_type": kind}))
            del types["represented_text"]
        self.column_types = {name: str(kind) for name, kind in types.items() if kind is not None}
        return tuple(name for name in DETECTED_ITEM_COLUMNS if name in types)

    def optional_columns(self, release_attempt_id: str) -> tuple[str, ...]:
        """The optional columns of a release as detected when it was pinned (detect if unseen)."""
        columns = self._columns.get(release_attempt_id)
        if columns is None:
            columns = self._detect_optional_columns()
            self._columns[release_attempt_id] = columns
        return columns

    def withheld(self, release: Release, serve_rights: Sequence[str],
                 admitted_licensed_works: Sequence[str] = ()) -> Release:
        """What the rights predicate withholds from this release, counted once per predicate.

        One aggregate over the same rows the search scans, under the same two-armed rights clause,
        so an admitted licensed work is not counted as withheld. It is release-level
        information (the internal-only works must stay visible as a number), so paying for it
        per request would be waste. The cache key is the release identity (a `--replace` changes
        it) and the predicate digest (a mode switch changes it).
        """
        key = (release_cache_identity(release), read_predicate_sha256(
            serve_rights, self.db_schema, admitted_licensed_works=admitted_licensed_works))
        if key not in self._withheld_cache:
            with _driver_errors("withheld count"):
                with self._pool.connection() as conn, conn.cursor() as cur:
                    # One aggregate per (release, predicate), then cached: a cold scan of ~330k rows passed the served
                    # 5 s timeout (seen in a parity run as QueryCanceled). SET LOCAL = this transaction only.
                    cur.execute(f"SET LOCAL statement_timeout = {RELEASE_COUNT_TIMEOUT_MS}")
                    cur.execute(
                        f"SELECT count(*) FILTER (WHERE NOT ({self.RIGHTS_CLAUSE}))"
                        f" FROM {self.db_schema}.items WHERE release_attempt_id = %(active)s",
                        {"serve_rights": list(serve_rights), "active": release.release_attempt_id,
                         "admitted": list(admitted_licensed_works)},
                    )
                    (rights,) = cur.fetchone()
            self._withheld_cache[key] = int(rights or 0)
        return Release(
            release_attempt_id=release.release_attempt_id, route_id=release.route_id,
            row_count=release.row_count, membership_counts=release.membership_counts,
            withheld_rights=self._withheld_cache[key], manifest_sha256=release.manifest_sha256,
            optional_columns=release.optional_columns, profile_sha256=dict(release.profile_sha256),
        )

    # -------------------------------------------------------- search

    # The store contract's read side. Do not "improve" it: the
    # order (cosine, then chunk_id) is the loop's exact-scan order, so ties break the same way.
    # Equality with the numpy exact scan was MEASURED (`test_store.py::
    # test_sql_pool_equals_the_numpy_pool`, PostgreSQL 16.2 + pgvector 0.6.2): identical ids and
    # order for all three fixture questions in every mode; 2 of 17 random unit vectors showed a
    # near-tie swap (cosine gap ≤ 2.2e-8, float rounding). The predicate is exactly three lines — the pinned release,
    # `evidence_eligible`, and the rights allow-list. The contract removed the
    # `excluded_reasons` clause; the flags still travel into the packet's `warnings`, they just no
    # longer decide retrievability. No vector index at v0 — exact scan by design.
    #
    # The rights line is two-armed. `serve_rights` is derived from `store.serving_mode`, and
    # production adds the licensed works a compliance decision admitted by id. Both arms stay
    # parameterized; the `::text[]` cast is what lets the empty production list (and the always
    # empty rnd list) compare without Postgres having to guess an array element type.
    RIGHTS_CLAUSE = ("(rights_state = ANY(%(serve_rights)s::text[])"
                     " OR (rights_state = 'licensed' AND work_id = ANY(%(admitted)s::text[])))")
    # Production arm. Added only when the mode refuses a basis AND the release's table has the
    # column; without the column every basis is NULL and the arm could not remove anything.
    # `IS DISTINCT FROM` keeps NULL-basis rows (ordinary evidence); the value is a bound parameter
    # (production: 'ocr_admitted_unverified').
    BASIS_CLAUSE = "evidence_basis IS DISTINCT FROM %(refused_basis)s::text"
    SEARCH_SQL = """
        SELECT chunk_id, parent_id, work_id, membership, rights_state, excluded_reasons,
               locator_ids, lineage_ids, payload, payload_sha256,
               {evidence_basis} AS evidence_basis, {page_start} AS page_start, {page_end} AS page_end,
               {context} AS context, {represented_text} AS represented_text,
               1 - (embedding <=> %(q)s) AS score
        FROM {schema}.items
        WHERE release_attempt_id = %(active)s
          AND evidence_eligible
          AND {rights}{basis}
        ORDER BY embedding <=> %(q)s, chunk_id
        LIMIT %(limit)s
    """

    def search_sql(self, columns: Sequence[str], refused_evidence_basis: str | None) -> str:
        """The statement for one release's column layout and one mode (optional OCR/page columns)."""
        selected = {name: (name if name in columns else placeholder)
                    for name, placeholder in DETECTED_ITEM_COLUMNS.items()}
        basis = ""
        if refused_evidence_basis is not None and "evidence_basis" in columns:
            basis = f"\n          AND {self.BASIS_CLAUSE}"
        return self.SEARCH_SQL.format(schema=self.db_schema, rights=self.RIGHTS_CLAUSE,
                                      basis=basis, **selected)
    # `meta.read_predicate_sha256` is computed over this description, not over the SQL text, so a
    # whitespace edit does not look like a contract change but a predicate edit does (the predicate digest).
    PREDICATE_NAME = "active+evidence_eligible+rights"
    PREDICATE_CONTRACT = "v0.1"

    def search(
        self,
        qvec: Sequence[float],
        release_attempt_id: str,
        serve_rights: Sequence[str],
        limit: int = CANDIDATE_POOL,
        admitted_licensed_works: Sequence[str] = (),
        refused_evidence_basis: str | None = None,
    ) -> list[Item]:
        vector = np.asarray(qvec, dtype=np.float32)
        params = {
            "q": vector, "active": release_attempt_id, "serve_rights": list(serve_rights),
            "admitted": list(admitted_licensed_works), "limit": int(limit),
            "refused_basis": refused_evidence_basis,
        }
        columns = self.optional_columns(release_attempt_id)
        # The vector type is registered once per pooled connection (`_register_vector_types`).
        with _driver_errors("search"):
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute(self.search_sql(columns, refused_evidence_basis), params)
                rows = cur.fetchall()
        return [self._row_item(row, rank) for rank, row in enumerate(rows, start=1)]

    @staticmethod
    def _row_item(row: Sequence[Any], rank: int) -> Item:
        """One SEARCH_SQL / FETCH_SQL row as an Item; `rank` is its dense rank (0 = not from the dense leg)."""
        (chunk_id, parent_id, work_id, membership, rights_state, excluded_reasons_col,
         locator_ids, lineage_ids, payload, payload_sha256, evidence_basis, page_start,
         page_end, context, represented_text, score) = row
        basis, start, end = optional_fields(evidence_basis, page_start, page_end, chunk_id)
        item = Item(
            chunk_id=chunk_id, parent_id=parent_id, work_id=work_id, membership=membership,
            rights_state=rights_state,
            payload=decode_payload(payload, payload_sha256, chunk_id),
            payload_sha256=payload_sha256,
            locator_ids=_json_list(locator_ids), lineage_ids=_json_list(lineage_ids),
            excluded_reasons=_json_list(excluded_reasons_col),
            score=float(score), dense_rank=rank, rank=rank,
            evidence_basis=basis, page_start=start, page_end=end,
        )
        item.context = context_value(context, chunk_id)
        item.represented_text = decode_represented_text(represented_text, chunk_id)
        return item

    def warm_up(self, release_attempt_id: str, serve_rights: Sequence[str],
                admitted_licensed_works: Sequence[str] = (), refused_evidence_basis: str | None = None,
                *, timeout_ms: int = 300_000) -> dict[str, Any]:
        """One served-shape exact search over the release, with a long statement timeout for THIS query only.

        Startup warm-up: the first live search on a cold database cache can exceed the
        served 5 s statement timeout (2.1 s cold at 49k rows; the store is now ~330k). `SET LOCAL` limits
        the longer timeout to this transaction; every request keeps the configured one. The query vector
        is one stored embedding of the release, so no model is needed."""
        columns = self.optional_columns(release_attempt_id)
        with _driver_errors("warm-up"):
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)}")
                cur.execute(f"SELECT embedding FROM {self.db_schema}.items"
                            " WHERE release_attempt_id = %(active)s LIMIT 1", {"active": release_attempt_id})
                rows = cur.fetchall()
                if not rows:
                    return {"rows": 0, "timeout_ms": int(timeout_ms)}
                cur.execute(self.search_sql(columns, refused_evidence_basis),
                            {"q": rows[0][0], "active": release_attempt_id, "serve_rights": list(serve_rights),
                             "admitted": list(admitted_licensed_works), "limit": CANDIDATE_POOL,
                             "refused_basis": refused_evidence_basis})
                return {"rows": len(cur.fetchall()), "timeout_ms": int(timeout_ms)}

    # -------------------------------------------------------- hybrid retrieval reads
    # All three are plain SELECTs as `ceres_reader` (read only, no schema change). The
    # lexical index covers ALL rows of the release (statistics independent of serving mode); the
    # read predicate is applied to each leg before truncation, from `predicate_ids`; rows a
    # lexical leg needs that the dense leg did not return are read with the predicate IN the SQL.

    def lexical_corpus(self, release_attempt_id: str) -> tuple[list[str], list[Any]]:
        """`(chunk_ids, represented_texts)` of every row of the release, chunk_id order; NULL texts
        come back as None (the index refuses them). A release without the column is refused."""
        if "represented_text" not in self.optional_columns(release_attempt_id):
            raise StoreError(f"release {release_attempt_id} has no items.represented_text column;"
                             " the hybrid lexical leg indexes it")
        with _driver_errors("lexical corpus"):
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute(f"SET LOCAL statement_timeout = {max(self.statement_timeout_ms, 300000)}")
                cur.execute(f"SELECT chunk_id, represented_text FROM {self.db_schema}.items"
                            " WHERE release_attempt_id = %(active)s ORDER BY chunk_id",
                            {"active": release_attempt_id})
                rows = cur.fetchall()
        return [row[0] for row in rows], [row[1] for row in rows]

    def predicate_ids(self, release_attempt_id: str, serve_rights: Sequence[str],
                      admitted_licensed_works: Sequence[str] = (),
                      refused_evidence_basis: str | None = None) -> frozenset[str]:
        """The chunk ids the read predicate passes (release, evidence_eligible, rights, basis arm)."""
        columns = self.optional_columns(release_attempt_id)
        basis = (f" AND {self.BASIS_CLAUSE}"
                 if refused_evidence_basis is not None and "evidence_basis" in columns else "")
        params = {"active": release_attempt_id, "serve_rights": list(serve_rights),
                  "admitted": list(admitted_licensed_works), "refused_basis": refused_evidence_basis}
        with _driver_errors("predicate ids"):
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute(f"SET LOCAL statement_timeout = {max(self.statement_timeout_ms, 300000)}")
                cur.execute(f"SELECT chunk_id FROM {self.db_schema}.items WHERE release_attempt_id = %(active)s"
                            f" AND evidence_eligible AND {self.RIGHTS_CLAUSE}{basis}", params)
                return frozenset(row[0] for row in cur.fetchall())

    FETCH_SQL = """
        SELECT chunk_id, parent_id, work_id, membership, rights_state, excluded_reasons,
               locator_ids, lineage_ids, payload, payload_sha256,
               {evidence_basis} AS evidence_basis, {page_start} AS page_start, {page_end} AS page_end,
               {context} AS context, {represented_text} AS represented_text,
               1 - (embedding <=> %(q)s) AS score
        FROM {schema}.items
        WHERE release_attempt_id = %(active)s
          AND chunk_id = ANY(%(ids)s::text[])
          AND evidence_eligible
          AND {rights}{basis}
    """

    def fetch(self, chunk_ids: Sequence[str], qvec: Sequence[float], release_attempt_id: str,
              serve_rights: Sequence[str], admitted_licensed_works: Sequence[str] = (),
              refused_evidence_basis: str | None = None) -> dict[str, Item]:
        """Rows by id under the SAME read predicate as `search` (in the SQL), keyed by chunk_id; an id
        the predicate does not pass is simply absent. `score` is the cosine to `qvec`; `dense_rank` is 0."""
        if not chunk_ids:
            return {}
        columns = self.optional_columns(release_attempt_id)
        selected = {name: (name if name in columns else placeholder)
                    for name, placeholder in DETECTED_ITEM_COLUMNS.items()}
        basis = (f"\n          AND {self.BASIS_CLAUSE}"
                 if refused_evidence_basis is not None and "evidence_basis" in columns else "")
        sql = self.FETCH_SQL.format(schema=self.db_schema, rights=self.RIGHTS_CLAUSE, basis=basis, **selected)
        params = {"q": np.asarray(qvec, dtype=np.float32), "active": release_attempt_id,
                  "ids": list(chunk_ids), "serve_rights": list(serve_rights),
                  "admitted": list(admitted_licensed_works), "refused_basis": refused_evidence_basis}
        with _driver_errors("fetch"):
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return {item.chunk_id: item for item in (self._row_item(row, 0) for row in rows)}

    def health(self) -> tuple[str, str]:
        try:
            release = self.active_release()
        except StoreError as exc:
            # A wrapped driver failure means the database is unreachable ("down"); a StoreError of
            # our own (zero or two active releases) means it answered inconsistently ("degraded").
            return ("down" if exc.__cause__ is not None else "degraded"), str(exc)
        except Exception as exc:                      # noqa: BLE001 — health never raises
            return "down", f"{exc.__class__.__name__}"
        return "ready", f"release {release.release_attempt_id} ({release.row_count} items)"


# ---------------------------------------------------------------- fixture store (tests, smoke)

class FixtureStore:
    """The same predicate and the same order, over an in-memory fixture.

    Loads `tests/fixtures/` (`items.jsonl` + `items.npy`, built from a local store) when
    the folder is there, and an inline fixture otherwise. It is the hermetic seam the CLI kept as
    `--fake-vectors` (the store contract's boundary): every stage above the store can be
    tested with no Postgres and no GPU.
    """

    # The columns the read predicate and the packet need. A row without one is a broken fixture,
    # and the store refuses it at construction instead of defaulting it: a permissive default
    # (missing `evidence_eligible` = eligible, missing release = the active one) would let a
    # fixture pass that the real SQL would filter, which defeats the fixture's job as the
    # contract oracle for `PgStore`.
    REQUIRED_FIELDS = ("release_attempt_id", "chunk_id", "parent_id", "work_id", "membership",
                       "rights_state", "evidence_eligible", "payload_sha256")

    def __init__(self, items: Iterable[dict[str, Any]], vectors: np.ndarray, release: Release) -> None:
        self.rows = [dict(row) for row in items]
        for index, row in enumerate(self.rows):
            missing = [name for name in self.REQUIRED_FIELDS if row.get(name) is None]
            if row.get("payload") is None and not row.get("payload_b64"):
                missing.append("payload")
            if missing:
                raise StoreError(f"fixture row {index} lacks {', '.join(missing)}")
            if not isinstance(row["evidence_eligible"], bool):
                raise StoreError(f"fixture row {index}: evidence_eligible must be a boolean")
        self.vectors = np.asarray(vectors, dtype=np.float32)
        if len(self.rows) != self.vectors.shape[0]:
            raise StoreError(
                f"fixture mismatch: {len(self.rows)} items against {self.vectors.shape[0]} vectors"
            )
        # A fixture whose rows carry the `represented_text` key is a release WITH the
        # column (a row may still hold None = SQL NULL); without the key the release lacks it. Only
        # this column is inferred from the rows, so every existing fixture's release is unchanged.
        if (any("represented_text" in row for row in self.rows)
                and "represented_text" not in release.optional_columns):
            from dataclasses import replace

            release = replace(release, optional_columns=(*release.optional_columns, "represented_text"))
        self._release = release

    # -------------------------------------------------------- loading

    @classmethod
    def from_dir(cls, directory: Path) -> "FixtureStore":
        """Load `items.jsonl` + `items.npy`; `MANIFEST.json` names the release when present."""
        directory = Path(directory)
        rows = [json.loads(line) for line in
                (directory / "items.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        vectors = np.load(directory / "items.npy").astype(np.float32)
        manifest_path = directory / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        release_id = (
            manifest.get("release_attempt_id")
            or (manifest.get("store") or {}).get("release_attempt_id")
            or (rows[0].get("release_attempt_id") if rows else None)
            or "fixture-release"
        )
        counts: dict[str, int] = {}
        for row in rows:
            membership = row.get("membership", "unknown")   # a missing value is refused in __init__
            counts[membership] = counts.get(membership, 0) + 1
        return cls(rows, vectors, Release(
            release_attempt_id=release_id,
            route_id=manifest.get("route_id", "unknown"),
            row_count=len(rows), membership_counts=counts,
        ))

    # -------------------------------------------------------- Store protocol

    def active_release(self) -> Release:
        return self._release

    def release(self, release_attempt_id: str) -> Release | None:
        """`PgStore.release` over the fixture: its one release by id, else None (eval only)."""
        return self._release if release_attempt_id == self._release.release_attempt_id else None

    def withheld(self, release: Release, serve_rights: Sequence[str],
                 admitted_licensed_works: Sequence[str] = ()) -> Release:
        rights = sum(1 for row in self.rows
                     if not serves(row["rights_state"], row["work_id"], serve_rights,
                                   admitted_licensed_works))
        return Release(
            release_attempt_id=release.release_attempt_id, route_id=release.route_id,
            row_count=release.row_count, membership_counts=release.membership_counts,
            withheld_rights=rights, profile_sha256=dict(release.profile_sha256),
            optional_columns=release.optional_columns,
        )

    def search(
        self,
        qvec: Sequence[float],
        release_attempt_id: str,
        serve_rights: Sequence[str],
        limit: int = CANDIDATE_POOL,
        admitted_licensed_works: Sequence[str] = (),
        refused_evidence_basis: str | None = None,
    ) -> list[Item]:
        vector = np.asarray(qvec, dtype=np.float32)
        if vector.shape[0] != self.vectors.shape[1]:
            raise StoreError(
                f"query vector has {vector.shape[0]} dimensions, the fixture has "
                f"{self.vectors.shape[1]}"
            )
        # Cosine similarity. The fixture vectors are L2-normalized (embed profile
        # `normalization_version: l2-float32-v1`), so the dot product is the cosine; normalize
        # anyway, because a hand-written test fixture need not be.
        norms = np.linalg.norm(self.vectors, axis=1)
        norms[norms == 0] = 1.0
        qnorm = float(np.linalg.norm(vector)) or 1.0
        scores = (self.vectors @ vector) / (norms * qnorm)

        keep: list[tuple[float, str, int]] = []
        for index, row in enumerate(self.rows):
            if self._passes(row, release_attempt_id, serve_rights, admitted_licensed_works,
                            refused_evidence_basis):
                keep.append((float(scores[index]), row["chunk_id"], index))
        # `ORDER BY embedding <=> q, chunk_id` = descending similarity, ascending chunk_id.
        keep.sort(key=lambda t: (-t[0], t[1]))
        return [self._row_item(self.rows[index], score, rank)
                for rank, (score, _chunk_id, index) in enumerate(keep[: int(limit)], start=1)]

    @staticmethod
    def _passes(row: dict[str, Any], release_attempt_id: str, serve_rights: Sequence[str],
                admitted_licensed_works: Sequence[str], refused_evidence_basis: str | None) -> bool:
        """The SQL read predicate, line for line; every field is present (checked in __init__)."""
        if row["release_attempt_id"] != release_attempt_id:
            return False
        if row["evidence_eligible"] is not True:
            return False
        # The same two-armed rights clause as `PgStore.RIGHTS_CLAUSE` .
        if not serves(row["rights_state"], row["work_id"], serve_rights, admitted_licensed_works):
            return False
        # Production arm, `evidence_basis IS DISTINCT FROM refused`: a row without the key
        # is a release without the column, and reads NULL.
        # No `excluded_reasons` clause : the flags reach the packet as `warnings`.
        return not (refused_evidence_basis is not None
                    and row.get("evidence_basis") == refused_evidence_basis)

    def _cosine(self, qvec: Sequence[float]) -> np.ndarray:
        vector = np.asarray(qvec, dtype=np.float32)
        norms = np.linalg.norm(self.vectors, axis=1)
        norms[norms == 0] = 1.0
        return (self.vectors @ vector) / (norms * (float(np.linalg.norm(vector)) or 1.0))

    def warm_up(self, release_attempt_id: str, serve_rights: Sequence[str],
                admitted_licensed_works: Sequence[str] = (), refused_evidence_basis: str | None = None,
                *, timeout_ms: int = 300_000) -> dict[str, Any]:
        """Nothing to warm in memory; the same shape as `PgStore.warm_up`."""
        return {"rows": 0, "timeout_ms": int(timeout_ms), "skipped": "fixture store"}

    # -------------------------------------------------------- hybrid retrieval reads (PgStore's, over the fixture)

    def lexical_corpus(self, release_attempt_id: str) -> tuple[list[str], list[Any]]:
        if "represented_text" not in self._release.optional_columns:
            raise StoreError(f"release {release_attempt_id} has no items.represented_text column;"
                             " the hybrid lexical leg indexes it")
        rows = sorted((row for row in self.rows if row["release_attempt_id"] == release_attempt_id),
                      key=lambda row: row["chunk_id"])
        return [row["chunk_id"] for row in rows], [row.get("represented_text") for row in rows]

    def predicate_ids(self, release_attempt_id: str, serve_rights: Sequence[str],
                      admitted_licensed_works: Sequence[str] = (),
                      refused_evidence_basis: str | None = None) -> frozenset[str]:
        return frozenset(row["chunk_id"] for row in self.rows
                         if self._passes(row, release_attempt_id, serve_rights, admitted_licensed_works,
                                         refused_evidence_basis))

    def fetch(self, chunk_ids: Sequence[str], qvec: Sequence[float], release_attempt_id: str,
              serve_rights: Sequence[str], admitted_licensed_works: Sequence[str] = (),
              refused_evidence_basis: str | None = None) -> dict[str, Item]:
        wanted = set(chunk_ids)
        scores = self._cosine(qvec) if wanted else None
        return {row["chunk_id"]: self._row_item(row, float(scores[index]), 0)
                for index, row in enumerate(self.rows)
                if row["chunk_id"] in wanted and self._passes(row, release_attempt_id, serve_rights,
                                                              admitted_licensed_works, refused_evidence_basis)}

    @staticmethod
    def _row_item(row: dict[str, Any], score: float, rank: int) -> Item:
        """One fixture row as an Item (`rank` = dense rank; 0 = not from the dense leg)."""
        payload = row.get("payload")
        if payload is None and row.get("payload_b64"):
            import base64
            payload = base64.b64decode(row["payload_b64"])
        basis, start, end = optional_fields(row.get("evidence_basis"), row.get("page_start"),
                                            row.get("page_end"), row["chunk_id"])
        item = Item(
            chunk_id=row["chunk_id"], parent_id=row["parent_id"], work_id=row["work_id"],
            membership=row["membership"], rights_state=row["rights_state"],
            payload=decode_payload(payload, row["payload_sha256"], row["chunk_id"]),
            payload_sha256=row["payload_sha256"],
            locator_ids=_json_list(row.get("locator_ids") or []),
            lineage_ids=_json_list(row.get("lineage_ids") or []),
            excluded_reasons=_json_list(row.get("excluded_reasons") or []),
            score=score, dense_rank=rank, rank=rank,
            evidence_basis=basis, page_start=start, page_end=end,
        )
        # A row without the key is a release without the `context` column: None.
        item.context = context_value(row.get("context"), row["chunk_id"])
        item.represented_text = decode_represented_text(row.get("represented_text"), row["chunk_id"])
        return item

    def health(self) -> tuple[str, str]:
        return "ready", f"fixture release {self._release.release_attempt_id} ({len(self.rows)} items)"
