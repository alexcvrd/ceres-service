"""Shared wiring: the inline fixture stack, a loadable config, and an ASGI client.

Two stores are available on purpose.
  * The inline fixture (`ceres.testing.build_inline_fixture`) is six crafted items where every dedup
    and predicate rule fires exactly once. Behavioural tests use it, because its expected outcome
    is readable in the fixture itself.
  * `tests/fixtures/` is 536 real items with the pipeline's own goldens. `test_parity.py` uses
    it, because only real goldens can prove Ceres reproduces route r1.

`tests/fixtures/` and `tools/` are read-only here: tests must not modify them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ceres import pipeline                                   # noqa: E402
from ceres.api import create_app                             # noqa: E402
from ceres.config import load_config                         # noqa: E402
from ceres.llm import FakeLLM                                # noqa: E402
from ceres.testing import build_inline_fixture, smoke_env     # noqa: E402  (one fake stack for all)

FIXTURES = ROOT / "tests" / "fixtures"
EXAMPLE_CONFIG = ROOT / "config.example.yaml"

requires_fixtures = pytest.mark.skipif(
    not (FIXTURES / "MANIFEST.json").is_file(),
    reason="the fixture corpus is not present (tests/fixtures/MANIFEST.json)",
)


# The golden answer rows were written by the FROZEN validator (route r1). Under
# ceres-validate-r2 one golden question changed: its attempt-1 quote "Elsewhere in Europe... temperatures
# were mildly above average (up to 3 °C)" passed every r2 guard, so r2 validated attempt 1. Under
# ceres-validate-r3 that change REVERTS: the omitted gap is ", except Ireland, southern Spain
# and Portugal, the Balkans and western and central Türkiye, " — a dropped place scope, the exact defect the
# r3 capitalized-word guard refuses. So r3 reproduces the golden rows unchanged and sends all six
# recorded attempts again. The r2 row is kept here only so a test can prove r3 refuses it.
R2_ONLY_GOLDEN_RESCUE = ("95d315efc4ecdbea6db2b9f02e9dcb9d9e78447a5b2cc285d75e1bbad58ee3b7", 1)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def env(tmp_path) -> dict[str, str]:
    """The environment `config.example.yaml` needs, rooted in this test's temp directory."""
    values = smoke_env(tmp_path)
    values.pop("CERES_API_KEY", None)          # tests that need a key set it themselves
    for name in list(values):
        if name.startswith("CERES_") and "__" in name:
            values.pop(name)                   # no inherited overrides in a test
    return values


@pytest.fixture
def config(env):
    """The shipped example config, loaded with authentication disabled: `serving_mode: rnd`."""
    return load_config(EXAMPLE_CONFIG, env, require_api_key=False)


@pytest.fixture
def production_env(env) -> dict[str, str]:
    """The production switch as an operator makes it: the mode and the decision that authorised it.

    Both go through `CERES_STORE__*` overrides, which is the shape the key takes on a real box
    (production is never the shipped default, and it never loads without a recorded
    compliance decision).
    """
    return dict(env, CERES_STORE__SERVING_MODE="production",
                CERES_STORE__PRODUCTION_DECISION="COMPLIANCE-EXAMPLE-1")


@pytest.fixture
def production_config(production_env):
    return load_config(EXAMPLE_CONFIG, production_env, require_api_key=False)


@pytest.fixture
def inline_stack():
    store, models, questions = build_inline_fixture()
    return store, models, questions


@pytest.fixture
def deps(config, inline_stack):
    store, models, _questions = inline_stack
    return pipeline.Deps.build(config, store, models, FakeLLM())


@pytest.fixture
def production_deps(production_config, inline_stack):
    """The same inline fixture served in production mode — c5 (`internal_research_only`) is out."""
    store, models, _questions = inline_stack
    return pipeline.Deps.build(production_config, store, models, FakeLLM())


@pytest.fixture
def app(deps, config):
    return create_app(deps, config)


@pytest.fixture
async def client(app):
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://ceres.test") as http:
        yield http


# ================================================================ disposable PostgreSQL
#
# pgserver 0.1.4 = PostgreSQL 16.2 + pgvector 0.6.2 in a wheel, no root (the same build as
# the operator CLI's pgserver tooling). One server per test session, in pytest's temp dir,
# started only when a test asks for the `pg` fixture. The CLI's contract DDL is copied below
# VERBATIM; `test_store.py::test_the_disposable_store_ddl_is_still_the_cli_ddl` compares it with
# the CLI file whenever that file is on disk.

import hashlib                                                # noqa: E402
from contextlib import contextmanager                         # noqa: E402

import numpy as np                                            # noqa: E402

CLI_LOAD_PY = ROOT.parent / "dds-operator-cli" / "dds" / "load.py"
DIM = 768                                                     # the operator CLI loader's DIM
IDENTIFIER = __import__("re").compile(r"^[a-z][a-z0-9_]*$")    # the operator CLI loader's IDENTIFIER


class LoadError(Exception):
    """Stand-in for `dds.load.LoadError`, so the copied function runs unchanged."""


# ---- copied verbatim from the operator CLI's `dds/load.py` (`ddl`),
# ---- file sha256 df512c1a48e8cb1796e45c911bd27dfa79bcf3834b714aca9c1c5f33388b0bc7. Do not edit.
def ddl(schema: str) -> list[str]:
    """Additive DDL for the Ceres serving store v0. Nothing here drops or rewrites anything."""
    if not IDENTIFIER.match(schema):
        raise LoadError(f"store.schema is not a plain identifier: {schema!r}")
    return [
        f"CREATE SCHEMA IF NOT EXISTS {schema}",
        f"""CREATE TABLE IF NOT EXISTS {schema}.releases (
            release_attempt_id text PRIMARY KEY,
            route_id text NOT NULL,
            chunk_profile_sha256 text NOT NULL,
            represent_profile_sha256 text NOT NULL,
            embed_profile_sha256 text NOT NULL,
            embed_attempt_dir text NOT NULL,
            row_count integer NOT NULL,
            loaded_at timestamptz NOT NULL DEFAULT now(),
            loaded_by text NOT NULL,
            manifest jsonb NOT NULL,
            active boolean NOT NULL DEFAULT false)""",
        f"ALTER TABLE {schema}.releases ADD COLUMN IF NOT EXISTS active boolean NOT NULL DEFAULT false",
        f"""CREATE TABLE IF NOT EXISTS {schema}.items (
            release_attempt_id text NOT NULL REFERENCES {schema}.releases(release_attempt_id),
            chunk_id text NOT NULL,
            parent_id text NOT NULL,
            work_id text NOT NULL,
            membership text NOT NULL,
            rights_state text NOT NULL,
            eligibility_state text NOT NULL,
            evidence_eligible boolean NOT NULL,
            payload bytea NOT NULL,
            payload_sha256 text NOT NULL,
            locator_ids jsonb NOT NULL,
            lineage_ids jsonb NOT NULL,
            excluded_reasons jsonb NOT NULL,
            representation_id text NOT NULL,
            embedding vector({DIM}) NOT NULL,
            PRIMARY KEY (release_attempt_id, chunk_id))""",
        f"ALTER TABLE {schema}.items ADD COLUMN IF NOT EXISTS membership text NOT NULL DEFAULT 'unknown'",
    ]


# ---- copied verbatim from the operator CLI's `dds/load.py` (`grant_statements`), same file.
def grant_statements(schema: str, roles: Sequence[str]) -> list[Any]:
    """The read grants for `store.read_roles`, as psycopg `sql.Composed` statements.

    Per role: USAGE on the schema, SELECT on every table it holds now, and the default privilege
    so a table a later load creates is readable too. Every one is idempotent in PostgreSQL, so
    they run on every load. Both identifiers are quoted with `sql.Identifier`, never formatted in.
    """
    from psycopg import sql

    target = sql.Identifier(schema)
    statements: list[Any] = []
    for role in roles:
        grantee = sql.Identifier(role)
        statements += [
            sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(target, grantee),
            sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(target, grantee),
            sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA {} GRANT SELECT ON TABLES TO {}").format(target, grantee),
        ]
    return statements
# ---- end of the verbatim copies.


# The optional OCR-basis columns: three nullable columns, `page_start`/`page_end` being
# the 0-based, inclusive PDF page span of the chunk's source section (NOT the first announcement's
# `page_indexes int[]`). The CLI's `ddl()` does not carry them yet (as read above), so the "present"
# layout adds them here with the exact statement the loader will run.
OPTIONAL_COLUMNS_DDL = ("ALTER TABLE {schema}.items ADD COLUMN IF NOT EXISTS evidence_basis text,"
                        " ADD COLUMN IF NOT EXISTS page_start integer,"
                        " ADD COLUMN IF NOT EXISTS page_end integer",)
# Per-column types, for layouts with only SOME of the optional columns (detection must tolerate any subset).
OPTIONAL_COLUMN_TYPES = {"evidence_basis": "text", "page_start": "integer", "page_end": "integer"}
READER_ROLE = "ceres_ro_test"
FIXTURE_RELEASE = "fixture-intake-mars-2026-cli-acceptance/release-attempt-0003"
EXTENDED_RELEASE = "c2-extended/release-attempt-0001"
RND_RIGHTS = ("redistributable", "internal_research_only", "licensed")
PRODUCTION_RIGHTS = ("redistributable",)


def fixture_rows() -> list[dict]:
    return read_jsonl(FIXTURES / "items.jsonl")


def fixture_vectors() -> np.ndarray:
    return np.load(FIXTURES / "items.npy").astype(np.float32)


def fixture_query_vectors() -> np.ndarray:
    return np.load(FIXTURES / "query-vectors.npy").astype(np.float32)


def extended_rows(*, with_basis: bool) -> list[dict]:
    """A test-only copy of the fixture with every predicate arm exercised (tests/fixtures untouched).

    By work (the fixture's five works, sorted): the 1st becomes `licensed` (the one production may admit),
    the 2nd `internal_research_only`, the 3rd carries `ocr_admitted_unverified` with
    a page span on every other row (NULL on the rest; `page_end - page_start` cycles 0, 1, 2 so a
    single-page span occurs) when `with_basis`, and every 10th row of
    the whole copy is `evidence_eligible = false`.
    """
    rows = [dict(row) for row in fixture_rows()]
    works = sorted({row["work_id"] for row in rows})
    licensed, internal, ocr = works[0], works[1], works[2]
    for index, row in enumerate(rows):
        row["release_attempt_id"] = EXTENDED_RELEASE
        if row["work_id"] == licensed:
            row["rights_state"] = "licensed"
        elif row["work_id"] == internal:
            row["rights_state"] = "internal_research_only"
        if index % 10 == 0:
            row["evidence_eligible"] = False
        if with_basis and row["work_id"] == ocr:
            row["evidence_basis"] = "ocr_admitted_unverified"
            if index % 2 == 0:
                row["page_start"], row["page_end"] = index % 40, index % 40 + index % 3
            else:
                row["page_start"] = row["page_end"] = None
    return rows


def extended_works() -> dict[str, str]:
    works = sorted({row["work_id"] for row in fixture_rows()})
    return {"licensed": works[0], "internal": works[1], "ocr": works[2]}


class PgHarness:
    """The disposable server: an admin (superuser) connection and a read-only reader role."""

    def __init__(self, server) -> None:
        self.server = server
        self.admin_dsn = server.get_uri()
        host = self.admin_dsn.split("host=", 1)[1]
        self.reader_dsn = f"postgresql://{READER_ROLE}@/postgres?host={host}"
        self._stores: list[Any] = []

    @contextmanager
    def admin(self, *, vector: bool = True):
        import psycopg
        from pgvector.psycopg import register_vector

        with psycopg.connect(self.admin_dsn, autocommit=True) as conn:
            if vector:
                register_vector(conn)
            yield conn

    def create_schema(self, schema: str, *, optional_columns: bool | Sequence[str] = False) -> None:
        """`optional_columns`: True = the loader's statement; a list = only those columns."""
        with self.admin() as conn:
            for statement in ddl(schema):
                conn.execute(statement)
            if optional_columns is True:
                for statement in OPTIONAL_COLUMNS_DDL:
                    conn.execute(statement.format(schema=schema))
            elif optional_columns:
                for name in optional_columns:
                    conn.execute(f"ALTER TABLE {schema}.items ADD COLUMN IF NOT EXISTS"
                                 f" {name} {OPTIONAL_COLUMN_TYPES[name]}")
            for statement in grant_statements(schema, [READER_ROLE]):
                conn.execute(statement)

    def load(self, schema: str, release_id: str, rows: Sequence[dict], vectors: np.ndarray, *,
             active: bool = False, manifest: dict | None = None, conn=None,
             profile_shas: dict[str, str] | None = None) -> None:
        """Insert one release and its items the way `dds load` shapes them (payload = UTF-8 bytes).

        `profile_shas` ({chunk, represent, embed} → sha256) fills the release row's three NOT NULL
        profile columns; the default is the placeholder shas 0…/1…/2… no configured route has."""
        shas = dict({"chunk": "0" * 64, "represent": "1" * 64, "embed": "2" * 64}, **(profile_shas or {}))
        from psycopg.types.json import Jsonb

        counts: dict[str, int] = {}
        for row in rows:
            counts[row["membership"]] = counts.get(row["membership"], 0) + 1
        manifest = dict(manifest or {"load_id": f"c2-{release_id}"}, membership_counts=counts)
        with (self.admin() if conn is None else _same(conn)) as db:
            columns = {r[0] for r in db.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = %s"
                " AND table_name = 'items'", (schema,)).fetchall()}
            db.execute(
                f"INSERT INTO {schema}.releases (release_attempt_id, route_id, chunk_profile_sha256,"
                " represent_profile_sha256, embed_profile_sha256, embed_attempt_dir, row_count,"
                " loaded_by, manifest, active) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, false)",
                (release_id, "c2-route", shas["chunk"], shas["represent"], shas["embed"], "/nowhere", len(rows),
                 "c2-test", Jsonb(manifest)))
            names = ["release_attempt_id", "chunk_id", "parent_id", "work_id", "membership",
                     "rights_state", "eligibility_state", "evidence_eligible", "payload",
                     "payload_sha256", "locator_ids", "lineage_ids", "excluded_reasons",
                     "representation_id", "embedding"]
            optional = [name for name in OPTIONAL_COLUMN_TYPES if name in columns]
            payload_column = "payload" if "payload" in columns else "payload_bytes"
            placeholders = ", ".join(["%s"] * (len(names) + len(optional)))
            statement = (f"INSERT INTO {schema}.items ("
                         + ", ".join([n if n != "payload" else payload_column for n in names] + optional)
                         + f") VALUES ({placeholders})")
            values = []
            for row, vector in zip(rows, vectors):
                payload = row.get("payload_bytes")
                if payload is None:
                    payload = row["payload"].encode("utf-8")
                values.append([
                    release_id, row["chunk_id"], row["parent_id"], row["work_id"], row["membership"],
                    row["rights_state"], row.get("eligibility_state", "eligible"),
                    row["evidence_eligible"], payload, row["payload_sha256"],
                    Jsonb(row.get("locator_ids") or []), Jsonb(row.get("lineage_ids") or []),
                    Jsonb(row.get("excluded_reasons") or []), row.get("representation_id", "c2"),
                    np.asarray(vector, dtype=np.float32),
                ] + [row.get(name) for name in optional])
            with db.cursor() as cur:
                cur.executemany(statement, values)
            if active:
                db.execute(f"UPDATE {schema}.releases SET active = (release_attempt_id = %s)",
                           (release_id,))

    def activate(self, schema: str, release_id: str | None) -> None:
        """`dds load --activate` (`load.py:743`); None deactivates every release."""
        with self.admin() as conn:
            conn.execute(f"UPDATE {schema}.releases SET active = (release_attempt_id = %s)",
                         (release_id,))

    def store(self, schema: str = "ceres_serve_v0", dsn: str | None = None, **kwargs):
        from ceres.store import PgStore

        kwargs.setdefault("pool_timeout_s", 5.0)
        store = PgStore(dsn or self.reader_dsn, schema, **kwargs)
        self._stores.append(store)
        return store

    def close(self) -> None:
        for store in self._stores:
            try:
                store.close()
            except Exception:                                   # noqa: BLE001
                pass


@contextmanager
def _same(conn):
    yield conn


@pytest.fixture(scope="session")
def pg(tmp_path_factory):
    """The session's disposable PostgreSQL, seeded from the fixture corpus.

    `ceres_serve_v0` (contract DDL only, optional columns ABSENT): the fixture release (active) and
    the extended copy (inactive, no basis possible). `c2_ocr` (DDL + the three optional columns PRESENT):
    the extended copy with OCR rows, active. Tests that mutate state make their own schema.
    """
    import pgserver

    server = pgserver.get_server(tmp_path_factory.mktemp("pg"), cleanup_mode="delete")
    harness = PgHarness(server)
    with harness.admin(vector=False) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(f"CREATE ROLE {READER_ROLE} LOGIN")
    vectors = fixture_vectors()
    harness.create_schema("ceres_serve_v0")
    harness.load("ceres_serve_v0", FIXTURE_RELEASE, fixture_rows(), vectors, active=True)
    harness.load("ceres_serve_v0", EXTENDED_RELEASE, extended_rows(with_basis=False), vectors)
    harness.create_schema("c2_ocr", optional_columns=True)
    harness.load("c2_ocr", EXTENDED_RELEASE, extended_rows(with_basis=True), vectors, active=True)
    yield harness
    harness.close()
    server.cleanup()


# ================================================================ PgStore without Postgres
# (shared by test_store.py and test_api.py)

class FakeCursor:
    def __init__(self, pool):
        self.pool = pool
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.pool.executed.append(sql)
        if "count(*)" in sql:
            rights = tuple(params["serve_rights"])
            self._rows = [(self.pool.withheld_by_rights.get(rights, 0),)]
        elif "FROM" in sql and ".releases" in sql:
            # `PgStore.RELEASE_COLUMNS` order: id, route, row_count, manifest, chunk/represent/embed shas.
            self._rows = [("rel-1", "route-x", 3, {"membership_counts": {"train_eligible": 3}},
                           "a" * 64, "b" * 64, "c" * 64)]
        else:
            self._rows = []

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0]


class FakeConnection:
    def __init__(self, pool):
        self.pool = pool

    def cursor(self):
        return FakeCursor(self.pool)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakePool:
    """Stands in for `psycopg_pool.ConnectionPool`: records statements, can raise on connect."""

    def __init__(self, raises: Exception | None = None, withheld_by_rights=None):
        self.raises = raises
        self.executed: list[str] = []
        self.withheld_by_rights = dict(withheld_by_rights or {})
        self.closed = False

    @contextmanager
    def connection(self):
        if self.raises is not None:
            raise self.raises
        yield FakeConnection(self)

    def close(self):
        self.closed = True


def pg_store(pool: FakePool):
    from ceres.store import PgStore

    store = PgStore("postgresql://ceres_reader@127.0.0.1/veridian_rag", open_pool=False)
    store._pool = pool
    return store
