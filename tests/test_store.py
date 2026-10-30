"""The store adapters: `PgStore` against real PostgreSQL, `FixtureStore` as its numpy oracle.

Every test names the bug it catches. Four sections:

  1. PgStore without Postgres (fake pool) and FixtureStore as a contract oracle — moved here
     from an earlier regression file, unchanged in substance.
  2. The optional OCR columns in the fixture store and the predicate identity.
  3. Disposable PostgreSQL 16.2 + pgvector 0.6.2 (pgserver; the `pg` fixture in conftest) for the
     cases the real store cannot produce: broken releases, corrupt bytes, timeouts, refused connections,
     exhausted pools, schema drift, mid-request switches, and SQL-vs-numpy parity.
  4. Opt-in live run against the serving database, marker `live_store`, skipped unless CERES_TEST_LIVE_DSN is set.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pytest

from ceres import pipeline
from ceres.api import create_app
from ceres.config import load_config
from ceres.contracts import AskRequest
from ceres.llm import FakeLLM
from ceres.models import FakeModels
from ceres.store import (
    CANDIDATE_POOL,
    OCR_ADMITTED_UNVERIFIED,
    OPTIONAL_ITEM_COLUMNS,
    FixtureStore,
    PgStore,
    Release,
    StoreError,
    decode_payload,
    read_predicate_sha256,
)

from conftest import (
    CLI_LOAD_PY,
    EXAMPLE_CONFIG,
    EXTENDED_RELEASE,
    FIXTURE_RELEASE,
    OPTIONAL_COLUMNS_DDL,
    PRODUCTION_RIGHTS,
    ROOT,
    RND_RIGHTS,
    FakePool,
    ddl,
    extended_rows,
    extended_works,
    fixture_query_vectors,
    fixture_rows,
    fixture_vectors,
    pg_store,
)

SENTINEL = "UPSTREAM_DETAIL_SENTINEL http://10.9.8.7:8215/v1 secret-path"
RND = list(RND_RIGHTS)
QUERY = np.array([1.0, 0.0], dtype=np.float32)


# ================================================================ 1a. PgStore without Postgres

def test_a_driver_error_becomes_a_store_error():
    """psycopg exceptions are normalized at the adapter boundary."""
    import psycopg
    import psycopg_pool

    for error in (psycopg.OperationalError(f"connection failed: {SENTINEL}"),
                  psycopg_pool.PoolTimeout("couldn't get a connection after 30 sec"),
                  TimeoutError("socket")):
        store = pg_store(FakePool(raises=error))
        with pytest.raises(StoreError) as caught:
            store.active_release()
        assert caught.value.__cause__ is error          # the private cause is kept
        assert "UPSTREAM_DETAIL_SENTINEL" not in str(caught.value)
        with pytest.raises(StoreError):
            store.search(np.zeros(8, dtype=np.float32), "rel-1", ["redistributable"])


def test_correctly_hashed_invalid_utf8_is_a_store_error():
    """Bug caught: bytes that hash correctly but are not UTF-8 decoded with replacement chars."""
    payload = b"\xff\xfe\xfa not utf-8"
    with pytest.raises(StoreError):
        decode_payload(payload, hashlib.sha256(payload).hexdigest(), "c" * 64)


def test_a_zero_withheld_count_is_cached():
    """Zero is the common rnd answer and must not re-run the aggregate."""
    pool = FakePool()
    store = pg_store(pool)
    release = store.active_release()
    assert store.withheld(release, RND).withheld_rights == 0
    assert store.withheld(release, RND).withheld_rights == 0
    assert sum("count(*)" in sql for sql in pool.executed) == 1


def test_the_withheld_cache_is_not_shared_between_predicates():
    """A count cached for one mode must not answer for another (keyed by the digest)."""
    pool = FakePool(withheld_by_rights={("redistributable",): 5})
    store = pg_store(pool)
    release = store.active_release()
    assert store.withheld(release, ["redistributable"]).withheld_rights == 5
    assert store.withheld(release, RND).withheld_rights == 0


def test_vector_types_are_registered_once_per_pooled_connection(monkeypatch):
    """`register_vector` ran on every search (a catalog round trip per request)."""
    import pgvector.psycopg
    import psycopg_pool

    registered = []
    monkeypatch.setattr(pgvector.psycopg, "register_vector", lambda conn: registered.append(conn))
    created = {}

    class RecordingPool:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(psycopg_pool, "ConnectionPool", RecordingPool)
    store = PgStore("postgresql://ceres_reader@127.0.0.1/veridian_rag", open_pool=False)
    assert callable(created.get("configure")), "the pool must register types on connect"
    store._pool = FakePool()
    store.search(np.zeros(8, dtype=np.float32), "rel-1", ["redistributable"])
    store.search(np.zeros(8, dtype=np.float32), "rel-1", ["redistributable"])
    assert registered == []                       # never per search
    created["configure"](object())
    assert len(registered) == 1


# ================================================================ 1b. FixtureStore as a contract oracle

def _row(label, **overrides):
    text = f"Passage {label} about sowing density in winter wheat fields. " * 3
    row = {"release_attempt_id": "rel-1",
           "chunk_id": hashlib.sha256(f"c1c/{label}".encode()).hexdigest(),
           "parent_id": f"p-{label}", "work_id": f"w-{label}", "membership": "train_eligible",
           "rights_state": "redistributable", "evidence_eligible": True, "payload": text,
           "payload_sha256": hashlib.sha256(text.encode()).hexdigest(),
           "locator_ids": [], "lineage_ids": [], "excluded_reasons": []}
    row.update(overrides)
    return row


def _fixture(rows, vectors=None):
    vectors = np.asarray(vectors if vectors is not None else [[1.0, 0.0]] * len(rows), dtype=np.float32)
    return FixtureStore(rows, vectors, Release("rel-1", "fixture", len(rows), {}))


@pytest.mark.parametrize("missing", ["release_attempt_id", "evidence_eligible", "rights_state",
                                     "work_id"])
def test_the_fixture_store_refuses_a_row_missing_a_predicate_field(missing):
    """A missing field is an error, never a permissive default."""
    row = _row("a")
    del row[missing]
    with pytest.raises(StoreError, match=missing):
        _fixture([row])


def test_tied_scores_do_not_reorder_between_runs():
    """Bug caught: equal cosine scores ordered by insertion instead of the SQL's `chunk_id`."""
    rows = [_row("x"), _row("y"), _row("z")]
    items = _fixture(rows).search(QUERY, "rel-1", RND)
    assert [item.chunk_id for item in items] == sorted(row["chunk_id"] for row in rows)


def test_ineligible_unknown_rights_and_other_release_rows_are_not_served():
    rows = [_row("ok"), _row("ineligible", evidence_eligible=False),
            _row("unknown", rights_state="unknown"),
            _row("other-release", release_attempt_id="rel-0")]
    store = _fixture(rows)
    for rights in (RND, ["redistributable"]):
        items = store.search(QUERY, "rel-1", rights)
        assert [item.chunk_id for item in items] == [rows[0]["chunk_id"]]


def test_a_corrupt_payload_fails_the_search_closed():
    rows = [_row("ok"), _row("corrupt", payload_sha256="0" * 64)]
    with pytest.raises(StoreError, match="payload_sha256"):
        _fixture(rows).search(QUERY, "rel-1", RND)


# ================================================================ 2. Optional OCR columns in the fixture store

def test_production_refuses_ocr_items_in_the_fixture_store_too():
    """The oracle must apply the same arm as the SQL, or parity proves nothing."""
    rows = [_row("plain"), _row("ocr", evidence_basis=OCR_ADMITTED_UNVERIFIED, page_start=4, page_end=5)]
    store = _fixture(rows)
    rnd = store.search(QUERY, "rel-1", RND)
    production = store.search(QUERY, "rel-1", ["redistributable"],
                              refused_evidence_basis=OCR_ADMITTED_UNVERIFIED)
    assert {item.evidence_basis for item in rnd} == {None, OCR_ADMITTED_UNVERIFIED}
    assert [item.chunk_id for item in production] == [rows[0]["chunk_id"]]
    ocr = next(item for item in rnd if item.evidence_basis)
    assert (ocr.page_start, ocr.page_end) == (4, 5)


def test_an_unknown_evidence_basis_is_refused_not_served():
    """A basis the contract does not know would pass the production arm (it names one value)
    and fail the packet schema's enum later with a 500. Refuse it at the store."""
    rows = [_row("new", evidence_basis="ocr_admitted_v2")]
    for refused in (None, OCR_ADMITTED_UNVERIFIED):
        with pytest.raises(StoreError, match="unknown evidence_basis"):
            _fixture(rows).search(QUERY, "rel-1", RND, refused_evidence_basis=refused)


def _ocr_span(page_start, page_end):
    return [_row("ocr", evidence_basis=OCR_ADMITTED_UNVERIFIED, page_start=page_start, page_end=page_end)]


def test_a_half_null_page_span_is_refused_not_logged():
    """One bound NULL is neither "unknown page base" (both NULL) nor a span; logging it would
    record a guessed or open-ended range. StoreError (→ 503)."""
    for start, end in ((3, None), (None, 3), (0, None), (None, 0)):
        with pytest.raises(StoreError, match="exactly one of page_start/page_end is NULL"):
            _fixture(_ocr_span(start, end)).search(QUERY, "rel-1", RND)


def test_a_negative_page_is_refused_not_logged():
    """Pages are 0-based; a negative bound is a broken load, not a page to cite."""
    for start, end in ((-1, 2), (-2, -1), (-1, -1)):
        with pytest.raises(StoreError, match="is negative"):
            _fixture(_ocr_span(start, end)).search(QUERY, "rel-1", RND)


def test_an_inverted_page_span_is_refused_not_logged():
    """`page_start > page_end` cannot be an inclusive span; logging it would cite nonsense."""
    for start, end in ((5, 4), (1, 0)):
        with pytest.raises(StoreError, match="page_start .* > page_end"):
            _fixture(_ocr_span(start, end)).search(QUERY, "rel-1", RND)


def test_a_non_integer_page_is_refused_not_logged():
    """A string, float, bool (an int subclass) or list is not a page number; accepting it
    would log `"3"`, `3.0`, `true` or `[3]` as a page."""
    for start, end in (("3", 4), (3, "4"), (3.0, 4), (1, 2.5), (True, 2), (0, True), ([3], [4])):
        with pytest.raises(StoreError, match="is not an integer"):
            _fixture(_ocr_span(start, end)).search(QUERY, "rel-1", RND)


def test_a_single_page_span_and_an_unknown_page_base_are_served():
    """Over-refusal: `start == end` (one page, page 0 included) and both NULL (legacy section,
    unknown page base) are valid; refusing them would 503 ordinary OCR citations."""
    for start, end in ((0, 0), (7, 7), (0, 3), (None, None)):
        items = _fixture(_ocr_span(start, end)).search(QUERY, "rel-1", RND)
        assert [(item.page_start, item.page_end) for item in items] == [(start, end)]


def test_the_ocr_arm_changes_the_production_digest_and_leaves_rnd_unchanged():
    """Production's predicate gained an arm, so its identity must change; rnd's did not."""
    rnd_before = read_predicate_sha256(RND, "ceres_serve_v0")
    rnd_after = read_predicate_sha256(RND, "ceres_serve_v0", refused_evidence_basis=None)
    production_before = read_predicate_sha256(["redistributable"], "ceres_serve_v0")
    production_after = read_predicate_sha256(["redistributable"], "ceres_serve_v0",
                                             refused_evidence_basis=OCR_ADMITTED_UNVERIFIED)
    assert rnd_after == rnd_before
    assert production_after != production_before


# ================================================================ 3. disposable PostgreSQL

def test_the_disposable_store_ddl_is_still_the_cli_ddl():
    """The copied contract DDL drifted from what the CLI loader creates.

    Every line of the copy must still be in the CLI's `ddl()`, in order. Lines the CLI ADDS are
    allowed (contract v0.2 adds optional columns) and are printed, so the next session sees them.
    """
    import inspect

    if not CLI_LOAD_PY.is_file():
        pytest.skip(f"the CLI checkout is not on this box: {CLI_LOAD_PY}")
    cli = CLI_LOAD_PY.read_text(encoding="utf-8")
    start = cli.index("def ddl(schema: str)")
    block = cli[start:cli.index("\ndef ", start + 1)]
    cli_lines = [line.strip() for line in block.splitlines() if line.strip()]
    ours = [line.strip() for line in inspect.getsource(ddl).splitlines() if line.strip()]
    position = 0
    for line in ours:
        assert line in cli_lines[position:], f"the CLI's ddl() no longer has: {line}"
        position = cli_lines.index(line, position) + 1
    added = [line for line in cli_lines if line not in ours]
    if added:
        print(f"\nCLI ddl() lines not in the disposable copy: {added}")


def _search_ids(store, qvec, release_id, rights, admitted=(), refused=None, limit=CANDIDATE_POOL):
    return [(item.chunk_id, item.score) for item in
            store.search(qvec, release_id, list(rights), limit, list(admitted),
                         refused_evidence_basis=refused)]


def _modes():
    licensed = extended_works()["licensed"]
    return {
        "rnd": (RND_RIGHTS, (), None),
        "production/admitted-empty": (PRODUCTION_RIGHTS, (), OCR_ADMITTED_UNVERIFIED),
        "production/admitted-licensed": (PRODUCTION_RIGHTS, (licensed,), OCR_ADMITTED_UNVERIFIED),
    }


# pgvector computes the cosine distance in its own float arithmetic, numpy in float32 then float64.
# Two rows whose cosines differ by less than this can legitimately come out in either order; the
# swap is then a near-tie and is REPORTED, never hidden. 1e-6 is ~45× the largest swap observed in
# the first runs (2.2e-8, i.e. float32 rounding of the numpy cosine).
NEAR_TIE = 1e-6


def test_sql_pool_equals_the_numpy_pool(pg):
    """PgStore.search ids AND order == FixtureStore.search on the same rows.

    Three releases × three modes × (3 fixture questions + 17 seeded random unit vectors): the
    fixture release; the extended copy with the optional columns ABSENT (`ceres_serve_v0`); the
    extended copy with them PRESENT and OCR rows (`c2_ocr`). The fixture questions must match
    exactly. A random vector may show a near-tie swap: allowed only when the two swapped rows'
    numpy cosines differ by ≤ NEAR_TIE, and printed with that gap. Anything else fails.
    """
    vectors = fixture_vectors()
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    rng = np.random.default_rng(20260924)
    fixture_questions = [q for q in fixture_query_vectors()]
    queries = fixture_questions + [
        v / np.linalg.norm(v) for v in rng.normal(size=(17, vectors.shape[1])).astype(np.float32)]
    layouts = {
        ("ceres_serve_v0", FIXTURE_RELEASE): fixture_rows(),
        ("ceres_serve_v0", EXTENDED_RELEASE): extended_rows(with_basis=False),
        ("c2_ocr", EXTENDED_RELEASE): extended_rows(with_basis=True),
    }
    failures, swaps, compared, exact = [], [], 0, 0
    for (schema, release_id), rows in layouts.items():
        sql = pg.store(schema)
        oracle = FixtureStore(rows, vectors, Release(release_id, "c2", len(rows), {}))
        for mode, (rights, admitted, refused) in _modes().items():
            for index, qvec in enumerate(queries):
                got = [c for c, _ in _search_ids(sql, qvec, release_id, rights, admitted, refused)]
                want = [c for c, _ in _search_ids(oracle, qvec, release_id, rights, admitted, refused)]
                compared += 1
                assert len(want) == CANDIDATE_POOL, f"{schema}/{mode}/q{index}: the pool is not full"
                if got == want:
                    exact += 1
                    continue
                q = qvec / np.linalg.norm(qvec)
                cosine = {row["chunk_id"]: float(unit[i] @ q) for i, row in enumerate(rows)}
                where = f"{schema}/{release_id} {mode} q{index}"
                if index < len(fixture_questions) or len(got) != len(want):
                    failures.append(f"{where}: differs on a fixture question or in length")
                    continue
                for rank, (a, b) in enumerate(zip(got, want), start=1):
                    if a != b:
                        gap = abs(cosine[a] - cosine[b])
                        swaps.append((where, rank, a[:12], b[:12], gap))
                        if gap > NEAR_TIE:
                            failures.append(f"{where} rank {rank}: sql {a[:12]} vs numpy {b[:12]},"
                                            f" cosine gap {gap:.3e} is not a near-tie")
    print(f"\nSQL-vs-numpy parity: {compared} pools compared, {exact} identical,"
          f" {compared - exact} with near-tie swaps; swapped positions:")
    for where, rank, a, b, gap in swaps:
        print(f"  {where} rank {rank}: sql {a}… numpy {b}… cosine gap {gap:.3e}")
    assert not failures, "\n".join(failures)


def test_empty_and_non_empty_text_array_parameters_execute(pg):
    """`::text[]` — an empty admission list must execute and a non-empty one must match."""
    store = pg.store("ceres_serve_v0")
    qvec = fixture_query_vectors()[0]
    licensed = extended_works()["licensed"]
    empty = store.search(qvec, EXTENDED_RELEASE, list(PRODUCTION_RIGHTS), 500, [])
    admitted = store.search(qvec, EXTENDED_RELEASE, list(PRODUCTION_RIGHTS), 500, [licensed])
    assert {item.rights_state for item in empty} == {"redistributable"}
    assert {item.rights_state for item in admitted} == {"redistributable", "licensed"}
    assert {item.work_id for item in admitted if item.rights_state == "licensed"} == {licensed}
    release = store.active_release()
    assert store.withheld(release, list(PRODUCTION_RIGHTS), []).withheld_rights == 0
    assert store.withheld(release, [], [licensed]).withheld_rights == len(fixture_rows())


def _app(config, store, stack=None):
    from ceres.testing import build_fixture_stack

    _store, models, _llm, asked = stack or build_fixture_stack()
    deps = pipeline.Deps.build(config, store, models, FakeLLM())
    return create_app(deps, config), asked


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ceres.test")


async def _ask_status(config, store) -> httpx.Response:
    app, asked = _app(config, store)
    question, lang = asked[0]
    async with _client(app) as http:
        return await http.post("/v1/ask", json={"question": question, "lang": lang})


def _two_release_schema(pg, schema, *, active):
    rows = fixture_rows()[:40]
    vectors = fixture_vectors()[:40]
    pg.create_schema(schema)
    pg.load(schema, "rel-a", rows, vectors)
    pg.load(schema, "rel-b", rows, vectors)
    with pg.admin() as conn:
        conn.execute(f"UPDATE {schema}.releases SET active = release_attempt_id = ANY(%s)",
                     (list(active),))


async def test_zero_active_releases_is_a_store_error_and_503(pg, config):
    """No active release must refuse, never fall back to another release."""
    _two_release_schema(pg, "c2_zero", active=[])
    store = pg.store("c2_zero")
    with pytest.raises(StoreError, match="0 active"):
        store.active_release()
    assert (await _ask_status(config, store)).status_code == 503


def test_the_release_manifest_is_read_from_the_named_release_not_the_active_one(pg):
    """ceres-eval pins a release by its manifest: reading another release's manifest, or inventing
    one for a missing release, would hide a reload under the same id."""
    _two_release_schema(pg, "c6_manifest", active=["rel-b"])
    with pg.admin() as conn:
        conn.execute("UPDATE c6_manifest.releases SET manifest = manifest || jsonb_build_object('marker', release_attempt_id)")
    store = pg.store("c6_manifest")
    assert store.release_manifest("rel-a")["marker"] == "rel-a"
    assert store.release_manifest("rel-b")["marker"] == "rel-b"
    assert store.release_manifest("no-such-release") is None


def test_a_named_release_is_read_whether_or_not_it_is_active(pg):
    """`release(id)` must return the INACTIVE release it names — the same Release
    (shape, manifest sha, optional columns) `active_release()` gives once it is activated — and
    None for an id the schema does not hold, never the active release instead."""
    _two_release_schema(pg, "v34_named", active=["rel-a"])
    with pg.admin() as conn:
        conn.execute("UPDATE v34_named.releases SET manifest = manifest || jsonb_build_object('marker', release_attempt_id)")
    store = pg.store("v34_named")
    staged = store.release("rel-b")
    assert staged is not None and staged.release_attempt_id == "rel-b"
    assert store.active_release().release_attempt_id == "rel-a"
    assert store.release("no-such-release") is None
    pg.activate("v34_named", "rel-b")
    fresh = pg.store("v34_named")                     # no cache shared with the first reader
    assert fresh.active_release() == staged


def test_a_named_release_detects_the_optional_columns_like_the_active_one(pg):
    """A staged release read by name must carry the optional columns its
    table has; without them `evidence_basis` would read NULL and OCR items would go unmarked."""
    pg.create_schema("v34_named_ocr", optional_columns=True)
    pg.load("v34_named_ocr", "rel-a", extended_rows(with_basis=True)[:5], fixture_vectors()[:5], active=True)
    pg.load("v34_named_ocr", "rel-b", extended_rows(with_basis=True), fixture_vectors())
    store = pg.store("v34_named_ocr")
    assert store.release("rel-b").optional_columns == tuple(OPTIONAL_ITEM_COLUMNS)
    assert store.optional_columns("rel-b") == tuple(OPTIONAL_ITEM_COLUMNS)


def test_a_named_release_reloaded_under_the_same_id_is_a_new_identity(pg):
    """`dds load --replace` keeps the id; `release(id)` must not answer from its
    cache with the old manifest sha, or the eval's reload refusal could never fire."""
    _two_release_schema(pg, "v34_named_reload", active=["rel-a"])
    store = pg.store("v34_named_reload")
    before = store.release("rel-b")
    with pg.admin() as conn:
        conn.execute("UPDATE v34_named_reload.releases SET manifest = manifest || '{\"reloaded\": 1}'"
                     " WHERE release_attempt_id = 'rel-b'")
    after = store.release("rel-b")
    assert before.release_attempt_id == after.release_attempt_id
    assert before.manifest_sha256 != after.manifest_sha256


async def test_two_active_releases_is_a_store_error_and_503(pg, config):
    """Two active releases must refuse, never pick the first row."""
    _two_release_schema(pg, "c2_two", active=["rel-a", "rel-b"])
    store = pg.store("c2_two")
    with pytest.raises(StoreError, match="2 active"):
        store.active_release()
    assert store.health()[0] == "degraded"
    assert (await _ask_status(config, store)).status_code == 503


def _corrupt_schema(pg):
    """Three releases of three rows each: sha mismatch, invalid UTF-8 (sha matching), NUL bytes."""
    if getattr(pg, "_corrupt_ready", False):
        return
    rows = fixture_rows()[:3]
    vectors = fixture_vectors()[:3]
    pg.create_schema("c2_bytes")
    bad_sha = [dict(row) for row in rows]
    bad_sha[1]["payload_sha256"] = "0" * 64
    pg.load("c2_bytes", "rel-bad-sha", bad_sha, vectors)
    invalid = [dict(row) for row in rows]
    raw = b"\xff\xfe\xfa valid sha, invalid utf-8"
    invalid[1].update(payload_bytes=raw, payload_sha256=hashlib.sha256(raw).hexdigest())
    pg.load("c2_bytes", "rel-bad-utf8", invalid, vectors)
    nul = [dict(row) for row in rows]
    text = "before\x00after — NUL inside valid UTF-8 " + rows[1]["payload"]
    nul[1].update(payload=text, payload_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
    pg.load("c2_bytes", "rel-nul", nul, vectors, active=True)
    pg._corrupt_ready = True
    pg._nul_text = text


async def test_a_corrupt_payload_in_postgres_is_refused_never_served(pg, config):
    """A sha mismatch, and correctly hashed invalid UTF-8, fail the request closed."""
    _corrupt_schema(pg)
    store = pg.store("c2_bytes")
    qvec = fixture_vectors()[1]
    for release_id in ("rel-bad-sha", "rel-bad-utf8"):
        with pytest.raises(StoreError, match="refusing to cite"):
            store.search(qvec, release_id, RND)
        pg.activate("c2_bytes", release_id)
        response = await _ask_status(config, pg.store("c2_bytes"))
        assert response.status_code == 503
        assert "packet" not in response.text and "evidence_text" not in response.text
    pg.activate("c2_bytes", "rel-nul")


def test_a_payload_with_nul_bytes_round_trips_exactly(pg):
    """NUL is valid UTF-8; the driver must not truncate the payload at it."""
    _corrupt_schema(pg)
    items = pg.store("c2_bytes").search(fixture_vectors()[1], "rel-nul", RND)
    served = next(item for item in items if "\x00" in item.payload)
    assert served.payload == pg._nul_text
    assert hashlib.sha256(served.payload.encode("utf-8")).hexdigest() == served.payload_sha256


def _slow_schema(pg):
    """`c2_slow.items` is a view over the fixture that sleeps `knob.sleep_s` once per statement."""
    if getattr(pg, "_slow_ready", False):
        return
    with pg.admin() as conn:
        conn.execute("CREATE SCHEMA c2_slow")
        conn.execute("CREATE TABLE c2_slow.releases (LIKE ceres_serve_v0.releases INCLUDING ALL)")
        conn.execute("INSERT INTO c2_slow.releases SELECT * FROM ceres_serve_v0.releases"
                     " WHERE release_attempt_id = %s", (FIXTURE_RELEASE,))
        conn.execute("CREATE TABLE c2_slow.knob (sleep_s float8 NOT NULL)")
        conn.execute("INSERT INTO c2_slow.knob VALUES (0)")
        conn.execute("CREATE FUNCTION c2_slow.nap() RETURNS boolean LANGUAGE plpgsql AS"
                     " $$BEGIN PERFORM pg_sleep((SELECT sleep_s FROM c2_slow.knob)); RETURN true; END$$")
        conn.execute("CREATE VIEW c2_slow.items AS SELECT * FROM ceres_serve_v0.items"
                     " WHERE (SELECT c2_slow.nap())")
        conn.execute("GRANT USAGE ON SCHEMA c2_slow TO ceres_ro_test")
        conn.execute("GRANT SELECT ON ALL TABLES IN SCHEMA c2_slow TO ceres_ro_test")
    pg._slow_ready = True


def _knob(pg, seconds):
    with pg.admin() as conn:
        conn.execute("UPDATE c2_slow.knob SET sleep_s = %s", (seconds,))


def test_a_statement_timeout_is_a_store_error_and_the_pool_recovers(pg):
    """A slow query must fail fast as StoreError and leave its pooled connection usable."""
    import psycopg

    _slow_schema(pg)
    store = pg.store("c2_slow", statement_timeout_ms=200, pool_max_size=1)
    qvec = fixture_query_vectors()[0]
    release = store.active_release()
    _knob(pg, 2.0)
    try:
        started = time.monotonic()
        with pytest.raises(StoreError) as caught:
            store.search(qvec, release.release_attempt_id, RND)
        assert time.monotonic() - started < 1.5
        assert isinstance(caught.value.__cause__, psycopg.errors.QueryCanceled)
    finally:
        _knob(pg, 0)
    # The same single connection serves the next request: nothing was left in a failed transaction.
    assert len(store.search(qvec, release.release_attempt_id, RND)) == CANDIDATE_POOL


async def test_connection_refused_starts_the_service_reports_store_down_and_answers_503(pg, config):
    """An unreachable store must not stop startup; /health says down; /v1/ask is 503."""
    store = pg.store(dsn="postgresql://ceres_ro_test@127.0.0.1:1/postgres",
                     pool_timeout_s=0.5, connect_timeout_s=1)
    app, asked = _app(config, store)
    question, lang = asked[0]
    async with _client(app) as http:
        health = (await http.get("/health")).json()
        answer = await http.post("/v1/ask", json={"question": question, "lang": lang})
    assert health["store"] == "down" and health["ready"] is False
    assert answer.status_code == 503
    assert "127.0.0.1" not in answer.text and "ceres_ro_test" not in answer.text


def test_pool_exhaustion_is_a_store_error_within_the_pool_timeout_and_recovers(pg):
    """With every connection taken, a request fails within `pool_timeout_s`, then recovers."""
    import psycopg_pool

    store = pg.store("ceres_serve_v0", pool_max_size=1, pool_timeout_s=0.5)
    qvec = fixture_query_vectors()[0]
    release = store.active_release()
    held = store._pool.getconn()
    try:
        started = time.monotonic()
        with pytest.raises(StoreError) as caught:
            store.search(qvec, release.release_attempt_id, RND)
        assert time.monotonic() - started < 2.0
        assert isinstance(caught.value.__cause__, psycopg_pool.PoolTimeout)
    finally:
        store._pool.putconn(held)
    assert store.search(qvec, release.release_attempt_id, RND)


async def test_schema_drift_is_a_store_error_not_a_crash(pg, config):
    """A renamed contract column, or a query of another width, is a store refusal (503)."""
    pg.create_schema("c2_drift")
    with pg.admin() as conn:
        conn.execute("ALTER TABLE c2_drift.items RENAME COLUMN payload TO payload_bytes")
    pg.load("c2_drift", "rel-drift", fixture_rows()[:5], fixture_vectors()[:5], active=True)
    store = pg.store("c2_drift")
    with pytest.raises(StoreError, match="search failed"):
        store.search(fixture_vectors()[0], "rel-drift", RND)
    assert (await _ask_status(config, store)).status_code == 503
    with pytest.raises(StoreError, match="search failed"):
        pg.store("ceres_serve_v0").search(np.ones(8, dtype=np.float32) / np.sqrt(8),
                                          FIXTURE_RELEASE, RND)


def test_the_adapter_session_is_read_only(pg):
    """Even with a role that could write, the adapter's sessions refuse writes."""
    import psycopg

    store = pg.store("ceres_serve_v0", dsn=pg.admin_dsn)       # superuser, on purpose
    with store._pool.connection() as conn:
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            conn.execute("UPDATE ceres_serve_v0.releases SET loaded_by = 'x'")


def _switch_schema(pg):
    """`c2_switch`: rel-a = 60 fixture rows, rel-b = the same rows under other chunk ids."""
    rows_a = fixture_rows()[:60]
    rows_b = [dict(row, chunk_id=hashlib.sha256(f"c2-b/{row['chunk_id']}".encode()).hexdigest())
              for row in rows_a]
    vectors = fixture_vectors()[:60]
    pg.create_schema("c2_switch")
    pg.load("c2_switch", "rel-a", rows_a, vectors, active=True)
    pg.load("c2_switch", "rel-b", rows_b, vectors)
    return {row["chunk_id"] for row in rows_a}, {row["chunk_id"] for row in rows_b}


@pytest.fixture
def switch_schema(pg):
    ids = getattr(pg, "_switch_ids", None)
    if ids is None:
        ids = pg._switch_ids = _switch_schema(pg)
    pg.activate("c2_switch", "rel-a")
    return ids


def test_a_switch_between_release_lookup_and_search_keeps_the_pinned_release(pg, switch_schema):
    """Design: the request PINS the release id it read; the search that follows
    returns only that release's rows even after `--activate` moved on (not fail closed)."""
    ids_a, ids_b = switch_schema
    store = pg.store("c2_switch")
    pinned = store.active_release()
    pg.activate("c2_switch", "rel-b")
    items = store.search(fixture_vectors()[0], pinned.release_attempt_id, RND)
    assert pinned.release_attempt_id == "rel-a"
    assert items and {item.chunk_id for item in items} <= ids_a


def test_a_switched_release_is_seen_by_the_next_request_not_cached(pg, switch_schema):
    """The release cache must not keep serving the release that was active before."""
    store = pg.store("c2_switch")
    assert store.active_release().release_attempt_id == "rel-a"
    pg.activate("c2_switch", "rel-b")
    assert store.active_release().release_attempt_id == "rel-b"


class SwitchingStore:
    """Delegates to a PgStore and activates another release right after the request pinned one."""

    def __init__(self, inner, pg, schema, to):
        self.inner, self.pg, self.schema, self.to = inner, pg, schema, to

    def active_release(self):
        release = self.inner.active_release()
        self.pg.activate(self.schema, self.to)
        return release

    def search(self, *args, **kwargs):
        return self.inner.search(*args, **kwargs)

    def withheld(self, *args, **kwargs):
        return self.inner.withheld(*args, **kwargs)

    def health(self):
        return self.inner.health()


def test_a_switch_mid_request_serves_one_release_end_to_end(pg, switch_schema, config):
    """Pipeline level: meta names release A and every context comes from A, although B was
    activated between the release lookup and the search; the next request is B's."""
    from ceres.testing import build_fixture_stack

    ids_a, ids_b = switch_schema
    _store, models, _llm, asked = build_fixture_stack()
    store = pg.store("c2_switch")
    deps = pipeline.Deps.build(config, SwitchingStore(store, pg, "c2_switch", "rel-b"), models, FakeLLM())
    question, lang = asked[0]
    first = pipeline.ask(deps, AskRequest(question=question, lang=lang))
    assert first.meta.release_attempt_id == "rel-a"
    assert first.packet["contexts"]
    assert {c["chunk_id"] for c in first.packet["contexts"]} <= ids_a
    deps.store = store
    second = pipeline.ask(deps, AskRequest(question=question, lang=lang))
    assert second.meta.release_attempt_id == "rel-b"
    assert {c["chunk_id"] for c in second.packet["contexts"]} <= ids_b


def test_a_replace_under_the_same_id_changes_the_withheld_cache_key(pg):
    """A loader `--replace` rewrites an attempt under the SAME id with a new
    manifest. The withheld count and the optional-column detection must follow the new rows."""
    rows = fixture_rows()[:30]
    vectors = fixture_vectors()[:30]
    pg.create_schema("c2_replace")
    pg.load("c2_replace", "rel-r", rows, vectors, active=True, manifest={"load_id": "first"})
    store = pg.store("c2_replace")
    before = store.active_release()
    assert store.withheld(before, list(PRODUCTION_RIGHTS)).withheld_rights == 0
    assert before.optional_columns == ()

    replaced = [dict(row, rights_state="internal_research_only") if index < 7 else row
                for index, row in enumerate(rows)]
    import psycopg

    with psycopg.connect(pg.admin_dsn) as conn:                  # one transaction, like `--replace`
        from pgvector.psycopg import register_vector

        register_vector(conn)
        for statement in OPTIONAL_COLUMNS_DDL:                # the loader's statement
            conn.execute(statement.format(schema="c2_replace"))
        for table in ("items", "releases"):
            conn.execute(f"DELETE FROM c2_replace.{table} WHERE release_attempt_id = %s", ("rel-r",))
        pg.load("c2_replace", "rel-r", replaced, vectors, active=True,
                manifest={"load_id": "second"}, conn=conn)
    after = store.active_release()
    assert after.release_attempt_id == before.release_attempt_id
    assert after.identity != before.identity
    assert store.withheld(after, list(PRODUCTION_RIGHTS)).withheld_rights == 7
    assert after.optional_columns == ("evidence_basis", "page_start", "page_end")


# ---------------------------------------------------------------- optional OCR columns in PostgreSQL

def test_absent_optional_columns_read_as_null(pg):
    """Older releases have no optional columns; the SELECT must not name them."""
    store = pg.store("ceres_serve_v0")
    assert store.active_release().optional_columns == ()
    items = store.search(fixture_query_vectors()[0], EXTENDED_RELEASE, RND, 500)
    assert items and all(item.evidence_basis is None and item.page_start is None
                         and item.page_end is None for item in items)


def test_present_optional_columns_are_selected_and_detected_once_per_release(pg):
    """The columns exist → basis and page span are read; information_schema is asked once."""
    store = pg.store("c2_ocr")
    release = store.active_release()
    store.active_release()
    items = store.search(fixture_query_vectors()[0], release.release_attempt_id, RND, 500)
    store.search(fixture_query_vectors()[1], release.release_attempt_id, RND, 500)
    assert release.optional_columns == ("evidence_basis", "page_start", "page_end")
    assert store.column_detections == 1
    ocr = [item for item in items if item.evidence_basis == OCR_ADMITTED_UNVERIFIED]
    assert ocr and {item.work_id for item in ocr} == {extended_works()["ocr"]}
    expected = {row["chunk_id"]: (row["page_start"], row["page_end"])
                for row in extended_rows(with_basis=True) if row.get("evidence_basis")}
    assert all((item.page_start, item.page_end) == expected[item.chunk_id] for item in ocr)
    assert any(item.page_start is not None for item in ocr) and any(item.page_start is None for item in ocr)
    assert any(item.page_start == item.page_end for item in ocr if item.page_start is not None)


def _ocr_slice(count: int = 12):
    """The first `count` eligible OCR rows of the extended copy (half with a page span) + vectors."""
    rows = extended_rows(with_basis=True)
    picked = [i for i, row in enumerate(rows) if row.get("evidence_basis") and row["evidence_eligible"]][:count]
    return [rows[i] for i in picked], fixture_vectors()[picked]


def test_any_subset_of_the_optional_columns_is_detected_and_read(pg):
    """Optional-column detection: each column is its own `ADD COLUMN IF NOT EXISTS`, so a table may carry any
    subset. Bug caught: detection that assumes all-or-none (the SELECT names an absent column → a
    500, or a present one reads NULL). A lone page bound with values is a half-null span → refused."""
    import itertools

    rows, vectors = _ocr_slice()
    names = tuple(OPTIONAL_ITEM_COLUMNS)
    subsets = [s for n in (1, 2) for s in itertools.combinations(names, n)]   # 6; 0 and 3 tested above
    for number, subset in enumerate(subsets):
        schema = f"c2b_subset_{number}"
        pg.create_schema(schema, optional_columns=list(subset))
        pg.load(schema, "rel-s", rows, vectors, active=True)
        store = pg.store(schema)
        assert store.active_release().optional_columns == subset, subset
        half_span = ("page_start" in subset) != ("page_end" in subset)
        if half_span:
            with pytest.raises(StoreError, match="exactly one of page_start/page_end is NULL"):
                store.search(vectors[0], "rel-s", RND, 50)
            continue
        items = {item.chunk_id: item for item in store.search(vectors[0], "rel-s", RND, 50)}
        assert set(items) == {row["chunk_id"] for row in rows}, subset
        for row in rows:
            item = items[row["chunk_id"]]
            for name in names:
                expected = row[name] if name in subset else None
                assert getattr(item, name) == expected, (subset, name)


def test_a_malformed_page_span_in_postgres_is_refused_not_served(pg):
    """Page bounds through the real adapter: integers from psycopg go through the same checks as the
    fixture store — an inverted span, a negative page, a half-null span each fail closed."""
    rows, vectors = _ocr_slice(4)
    for number, (start, end, message) in enumerate(((5, 4, "page_start .* > page_end"),
                                                    (-1, 2, "is negative"),
                                                    (3, None, "exactly one of page_start/page_end"))):
        schema = f"c2b_malformed_{number}"
        broken = [dict(rows[0], page_start=start, page_end=end)] + rows[1:]
        pg.create_schema(schema, optional_columns=True)
        pg.load(schema, "rel-m", broken, vectors, active=True)
        with pytest.raises(StoreError, match=message):
            pg.store(schema).search(vectors[0], "rel-m", RND, 50)


def test_production_refuses_ocr_items_and_rnd_serves_them(pg):
    """The production arm removes every OCR item; rnd keeps them."""
    store = pg.store("c2_ocr")
    ocr_row = next(i for i, row in enumerate(extended_rows(with_basis=True))
                   if row.get("evidence_basis") and row["evidence_eligible"])
    qvec = fixture_vectors()[ocr_row]
    rnd = store.search(qvec, EXTENDED_RELEASE, RND, 500)
    production = store.search(qvec, EXTENDED_RELEASE, list(PRODUCTION_RIGHTS), 500,
                              refused_evidence_basis=OCR_ADMITTED_UNVERIFIED)
    rnd_with_production_rights = store.search(qvec, EXTENDED_RELEASE, list(PRODUCTION_RIGHTS), 500)
    assert rnd[0].evidence_basis == OCR_ADMITTED_UNVERIFIED       # the query is that row's vector
    assert all(item.evidence_basis is None for item in production)
    assert any(item.evidence_basis for item in rnd_with_production_rights)   # the arm did it, not rights


def test_the_ocr_mark_travels_from_postgres_to_packet_and_meta(pg, config, tmp_path):
    """Basis read in SQL → Item → packet context → meta.citation_marks → the citation log."""
    rows = extended_rows(with_basis=True)
    index = next(i for i, row in enumerate(rows)
                 if row.get("evidence_basis") and row["evidence_eligible"] and row["page_start"]
                 and len(row["payload"].split()) >= 21)
    question = "Which figure does the scanned vegetable-growing treatise give?"
    qid = __import__("ceres.contracts", fromlist=["question_id_for"]).question_id_for(question, "en")
    models = FakeModels({qid: fixture_vectors()[index]}, {question: qid}, dim=768)
    deps = pipeline.Deps.build(config, pg.store("c2_ocr"), models, FakeLLM())
    response = pipeline.ask(deps, AskRequest(question=question, lang="en"))
    first = response.packet["contexts"][0]
    assert response.answer["state"] == "VALIDATED"
    assert first["chunk_id"] == rows[index]["chunk_id"]
    assert first["evidence_basis"] == OCR_ADMITTED_UNVERIFIED
    assert [mark.model_dump() for mark in response.meta.citation_marks] == [{
        "evidence_id": "E1", "chunk_id": rows[index]["chunk_id"], "work_id": rows[index]["work_id"],
        "evidence_basis": OCR_ADMITTED_UNVERIFIED}]
    log = config.ocr_citation_log.read_text(encoding="utf-8").splitlines()
    line = json.loads(log[-1])
    assert line["chunk_id"] == rows[index]["chunk_id"]
    assert (line["page_start"], line["page_end"]) == (rows[index]["page_start"], rows[index]["page_end"])


# ================================================================ 4. live serving database (opt-in)

LIVE_DSN = os.environ.get("CERES_TEST_LIVE_DSN")
LIVE_DEFAULT_DSN = "postgresql://ceres_reader@db.invalid/veridian_rag"   # password: ~/.pgpass
LIVE_ROUTE_FILE = os.environ.get("CERES_TEST_LIVE_ROUTE_FILE")
LIVE_SCHEMA = os.environ.get("CERES_TEST_LIVE_SCHEMA", "ceres_serve_v0")
LIVE_QUERIES = 20
LIVE_SEED = 20260924
requires_live = pytest.mark.skipif(
    not LIVE_DSN, reason=f"CERES_TEST_LIVE_DSN is not set (documented default: {LIVE_DEFAULT_DSN};"
                         " password from ~/.pgpass, never in code)")


def _percentile(values, q):
    ordered = sorted(values)
    return float(np.percentile(np.asarray(ordered), q, method="linear")) if ordered else None


def _route_profiles(route_file: Path) -> dict[str, str]:
    """sha256 of the chunk/represent/embed profiles a route file names (relative to the pipeline root)."""
    import yaml

    doc = yaml.safe_load(route_file.read_text(encoding="utf-8"))
    loop_root = route_file.resolve().parent.parent.parent       # routes → rag → pipeline root
    shas = {}
    for slot in ("chunk", "represent", "embed"):
        path = Path(doc["profiles"][slot])
        path = path if path.is_absolute() else loop_root / path
        shas[slot] = hashlib.sha256(path.read_bytes()).hexdigest()
    return shas


@pytest.fixture(scope="module")
def live_run():
    """One read-only run against the serving database, written to `runtime/c2-live/<UTC stamp>.json`.

    Release-agnostic: nothing here names a release id, a row count or a chunk id. Each query pins
    the active release the way a request does, searches it, and is then checked by an independent
    query on a separate connection (same role) that the returned rows are rows OF that release,
    evidence-eligible and inside the mode's rights.
    """
    import psycopg

    from ceres.config import SERVE_RIGHTS_BY_MODE

    started = datetime.now(timezone.utc)
    store = PgStore(LIVE_DSN, LIVE_SCHEMA, pool_max_size=2)
    check = psycopg.connect(LIVE_DSN, options="-c default_transaction_read_only=on"
                                              " -c statement_timeout=30000", autocommit=True)
    record: dict = {"started_at": started.isoformat(), "schema": LIVE_SCHEMA,
                    "dsn": LIVE_DSN.split("@", 1)[-1], "seed": LIVE_SEED, "queries_per_mode": LIVE_QUERIES,
                    "pool": CANDIDATE_POOL, "python": platform.python_version(),
                    "psycopg": psycopg.__version__, "modes": {}, "violations": []}

    def active_rows():
        return check.execute(
            f"SELECT release_attempt_id, route_id, chunk_profile_sha256, represent_profile_sha256,"
            f" embed_profile_sha256, row_count, loaded_at FROM {LIVE_SCHEMA}.releases WHERE active").fetchall()

    record["active_before"] = [[str(v) for v in row] for row in active_rows()]
    if LIVE_ROUTE_FILE:
        record["route_file"] = LIVE_ROUTE_FILE
        record["route_profiles"] = _route_profiles(Path(LIVE_ROUTE_FILE))
    rng = np.random.default_rng(LIVE_SEED)
    vectors = rng.normal(size=(LIVE_QUERIES, 768)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    pinned_ids = set()
    try:
        for mode in ("rnd", "production"):
            rights = list(SERVE_RIGHTS_BY_MODE[mode])
            refused = OCR_ADMITTED_UNVERIFIED if mode == "production" else None
            timings, counts, verified, bases, per_query = [], [], 0, {}, []
            for qvec in vectors:
                release = store.active_release()
                pinned_ids.add(release.release_attempt_id)
                began = time.perf_counter()
                items = store.search(qvec, release.release_attempt_id, rights,
                                     refused_evidence_basis=refused)
                timings.append((time.perf_counter() - began) * 1000)
                per_query.append({"release_attempt_id": release.release_attempt_id,
                                  "row_count": release.row_count, "ms": round(timings[-1], 3),
                                  "rows": len(items)})
                counts.append(len(items))
                for item in items:
                    if hashlib.sha256(item.payload.encode("utf-8")).hexdigest() == item.payload_sha256:
                        verified += 1
                    bases[str(item.evidence_basis)] = bases.get(str(item.evidence_basis), 0) + 1
                rows = check.execute(
                    f"SELECT chunk_id, evidence_eligible, rights_state FROM {LIVE_SCHEMA}.items"
                    " WHERE release_attempt_id = %s AND chunk_id = ANY(%s)",
                    (release.release_attempt_id, [item.chunk_id for item in items])).fetchall()
                if len(rows) != len(items) or len(items) > CANDIDATE_POOL:
                    record["violations"].append({"mode": mode, "why": "rows not of the pinned release"
                                                 " or pool exceeded", "returned": len(items),
                                                 "found_in_release": len(rows)})
                for chunk_id, eligible, rights_state in rows:
                    if not eligible or rights_state not in rights:
                        record["violations"].append({"mode": mode, "chunk_id": chunk_id,
                                                     "eligible": eligible, "rights": rights_state})
                if refused and any(item.evidence_basis == refused for item in items):
                    record["violations"].append({"mode": mode, "why": "OCR item served in production"})
            by_release = {}
            for query in per_query:
                by_release.setdefault(query["release_attempt_id"], []).append(query["ms"])
            record["modes"][mode] = {
                "rights": rights, "refused_evidence_basis": refused,
                "latency_ms": {"p50": _percentile(timings, 50), "p95": _percentile(timings, 95),
                               "max": max(timings), "mean": statistics.fmean(timings)},
                # The active release can move during a run (the loader activates releases); every
                # measurement is attributed to the release it was pinned to.
                "latency_ms_by_release": {rid: {"n": len(ms), "p50": _percentile(ms, 50),
                                                "p95": _percentile(ms, 95), "max": max(ms)}
                                          for rid, ms in by_release.items()},
                "queries": per_query,
                "rows_per_query": {"min": min(counts), "max": max(counts)},
                "payloads": sum(counts), "payload_sha_verified": verified, "evidence_basis": bases,
            }
        record["optional_columns"] = list(store.optional_columns(sorted(pinned_ids)[0]))
    finally:
        record["active_after"] = [[str(v) for v in row] for row in active_rows()]
        record["pinned_release_ids"] = sorted(pinned_ids)
        record["release_switch_observed"] = (len(pinned_ids) > 1
                                             or record["active_before"] != record["active_after"])
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        out = ROOT / "runtime" / "c2-live"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{started.strftime('%Y%m%dT%H%M%SZ')}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        record["artifact"] = str(path)
        store.close()
        check.close()
    return record


@pytest.mark.live_store
@requires_live
def test_live_exactly_one_active_release_matching_the_configured_route_profiles(live_run):
    """The serving database has exactly one active release, before and after the run; when a route file
    is configured, the release's chunk/represent/embed profile sha256s are that route's (only those
    three: never `index`, never `releases.route_id`)."""
    for key in ("active_before", "active_after"):
        assert len(live_run[key]) == 1, f"{key}: {len(live_run[key])} active releases"
    if "route_profiles" in live_run:
        for snapshot in (live_run["active_before"][0], live_run["active_after"][0]):
            store_profiles = dict(zip(("chunk", "represent", "embed"), snapshot[2:5]))
            assert store_profiles == live_run["route_profiles"], (snapshot[0], store_profiles)


@pytest.mark.live_store
@requires_live
def test_live_search_returns_only_servable_rows_of_the_pinned_release(live_run):
    """≤ pool rows, all of the pinned release, eligible, in the mode's rights, no OCR
    in production, every payload sha verified; p50/p95 recorded in the artifact."""
    assert live_run["violations"] == []
    for mode, result in live_run["modes"].items():
        assert 0 < result["rows_per_query"]["min"] <= result["rows_per_query"]["max"] <= CANDIDATE_POOL
        assert result["payload_sha_verified"] == result["payloads"], mode
        assert result["latency_ms"]["p50"] is not None and result["latency_ms"]["p95"] is not None
    assert Path(live_run["artifact"]).is_file()


# ---------------------------------------------------------------- ROUTE-STORE-GUARD in PostgreSQL

GUARD_PROFILES = ROOT / "tests" / "fixtures" / "profiles"
GUARD_FILES = {"chunk": "chunk-bridge-v3.1.yaml", "represent": "verbatim-v1.yaml", "embed": "jina-768-v1.yaml"}


async def test_a_replace_to_another_chunk_profile_under_the_same_id_is_refused_in_postgres(pg, env):
    """Bug: a loader replace re-points `all/release-attempt-0008` to the v3.4 chunks under the same id;
    serving read only (id, route_id, row_count, manifest) and answered under the configured v3.3
    route. Now the releases row's real profile shas are read by `active_release` AND `release(id)`,
    a matching release serves ("ok"), and after the replace /v1/ask is 503 route_store_mismatch
    and /health is not ready."""
    shas = {slot: hashlib.sha256((GUARD_PROFILES / name).read_bytes()).hexdigest()
            for slot, name in GUARD_FILES.items()}
    schema, release_id = "c2_route_guard", "all/release-attempt-0008"
    rows, vectors = fixture_rows()[:40], fixture_vectors()[:40]
    pg.create_schema(schema)
    pg.load(schema, release_id, rows, vectors, active=True, manifest={"load_id": "v33"},
            profile_shas=shas)
    store = pg.store(schema)
    assert store.active_release().profile_sha256 == shas
    assert store.release(release_id).profile_sha256 == shas
    overrides = {f"CERES_ROUTE__PROFILES__{slot.upper()}": str(GUARD_PROFILES / name)
                 for slot, name in GUARD_FILES.items()}
    config = load_config(EXAMPLE_CONFIG, dict(env, **overrides), require_api_key=False)
    app, asked = _app(config, store)
    question, lang = asked[0]
    async with _client(app) as http:
        before = await http.post("/v1/ask", json={"question": question, "lang": lang})
        info_before = (await http.get("/v1/info")).json()

        import psycopg
        from pgvector.psycopg import register_vector

        with psycopg.connect(pg.admin_dsn) as conn:              # one transaction, like `--replace`
            register_vector(conn)
            for table in ("items", "releases"):
                conn.execute(f"DELETE FROM {schema}.{table} WHERE release_attempt_id = %s", (release_id,))
            pg.load(schema, release_id, rows, vectors, active=True, manifest={"load_id": "v34"},
                    conn=conn, profile_shas=dict(shas, chunk="e" * 64))
        after = await http.post("/v1/ask", json={"question": question, "lang": lang})
        health = (await http.get("/health")).json()
        info_after = (await http.get("/v1/info")).json()
    assert before.status_code == 200 and info_before["route_store_check"] == "ok"
    assert store.active_release().release_attempt_id == release_id
    assert store.active_release().profile_sha256["chunk"] == "e" * 64
    assert after.status_code == 503 and after.json()["detail"]["code"] == "route_store_mismatch"
    assert "e" * 64 not in after.text
    assert health["store"] == "route_store_mismatch" and health["ready"] is False
    assert info_after["route_store_check"] == "mismatch"


async def test_served_meta_names_the_load_so_two_loads_of_one_release_id_are_told_apart(pg, env):
    """Bug: meta carried only release_attempt_id, so answers before and after a
    same-id `--replace` looked identical. Now meta.release_manifest_sha256 names the load, and it changes."""
    shas = {slot: hashlib.sha256((GUARD_PROFILES / name).read_bytes()).hexdigest()
            for slot, name in GUARD_FILES.items()}
    schema, release_id = "f1_load_identity", "all/release-attempt-0008"
    rows, vectors = fixture_rows()[:40], fixture_vectors()[:40]
    pg.create_schema(schema)
    pg.load(schema, release_id, rows, vectors, active=True, manifest={"load_id": "load-a"}, profile_shas=shas)
    store = pg.store(schema)
    overrides = {f"CERES_ROUTE__PROFILES__{slot.upper()}": str(GUARD_PROFILES / name)
                 for slot, name in GUARD_FILES.items()}
    config = load_config(EXAMPLE_CONFIG, dict(env, **overrides), require_api_key=False)
    app, asked = _app(config, store)
    question, lang = asked[0]
    async with _client(app) as http:
        before = await http.post("/v1/ask", json={"question": question, "lang": lang})
        import psycopg
        from pgvector.psycopg import register_vector
        with psycopg.connect(pg.admin_dsn) as conn:
            register_vector(conn)
            for table in ("items", "releases"):
                conn.execute(f"DELETE FROM {schema}.{table} WHERE release_attempt_id = %s", (release_id,))
            pg.load(schema, release_id, rows, vectors, active=True, manifest={"load_id": "load-b"},
                    conn=conn, profile_shas=shas)
        after = await http.post("/v1/ask", json={"question": question, "lang": lang})
    assert before.status_code == 200 and after.status_code == 200
    first, second = before.json()["meta"], after.json()["meta"]
    assert first["release_attempt_id"] == second["release_attempt_id"] == release_id
    assert first["release_manifest_sha256"] and second["release_manifest_sha256"]
    assert first["release_manifest_sha256"] != second["release_manifest_sha256"]


async def test_a_same_manifest_replace_with_other_profile_shas_is_still_refused(pg, env):
    """Bug: the release cache was keyed on (id, manifest sha) only, so a replace
    that changed the profile-sha columns but not the manifest bytes served the stale 'ok' route-store verdict."""
    shas = {slot: hashlib.sha256((GUARD_PROFILES / name).read_bytes()).hexdigest()
            for slot, name in GUARD_FILES.items()}
    schema, release_id = "f2_same_manifest", "all/release-attempt-0008"
    rows, vectors = fixture_rows()[:40], fixture_vectors()[:40]
    pg.create_schema(schema)
    pg.load(schema, release_id, rows, vectors, active=True, manifest={"load_id": "same"}, profile_shas=shas)
    store = pg.store(schema)
    overrides = {f"CERES_ROUTE__PROFILES__{slot.upper()}": str(GUARD_PROFILES / name)
                 for slot, name in GUARD_FILES.items()}
    config = load_config(EXAMPLE_CONFIG, dict(env, **overrides), require_api_key=False)
    app, asked = _app(config, store)
    question, lang = asked[0]
    async with _client(app) as http:
        before = await http.post("/v1/ask", json={"question": question, "lang": lang})
        import psycopg
        from pgvector.psycopg import register_vector
        with psycopg.connect(pg.admin_dsn) as conn:
            register_vector(conn)
            for table in ("items", "releases"):
                conn.execute(f"DELETE FROM {schema}.{table} WHERE release_attempt_id = %s", (release_id,))
            pg.load(schema, release_id, rows, vectors, active=True, manifest={"load_id": "same"},
                    conn=conn, profile_shas=dict(shas, chunk="e" * 64))
        after = await http.post("/v1/ask", json={"question": question, "lang": lang})
    assert before.status_code == 200
    assert after.status_code == 503 and after.json()["detail"]["code"] == "route_store_mismatch"


def test_the_withheld_cache_key_separates_releases_that_differ_only_in_profile_shas():
    """Bug: withheld() keyed on (id, manifest sha) alone, so a same-manifest
    replace could serve the previous release's rights-withheld count."""
    from ceres.store import Release, release_cache_identity
    a = Release("all/release-attempt-0008", "route", 10, {}, manifest_sha256="m" * 64,
                profile_sha256={"chunk": "a" * 64, "represent": "b" * 64, "embed": "c" * 64})
    b = Release("all/release-attempt-0008", "route", 10, {}, manifest_sha256="m" * 64,
                profile_sha256={"chunk": "e" * 64, "represent": "b" * 64, "embed": "c" * 64})
    assert a.identity == b.identity
    assert release_cache_identity(a) != release_cache_identity(b)
