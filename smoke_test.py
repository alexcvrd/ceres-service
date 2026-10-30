#!/usr/bin/env python3
"""Scaffold smoke test: the whole route over fake transports. No GPU, no Postgres, no network.

It builds the app exactly as `ceres-api --fake --no-auth` would, calls `/health`, `/v1/info` and
`/v1/ask` for one EN and one RO question through an in-process ASGI transport, and validates the
packet and the answer row against the shipped schemas. A real-model acceptance smoke is a separate
thing and is not part of this script.

The fake stack it runs over is `ceres/testing.py`: the same module
`ceres-api --fake` serves and `tests/conftest.py` builds on, so the fixture the smoke test proves
is the fixture the service fakes and the unit tests use.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ceres import pipeline                                   # noqa: E402
from ceres.api import create_app                             # noqa: E402
from ceres.config import load_config                         # noqa: E402
from ceres.contracts import ANSWER_SCHEMA, PACKET_SCHEMA, schema_errors  # noqa: E402
# The fake stack lives in the package; re-exported here for callers of the old names.
from ceres.testing import (                                  # noqa: E402,F401
    CHUNK,
    EN_QUESTION,
    FIXTURE_DIR,
    LABEL_BY_CHUNK,
    RO_QUESTION,
    build_fixture_stack,
    build_inline_fixture,
    golden_rerank_scores,
    read_jsonl,
    smoke_env,
)


async def _run(app, questions: list[tuple[str, str]]) -> tuple[list[str], int]:
    import httpx

    lines: list[str] = []
    failures = 0
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://ceres.smoke") as client:
        health = await client.get("/health")
        health.raise_for_status()
        body = health.json()
        lines.append(f"health      : api={body['api']} ready={body['ready']} store={body['store']} "
                     f"model={body['model']} embedder={body['embedder']} reranker={body['reranker']}")

        info = await client.get("/v1/info")
        info.raise_for_status()
        meta = info.json()
        lines.append(f"info        : version={meta['version']} route={meta['route_id']} "
                     f"({meta['route_status']}) judged={meta['judged_route_id'][:12]}… "
                     f"release={meta['release_attempt_id']} adapters={meta['adapters']}")

        for question, lang in questions:
            response = await client.post("/v1/ask", json={"question": question, "lang": lang})
            if response.status_code != 200:
                failures += 1
                lines.append(f"ask [{lang}]   : HTTP {response.status_code} {response.text[:120]}")
                continue
            payload = response.json()
            packet_errors = schema_errors(payload["packet"], PACKET_SCHEMA)
            answer_errors = schema_errors(payload["answer"], ANSWER_SCHEMA)
            failures += len(packet_errors) + len(answer_errors)
            if payload["answer"]["state"] != "VALIDATED":
                failures += 1
            decision = (payload["answer"].get("candidate") or {}).get("decision", "-")
            lines.append(
                f"ask [{lang}]   : {payload['answer']['state']}/{decision} "
                f"contexts={len(payload['packet']['contexts'])} "
                f"attempts={payload['answer']['attempts']} "
                f"total={payload['meta']['latency_ms']['total']} ms "
                f"schema_errors={len(packet_errors) + len(answer_errors)}"
            )
    return lines, failures


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ceres-smoke-") as tmp:
        runtime_root = Path(tmp)
        config = load_config(HERE / "config.example.yaml", smoke_env(runtime_root),
                             require_api_key=False)     # explicit --no-auth for the smoke path
        store, models, llm, questions = build_fixture_stack()
        deps = pipeline.Deps.build(config, store, models, llm)
        app = create_app(deps, config)
        lines, failures = asyncio.run(_run(app, questions))
        raw = list((runtime_root / "executions").rglob("*.json"))
        lines.append(f"raw events  : {len(raw)} written under {runtime_root.name}/executions/"
                     f"  |  auth={'required' if config.auth_required else 'disabled (--no-auth)'}")
    print("\n".join(lines))
    print("SMOKE OK" if failures == 0 else f"SMOKE FAILED ({failures} problems)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
