#!/usr/bin/env python3
"""Ask Ceres one question in fake mode and print the answer with its citations.

Fake mode replaces the store, the embedder, the reranker and the answer model with small in-process fixtures, so this runs
on any machine after `bash setup_venv.sh`, with no GPU, database or model download. The request goes through the real HTTP
API, retrieval pipeline and citation validator.

Usage: .venv/bin/python demo.py ["your question"] [--lang en|ro]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("question", nargs="?", help="defaults to the fixture's English question")
    ap.add_argument("--lang", default="en", choices=["en", "ro"])
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore", message=".*httpx.*")      # a dependency notice from starlette.testclient, not ours
    from fastapi.testclient import TestClient

    from ceres.api import build_deps, create_app
    from ceres.config import load_config
    from ceres.testing import EN_QUESTION, RO_QUESTION

    question = args.question or (RO_QUESTION if args.lang == "ro" else EN_QUESTION)
    with tempfile.TemporaryDirectory(prefix="ceres-demo-") as runtime:
        env = dict(os.environ, CERES_RUNTIME_ROOT=runtime, DDS_LOOP_ROOT=runtime, DDS_GPU_LOCK=str(Path(runtime) / "gpu.lock"),
                   DDS_RUNTIME_ROOT=runtime)
        config = load_config(HERE / "config.example.yaml", env, require_api_key=False)
        app = create_app(build_deps(config, fake=True), config)
        with TestClient(app) as client:
            reply = client.post("/v1/ask", json={"question": question, "lang": args.lang})
    if reply.status_code != 200:
        print(f"HTTP {reply.status_code}: {reply.text}", file=sys.stderr)
        return 1
    body = reply.json()
    candidate = (body.get("answer") or {}).get("candidate") or {}
    print(f"Question: {question}\n")
    print(f"Decision: {candidate.get('decision')}")
    print(f"Answer:   {candidate.get('user_answer')}\n")
    contexts = {c["evidence_id"]: c for c in (body.get("packet") or {}).get("contexts", [])}
    for cite in candidate.get("citations") or []:
        passage = contexts.get(cite.get("evidence_id"), {})
        print(f"[{cite.get('evidence_id')}] “{cite.get('support_quote')}”")
        print(f"      from passage {passage.get('chunk_id', '?')[:40]}…")
    print(f"\nRoute: {(body.get('meta') or {}).get('route_id') or 'unassigned'}  (fake mode: fixture store, models and answerer;"
          " the answer shows the pipeline and citation format, not answer quality)")
    if os.environ.get("CERES_DEMO_JSON"):
        print(json.dumps(body, indent=1, ensure_ascii=False)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
