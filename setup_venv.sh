#!/usr/bin/env bash
# Venv for ceres-service. Idempotent, uv-based, frozen: the lock decides, never the resolver.
# Default: the API + tests stack (no torch). `--models` (or CERES_WITH_MODELS=1) adds the `models`
# extra — torch, transformers, sentence-transformers — needed by `ceres-eval` embed/rerank phases.
# `uv sync` is an EXACT sync: without the flag it removes torch from an existing venv, by design.
set -euo pipefail
EXTRAS=(--extra dev)
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --models) EXTRAS+=(--extra models) ;;
    *) ARGS+=("$arg") ;;
  esac
done
[ "${CERES_WITH_MODELS:-0}" = 1 ] && EXTRAS+=(--extra models)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
[ -x "$UV" ] || { echo "uv not found (expected on PATH or at ~/.local/bin/uv)"; exit 1; }

# UV_CACHE_DIR comes from the environment (it may be on a separate drive); this script never
# sets a path of its own, so the same commands work on the server.
[ -d .venv ] || "$UV" venv .venv --python 3.12
"$UV" sync --frozen "${EXTRAS[@]}"

# Keep the pip-style lock beside uv.lock: the install path on the server may not have uv. It is a
# committed artefact, so it is regenerated only when it is missing — never silently rewritten.
[ -f requirements.lock ] || "$UV" export --frozen --extra dev --no-emit-project --no-hashes \
  --format requirements.txt -o requirements.lock

.venv/bin/python smoke_test.py ${ARGS[@]+"${ARGS[@]}"}
