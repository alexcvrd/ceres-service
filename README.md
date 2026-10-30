# ceres-service

Ceres is Veridian's agronomy question-answering service. It answers a question in English or Romanian from passages
of a document release built by DDS, Veridian's document data stack (released separately as `ceres-dds`). The release
lives in Postgres with pgvector. For each question Ceres finds candidate passages with a hybrid search, reranks them,
gives the best five to an answer model behind an OpenAI-compatible server (for example `llama-server`), and checks
every citation in the reply against the passage it cites. The result is a JSON answer with its passages, its citations
and the metadata needed to audit it later.

Version 0.2.0. Python 3.12 or later.

## Limitations

Read these before relying on an answer.

- **Citations are not hardened for production.** In soft citation mode (the bundled serving template) a failed
  citation is dropped and the answer is kept, so an answer can arrive with no citation at all. No check confirms that a
  quote actually supports the claim it is attached to.
- **Romanian questions are answered from English sources.** The current release holds English documents. Romanian
  answer quality has not been evaluated.
- **One answer at a time on one GPU.** The answer server, embedder and reranker share one GPU; concurrent clients queue
  and may get HTTP 429.
- **No streaming and no monitoring.** The answer arrives as one response; there is no metrics endpoint or alerting.
- **`ceres-eval` is an internal evaluation tool.** It answers a question bank phase by phase against a DDS checkout and
  writes only into the existing folder given with `--answers-dir` (inside the DDS loop); without it, use `--scratch`. It is not part of the
  serving API.

**What `VALIDATED` proves.** `answer.state: VALIDATED` means the reply passed the structure checks (keys, enums,
language, length limits) and that every citation it still carries quotes text found in the cited passage, with its
claim found in the answer. It does **not** prove that the answer is correct or complete, that each quote supports its
claim, that every statement in the answer is cited, or (in soft mode) that any citation survived — check
`meta.dropped_citations` and the citation count. `serving_mode: production` controls which documents may be served; it
does not remove these limitations.

## Terms

| Term | Meaning |
|---|---|
| **Release** | One versioned set of passages that DDS loaded into the store. Exactly one release is active at a time. |
| **Passage** | One chunk of a source document. Passages in an answer packet are labelled `E1`…`E5`. |
| **Profile** | A YAML file that fixes one step of the recipe: chunk, represent, embed, index, retrieve, rerank or answer. |
| **Route** | The seven profiles together. Its id (`route_id`) is the sha256 of the seven profile hashes, so any change to any step gives a new id. |
| **Citation mode** | How a failed citation is handled: `strict` refuses the reply, `soft` drops the citation and keeps the answer. |

## Features

- `POST /v1/ask`: question in, answer with citations out. API-key authentication.
- Hybrid first stage: dense vector search and BM25 lexical search, merged into a pool of up to 150 passages.
- Reranking with a cross-encoder-style reranker (Qwen3-Reranker-0.6B) to the best five passages.
- Grounded answer prompt (version 3) with a fixed JSON reply format: decision, answer, at most two citations,
  missing information, confidence and reason code.
- Citation validator: every citation quote must be found in the cited passage. Soft citation mode by default in the
  bundled serving template.
- Route identity and a route–store guard: the service refuses a release that was built for another route.
- Optional conversation memory with example user profiles (personas).
- Optional situational split: a question that describes the asker's own situation is searched on its core question.
- Optional readable sources: title, authors and PDF page for each passage.
- Fake mode: a complete in-process stack (fixture store, fake models, fake answerer) for demos and tests.
- Raw answer-model request and response saved per answer, structured JSON logs.

## Quick start (no GPU, no database)

```bash
bash setup_venv.sh                # uv venv + frozen sync of the API and test stack, then a smoke test
.venv/bin/python -m pytest -q     # the offline test suite
.venv/bin/python demo.py          # one question through the real HTTP API, in fake mode
```

`setup_venv.sh` needs [`uv`](https://docs.astral.sh/uv/) on the `PATH` (or at `~/.local/bin/uv`). It ends with the smoke
test:

```
health      : api=ok ready=True store=ready model=ready embedder=fake reranker=fake
info        : version=0.2.0 route=None (unassigned) judged=73c2e7911d63… release=fixture-intake-mars-2026-cli-acceptance/release-attempt-0003 …
ask [en]   : VALIDATED/ANSWER contexts=5 attempts=1 total=29.594 ms schema_errors=0
ask [ro]   : VALIDATED/ANSWER contexts=5 attempts=1 total=32.404 ms schema_errors=0
SMOKE OK
```

The test suite reports `595 passed, 18 skipped` on a machine without GPU or live store. `demo.py` prints:

```
Question: What sowing density does the reference give for winter wheat?

Decision: ANSWER
Answer:   Reference E1 states: cropping season has started wetter and colder than usual. Similarly to the

[E1] “cropping season has started wetter and colder than usual. Similarly to the”
      from passage 78b933868c10db1fdc79e0c6ae21289335099656…

Route: unassigned  (fake mode: fixture store, models and answerer; the answer shows the pipeline and citation format, not answer quality)
```

Fake mode shows the pipeline and the reply format, not answer quality: the fake answerer quotes the first passage.
Ask your own question with `.venv/bin/python demo.py "your question" --lang en` (or `--lang ro`). Set
`CERES_DEMO_JSON=1` to also print the start of the full JSON response.

## Running live

Live mode **requires** a GPU, a loaded DDS release and model weights. Nothing in this section runs on a laptop.

### Prerequisites

1. **A DDS release in Postgres + pgvector.** The DDS tools create and load the schema (default `ceres_serve_v0`). Ceres
   only reads it. The active release must be built with the same chunk, represent and embed profiles as the ones
   bundled with the served route (compared byte for byte by hash); the route–store guard refuses to serve any other
   release. It must also carry the represented text the reranker reads. Ceres connects with a password-less DSN; libpq reads the
   password from `~/.pgpass`. A read-only database role is enough.
2. **An answer-model server.** An OpenAI-compatible chat server, for example `llama-server`, that lists the configured
   model name (`expect_model`, `qwen38` in the template) in `GET /v1/models` (until it does, `/health` is not ready) and runs with an 8192-token context (the
   answer profile's `context_tokens`). Ceres does not start or stop it.
3. **The embedder and reranker weights in a Hugging Face cache.** The route loads
   `jinaai/jina-embeddings-v5-text-nano` (revision `8a7f00aac812071b69403df470f1038ec85f8925`) and
   `Qwen/Qwen3-Reranker-0.6B` (revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`) on CUDA, offline, and checks the
   hashes of their weights and tokenizer files (not of the model code the snapshot carries, which runs with
   `trust_remote_code`; use a cache you trust). Set `CERES_MODEL_CACHE` to the cache root (the directory that holds `hub/`) so both are found there.
4. **The models extra.** Install torch and the model libraries: `bash setup_venv.sh --models`. Without `--models`, a
   later `setup_venv.sh` removes them again (the sync is exact).

### Configure and start

```bash
cp config.served.example.yaml config.yaml          # config.yaml is git-ignored
export CERES_STORE_DSN="postgresql://ceres_reader@db-host/veridian_rag"   # no password: libpq reads ~/.pgpass
export CERES_GPU_LOCK="$PWD/runtime/gpu.lock"
export CERES_RUNTIME_ROOT="$PWD/runtime"
export CERES_MODEL_CACHE="$HOME/.cache/huggingface"
export CERES_API_KEY="$(openssl rand -hex 32)"     # clients send it as X-API-Key
.venv/bin/ceres-api --config config.yaml
```

`.venv/bin/ceres-api --help`:

```
usage: ceres-api [-h] [--config CONFIG] [--host HOST] [--port PORT] [--fake]
                 [--no-auth] [--log-level LOG_LEVEL]

Serve the Ceres API

options:
  -h, --help            show this help message and exit
  --config CONFIG       config file (default: $CERES_CONFIG)
  --host HOST
  --port PORT
  --fake                fixture store and fake models; without it: PgStore +
                        the served route's models
  --no-auth             serve /v1/* without an API key — smoke tests only
  --log-level LOG_LEVEL
```

Without `--config`, Ceres reads `$CERES_CONFIG`, then `config.yaml` in the repository folder.

### Health check

```bash
curl -s http://127.0.0.1:8300/health
```

`"ready": true` means the store, the answer model, the embedder and the reranker are all usable. Right after start the
store reports `"warming"` until the startup warm-up ends. After a restart, also send a few warm-up questions before
real traffic.

## API summary

All `/v1/*` routes need the `X-API-Key` header.

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` | none | Liveness and readiness of store, answer model, embedder and reranker. |
| `GET` | `/v1/info` | key | What this process serves: version, route, release, profile hashes, config digest. |
| `GET` | `/v1/profiles` | key | The example personas available to conversation memory. |
| `POST` | `/v1/ask` | key | Answer one question: `{"question": "…", "lang": "en" \| "ro"}`. |

## Configuration overview

One YAML file, validated at startup.

- `config.served.example.yaml` — template for serving the bundled route against a DDS release.
- `config.example.yaml` — development template used by `demo.py`, the smoke test and fake mode.
- `${UPPER}` placeholders come from the environment; `${lower}` placeholders come from `paths` in the same file.
- Any scalar key can be overridden with `CERES_<SECTION>__<KEY>`, for example `CERES_API__PORT=8301`.
- Secrets never go in the file. The API key comes from the environment; the database password from `~/.pgpass`. A
  secret-looking value stops startup.

## Project layout

```
ceres/                  the package
  api.py                HTTP routes, admission, body limit, `ceres-api` entry point
  pipeline.py           one question end to end: retrieve, rerank, packet, answer, validate
  hybrid.py, hybrid_v1.py   dense + BM25 hybrid retrieval
  store.py              the Postgres/pgvector reader and the fixture store
  models.py             in-process embedder and reranker
  llm.py                OpenAI-compatible answer-model client
  memory.py             conversation memory, personas, situational split
  sources.py            readable sources (title, authors, page)
  config.py             config loading, placeholders, overrides, secret refusal
  contracts.py          request/response models and the answer schemas
  eval.py               `ceres-eval`: answer a question bank phase by phase
  testing.py            the fake stack used by fake mode and the tests
  assets/loop-v0/       the bundled route assets (route file, seven profiles, prompt)
  memory_prompts/       prompts for memory rewrite and situational split
  schemas/              JSON schemas of the answer packet and answer row
personas/               example personas for conversation memory (fictional)
tests/                  pytest suite and golden fixtures
tools/                  fixture builder and rerank parity check
demo.py                 one question in fake mode
smoke_test.py           offline smoke test, run by setup_venv.sh
setup_venv.sh           idempotent uv-based venv setup
```

## Testing

```bash
.venv/bin/python -m pytest -q
```

The suite runs offline on fixtures and fakes. Store tests start a disposable Postgres with pgvector (`pgserver`, part
of the `dev` extra). Two groups are opt-in: `live_store` tests run only when `CERES_TEST_LIVE_DSN` is set, and `models`
tests load the real embedder and reranker on CPU only when `CERES_TEST_MODELS=1` (requires the models extra and the
weights).

## Test data sources

The test data quotes passages, tables or page text from 6 public JRC MARS Bulletins (European Commission, Joint Research Centre), for testing only. Of these 6 documents, 5 state CC BY 4.0, 1 carries no reuse statement, so its reuse rests on the Commission's reuse decision 2011/833/EU. The rest of the test data is invented or recorded from this software's own runs. Third-party material keeps its own terms; the proprietary licence covers the rest of the repository.

## License

Proprietary. Copyright Veridian. See [LICENSE](LICENSE).
