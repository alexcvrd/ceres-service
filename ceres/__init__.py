"""Ceres v0 — the serving side of DDS route r1.

The package is four adapters behind three protocols (`store.Store`, `models.Models`, `llm.LLM`)
and one module of pure functions (`pipeline`) that composes them. `api` is the only module that
knows about HTTP. The fake adapters ship in the package; PgStore, the real models and the real LLM swap
implementations without touching `pipeline.py` or `api.py`.
"""

__version__ = "0.2.0"
