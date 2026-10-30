"""Route-freeze checks (v3.5): one token floor and the declared reranker text.

Mirrors of the loop's own rules, not re-interpretations (a rule that drifts from the loop would let
Ceres serve a route the loop refuses to freeze):

  * `check_token_floor` = `loop-v0/04_rag/run_route.py: check_route_contract`, token-floor half. When the
    chunk profile declares `min_child_tokens` (chunk-bridge-v3.5 and later), the retrieve profile's
    `min_context_tokens` (absent = 0) must equal it, and so must a represent profile's
    `min_body_tokens` when it declares one. A chunk profile WITHOUT the key (v3.4 and earlier)
    carries no floor and is never refused here.
  * `rerank_text_choice` = `slot_adapter.py: rerank_text_choice` (RERANK_TEXT_CHOICES): the rerank
    profile's `rerank_text` (absent = payload) is `payload` or `represented`, and the identity
    rerank profile (`none-v1`, `operation: identity`) must not declare it at all.
  * `check_route_contract` = `run_route.check_route_contract`, rerank_text half: the ROUTE's `rerank_text`
    (absent = payload) must be one of the choices, a route that declares it must have a rerank
    slot, and it must equal its rerank profile's value.

Pure functions over parsed YAML; callers turn `RouteContractError` into their own refusal
(`eval.Refused`, `pipeline.PipelineError`).
"""

from __future__ import annotations

from typing import Any, Mapping

RERANK_TEXT_CHOICES = ("payload", "represented")
RERANK_TEXT_PAYLOAD, RERANK_TEXT_REPRESENTED = RERANK_TEXT_CHOICES
RERANK_IDENTITY_PROFILE_ID = "none-v1"          # slot_adapter.RERANK_IDENTITY


class RouteContractError(ValueError):
    """A route or profile violates a v3.5 contract value. The message names the rule."""


def _is_identity(profile: Mapping[str, Any]) -> bool:
    return (profile.get("profile_id") == RERANK_IDENTITY_PROFILE_ID
            or profile.get("operation") == "identity")


def rerank_text_choice(profile: Mapping[str, Any] | None) -> str:
    """Which text the reranker scores: the rerank profile's `rerank_text`, absent = payload."""
    profile = profile or {}
    value = profile.get("rerank_text", RERANK_TEXT_PAYLOAD)
    if value not in RERANK_TEXT_CHOICES:
        raise RouteContractError(f"rerank_text must be one of {list(RERANK_TEXT_CHOICES)}: {value!r}")
    if _is_identity(profile) and "rerank_text" in profile:
        raise RouteContractError("the identity rerank profile scores nothing; it must not declare rerank_text")
    return value


def check_token_floor(chunk: Mapping[str, Any] | None, retrieve: Mapping[str, Any] | None,
                      represent: Mapping[str, Any] | None = None, *, route_name: str = "?") -> None:
    """One floor shared by chunking, representation and retrieval (v3.5+ chunk profiles only)."""
    chunk, retrieve, represent = chunk or {}, retrieve or {}, represent or {}
    floor = chunk.get("min_child_tokens")
    if floor is None:
        return                                      # v3.4 and earlier: no floor declared, nothing to refuse
    context_floor = retrieve.get("min_context_tokens", 0)
    if context_floor != floor:
        raise RouteContractError(
            f"route {route_name}: retrieve min_context_tokens {context_floor!r} differs from the chunk"
            f" profile's min_child_tokens {floor!r} (one floor)")
    body_floor = represent.get("min_body_tokens")
    if body_floor is not None and body_floor != floor:
        raise RouteContractError(
            f"route {route_name}: represent min_body_tokens {body_floor!r} differs from the chunk"
            f" profile's min_child_tokens {floor!r}")


def check_route_contract(route: Mapping[str, Any], profiles: Mapping[str, Mapping[str, Any] | None]) -> str:
    """Both halves over a route declaration and its parsed slot profiles; returns the rerank text."""
    name = str(route.get("route_name") or "?")
    check_token_floor(profiles.get("chunk"), profiles.get("retrieve"), profiles.get("represent"),
                      route_name=name)
    declared = route.get("rerank_text", RERANK_TEXT_PAYLOAD)
    if declared not in RERANK_TEXT_CHOICES:
        raise RouteContractError(f"route rerank_text must be one of {list(RERANK_TEXT_CHOICES)}: {declared!r}")
    if "rerank_text" in route and profiles.get("rerank") is None:
        raise RouteContractError(f"route {name} declares rerank_text but has no rerank slot")
    used = rerank_text_choice(profiles.get("rerank"))
    if declared != used:
        raise RouteContractError(f"route {name}: rerank_text {declared!r} differs from its rerank profile's {used!r}")
    return used


__all__ = ["RERANK_TEXT_CHOICES", "RERANK_TEXT_PAYLOAD", "RERANK_TEXT_REPRESENTED", "RouteContractError",
           "rerank_text_choice", "check_token_floor", "check_route_contract"]
