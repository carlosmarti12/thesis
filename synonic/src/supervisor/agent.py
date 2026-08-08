"""
The supervisor: one LLM call per turn that inspects SupervisorState and
returns a validated SupervisorDecision. It never calls tools itself - that
is executor.execute_decision's job, so a malformed or adversarial decision
can't run anything outside the allowed action set.
"""

import json

from ..llm import ask_llm
from .prompts import SUPERVISOR_SYSTEM_PROMPT, build_supervisor_prompt
from .schemas import SupervisorDecision
from .state import SupervisorState


def _extract_json_object(text: str) -> dict:
    text = str(text).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in supervisor output.")

    return json.loads(text[start : end + 1])


def supervisor_agent(state: SupervisorState, model: str | None = None) -> SupervisorDecision:
    """Ask the LLM to decide the next action. Raises on malformed/invalid
    output - callers (runner.py) must catch and fall back to fallback_decision."""
    prompt = build_supervisor_prompt(state)
    response = ask_llm(SUPERVISOR_SYSTEM_PROMPT, prompt, model=model, temperature=0.0)
    data = _extract_json_object(response)
    return SupervisorDecision.model_validate(data)


def _last_index(history: list[dict], action: str) -> int:
    for i in range(len(history) - 1, -1, -1):
        if history[i]["action"] == action:
            return i
    return -1


def _pipeline_freshness(history: list[dict]) -> dict:
    """Whether merge/score/finalize each last ran *after* every stage that
    feeds it, going purely by order of appearance in action_history.

    This is what catches the ordering bug found in production: the LLM is
    free to call e.g. retrieve -> merge -> generate -> score -> finalize.
    That "merge" call happened before "generate" existed, so the candidates
    it generated never made it into merged_candidates/scored_candidates -
    they're silently orphaned. Checking only "was merge attempted at least
    once" (the old logic) can't see this, because merge WAS attempted - just
    with incomplete inputs. Freshness makes "generate" running after "merge"
    invalidate merge (and everything downstream of it) again, without ever
    having to physically clear merged_candidates/scored_candidates."""
    last_retrieve = _last_index(history, "retrieve")
    last_generate = _last_index(history, "generate")
    last_merge = _last_index(history, "merge")
    last_score = _last_index(history, "score")
    last_finalize = _last_index(history, "finalize")

    merge_fresh = last_merge != -1 and last_merge > last_retrieve and last_merge > last_generate
    score_fresh = merge_fresh and last_score != -1 and last_score > last_merge
    finalize_fresh = score_fresh and last_finalize != -1 and last_finalize > last_score

    return {"merge": merge_fresh, "score": score_fresh, "finalize": finalize_fresh}


def fallback_decision(state: SupervisorState) -> SupervisorDecision:
    """Deterministic backstop: retrieve -> generate -> merge -> score ->
    finalize -> finish. retrieve/generate are gated on whether they were
    *attempted* (not on non-empty result, since a source legitimately
    returning empty shouldn't be retried forever); merge/score/finalize are
    gated on _pipeline_freshness instead, so a stale merge (see above) gets
    re-run rather than treated as already done. Guarantees the pipeline
    always makes forward progress even if the supervisor LLM misbehaves."""
    history = state["action_history"]
    attempted = {item["action"] for item in history}
    fresh = _pipeline_freshness(history)

    if "retrieve" not in attempted:
        return SupervisorDecision(
            action="retrieve", reason="No retrieval performed yet.", parameters={"top_k": 12}, confidence=0.8
        )

    if "generate" not in attempted:
        return SupervisorDecision(
            action="generate", reason="No generated candidates yet.", parameters={"n": 6}, confidence=0.7
        )

    if not fresh["merge"]:
        return SupervisorDecision(
            action="merge",
            reason="Merge is missing, or stale relative to retrieve/generate.",
            parameters={"cap": 30},
            confidence=0.9,
        )

    if not fresh["score"]:
        return SupervisorDecision(
            action="score", reason="Score is missing, or stale relative to merge.", parameters={}, confidence=0.9
        )

    if not fresh["finalize"]:
        return SupervisorDecision(
            action="finalize",
            reason="Finalize is missing, or stale relative to score.",
            parameters={"top_k": 5},
            confidence=0.9,
        )

    return SupervisorDecision(
        action="finish", reason="Pipeline is fresh and final candidates already exist.", parameters={}, confidence=1.0
    )


def repeated_action(state: SupervisorState, decision: SupervisorDecision) -> bool:
    """True if the exact (action, parameters) pair already ran - guards
    against the LLM looping on a call that can't change the outcome."""
    signature = (decision.action, tuple(sorted(decision.parameters.items())))

    for item in state["action_history"]:
        previous = (item["action"], tuple(sorted(item["parameters"].items())))
        if previous == signature:
            return True

    return False


def can_finish(state: SupervisorState) -> bool:
    """Objective gate on "finish": there must actually be final candidates,
    produced by a "finalize" that is *fresh* relative to score/merge/
    retrieve/generate - not merely present from an earlier, now-stale run
    (see _pipeline_freshness). This is what forces a re-merge/re-score/
    re-finalize when e.g. "generate" runs after "merge" already did.

    Note: evaluate_candidate_quality's stricter score thresholds are computed
    by executor.execute_decision and logged (state["confidence"]) for
    observability, but don't block finishing here - this task always needs a
    top-5 list back for evaluation, even when no candidate clears a high
    quality bar, so gating on quality would risk looping forever on terms
    with no good synonym in the vocabulary."""
    if not (state["final_candidates"] and state["scored_candidates"]):
        return False

    return _pipeline_freshness(state["action_history"])["finalize"]
