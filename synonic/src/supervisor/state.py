"""
Explicit, inspectable state for the supervisor loop. The supervisor LLM only
ever sees this dict rendered as text (prompts.build_supervisor_prompt) - no
hidden closures, no state it can't account for.
"""

from typing import TypedDict


class SupervisorState(TypedDict):
    term: str

    generated_candidates: list[str]
    retrieved_candidates: list[str]
    merged_candidates: list[str]
    scored_candidates: list[dict]
    final_candidates: list[str]

    current_action: str | None
    action_parameters: dict
    action_history: list[dict]
    pending_decision: dict

    iteration: int
    llm_calls: int
    confidence: float
    finished: bool
    error: str | None


def empty_supervisor_state(term: str) -> SupervisorState:
    return {
        "term": term,
        "generated_candidates": [],
        "retrieved_candidates": [],
        "merged_candidates": [],
        "scored_candidates": [],
        "final_candidates": [],
        "current_action": None,
        "action_parameters": {},
        "action_history": [],
        "pending_decision": {},
        "iteration": 0,
        "llm_calls": 0,
        "confidence": 0.0,
        "finished": False,
        "error": None,
    }
