"""
The only place allowed to turn a SupervisorDecision into an actual tool
call. The supervisor (agent.py) never calls agents.py functions directly -
it only produces a decision, and every LLM-supplied parameter is clamped
here so a bad decision can't request unbounded work (e.g. top_k=1000000).
"""

from ..agents import (
    generator_agent,
    merge_agent,
    ranking_agent,
    retrieval_agent,
    verification_agent,
)
from .schemas import SupervisorDecision
from .state import SupervisorState
from .tools import evaluate_candidate_quality


def execute_decision(
    state: SupervisorState,
    decision: SupervisorDecision,
    *,
    embedder,
    candidate_pool_terms,
    candidate_pool_embeddings,
    model: str | None = None,
) -> dict:
    """Run the tool decision.action refers to. Returns a partial state
    update dict (merged into SupervisorState by runner.apply_tool_result)."""
    action = decision.action
    parameters = decision.parameters or {}

    if action == "retrieve":
        top_k = max(1, min(int(parameters.get("top_k", 10)), 50))
        candidates = retrieval_agent(
            state["term"], embedder, candidate_pool_terms, candidate_pool_embeddings, top_k=top_k
        )
        return {"retrieved_candidates": candidates}

    if action == "generate":
        n = max(1, min(int(parameters.get("n", 6)), 20))
        candidates = generator_agent(state["term"], model=model, n=n)
        return {"generated_candidates": candidates}

    if action == "merge":
        cap = max(5, min(int(parameters.get("cap", 30)), 100))
        order = parameters.get("order", "retrieved_first")
        if order not in {"retrieved_first", "generated_first"}:
            order = "retrieved_first"
        merged = merge_agent(
            state["term"],
            state["generated_candidates"],
            state["retrieved_candidates"],
            order=order,
            cap=cap,
        )
        return {"merged_candidates": merged}

    if action == "score":
        candidates = state["merged_candidates"] or state["retrieved_candidates"] or state["generated_candidates"]
        scored = verification_agent(state["term"], candidates, embedder)
        quality = evaluate_candidate_quality(scored)
        return {"scored_candidates": scored, "confidence": quality["confidence"]}

    if action == "finalize":
        top_k = max(1, min(int(parameters.get("top_k", 5)), 10))
        final = ranking_agent(state["scored_candidates"], top_k=top_k)
        return {"final_candidates": final}

    if action == "finish":
        return {}

    raise ValueError(f"Unknown action: {action}")
