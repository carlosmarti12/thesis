"""
Plain-Python control loop for the supervisor (decide -> execute -> observe ->
repeat). Verified here before trusting the LangGraph wrapper in graph.py,
which reuses these exact functions for its control flow.
"""

from .agent import can_finish, fallback_decision, repeated_action, supervisor_agent
from .executor import execute_decision
from .schemas import ALLOWED_ACTIONS, SupervisorDecision
from .state import SupervisorState, empty_supervisor_state

MAX_ITERATIONS = 10


def apply_tool_result(state: SupervisorState, decision: SupervisorDecision, result: dict) -> SupervisorState:
    """Merge a tool's result into state and record what happened, so the next
    supervisor turn (and any post-hoc log analysis) can see the full history."""
    updated = dict(state)
    updated.update(result)

    history_entry = {
        "iteration": state["iteration"],
        "action": decision.action,
        "parameters": decision.parameters,
        "reason": decision.reason,
        "result_summary": {
            key: (len(value) if isinstance(value, list) else value) for key, value in result.items()
        },
    }

    updated["action_history"] = [*state["action_history"], history_entry]
    updated["current_action"] = decision.action
    updated["action_parameters"] = decision.parameters
    updated["iteration"] = state["iteration"] + 1

    if decision.action == "generate":
        updated["llm_calls"] = state.get("llm_calls", 0) + 1

    return updated


def force_finish(
    state: SupervisorState,
    *,
    embedder,
    candidate_pool_terms,
    candidate_pool_embeddings,
    model: str | None,
) -> SupervisorState:
    """Deterministic emergency exit when max_iterations is hit: finalize
    whatever has been scored so far instead of returning nothing."""
    if state["scored_candidates"] and not state["final_candidates"]:
        decision = SupervisorDecision(
            action="finalize",
            reason="Iteration limit reached; forcing finalize on existing scores.",
            parameters={"top_k": 5},
        )
        result = execute_decision(
            state,
            decision,
            embedder=embedder,
            candidate_pool_terms=candidate_pool_terms,
            candidate_pool_embeddings=candidate_pool_embeddings,
            model=model,
        )
        state = apply_tool_result(state, decision, result)

    state = dict(state)
    state["finished"] = True
    state["error"] = state.get("error") or "max_iterations_reached"
    return state


def run_supervisor(
    term: str,
    *,
    embedder,
    candidate_pool_terms,
    candidate_pool_embeddings,
    model: str | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> SupervisorState:
    state = empty_supervisor_state(term)

    while not state["finished"]:
        if state["iteration"] >= max_iterations:
            state = force_finish(
                state,
                embedder=embedder,
                candidate_pool_terms=candidate_pool_terms,
                candidate_pool_embeddings=candidate_pool_embeddings,
                model=model,
            )
            break

        try:
            decision = supervisor_agent(state, model=model)
            state = dict(state)
            state["llm_calls"] = state.get("llm_calls", 0) + 1
        except Exception as exc:
            decision = fallback_decision(state)
            state = dict(state)
            state["error"] = f"supervisor_llm_failed: {exc}"

        if decision.action not in ALLOWED_ACTIONS:
            decision = fallback_decision(state)

        if repeated_action(state, decision):
            decision = fallback_decision(state)

        if decision.action == "finish" and not can_finish(state):
            decision = fallback_decision(state)

        result = execute_decision(
            state,
            decision,
            embedder=embedder,
            candidate_pool_terms=candidate_pool_terms,
            candidate_pool_embeddings=candidate_pool_embeddings,
            model=model,
        )

        state = apply_tool_result(state, decision, result)

        if decision.action == "finish":
            state["finished"] = True

    return state


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    from ..open_vocab import get_wordnet_index

    term = "stocks"
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index_terms, index_embeddings = get_wordnet_index(embedder, "sentence-transformers/all-MiniLM-L6-v2")

    final_state = run_supervisor(term, embedder=embedder, candidate_pool_terms=index_terms, candidate_pool_embeddings=index_embeddings)

    print("Final candidates:", final_state["final_candidates"])
    print("Iterations:", final_state["iteration"], "LLM calls:", final_state["llm_calls"])
    print("Action history:")
    for entry in final_state["action_history"]:
        print(" ", entry)
