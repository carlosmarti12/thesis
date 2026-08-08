"""
LangGraph wrapper around the same supervisor/tool functions runner.py uses
in a plain Python loop. Prefer run_supervisor (runner.py) for scripted/eval
use - this expresses the identical control flow as a graph, useful once the
manual loop is verified and you want LangGraph's tracing/visualization.
"""

from langgraph.graph import END, START, StateGraph

from .agent import can_finish, fallback_decision, repeated_action, supervisor_agent
from .executor import execute_decision
from .runner import MAX_ITERATIONS, apply_tool_result, force_finish
from .schemas import ALLOWED_ACTIONS, SupervisorDecision
from .state import SupervisorState, empty_supervisor_state


def build_supervisor_graph(
    embedder,
    candidate_pool_terms,
    candidate_pool_embeddings,
    model: str | None = None,
    max_iterations: int = MAX_ITERATIONS,
):
    graph = StateGraph(SupervisorState)

    def supervisor_node(state: SupervisorState) -> dict:
        try:
            decision = supervisor_agent(state, model=model)
            llm_calls = state.get("llm_calls", 0) + 1
        except Exception as exc:
            decision = fallback_decision(state)
            llm_calls = state.get("llm_calls", 0)
            state = {**state, "error": f"supervisor_llm_failed: {exc}"}

        if decision.action not in ALLOWED_ACTIONS:
            decision = fallback_decision(state)

        if repeated_action(state, decision):
            decision = fallback_decision(state)

        if decision.action == "finish" and not can_finish(state):
            decision = fallback_decision(state)

        return {"pending_decision": decision.model_dump(), "llm_calls": llm_calls}

    def tool_node(state: SupervisorState) -> dict:
        decision = SupervisorDecision.model_validate(state["pending_decision"])
        result = execute_decision(
            state,
            decision,
            embedder=embedder,
            candidate_pool_terms=candidate_pool_terms,
            candidate_pool_embeddings=candidate_pool_embeddings,
            model=model,
        )
        updated = apply_tool_result(state, decision, result)
        if decision.action == "finish":
            updated["finished"] = True
        return updated

    def force_finalize_node(state: SupervisorState) -> dict:
        return force_finish(
            state,
            embedder=embedder,
            candidate_pool_terms=candidate_pool_terms,
            candidate_pool_embeddings=candidate_pool_embeddings,
            model=model,
        )

    def route_after_supervisor(state: SupervisorState) -> str:
        if state["iteration"] >= max_iterations:
            return "force_finalize"
        if state["pending_decision"].get("action") == "finish":
            return "finish"
        return "execute_tool"

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("execute_tool", tool_node)
    graph.add_node("force_finalize", force_finalize_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"execute_tool": "execute_tool", "finish": END, "force_finalize": "force_finalize"},
    )
    graph.add_edge("execute_tool", "supervisor")
    graph.add_edge("force_finalize", END)

    return graph.compile()


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    from ..open_vocab import get_wordnet_index

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index_terms, index_embeddings = get_wordnet_index(embedder, "sentence-transformers/all-MiniLM-L6-v2")

    app = build_supervisor_graph(embedder, index_terms, index_embeddings)
    result = app.invoke(empty_supervisor_state("stocks"))
    print("Final candidates:", result["final_candidates"])
