from typing import TypedDict, List

from langgraph.graph import StateGraph, START, END

from src.agents import generator_agent, filter_agent


class SynonymState(TypedDict, total=False):
    term: str
    topic: str
    raw_candidates: List[str]    # 10 from Agent 1
    filtered_candidates: List[str]  # best 8 from Agent 2
    prediction: str              # top-1 (first in filtered list)
    log: List[str]


def _generator_node(state: SynonymState) -> dict:
    candidates = generator_agent(state["term"])
    return {
        "raw_candidates": candidates,
        "log": state.get("log", []) + [f"[generator] {len(candidates)} candidates"],
    }


def _filter_node(state: SynonymState) -> dict:
    filtered = filter_agent(state["term"], state.get("raw_candidates", []))
    prediction = filtered[0] if filtered else ""
    return {
        "filtered_candidates": filtered,
        "prediction": prediction,
        "log": state.get("log", []) + [f"[filter] kept {len(filtered)} candidates, top='{prediction}'"],
    }


def build_graph():
    """
    2-agent sequential graph:
        generator → filter → END
    """
    g = StateGraph(SynonymState)
    g.add_node("generator", _generator_node)
    g.add_node("filter", _filter_node)
    g.add_edge(START, "generator")
    g.add_edge("generator", "filter")
    g.add_edge("filter", END)
    return g.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({"term": "stocks", "topic": "test", "log": []})
    print("Raw candidates: ", result.get("raw_candidates"))
    print("Filtered (best 8):", result.get("filtered_candidates"))
    print("Prediction:", result.get("prediction"))
    print("Log:", result.get("log"))
