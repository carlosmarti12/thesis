from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, START, END

from src.agents import (
    synonym_generator_agent,
    domain_critic_agent,
    semantic_ranker_agent,
    final_selector_agent,
)


class SynonymState(TypedDict, total=False):
    term: str
    topic: str
    candidates: List[str]
    filtered_candidates: List[str]
    ranked_candidates: List[Dict[str, Any]]
    final_answer: str
    log: List[str]


def generator_node(state: SynonymState) -> SynonymState:
    """
    LangGraph node 1:
    Generates candidate synonyms.
    """

    term = state["term"]
    topic = state.get("topic", "")

    candidates = synonym_generator_agent(term=term, topic=topic)

    log = state.get("log", [])
    log.append(f"Generator produced {len(candidates)} candidates.")

    return {
        "candidates": candidates,
        "log": log,
    }


def critic_node(state: SynonymState) -> SynonymState:
    """
    LangGraph node 2:
    Filters candidates using a domain critic agent.
    """

    term = state["term"]
    topic = state.get("topic", "")
    candidates = state.get("candidates", [])

    filtered_candidates = domain_critic_agent(
        term=term,
        candidates=candidates,
        topic=topic,
    )

    log = state.get("log", [])
    log.append(f"Critic kept {len(filtered_candidates)} candidates.")

    return {
        "filtered_candidates": filtered_candidates,
        "log": log,
    }


def ranker_node(state: SynonymState) -> SynonymState:
    """
    LangGraph node 3:
    Ranks candidates using semantic similarity.
    """

    term = state["term"]
    filtered_candidates = state.get("filtered_candidates", [])

    ranked_candidates = semantic_ranker_agent(
        term=term,
        candidates=filtered_candidates,
    )

    log = state.get("log", [])
    log.append("Ranker ordered candidates by semantic similarity.")

    return {
        "ranked_candidates": ranked_candidates,
        "log": log,
    }


def final_selector_node(state: SynonymState) -> SynonymState:
    """
    LangGraph node 4:
    Chooses the final answer.
    """

    term = state["term"]
    topic = state.get("topic", "")
    ranked_candidates = state.get("ranked_candidates", [])

    final_answer = final_selector_agent(
        term=term,
        ranked_candidates=ranked_candidates,
        topic=topic,
    )

    log = state.get("log", [])
    log.append(f"Final selector chose: {final_answer}")

    return {
        "final_answer": final_answer,
        "log": log,
    }


def build_synonym_graph():
    """
    Builds and compiles the LangGraph MAS architecture.
    """

    graph_builder = StateGraph(SynonymState)

    graph_builder.add_node("generator", generator_node)
    graph_builder.add_node("critic", critic_node)
    graph_builder.add_node("ranker", ranker_node)
    graph_builder.add_node("final_selector", final_selector_node)

    graph_builder.add_edge(START, "generator")
    graph_builder.add_edge("generator", "critic")
    graph_builder.add_edge("critic", "ranker")
    graph_builder.add_edge("ranker", "final_selector")
    graph_builder.add_edge("final_selector", END)

    return graph_builder.compile()


if __name__ == "__main__":
    app = build_synonym_graph()

    result = app.invoke(
        {
            "term": "stocks",
            "topic": "finance",
            "log": [],
        }
    )

    print("\n=== RESULT ===")
    print("Term:", result["term"])
    print("Topic:", result.get("topic"))
    print("Candidates:", result.get("candidates"))
    print("Filtered:", result.get("filtered_candidates"))
    print("Ranked:", result.get("ranked_candidates"))
    print("Final answer:", result.get("final_answer"))

    print("\n=== LOG ===")
    for item in result.get("log", []):
        print("-", item)