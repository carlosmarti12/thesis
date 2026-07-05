from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, START, END

from src.agents import (
    domain_inference_agent,
    synonym_generator_agent,
    domain_critic_agent,
    semantic_ranker_agent,
    llm_ranking_agent,
    select_final_candidates,
)


FINAL_K = 5


class SynonymState(TypedDict, total=False):
    term: str
    topic: str                          # opaque dataset ID, kept for logging/grouping
    domain: str                         # inferred domain name (e.g., "finance", "law")
    candidates: List[str]
    filtered_candidates: List[str]
    ranked_candidates: List[Dict[str, Any]]        # retrieval: embedding-based ranking
    llm_ranked_candidates: List[str]               # LLM-based re-ranking (agentic)
    final_candidates: List[str]
    num_final_candidates: int
    log: List[str]


def domain_inference_node(state: SynonymState) -> SynonymState:
    """
    LangGraph node 1:
    Infers the academic domain from the input term.
    """
    term = state["term"]
    domain = domain_inference_agent(term=term)

    log = state.get("log", [])
    log.append(f"Domain inferred: {domain}")

    return {"domain": domain, "log": log}


def generator_node(state: SynonymState) -> SynonymState:
    """
    LangGraph node 2:
    Generates candidate synonyms using the inferred domain.
    """
    term = state["term"]
    domain = state.get("domain", "academic taxonomy")

    candidates = synonym_generator_agent(term=term, domain=domain)

    log = state.get("log", [])
    log.append(f"Generator produced {len(candidates)} candidates.")

    return {"candidates": candidates, "log": log}


def critic_node(state: SynonymState) -> SynonymState:
    """
    LangGraph node 3:
    Filters candidates using a domain critic agent.
    """
    term = state["term"]
    domain = state.get("domain", "academic taxonomy")
    candidates = state.get("candidates", [])

    filtered_candidates = domain_critic_agent(
        term=term,
        candidates=candidates,
        domain=domain,
    )

    log = state.get("log", [])
    log.append(f"Critic kept {len(filtered_candidates)} candidates.")

    return {"filtered_candidates": filtered_candidates, "log": log}


def ranker_node(state: SynonymState) -> SynonymState:
    """
    LangGraph node 4 (retrieval):
    Ranks candidates by embedding semantic similarity (deterministic, no LLM call).
    """
    term = state["term"]
    filtered_candidates = state.get("filtered_candidates", [])

    ranked_candidates = semantic_ranker_agent(
        term=term,
        candidates=filtered_candidates,
    )

    log = state.get("log", [])
    log.append("Retrieval ranker ordered candidates by embedding similarity.")

    return {"ranked_candidates": ranked_candidates, "log": log}


def llm_ranker_node(state: SynonymState) -> SynonymState:
    """
    LangGraph node 5 (LLM ranking, agentic):
    Re-ranks the embedding-ranked candidates using an LLM that also weighs
    domain fit. This is the retrieval + generation + LLM-ranking combination
    that produced the best experimental results, so it remains an LLM call
    rather than a purely deterministic step.
    """
    term = state["term"]
    domain = state.get("domain", "academic taxonomy")
    ranked_candidates = state.get("ranked_candidates", [])

    llm_ranked_candidates = llm_ranking_agent(
        term=term,
        ranked_candidates=ranked_candidates,
        domain=domain,
    )

    log = state.get("log", [])
    log.append(f"LLM ranker reordered {len(llm_ranked_candidates)} candidates.")

    return {"llm_ranked_candidates": llm_ranked_candidates, "log": log}


def finalize_node(state: SynonymState) -> SynonymState:
    """
    LangGraph node 6:
    Deterministically selects up to FINAL_K candidates, preferring the LLM's
    ranked order and padding from earlier stages (embedding ranking, filtered
    candidates, raw generated candidates) if fewer than FINAL_K survived.
    Never touches the ground truth.
    """
    final_candidates = select_final_candidates(
        llm_ranked_candidates=state.get("llm_ranked_candidates", []),
        ranked_candidates=state.get("ranked_candidates", []),
        filtered_candidates=state.get("filtered_candidates", []),
        candidates=state.get("candidates", []),
        k=FINAL_K,
    )
    num_final_candidates = len(final_candidates)

    log = state.get("log", [])
    log.append(f"num_final_candidates={num_final_candidates}: {final_candidates}")

    return {
        "final_candidates": final_candidates,
        "num_final_candidates": num_final_candidates,
        "log": log,
    }


def build_synonym_graph():
    """
    Builds and compiles the MAS v2 pipeline:
    domain_inference → generator → critic → ranker (retrieval) → llm_ranker (agentic) → finalize

    Input: only `term` (row["en"]). The ground truth (`en_synonym`) never
    enters this graph — it is used exclusively afterwards, for evaluation.
    """
    graph_builder = StateGraph(SynonymState)

    graph_builder.add_node("domain_inference", domain_inference_node)
    graph_builder.add_node("generator", generator_node)
    graph_builder.add_node("critic", critic_node)
    graph_builder.add_node("ranker", ranker_node)
    graph_builder.add_node("llm_ranker", llm_ranker_node)
    graph_builder.add_node("finalize", finalize_node)

    graph_builder.add_edge(START, "domain_inference")
    graph_builder.add_edge("domain_inference", "generator")
    graph_builder.add_edge("generator", "critic")
    graph_builder.add_edge("critic", "ranker")
    graph_builder.add_edge("ranker", "llm_ranker")
    graph_builder.add_edge("llm_ranker", "finalize")
    graph_builder.add_edge("finalize", END)

    return graph_builder.compile()


if __name__ == "__main__":
    app = build_synonym_graph()

    result = app.invoke(
        {
            "term": "stocks",
            "topic": "topic-2484",
            "log": [],
        }
    )

    print("\n=== RESULT ===")
    print("Term:    ", result["term"])
    print("Domain:  ", result.get("domain"))
    print("Candidates:", result.get("candidates"))
    print("Filtered:", result.get("filtered_candidates"))
    print("Ranked (retrieval):", result.get("ranked_candidates"))
    print("LLM ranked:", result.get("llm_ranked_candidates"))
    print("Final top-5:", result.get("final_candidates"))
    print("num_final_candidates:", result.get("num_final_candidates"))

    print("\n=== LOG ===")
    for item in result.get("log", []):
        print("-", item)
