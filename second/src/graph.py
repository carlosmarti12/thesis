from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, START, END

from src.agents import (
    domain_inference_agent,
    synonym_generator_agent,
    domain_critic_agent,
    semantic_ranker_agent,
)


class SynonymState(TypedDict, total=False):
    term: str
    topic: str                              # opaque dataset ID, kept for logging/grouping
    domain: str                             # inferred by Agent 1
    candidates: List[str]                   # raw output from Agent 2 (dual-strategy generator)
    filtered_candidates: List[str]          # after Agent 3 domain critic
    ranked_candidates: List[Dict[str, Any]] # final ranked list from Agent 4
    log: List[str]


def domain_inference_node(state: SynonymState) -> SynonymState:
    term = state["term"]
    domain = domain_inference_agent(term=term)

    log = state.get("log", [])
    log.append(f"[Agent 1] Domain inferred: {domain}")

    return {"domain": domain, "log": log}


def generator_node(state: SynonymState) -> SynonymState:
    term = state["term"]
    domain = state.get("domain", "academic taxonomy")

    candidates = synonym_generator_agent(term=term, domain=domain)

    log = state.get("log", [])
    log.append(f"[Agent 2] Generator produced {len(candidates)} candidates (dual-strategy).")

    return {"candidates": candidates, "log": log}


def critic_node(state: SynonymState) -> SynonymState:
    term = state["term"]
    domain = state.get("domain", "academic taxonomy")
    candidates = state.get("candidates", [])

    filtered_candidates = domain_critic_agent(
        term=term,
        candidates=candidates,
        domain=domain,
    )

    log = state.get("log", [])
    log.append(f"[Agent 3] Critic kept {len(filtered_candidates)} candidates.")

    return {"filtered_candidates": filtered_candidates, "log": log}


def ranker_node(state: SynonymState) -> SynonymState:
    term = state["term"]
    filtered_candidates = state.get("filtered_candidates", [])

    ranked_candidates = semantic_ranker_agent(
        term=term,
        candidates=filtered_candidates,
    )

    log = state.get("log", [])
    log.append(f"[Agent 4] Ranker ordered {len(ranked_candidates)} candidates by semantic similarity.")

    return {"ranked_candidates": ranked_candidates, "log": log}


def build_synonym_graph():
    """
    Builds and compiles the 4-agent MAS pipeline:

        domain_inference → generator → critic → ranker

    Agent 1 (domain_inference): infers the academic domain from the input term.
    Agent 2 (generator):        dual-strategy generation — lexical synonyms +
                                compound/domain forms — for maximum candidate diversity.
    Agent 3 (critic):           domain expert filters out domain-irrelevant candidates.
    Agent 4 (ranker):           deterministic embedding-based ranking; no LLM call.

    Evaluation: top-k match — correct if the ground-truth synonym appears anywhere
    in the final ranked list. Both exact and fuzzy (token_set_ratio >= 80) variants
    are reported.
    """
    graph_builder = StateGraph(SynonymState)

    graph_builder.add_node("domain_inference", domain_inference_node)
    graph_builder.add_node("generator", generator_node)
    graph_builder.add_node("critic", critic_node)
    graph_builder.add_node("ranker", ranker_node)

    graph_builder.add_edge(START, "domain_inference")
    graph_builder.add_edge("domain_inference", "generator")
    graph_builder.add_edge("generator", "critic")
    graph_builder.add_edge("critic", "ranker")
    graph_builder.add_edge("ranker", END)

    return graph_builder.compile()


if __name__ == "__main__":
    app = build_synonym_graph()

    result = app.invoke({"term": "stocks", "topic": "topic-2484", "log": []})

    print("\n=== RESULT ===")
    print("Term:      ", result["term"])
    print("Domain:    ", result.get("domain"))
    print("Generated: ", result.get("candidates"))
    print("Filtered:  ", result.get("filtered_candidates"))
    print("Ranked:    ", result.get("ranked_candidates"))

    print("\n=== LOG ===")
    for item in result.get("log", []):
        print("-", item)
