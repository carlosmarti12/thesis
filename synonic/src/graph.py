"""
MAS graph construction: wires the standalone agent functions from agents.py
into three LangGraph pipeline variants.

- "base"        (was mas_langgraph / v1): generator -> retrieval -> merge ->
                 verification (0.75*semantic + 0.25*fuzzy) -> ranking (top 5).
- "llm_ranker"  (was mas_v2_llm_ranker / v2): retrieval -> generator -> merge
                 -> an LLM re-orders the full candidate list. Confirmed
                 negative ablation in the original experiments - kept for
                 comparison, not because it's expected to win.
- "safe_hybrid" (was mas_v3_safe_hybrid / v3): v2 + a safe_finalizer_agent
                 blending 0.9*embedding_sim + 0.1*llm_rank_bonus.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import (
    generator_agent,
    llm_ranking_agent,
    merge_agent,
    ranking_agent,
    retrieval_agent,
    safe_finalizer_agent,
    verification_agent,
)


class SynonymState(TypedDict):
    term: str
    generated_candidates: list[str]
    retrieved_candidates: list[str]
    merged_candidates: list[str]
    verified_candidates: list[dict]
    llm_ranked_candidates: list[str]
    final_candidates: list[str]


def empty_state(term: str) -> SynonymState: # creas el estado inicial para un término dado, con listas vacías para los candidatos
    return {
        "term": term,
        "generated_candidates": [],
        "retrieved_candidates": [],
        "merged_candidates": [],
        "verified_candidates": [],
        "llm_ranked_candidates": [],
        "final_candidates": [],
    }

                    # embeding, lista de palabras candidatas, lista de embeddings de las palabras candidatas, modelo LLM, variante del grafo
def build_mas_graph(embedder, candidate_pool_terms, candidate_pool_embeddings, model: str | None = None, variant: str = "base"):
    if variant not in {"base", "llm_ranker", "safe_hybrid"}:
        raise ValueError(f"Unknown MAS variant: {variant}")

    def generator_node(state: SynonymState, n: int): # genera candidatos usando el agente generador, devuelve un diccionario con la clave "generated_candidates" y la lista de candidatos generados
        return {"generated_candidates": generator_agent(state["term"], model=model, n=n)}

    def retrieval_node(state: SynonymState, top_k: int): # Este nodo busca los términos más parecidos dentro del índice de WordNet.
        candidates = retrieval_agent(
            state["term"], embedder, candidate_pool_terms, candidate_pool_embeddings, top_k=top_k
        )
        return {"retrieved_candidates": candidates}

    def merge_node(state: SynonymState, order: str, cap: int | None): # Este nodo combina los candidatos generados y recuperados.
        merged = merge_agent(
            state["term"],
            state.get("generated_candidates", []),
            state.get("retrieved_candidates", []),
            order=order,
            cap=cap,
        )
        return {"merged_candidates": merged}

    def verification_node(state: SynonymState):  # Este nodo evalúa cada candidato
        return {"verified_candidates": verification_agent(state["term"], state.get("merged_candidates", []), embedder)}

    def ranking_node(state: SynonymState):
        return {"final_candidates": ranking_agent(state.get("verified_candidates", []), top_k=5)}

    def llm_ranking_node(state: SynonymState, keep_all_via_append: bool):
        ordered = llm_ranking_agent(
            state["term"], state.get("merged_candidates", []), model=model, keep_all_via_append=keep_all_via_append
        )
        key = "final_candidates" if keep_all_via_append else "llm_ranked_candidates"
        return {key: ordered}

    def safe_finalizer_node(state: SynonymState):
        final = safe_finalizer_agent(
            state["term"],
            state.get("merged_candidates", []),
            state.get("llm_ranked_candidates", []),
            embedder,
            top_k=5,
        )
        return {"final_candidates": final}

    graph = StateGraph(SynonymState)

    if variant == "base":
        graph.add_node("generator_agent", lambda s: generator_node(s, n=8))
        graph.add_node("retrieval_agent", lambda s: retrieval_node(s, top_k=8))
        graph.add_node("merge_agent", lambda s: merge_node(s, order="generated_first", cap=None))
        graph.add_node("verification_agent", verification_node)
        graph.add_node("ranking_agent", ranking_node)

        graph.add_edge(START, "generator_agent")
        graph.add_edge("generator_agent", "retrieval_agent")
        graph.add_edge("retrieval_agent", "merge_agent")
        graph.add_edge("merge_agent", "verification_agent")
        graph.add_edge("verification_agent", "ranking_agent")
        graph.add_edge("ranking_agent", END)

    elif variant == "llm_ranker":
        graph.add_node("retrieval_agent", lambda s: retrieval_node(s, top_k=15))
        graph.add_node("generator_agent", lambda s: generator_node(s, n=5))
        graph.add_node("merge_agent", lambda s: merge_node(s, order="retrieved_first", cap=20))
        graph.add_node("llm_ranking_agent", lambda s: llm_ranking_node(s, keep_all_via_append=True))

        graph.add_edge(START, "retrieval_agent")
        graph.add_edge("retrieval_agent", "generator_agent")
        graph.add_edge("generator_agent", "merge_agent")
        graph.add_edge("merge_agent", "llm_ranking_agent")
        graph.add_edge("llm_ranking_agent", END)

    else:  # safe_hybrid
        graph.add_node("retrieval_agent", lambda s: retrieval_node(s, top_k=15))
        graph.add_node("generator_agent", lambda s: generator_node(s, n=5))
        graph.add_node("merge_agent", lambda s: merge_node(s, order="retrieved_first", cap=20))
        graph.add_node("llm_ranking_agent", lambda s: llm_ranking_node(s, keep_all_via_append=False))
        graph.add_node("safe_finalizer_agent", safe_finalizer_node)

        graph.add_edge(START, "retrieval_agent")
        graph.add_edge("retrieval_agent", "generator_agent")
        graph.add_edge("generator_agent", "merge_agent")
        graph.add_edge("merge_agent", "llm_ranking_agent")
        graph.add_edge("llm_ranking_agent", "safe_finalizer_agent")
        graph.add_edge("safe_finalizer_agent", END)

    return graph.compile()


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    from .open_vocab import get_wordnet_index

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index_terms, index_embeddings = get_wordnet_index(embedder, "sentence-transformers/all-MiniLM-L6-v2")

    app = build_mas_graph(embedder, index_terms, index_embeddings, variant="base")
    result = app.invoke(empty_state("stocks"))
    print("Final candidates:", result["final_candidates"])
