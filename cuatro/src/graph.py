import operator
import warnings
from typing import Annotated, TypedDict, List, Dict, Any

import numpy as np
from langgraph.graph import StateGraph, START, END

from src.agents import (
    domain_classifier_agent,
    lexical_generator_agent,
    semantic_generator_agent,
    contextual_generator_agent,
    critic_agent,
    reflection_agent,
)
from src.embeddings import embed, batch_cosine_sim

# Pre-filter pool size sent to the critic — reduces prompt length and improves quality.
_PRE_FILTER_TOP_N = 15

# Composite score weights: embedding similarity + cross-agent agreement + domain proximity.
_W_EMBED = 0.5
_W_AGREE = 0.3
_W_DOMAIN = 0.2


class SynonymState(TypedDict, total=False):
    # Input
    term: str
    topic: str

    # After domain_classifier
    domain: str
    term_type: str
    domain_embedding: List[float]   # cached to avoid recomputing in reflection loop

    # After parallel generators (distinct keys — no write conflict)
    lexical_candidates: List[str]
    semantic_candidates: List[str]
    contextual_candidates: List[str]

    # After aggregator
    candidate_pool: List[str]
    candidate_scores: Dict[str, Dict[str, float]]

    # After critic
    filtered_candidates: List[str]

    # After consensus_ranker
    ranked_candidates: List[Dict[str, Any]]
    top_score: float
    prediction: str

    # Reflection control
    has_reflected: bool

    # Log — reducer required: all 3 parallel generators write to this field simultaneously.
    log: Annotated[List[str], operator.add]


def _domain_classifier_node(state: SynonymState) -> dict:
    term = state["term"]
    domain, term_type = domain_classifier_agent(term)
    domain_embedding = embed(domain).tolist()
    return {
        "domain": domain,
        "term_type": term_type,
        "domain_embedding": domain_embedding,
        "log": [f"[domain_classifier] domain='{domain}', type='{term_type}'"],
    }


def _lexical_generator_node(state: SynonymState) -> dict:
    term = state["term"]
    domain = state.get("domain", "academic taxonomy")
    term_type = state.get("term_type", "simple")
    candidates = lexical_generator_agent(term, domain, term_type)
    return {
        "lexical_candidates": candidates,
        "log": [f"[lexical_generator] {len(candidates)} candidates"],
    }


def _semantic_generator_node(state: SynonymState) -> dict:
    term = state["term"]
    domain = state.get("domain", "academic taxonomy")
    candidates = semantic_generator_agent(term, domain)
    return {
        "semantic_candidates": candidates,
        "log": [f"[semantic_generator] {len(candidates)} candidates"],
    }


def _contextual_generator_node(state: SynonymState) -> dict:
    term = state["term"]
    domain = state.get("domain", "academic taxonomy")
    candidates = contextual_generator_agent(term, domain)
    return {
        "contextual_candidates": candidates,
        "log": [f"[contextual_generator] {len(candidates)} candidates"],
    }


def _aggregator_node(state: SynonymState) -> dict:
    term = state["term"]
    domain_emb = np.array(state.get("domain_embedding", []), dtype=np.float32)

    lex = state.get("lexical_candidates", [])
    sem = state.get("semantic_candidates", [])
    ctx = state.get("contextual_candidates", [])

    # Track which agents proposed each candidate (for agreement scoring)
    agreement_count: Dict[str, int] = {}
    for source in [lex, sem, ctx]:
        for c in source:
            key = c.lower().strip()
            agreement_count[key] = agreement_count.get(key, 0) + 1

    # Deduplicate, preserving first-seen casing
    seen: set = set()
    pool: List[str] = []
    original_key = term.lower().strip()
    for c in lex + sem + ctx:
        key = c.lower().strip()
        if key and key != original_key and key not in seen:
            seen.add(key)
            pool.append(c)

    if not pool:
        return {
            "candidate_pool": [],
            "candidate_scores": {},
            "log": ["[aggregator] empty pool — no candidates generated"],
        }

    # Embed all candidates and the term
    term_emb = embed(term)
    cand_embs = np.stack([embed(c) for c in pool], axis=0)
    embed_sims = batch_cosine_sim(term_emb, cand_embs)

    # Domain similarity (weak signal — domain embedding vs candidate)
    domain_sims: np.ndarray
    if len(domain_emb) > 0:
        domain_sims = batch_cosine_sim(domain_emb, cand_embs)
    else:
        domain_sims = np.zeros(len(pool))

    # Build scores dict and composite score
    scores: Dict[str, Dict[str, float]] = {}
    composite: List[float] = []
    for i, c in enumerate(pool):
        key = c.lower().strip()
        agree_norm = agreement_count.get(key, 1) / 3.0
        es = float(embed_sims[i])
        ds = float(domain_sims[i])
        comp = _W_EMBED * es + _W_AGREE * agree_norm + _W_DOMAIN * ds
        scores[c] = {
            "embedding_sim": es,
            "agreement": agreement_count.get(key, 1),
            "domain_sim": ds,
            "composite": comp,
        }
        composite.append(comp)

    # Pre-filter: keep top-N before sending to critic
    if len(pool) > _PRE_FILTER_TOP_N:
        top_indices = sorted(range(len(pool)), key=lambda i: composite[i], reverse=True)
        pool = [pool[i] for i in top_indices[:_PRE_FILTER_TOP_N]]

    return {
        "candidate_pool": pool,
        "candidate_scores": scores,
        "log": [f"[aggregator] pool={len(pool)} (lex={len(lex)}, sem={len(sem)}, ctx={len(ctx)})"],
    }


def _critic_node(state: SynonymState, use_strong_model: bool = False) -> dict:
    term = state["term"]
    domain = state.get("domain", "academic taxonomy")
    pool = state.get("candidate_pool", [])

    filtered = critic_agent(term, pool, domain, use_strong_model=use_strong_model)

    return {
        "filtered_candidates": filtered,
        "log": [f"[critic] kept {len(filtered)}/{len(pool)} candidates"],
    }


def _consensus_ranker_node(state: SynonymState) -> dict:
    filtered = state.get("filtered_candidates", [])
    scores = state.get("candidate_scores", {})

    if not filtered:
        return {
            "ranked_candidates": [],
            "top_score": 0.0,
            "prediction": "",
            "log": ["[ranker] no candidates to rank"],
        }

    ranked = sorted(
        filtered,
        key=lambda c: scores.get(c, {}).get("composite", 0.0),
        reverse=True,
    )

    ranked_with_scores = [
        {
            "candidate": c,
            **scores.get(c, {"composite": 0.0, "embedding_sim": 0.0, "agreement": 1, "domain_sim": 0.0}),
        }
        for c in ranked
    ]

    top_score = ranked_with_scores[0]["composite"] if ranked_with_scores else 0.0
    prediction = ranked[0] if ranked else ""

    return {
        "ranked_candidates": ranked_with_scores,
        "top_score": top_score,
        "prediction": prediction,
        "log": [f"[ranker] top='{prediction}' (score={top_score:.3f})"],
    }


def _reflection_node(state: SynonymState) -> dict:
    term = state["term"]
    domain = state.get("domain", "academic taxonomy")
    ranked = state.get("ranked_candidates", [])
    existing_pool = state.get("candidate_pool", [])

    top_candidates = [r["candidate"] for r in ranked[:3]]
    new_candidates = reflection_agent(term, domain, top_candidates, existing_pool)

    # Merge new candidates into state fields so aggregator can re-score them
    # We append to lexical_candidates as a convenience (aggregator merges all three)
    existing_lex = state.get("lexical_candidates", [])
    existing_sem = state.get("semantic_candidates", [])
    merged_new = new_candidates

    return {
        "lexical_candidates": existing_lex + merged_new,
        "semantic_candidates": existing_sem,
        "has_reflected": True,
        "log": [f"[reflection] added {len(new_candidates)} new candidates"],
    }


def build_synonym_graph(
    use_strong_model: bool = False,
    use_reflection: bool = False,
    reflection_threshold: float = 0.45,
):
    """
    Build and compile the parallel MAS graph.

    Topology:
        domain_classifier
              │
        ┌─────┼─────┐
        ▼     ▼     ▼
       lex   sem   ctx    (parallel fan-out)
        └─────┼─────┘
              ▼
          aggregator      (fan-in barrier)
              │
           critic
              │
        consensus_ranker
              │
     [optional reflection loop if top_score < threshold]
    """
    graph_builder = StateGraph(SynonymState)

    # Register nodes (wrap critic to close over use_strong_model)
    def critic_node_closure(state: SynonymState) -> dict:
        return _critic_node(state, use_strong_model=use_strong_model)

    graph_builder.add_node("domain_classifier", _domain_classifier_node)
    graph_builder.add_node("lexical_generator", _lexical_generator_node)
    graph_builder.add_node("semantic_generator", _semantic_generator_node)
    graph_builder.add_node("contextual_generator", _contextual_generator_node)
    graph_builder.add_node("aggregator", _aggregator_node)
    graph_builder.add_node("critic", critic_node_closure)
    graph_builder.add_node("consensus_ranker", _consensus_ranker_node)

    # Fan-out: domain_classifier → 3 parallel generators
    graph_builder.add_edge(START, "domain_classifier")
    graph_builder.add_edge("domain_classifier", "lexical_generator")
    graph_builder.add_edge("domain_classifier", "semantic_generator")
    graph_builder.add_edge("domain_classifier", "contextual_generator")

    # Fan-in: all 3 generators → aggregator (synchronization barrier)
    graph_builder.add_edge(
        ["lexical_generator", "semantic_generator", "contextual_generator"],
        "aggregator",
    )
    graph_builder.add_edge("aggregator", "critic")
    graph_builder.add_edge("critic", "consensus_ranker")

    if use_reflection:
        graph_builder.add_node("reflection", _reflection_node)

        def should_reflect(state: SynonymState) -> str:
            if not state.get("has_reflected", False):
                if state.get("top_score", 1.0) < reflection_threshold:
                    return "reflection"
            return END

        graph_builder.add_conditional_edges("consensus_ranker", should_reflect)
        graph_builder.add_edge("reflection", "aggregator")
    else:
        graph_builder.add_edge("consensus_ranker", END)

    return graph_builder.compile()


if __name__ == "__main__":
    app = build_synonym_graph()
    result = app.invoke({"term": "stocks", "topic": "topic-2484", "log": [], "has_reflected": False})

    print("\n=== RESULT ===")
    print("Term:       ", result.get("term"))
    print("Domain:     ", result.get("domain"))
    print("Term type:  ", result.get("term_type"))
    print("Prediction: ", result.get("prediction"))
    print("Top score:  ", result.get("top_score"))
    print("Ranked:     ", result.get("ranked_candidates"))

    print("\n=== LOG ===")
    for entry in result.get("log", []):
        print("-", entry)
