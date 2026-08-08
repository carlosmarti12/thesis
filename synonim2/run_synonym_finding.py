"""
Synonym Finding (tarea nueva, distinta de la de run_experiment.py):

    Dado un término de `en`, encontrar su sinónimo ELIGIENDO de la lista
    de candidatos - el pool de en_synonym de todo el dataset (mismo pool
    que ya construye run_experiment.py, cacheado en data/cache/). Métrica
    principal: hit@1 (también se registran hit@3, hit@5, ndcg@3, ndcg@5 y
    mrr con la misma rsc/evaluation.py de la tarea actual).

Métodos incluidos - deliberadamente solo los que rindieron bien en
iteraciones previas de este proyecto (ver memoria de la tesis en
synonic/EXPERIMENTS_LOG.md y new/thesis_results/): generación libre por LLM
y retrieval por embeddings, en sus combinaciones. NO incluye same_term
(siempre da hit@1=0 - en nunca es igual a en_synonym en este dataset) ni
los grafos MAS estilo LangGraph (mas_base/mas_llm_ranker/mas_safe_hybrid en
new/src2 y synonic/) - en cada comparación anterior fueron el peor o
segundo peor grupo de métodos, y añaden una capa de orquestación (LangGraph)
que este proyecto no necesita.

  - embedding_only     - retrieval puro, sin LLM. Baseline rápido y fuerte,
                          útil para saber cuánto aporta cada LLM por encima.
  - llm_zero_shot       - el LLM propone el sinónimo a ciegas (sin ver el
                          pool), anclado después al pool real por embeddings
                          (rsc/agents/zero_shot_agent.py).
  - agent_tool_calling  - agente con tool-calling NATIVO de Ollama: decide
                          en tiempo real cuántas veces llamar a
                          retrieve_candidates antes de finalize
                          (rsc/agents/tool_agent.py) - la estructura
                          "agente + tools" pedida, no una secuencia fija.
  - hybrid_pipeline     - el pipeline ya existente de este proyecto
                          (generator_agent -> embedding_retrieval ->
                          reranker_agent_local), reutilizado tal cual como
                          punto de comparación.

Soporta resume (por (term, method)) y guarda incrementalmente, igual que
run_experiment.py.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.agents.generator_agent import generar_candidatos
from src.agents.reranker_agent_local import reordenar_candidatos
from src.agents.tool_agent import buscar_sinonimo_con_tools
from src.agents.zero_shot_agent import generar_respuesta_zero_shot
from src.evaluation import compute_metrics, summarize_results
from src.tools.embedding_retrieval import get_embedder, get_solution_index, retrieve_top_k

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "synonyms_clean.csv"
RESULTS_DIR = ROOT / "results"

ALL_METHODS = ["embedding_only", "llm_zero_shot", "agent_tool_calling", "hybrid_pipeline"]


def get_candidates(method: str, term: str, embedder, index_terms, index_embeddings) -> list[str]:
    if method == "embedding_only":
        return retrieve_top_k(
            [term], index_terms, index_embeddings, embedder=embedder,
            top_k=5, exclude_terms=[term],
        )

    if method == "llm_zero_shot":
        guesses = generar_respuesta_zero_shot(term)
        queries = guesses if guesses else [term]
        return retrieve_top_k(
            queries, index_terms, index_embeddings, embedder=embedder,
            top_k=5, exclude_terms=[term],
        )

    if method == "agent_tool_calling":
        return buscar_sinonimo_con_tools(
            term, index_terms, index_embeddings, embedder=embedder, top_k_final=5,
        )

    if method == "hybrid_pipeline":
        queries = generar_candidatos(term)
        top5 = retrieve_top_k(
            queries, index_terms, index_embeddings, embedder=embedder,
            top_k=5, exclude_terms=[term],
        )
        return reordenar_candidatos(term, top5)

    raise ValueError(f"Unknown method: {method}")


def run(methods: list[str], limit: int, resume: bool = True) -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_csv(DATA_PATH)
    df = full_df.head(limit) if limit > 0 else full_df

    out_path = RESULTS_DIR / "synonym_finding_comparison.csv"
    rows: list = []
    done_keys: set = set()

    if resume and out_path.exists():
        existing = pd.read_csv(out_path)
        rows = existing.to_dict("records")
        done_keys = set(zip(existing["term"].astype(str), existing["method"]))
        print(f"Resuming: {len(done_keys)} (term, method) pair(s) already saved.")

    print("Loading embedding model...")
    embedder = get_embedder()

    print("Building candidate pool from en_synonym (full dataset, cached to disk)...")
    # El pool sale siempre del dataset COMPLETO, no de `df` (que puede estar
    # recortado por --limit) - igual que en run_experiment.py, para no
    # encoger artificialmente el candidate pool al hacer un smoke test.
    pool_terms = full_df["en_synonym"].dropna().astype(str).tolist()
    index_terms, index_embeddings = get_solution_index(pool_terms, embedder=embedder)
    print(f"Rows: {len(df)} | pool size: {len(index_terms)}")

    for method in methods:
        print(f"\n=== {method} ===")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            term = str(row["en"])

            if (term, method) in done_keys:
                continue

            start = time.time()
            try:
                candidates = get_candidates(method, term, embedder, index_terms, index_embeddings)
            except Exception as e:
                tqdm.write(f"ERROR [{method}] con '{term}': {e}")
                candidates = []
            elapsed = time.time() - start

            # en_synonym se lee aquí, después de que el método ya produjo
            # sus candidatos - nunca antes.
            ground_truth = str(row["en_synonym"])
            metrics = compute_metrics(ground_truth, candidates)

            rows.append({
                "term": term,
                "ground_truth": ground_truth,
                "method": method,
                "prediction": candidates[0] if candidates else "",
                "candidates": json.dumps(candidates, ensure_ascii=False),
                "num_candidates": len(candidates),
                "time_seconds": elapsed,
                **metrics,
            })
            done_keys.add((term, method))

            pd.DataFrame(rows).to_csv(out_path, index=False)

    results_df = pd.DataFrame(rows)

    summary_rows = []
    for method in methods:
        method_df = results_df[results_df["method"] == method]
        if len(method_df):
            summary_rows.append(summarize_results(method_df, method=method))

    summary_path = RESULTS_DIR / "synonym_finding_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print("\nSummary:")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nSaved detailed results to: {out_path}")
    print(f"Saved summary to: {summary_path}")

    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=ALL_METHODS)
    parser.add_argument("--limit", type=int, default=0, help="Max terms (0 = full dataset)")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    run(methods=args.methods, limit=args.limit, resume=not args.no_resume)


if __name__ == "__main__":
    main()
