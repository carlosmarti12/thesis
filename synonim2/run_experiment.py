"""
Pipeline completo: agente generador -> herramienta de embeddings ->
agente reranker -> evaluación (hit@k/ndcg@k/mrr contra en_synonym).

    1. rsc/agents/generator_agent.py  - genera 9 variantes + el término
       original (10 queries) con un LLM local (llama3.2:3b vía Ollama).
    2. rsc/tools/embedding_retrieval.py - embebe esas 10 queries y el pool
       de en_synonym, devuelve los 5 candidatos más cercanos.
    3. rsc/agents/reranker_agent_local.py - un LLM (qwen3.5:2b, local vía
       Ollama) reordena esos 5 de mejor a peor. La variante que llama a la
       API de OpenRouter sigue disponible en rsc/agents/reranker_agent.py
       para comparar velocidad/calidad, pero no la usa este pipeline.

Soporta resume: si el CSV de salida ya existe, los términos ya procesados
se saltan y el archivo se guarda incrementalmente tras cada término.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.agents.generator_agent import AGENTE_GENERADOR, generar_candidatos
from src.agents.reranker_agent_local import AGENTE_RERANKER_LOCAL, reordenar_candidatos
from src.evaluation import compute_metrics, summarize_results
from src.tools.embedding_retrieval import get_embedder, get_solution_index, retrieve_top_k

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "synonyms_clean.csv"
RESULTS_DIR = ROOT / "results"

# Identifica la combinación generador+reranker de esta corrida, para poder
# comparar resúmenes de distintas ejecuciones (p.ej. si se cambia el modelo).
METHOD_NAME = (
    f"gen_{AGENTE_GENERADOR['modelo']}__rerank_{AGENTE_RERANKER_LOCAL['modelo']}"
).replace("/", "_").replace(":", "_")


def run(limit: int, output_path: str, resume: bool = True) -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_csv(DATA_PATH)
    df = full_df.head(limit) if limit > 0 else full_df

    out_path = Path(output_path)
    rows: list = []
    done_terms: set = set()

    if resume and out_path.exists():
        existing = pd.read_csv(out_path)
        rows = existing.to_dict("records")
        done_terms = set(existing["term"].astype(str))
        print(f"Resuming: {len(done_terms)} term(s) already saved.")

    print("Loading embedding model...")
    embedder = get_embedder()

    print("Building solution index from en_synonym (full dataset, cached to disk)...")
    # El pool sale siempre del dataset COMPLETO, no de `df` (que puede estar
    # recortado por --limit) - si no, --limit reduciría también el pool de
    # candidatos y haría el retrieval artificialmente más fácil.
    pool_terms = full_df["en_synonym"].dropna().astype(str).tolist()
    index_terms, index_embeddings = get_solution_index(pool_terms, embedder=embedder)

    print(f"Rows: {len(df)} | pool size: {len(index_terms)}")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        term = str(row["en"])

        if term in done_terms:
            continue

        start = time.time()
        try:
            queries = generar_candidatos(term)
            top5 = retrieve_top_k(
                queries, index_terms, index_embeddings, embedder=embedder,
                top_k=5, exclude_terms=[term],
            )
            candidatos_finales = reordenar_candidatos(term, top5)
        except Exception as e:
            tqdm.write(f"ERROR con '{term}': {e}")
            candidatos_finales = []
        elapsed = time.time() - start

        # en_synonym se lee aquí, después de que el pipeline ya produjo sus candidatos.
        ground_truth = str(row["en_synonym"])
        metrics = compute_metrics(ground_truth, candidatos_finales)

        rows.append({
            "term": term,
            "ground_truth": ground_truth,
            "prediction": candidatos_finales[0] if candidatos_finales else "",
            "candidates": json.dumps(candidatos_finales, ensure_ascii=False),
            "num_candidates": len(candidatos_finales),
            "time_seconds": elapsed,
            **metrics,
        })
        done_terms.add(term)

        pd.DataFrame(rows).to_csv(out_path, index=False)

    results_df = pd.DataFrame(rows)

    summary = summarize_results(results_df, method=METHOD_NAME)
    output_json = out_path.with_name(out_path.stem + "_summary.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary:")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"\nSaved detailed results to: {out_path}")
    print(f"Saved summary to: {output_json}")

    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max terms (0 = full dataset)")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR / "pipeline_results.csv"))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    run(limit=args.limit, output_path=args.output, resume=not args.no_resume)


if __name__ == "__main__":
    main()
