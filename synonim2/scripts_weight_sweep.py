"""
Barrido eficiente del peso "original_weight" de la fusión ponderada
(Método 1) sobre val: llama a generar_candidatos() y encode_queries() UNA
SOLA VEZ por término (lo caro: 1 llamada LLM + 1 embedding), y reutiliza esa
misma matriz de scores para probar varios pesos en memoria (barato) - evita
194 llamadas LLM redundantes por cada valor de peso que se pruebe.

No toca train ni test. Solo val, para elegir el peso final.
"""

import time

import pandas as pd

from src.agents.generator_agent import generar_candidatos
from src.evaluation import compute_metrics, summarize_results
from src.methods import EMBEDDER_E5
from src.tools.embedding_retrieval import encode_queries, get_embedder, get_solution_index
from src.tools.multi_query_fusion import combine_scores

WEIGHTS_TO_TEST = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]

full_df = pd.read_csv("data/synonyms_clean.csv")
val_df = pd.read_csv("data/splits/val.csv")

pool_terms = full_df["en_synonym"].dropna().astype(str).tolist()
embedder = get_embedder(EMBEDDER_E5)
index_terms, index_embeddings = get_solution_index(
    pool_terms, embedder=embedder, embedder_name=EMBEDDER_E5,
    cache_name=f"solution_pool_{EMBEDDER_E5.split('/')[-1]}",
)

rows_by_weight = {w: [] for w in WEIGHTS_TO_TEST}
gen_time_total = 0.0

for _, row in val_df.iterrows():
    term = str(row["en"])
    ground_truth = str(row["en_synonym"])

    t0 = time.time()
    queries = generar_candidatos(term)
    query_embeddings = encode_queries(queries, embedder, embedder_name=EMBEDDER_E5)
    gen_time_total += time.time() - t0

    for w in WEIGHTS_TO_TEST:
        candidates = combine_scores(
            "weighted", query_embeddings, index_terms, index_embeddings,
            exclude_terms=[term], top_k=5, original_weight=w, other_weight=1.0,
        )
        metrics = compute_metrics(ground_truth, candidates)
        rows_by_weight[w].append({"term": term, **metrics, "time_seconds": 0.0})

print(f"Total LLM+embedding time (shared across all weights): {gen_time_total:.1f}s "
      f"for {len(val_df)} terms ({gen_time_total/len(val_df):.2f}s/term)\n")

summaries = []
for w in WEIGHTS_TO_TEST:
    df = pd.DataFrame(rows_by_weight[w])
    summary = summarize_results(df, method=f"weight={w}")
    summaries.append(summary)

summary_df = pd.DataFrame(summaries)[["method", "hit@1", "hit@3", "hit@5", "mrr", "ndcg@3", "ndcg@5", "n_hit@1"]]
print(summary_df.to_string(index=False))
summary_df.to_csv("results/eval/weight_sweep_val.csv", index=False)
