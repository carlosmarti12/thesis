"""
Barrido del número de variantes generadas por el LLM (n en generar_candidatos,
por defecto 9) a temperature=0 - genera SIEMPRE con n=9 (el máximo) una sola
vez por término y luego recorta la lista para simular n menores, evitando
llamadas LLM redundantes (mismo patrón que scripts_weight_sweep_t0.py).

candidatos[0] es siempre el término original (ver generar_candidatos), así
que candidatos[:n+1] = término original + hasta n variantes.

Solo val. Peso de fusión fijo = LLM_EXPANSION_WEIGHTED_T0_WEIGHT (elegido en
el retune de temperature=0, ver EXPERIMENTS_LOG.md).
"""

import time

import pandas as pd

from src.agents.generator_agent import AGENTE_GENERADOR_T0, generar_candidatos
from src.evaluation import compute_metrics, summarize_results
from src.methods import EMBEDDER_E5, LLM_EXPANSION_WEIGHTED_T0_WEIGHT
from src.tools.embedding_retrieval import encode_queries, get_embedder, get_solution_index
from src.tools.multi_query_fusion import combine_scores

N_VALUES_TO_TEST = [3, 5, 7, 9]

full_df = pd.read_csv("data/synonyms_clean.csv")
val_df = pd.read_csv("data/splits/val.csv")

pool_terms = full_df["en_synonym"].dropna().astype(str).tolist()
embedder = get_embedder(EMBEDDER_E5)
index_terms, index_embeddings = get_solution_index(
    pool_terms, embedder=embedder, embedder_name=EMBEDDER_E5,
    cache_name=f"solution_pool_{EMBEDDER_E5.split('/')[-1]}",
)

rows_by_n = {n: [] for n in N_VALUES_TO_TEST}
gen_time_total = 0.0

for _, row in val_df.iterrows():
    term = str(row["en"])
    ground_truth = str(row["en_synonym"])

    t0 = time.time()
    # n=9 (máximo) una sola vez - candidatos[0] es siempre el término original.
    candidatos_full = generar_candidatos(term, n=9, agente=AGENTE_GENERADOR_T0)
    gen_time_total += time.time() - t0

    for n in N_VALUES_TO_TEST:
        candidatos_n = candidatos_full[: n + 1]
        query_embeddings = encode_queries(candidatos_n, embedder, embedder_name=EMBEDDER_E5)
        candidates = combine_scores(
            "weighted", query_embeddings, index_terms, index_embeddings,
            exclude_terms=[term], top_k=5, original_weight=LLM_EXPANSION_WEIGHTED_T0_WEIGHT,
        )
        metrics = compute_metrics(ground_truth, candidates)
        rows_by_n[n].append({"term": term, **metrics, "time_seconds": 0.0})

print(f"Total LLM+embedding time (shared across all n): {gen_time_total:.1f}s "
      f"for {len(val_df)} terms ({gen_time_total/len(val_df):.2f}s/term)\n")

summaries = []
for n in N_VALUES_TO_TEST:
    df = pd.DataFrame(rows_by_n[n])
    summary = summarize_results(df, method=f"n_variants={n}")
    summaries.append(summary)

summary_df = pd.DataFrame(summaries)[["method", "hit@1", "hit@3", "hit@5", "mrr", "ndcg@3", "ndcg@5", "n_hit@1"]]
print(summary_df.to_string(index=False))
summary_df.to_csv("results/eval/variant_count_sweep_val.csv", index=False)
