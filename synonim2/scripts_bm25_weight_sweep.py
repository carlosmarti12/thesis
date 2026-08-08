"""
Barrido del peso "bm25_weight" para la fusión ponderada por SCORE (no RRF)
de BM25 + e5 (src/tools/fusion.py::weighted_score_fusion) - solo se había
probado RRF antes para esta combinación (y perdió siempre frente a e5 solo).
No usa ningún LLM - barato, corre entero en segundos.

Solo val, para elegir el peso final antes de fijar el método
hybrid_weighted_bm25_e5 en src/methods.py.
"""

import pandas as pd

from src.evaluation import compute_metrics, summarize_results
from src.methods import EMBEDDER_E5
from src.tools.bm25_retrieval import build_bm25_index, retrieve_bm25_scored
from src.tools.embedding_retrieval import get_embedder, get_solution_index, retrieve_scored
from src.tools.fusion import weighted_score_fusion

BM25_WEIGHTS_TO_TEST = [0.1, 0.25, 0.5, 1.0, 2.0]

full_df = pd.read_csv("data/synonyms_clean.csv")
val_df = pd.read_csv("data/splits/val.csv")

pool_terms = full_df["en_synonym"].dropna().astype(str).tolist()
embedder = get_embedder(EMBEDDER_E5)
emb_terms, emb_embeddings = get_solution_index(
    pool_terms, embedder=embedder, embedder_name=EMBEDDER_E5,
    cache_name=f"solution_pool_{EMBEDDER_E5.split('/')[-1]}",
)
bm25_terms, bm25 = build_bm25_index(pool_terms)

rows_by_weight = {w: [] for w in BM25_WEIGHTS_TO_TEST}

for _, row in val_df.iterrows():
    term = str(row["en"])
    ground_truth = str(row["en_synonym"])

    emb_scored = retrieve_scored(
        [term], emb_terms, emb_embeddings, embedder=embedder, embedder_name=EMBEDDER_E5,
        top_n=20, exclude_terms=[term],
    )
    bm25_scored = retrieve_bm25_scored(term, bm25_terms, bm25, top_n=20, exclude_terms=[term])

    for w in BM25_WEIGHTS_TO_TEST:
        candidates = weighted_score_fusion(bm25_scored, emb_scored, bm25_weight=w, emb_weight=1.0, top_k=5)
        metrics = compute_metrics(ground_truth, candidates)
        rows_by_weight[w].append({"term": term, **metrics, "time_seconds": 0.0})

summaries = []
for w in BM25_WEIGHTS_TO_TEST:
    df = pd.DataFrame(rows_by_weight[w])
    summary = summarize_results(df, method=f"bm25_weight={w}")
    summaries.append(summary)

summary_df = pd.DataFrame(summaries)[["method", "hit@1", "hit@3", "hit@5", "mrr", "ndcg@3", "ndcg@5", "n_hit@1"]]
print(summary_df.to_string(index=False))
summary_df.to_csv("results/eval/bm25_weighted_fusion_sweep_val.csv", index=False)
