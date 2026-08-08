"""
Calcula el margen (score rank1 - score rank2) de la fusión ponderada de
llm_expansion_weighted_t0 para cada término de val, y lo cruza con si el
rank1 fue correcto - para elegir GATE_THRESHOLD del rerank condicionado por
confianza (Fase C.5, src/methods.py::llm_expansion_gated_rerank_qwen).

No usa la API de pago - solo generación local (Ollama) + embeddings, cero
coste. Reutiliza combine_scores_scored en vez de combine_scores para poder
leer el margen crudo, no solo el top-k.
"""

import pandas as pd

from src.agents.generator_agent import AGENTE_GENERADOR_T0, generar_candidatos
from src.evaluation import compute_metrics
from src.methods import EMBEDDER_E5, LLM_EXPANSION_WEIGHTED_T0_WEIGHT
from src.tools.embedding_retrieval import encode_queries, get_embedder, get_solution_index
from src.tools.multi_query_fusion import combine_scores_scored

full_df = pd.read_csv("data/synonyms_clean.csv")
val_df = pd.read_csv("data/splits/val.csv")

pool_terms = full_df["en_synonym"].dropna().astype(str).tolist()
embedder = get_embedder(EMBEDDER_E5)
index_terms, index_embeddings = get_solution_index(
    pool_terms, embedder=embedder, embedder_name=EMBEDDER_E5,
    cache_name=f"solution_pool_{EMBEDDER_E5.split('/')[-1]}",
)

rows = []
for _, row in val_df.iterrows():
    term = str(row["en"])
    ground_truth = str(row["en_synonym"])

    queries = generar_candidatos(term, agente=AGENTE_GENERADOR_T0)
    query_embeddings = encode_queries(queries, embedder, embedder_name=EMBEDDER_E5)
    top5, margin = combine_scores_scored(
        "weighted", query_embeddings, index_terms, index_embeddings,
        exclude_terms=[term], top_k=5, original_weight=LLM_EXPANSION_WEIGHTED_T0_WEIGHT,
    )
    metrics = compute_metrics(ground_truth, top5)
    rows.append({"term": term, "margin": margin, **metrics})

df = pd.DataFrame(rows)
df.to_csv("results/eval/margin_analysis_val.csv", index=False)

print("Margin stats by hit@1 correctness:")
print(df.groupby("hit@1")["margin"].describe().to_string())

print("\nCandidate thresholds - coverage and precision of the 'ambiguous' bucket:")
for thresh in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]:
    bucket = df[df["margin"] < thresh]
    n = len(bucket)
    n_wrong = (~bucket["hit@1"]).sum()
    n_right = bucket["hit@1"].sum()
    print(f"  threshold={thresh:.2f}: n_ambiguous={n:3d}  wrong_in_bucket={n_wrong:3d}  "
          f"right_in_bucket={n_right:3d}  (risk of disturbing an already-correct rank1)")
