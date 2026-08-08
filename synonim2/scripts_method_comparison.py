"""
Compara un método candidato contra el baseline actual
(llm_expansion_weighted_t0_rerank_qwen) fila por fila, en el mismo split,
para reportar exactamente las métricas pedidas por el asesor 2026-07-22:
retrieval misses recuperados, ranking errors corregidos, regresiones, y
mejora neta de rank-1 - además de las 6 métricas estándar del summary JSON.

Uso:
    python scripts_method_comparison.py --method <name> --split val
Requiere que ambos results/eval/<method>__<split>.csv y
results/eval/llm_expansion_weighted_t0_rerank_qwen__<split>.csv existan.
"""

import argparse
import json

import pandas as pd


def gold_rank(row) -> int:
    try:
        candidates = json.loads(row["candidates"])
    except Exception:
        return -1
    gt = row["ground_truth"]
    for i, c in enumerate(candidates, start=1):
        if c == gt:
            return i
    return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--baseline", default="llm_expansion_weighted_t0_rerank_qwen")
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    cand = pd.read_csv(f"results/eval/{args.method}__{args.split}.csv")
    base = pd.read_csv(f"results/eval/{args.baseline}__{args.split}.csv")
    cand["gold_rank"] = cand.apply(gold_rank, axis=1)
    base["gold_rank"] = base.apply(gold_rank, axis=1)

    merged = base[["term", "ground_truth", "gold_rank"]].rename(columns={"gold_rank": "base_rank"}).merge(
        cand[["term", "ground_truth", "gold_rank", "prediction"]].rename(
            columns={"gold_rank": "cand_rank", "prediction": "cand_prediction"}),
        on=["term", "ground_truth"], how="inner",
    )
    if len(merged) != len(base) or len(merged) != len(cand):
        print(f"WARNING: size mismatch base={len(base)} cand={len(cand)} merged={len(merged)}")

    def status(rank):
        if rank == 1:
            return "correct"
        if rank == -1:
            return "retrieval_miss"
        return "ranking_error"

    merged["base_status"] = merged["base_rank"].apply(status)
    merged["cand_status"] = merged["cand_rank"].apply(status)

    retrieval_miss_recovered = int(((merged["base_status"] == "retrieval_miss") &
                                     (merged["cand_status"] != "retrieval_miss")).sum())
    ranking_error_corrected = int(((merged["base_status"] == "ranking_error") &
                                    (merged["cand_status"] == "correct")).sum())
    regressions = int(((merged["base_status"] == "correct") &
                        (merged["cand_status"] != "correct")).sum())
    other_worsened = int(((merged["base_status"] == "ranking_error") &
                           (merged["cand_status"] == "retrieval_miss")).sum())
    net_rank1 = int((merged["cand_status"] == "correct").sum()) - int((merged["base_status"] == "correct").sum())

    print(f"=== {args.method} vs {args.baseline} ({args.split}, n={len(merged)}) ===")
    print(f"  retrieval misses recovered (miss -> in top5): {retrieval_miss_recovered}")
    print(f"  ranking errors corrected (top5-not-1 -> rank1): {ranking_error_corrected}")
    print(f"  regressions (was correct, now not):            {regressions}")
    print(f"  ranking errors worsened into misses:            {other_worsened}")
    print(f"  net rank-1 change:                              {net_rank1:+d} / {len(merged)}")

    print(f"\n  Recovered examples:")
    rec = merged[(merged["base_status"] == "retrieval_miss") & (merged["cand_status"] != "retrieval_miss")]
    for _, r in rec.head(10).iterrows():
        print(f"    {r['term']!r} -> gold {r['ground_truth']!r} | base=miss, cand_rank={r['cand_rank']} ({r['cand_prediction']!r})")

    print(f"\n  Ranking-errors-corrected examples:")
    fix = merged[(merged["base_status"] == "ranking_error") & (merged["cand_status"] == "correct")]
    for _, r in fix.head(10).iterrows():
        print(f"    {r['term']!r} -> gold {r['ground_truth']!r} | base_rank={r['base_rank']} -> rank1")

    print(f"\n  Regression examples:")
    reg = merged[(merged["base_status"] == "correct") & (merged["cand_status"] != "correct")]
    for _, r in reg.head(10).iterrows():
        print(f"    {r['term']!r} -> gold {r['ground_truth']!r} | base=rank1 -> cand_rank={r['cand_rank']} ({r['cand_prediction']!r})")


if __name__ == "__main__":
    main()
