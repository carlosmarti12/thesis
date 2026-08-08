"""
Análisis de errores extendido (2026-07-22, pedido explícito del asesor):
separa retrieval-miss vs ranking-error, cruza con más dimensiones que
scripts_failure_analysis.py (abreviatura, abreviatura INVERSA, solape
léxico, variante morfológica proxy, longitud), y añade el rank del gold
ANTES y DESPUÉS del reranking (no solo después).

Requiere un par de CSVs emparejables por (term, ground_truth):
  --pre  = salida de llm_expansion_weighted_t0 (sin rerank)
  --post = salida de llm_expansion_weighted_t0_rerank_qwen (con rerank)
ambos deben venir del mismo split para que el emparejamiento tenga sentido.

Limitaciones documentadas explícitamente (no se ocultan):
- No hay columna `topic`/dominio en el dataset (solo en/en_synonym, ver
  README.md) - no es posible una categorización real por área académica
  sin anotarla a mano; se omite en vez de forzar un proxy poco fiable.
- No hay WordNet/nltk instalado en este entorno (ver memoria
  `env-pip-install-reverts`) - la polisemia se aproxima con solape léxico
  bajo + longitud corta (single_word), no con conteo de synsets; es un
  proxy más débil, se documenta como tal.
"""

import argparse
import json
from difflib import SequenceMatcher

import pandas as pd


def categorize(term: str) -> str:
    words = str(term).split()
    if len(words) == 1:
        token = words[0]
        if token.isupper() or len(token) <= 5:
            return "abbreviation_like"
        return "single_word"
    if "-" in str(term) or len(words) == 2:
        return "compound"
    return "multi_word_phrase"


def is_abbrev_like(text: str) -> bool:
    words = str(text).split()
    if len(words) != 1:
        return False
    token = words[0]
    return token.isupper() or len(token) <= 5


def lexical_overlap(a: str, b: str) -> float:
    """Jaccard sobre tokens en minúscula - proxy simple de distancia léxica."""
    ta = set(str(a).lower().split())
    tb = set(str(b).lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def shares_stem(a: str, b: str, prefix_len: int = 5) -> bool:
    """Proxy de variante morfológica: alguna palabra de a y alguna de b
    comparten un prefijo de >= prefix_len caracteres (p.ej. investment/investing)."""
    wa = [w.lower() for w in str(a).split()]
    wb = [w.lower() for w in str(b).split()]
    for x in wa:
        for y in wb:
            if len(x) >= prefix_len and len(y) >= prefix_len and x[:prefix_len] == y[:prefix_len]:
                return True
    return False


def rank_of_gold(row) -> int:
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
    parser.add_argument("--pre", default="results/eval/llm_expansion_weighted_t0__val.csv")
    parser.add_argument("--post", default="results/eval/llm_expansion_weighted_t0_rerank_qwen__val.csv")
    parser.add_argument("--label", default="val")
    parser.add_argument("--out-prefix", default="results/eval/failure_v2")
    args = parser.parse_args()

    pre = pd.read_csv(args.pre)
    post = pd.read_csv(args.post)
    pre["gold_rank_pre"] = pre.apply(rank_of_gold, axis=1)
    post["gold_rank_post"] = post.apply(rank_of_gold, axis=1)

    merged = pre[["term", "ground_truth", "gold_rank_pre"]].merge(
        post[["term", "ground_truth", "gold_rank_post", "prediction", "candidates"]],
        on=["term", "ground_truth"], how="inner",
    )
    if len(merged) != len(pre) or len(merged) != len(post):
        print(f"WARNING: merge size mismatch - pre={len(pre)} post={len(post)} merged={len(merged)}")

    merged["category"] = merged["term"].astype(str).apply(categorize)
    merged["reverse_abbrev"] = merged.apply(
        lambda r: is_abbrev_like(r["ground_truth"]) and not is_abbrev_like(r["term"]), axis=1
    )
    merged["lexical_overlap"] = merged.apply(lambda r: lexical_overlap(r["term"], r["ground_truth"]), axis=1)
    merged["shares_stem"] = merged.apply(lambda r: shares_stem(r["term"], r["ground_truth"]), axis=1)
    merged["term_words"] = merged["term"].astype(str).apply(lambda t: len(t.split()))
    merged["term_chars"] = merged["term"].astype(str).apply(len)

    def failtype(row):
        if row["gold_rank_post"] == 1:
            return "correct"
        if row["gold_rank_post"] == -1:
            return "retrieval_miss"
        return "ranking_error"

    merged["failure_type"] = merged.apply(failtype, axis=1)

    n = len(merged)
    print(f"=== {args.label}: n={n} ===")
    vc = merged["failure_type"].value_counts()
    for k in ("correct", "ranking_error", "retrieval_miss"):
        c = int(vc.get(k, 0))
        print(f"  {k}: {c} ({c/n:.1%})")

    print(f"\n=== By term category ===")
    pivot = pd.crosstab(merged["category"], merged["failure_type"])
    pivot["total"] = pivot.sum(axis=1)
    for col in ("correct", "ranking_error", "retrieval_miss"):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["ranking_error_rate"] = (pivot["ranking_error"] / pivot["total"]).round(3)
    pivot["retrieval_miss_rate"] = (pivot["retrieval_miss"] / pivot["total"]).round(3)
    print(pivot[["total", "correct", "ranking_error", "retrieval_miss",
                 "ranking_error_rate", "retrieval_miss_rate"]].to_string())

    print(f"\n=== Reverse-abbreviation (gold is the abbreviation, term is not) ===")
    rev = merged[merged["reverse_abbrev"]]
    print(f"  n={len(rev)}")
    if len(rev):
        print(rev["failure_type"].value_counts().to_string())
        print("  examples:")
        for _, r in rev.head(10).iterrows():
            print(f"    {r['term']!r} -> gold {r['ground_truth']!r} | pre_rank={r['gold_rank_pre']} post_rank={r['gold_rank_post']} | {r['failure_type']}")

    print(f"\n=== Lexical overlap (Jaccard token) vs failure type ===")
    bins = pd.cut(merged["lexical_overlap"], bins=[-0.01, 0.0, 0.33, 1.01], labels=["none", "low", "high"])
    print(pd.crosstab(bins, merged["failure_type"], normalize="index").round(3).to_string())

    print(f"\n=== Shares morphological stem (proxy) vs failure type ===")
    print(pd.crosstab(merged["shares_stem"], merged["failure_type"]).to_string())

    print(f"\n=== Term length (word count) vs failure type ===")
    print(pd.crosstab(merged["term_words"].clip(upper=4), merged["failure_type"]).to_string())

    print(f"\n=== Rank movement: pre-rerank gold rank -> post-rerank gold rank (ranking_error + correct rows where gold was in pre top5) ===")
    moved = merged[(merged["gold_rank_pre"] != -1)]
    movement = moved.groupby(["gold_rank_pre", "gold_rank_post"]).size().reset_index(name="n")
    print(movement.to_string(index=False))

    n_improved = int((moved["gold_rank_post"] < moved["gold_rank_pre"]).sum())
    n_worsened = int(((moved["gold_rank_post"] > moved["gold_rank_pre"]) | (moved["gold_rank_post"] == -1)).sum())
    n_same = int((moved["gold_rank_post"] == moved["gold_rank_pre"]).sum())
    print(f"\n  rerank improved rank: {n_improved}, worsened/lost: {n_worsened}, unchanged: {n_same} (of {len(moved)} rows where gold was retrieved pre-rerank)")

    print(f"\n=== Sample ranking-error cases with pre->post rank (closest misses) ===")
    re_cases = merged[merged["failure_type"] == "ranking_error"].sort_values("gold_rank_post")
    for _, r in re_cases.head(15).iterrows():
        print(f"  [{r['category']}] {r['term']!r} -> gold {r['ground_truth']!r}, predicted {r['prediction']!r} | pre_rank={r['gold_rank_pre']} post_rank={r['gold_rank_post']}")

    out_cols = ["term", "ground_truth", "prediction", "category", "reverse_abbrev",
                "lexical_overlap", "shares_stem", "term_words", "gold_rank_pre",
                "gold_rank_post", "failure_type"]
    merged[out_cols].to_csv(f"{args.out_prefix}_{args.label}_detail.csv", index=False)
    pivot.to_csv(f"{args.out_prefix}_{args.label}_by_category.csv")
    print(f"\nSaved: {args.out_prefix}_{args.label}_detail.csv, {args.out_prefix}_{args.label}_by_category.csv")


if __name__ == "__main__":
    main()
