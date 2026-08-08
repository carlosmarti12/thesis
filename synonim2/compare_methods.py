"""
Compara un método nuevo contra un baseline (por defecto embedding_e5) sobre
el mismo split: cuenta en cuántos términos el gold "sube" a rank 1 (el
baseline no acertaba en top-1 y el método nuevo sí) y en cuántos "empeora"
(el baseline acertaba en top-1 y el método nuevo ya no). Requiere que ambos
métodos ya se hayan corrido con run_eval.py sobre el mismo split (lee
results/eval/{method}__{split}.csv).

Uso:
    python compare_methods.py --method llm_expansion_rerank --split val
    python compare_methods.py --method llm_expansion_rerank --baseline embedding_e5 --split val --examples 10
"""

import argparse
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results" / "eval"


def compare(method: str, baseline: str, split: str, n_examples: int = 8) -> dict:
    method_path = RESULTS_DIR / f"{method}__{split}.csv"
    baseline_path = RESULTS_DIR / f"{baseline}__{split}.csv"
    if not method_path.exists():
        raise FileNotFoundError(f"Missing {method_path} - run `python run_eval.py --method {method} --split {split}` first.")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing {baseline_path} - run `python run_eval.py --method {baseline} --split {split}` first.")

    m = pd.read_csv(method_path)
    b = pd.read_csv(baseline_path)
    # Merge on (term, ground_truth), no solo term: hay 10 términos `en`
    # duplicados en el dataset con dos en_synonym distintos (ver
    # data/make_splits.py) - mergear solo por term produce un cruce
    # many-to-many espurio para esas filas si ambas caen en el mismo split.
    merged = m.merge(b, on=["term", "ground_truth"], suffixes=("_new", "_base"))
    if len(merged) != len(m) or len(merged) != len(b):
        print(f"WARNING: row count mismatch after merge on (term, ground_truth) - "
              f"method={len(m)} baseline={len(b)} merged={len(merged)}. "
              f"Check both were run on the exact same split file.")

    moved_to_rank1 = merged[(merged["hit@1_base"] == False) & (merged["hit@1_new"] == True)]  # noqa: E712
    regressed_from_rank1 = merged[(merged["hit@1_base"] == True) & (merged["hit@1_new"] == False)]  # noqa: E712

    print(f"\n{method} vs {baseline} on {split} (n={len(merged)})")
    print(f"  moved gold INTO rank 1 (base wrong, new correct): {len(moved_to_rank1)}")
    print(f"  regressed FROM rank 1 (base correct, new wrong):  {len(regressed_from_rank1)}")
    print(f"  net rank@1 change: {len(moved_to_rank1) - len(regressed_from_rank1):+d}")

    if n_examples and len(moved_to_rank1):
        print(f"\n  Examples moved into rank 1 (showing up to {n_examples}):")
        for _, r in moved_to_rank1.head(n_examples).iterrows():
            print(f"    {r['term']!r:40s} gold={r['ground_truth']!r:35s} "
                  f"{baseline}={r['prediction_base']!r:30s} {method}={r['prediction_new']!r}")

    if n_examples and len(regressed_from_rank1):
        print(f"\n  Examples regressed from rank 1 (showing up to {n_examples}):")
        for _, r in regressed_from_rank1.head(n_examples).iterrows():
            print(f"    {r['term']!r:40s} gold={r['ground_truth']!r:35s} "
                  f"{baseline}={r['prediction_base']!r:30s} {method}={r['prediction_new']!r}")

    return {
        "method": method,
        "baseline": baseline,
        "split": split,
        "n": len(merged),
        "n_moved_to_rank1": len(moved_to_rank1),
        "n_regressed_from_rank1": len(regressed_from_rank1),
        "net_rank1_change": len(moved_to_rank1) - len(regressed_from_rank1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--baseline", default="embedding_e5")
    parser.add_argument("--split", required=True, choices=["train", "val", "test", "full"])
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()
    compare(args.method, args.baseline, args.split, n_examples=args.examples)


if __name__ == "__main__":
    main()
