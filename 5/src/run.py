import argparse
import json
import re
from pathlib import Path
from typing import List, Union

import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

from src.data import load_grouped
from src.graph import build_graph

FUZZY_THRESHOLD = 80


# ── helpers ──────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text)


def _hit(candidates: List[str], ground_truths: List[str], k: int, fuzzy: bool = False) -> bool:
    golds = [_norm(g) for g in ground_truths]
    for c in candidates[:k]:
        pred = _norm(c)
        for gold in golds:
            if pred == gold:
                return True
            if fuzzy and fuzz.token_set_ratio(pred, gold) >= FUZZY_THRESHOLD:
                return True
    return False


def _mrr(candidates: List[str], ground_truths: List[str], fuzzy: bool = False) -> float:
    golds = [_norm(g) for g in ground_truths]
    for rank, c in enumerate(candidates, 1):
        pred = _norm(c)
        for gold in golds:
            if pred == gold or (fuzzy and fuzz.token_set_ratio(pred, gold) >= FUZZY_THRESHOLD):
                return 1.0 / rank
    return 0.0


# ── main run ─────────────────────────────────────────────────────────────────

def run(
    limit: int = 0,
    output: str = "results/results.csv",
    resume: bool = True,
) -> pd.DataFrame:
    df = load_grouped()
    if limit > 0:
        df = df.head(limit)

    rows: list = []
    done: set = set()
    out = Path(output)

    if resume and out.exists():
        existing = pd.read_csv(out)
        rows = existing.to_dict("records")
        done = set(existing["term"].astype(str))
        print(f"Resuming — {len(done)} term(s) already done.")

    app = build_graph()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Running"):
        term = str(row["en"])
        topic = str(row["topic"])
        ground_truths: List[str] = row["en_synonyms"]

        if term in done:
            continue

        try:
            result = app.invoke({"term": term, "topic": topic, "log": []})
        except Exception as e:
            tqdm.write(f"ERROR '{term}': {e}")
            result = {}

        candidates: List[str] = result.get("filtered_candidates", result.get("raw_candidates", []))
        prediction = candidates[0] if candidates else ""

        rows.append({
            "topic": topic,
            "term": term,
            "ground_truths": json.dumps(ground_truths, ensure_ascii=False),
            "prediction": prediction,
            "candidates": json.dumps(candidates, ensure_ascii=False),
            "exact_match": any(_norm(prediction) == _norm(g) for g in ground_truths),
            "top_3": _hit(candidates, ground_truths, k=3),
            "top_5": _hit(candidates, ground_truths, k=5),
            "top_8": _hit(candidates, ground_truths, k=8),
            "top_8_fuzzy": _hit(candidates, ground_truths, k=8, fuzzy=True),
            "mrr": _mrr(candidates, ground_truths),
            "mrr_fuzzy": _mrr(candidates, ground_truths, fuzzy=True),
        })
        done.add(term)

        out.parent.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)

    return pd.DataFrame(rows)


def print_summary(results: pd.DataFrame) -> None:
    print("\n=== RESULTS ===")
    print(f"Terms evaluated: {len(results)}")
    for col in ["exact_match", "top_3", "top_5", "top_8", "top_8_fuzzy", "mrr", "mrr_fuzzy"]:
        if col in results.columns:
            print(f"  {col:<20} {results[col].mean():.4f}")

    print("\n=== SAMPLE (first 15) ===")
    cols = ["term", "ground_truths", "prediction", "top_8", "top_8_fuzzy", "mrr"]
    print(results[[c for c in cols if c in results.columns]].head(15).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = full dataset")
    parser.add_argument("--output", type=str, default="results/results.csv")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    results = run(limit=args.limit, output=args.output, resume=not args.no_resume)
    print_summary(results)
    print(f"\nSaved to: {args.output}")
