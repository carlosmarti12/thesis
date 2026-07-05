import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="First results CSV")
    parser.add_argument("--b", required=True, help="Second results CSV")
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    a = pd.read_csv(args.a)
    b = pd.read_csv(args.b)

    merged = a.merge(
        b,
        on=["topic", "term", "ground_truth"],
        suffixes=("_a", "_b"),
    )

    print(f"Compared rows: {len(merged)}")
    print()

    a_wins = merged[
        (merged["exact_any_a"] == True) &
        (merged["exact_any_b"] == False)
    ]

    b_wins = merged[
        (merged["exact_any_a"] == False) &
        (merged["exact_any_b"] == True)
    ]

    both_hit = merged[
        (merged["exact_any_a"] == True) &
        (merged["exact_any_b"] == True)
    ]

    both_fail = merged[
        (merged["exact_any_a"] == False) &
        (merged["exact_any_b"] == False)
    ]

    print(f"{args.name_a} finds GT, {args.name_b} misses: {len(a_wins)}")
    print(f"{args.name_b} finds GT, {args.name_a} misses: {len(b_wins)}")
    print(f"Both find GT: {len(both_hit)}")
    print(f"Both miss GT: {len(both_fail)}")
    print()

    print("=" * 100)
    print(f"Examples where {args.name_a} wins")
    print("=" * 100)

    for _, row in a_wins.head(args.top).iterrows():
        print(f"Term: {row['term']}")
        print(f"Ground truth: {row['ground_truth']}")
        print(f"{args.name_a} prediction: {row['prediction_a']}")
        print(f"{args.name_a} candidates: {row['candidates_a']}")
        print(f"{args.name_b} prediction: {row['prediction_b']}")
        print(f"{args.name_b} candidates: {row['candidates_b']}")
        print()

    print("=" * 100)
    print(f"Examples where {args.name_b} wins")
    print("=" * 100)

    for _, row in b_wins.head(args.top).iterrows():
        print(f"Term: {row['term']}")
        print(f"Ground truth: {row['ground_truth']}")
        print(f"{args.name_a} prediction: {row['prediction_a']}")
        print(f"{args.name_a} candidates: {row['candidates_a']}")
        print(f"{args.name_b} prediction: {row['prediction_b']}")
        print(f"{args.name_b} candidates: {row['candidates_b']}")
        print()

    print("=" * 100)
    print("Worst disagreements by semantic similarity difference")
    print("=" * 100)

    if "semantic_similarity_a" in merged.columns and "semantic_similarity_b" in merged.columns:
        merged["semantic_diff_a_minus_b"] = (
            merged["semantic_similarity_a"] - merged["semantic_similarity_b"]
        )

        print(f"\nCases where {args.name_a} has much higher semantic similarity:")
        better_a = merged.sort_values("semantic_diff_a_minus_b", ascending=False)
        for _, row in better_a.head(args.top).iterrows():
            print(f"Term: {row['term']}")
            print(f"Ground truth: {row['ground_truth']}")
            print(f"{args.name_a} semantic: {row['semantic_similarity_a']}")
            print(f"{args.name_a} candidates: {row['candidates_a']}")
            print(f"{args.name_b} semantic: {row['semantic_similarity_b']}")
            print(f"{args.name_b} candidates: {row['candidates_b']}")
            print()

        print(f"\nCases where {args.name_b} has much higher semantic similarity:")
        better_b = merged.sort_values("semantic_diff_a_minus_b", ascending=True)
        for _, row in better_b.head(args.top).iterrows():
            print(f"Term: {row['term']}")
            print(f"Ground truth: {row['ground_truth']}")
            print(f"{args.name_a} semantic: {row['semantic_similarity_a']}")
            print(f"{args.name_a} candidates: {row['candidates_a']}")
            print(f"{args.name_b} semantic: {row['semantic_similarity_b']}")
            print(f"{args.name_b} candidates: {row['candidates_b']}")
            print()


if __name__ == "__main__":
    main()