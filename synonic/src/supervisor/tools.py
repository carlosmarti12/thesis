"""
The one genuinely new deterministic tool this package adds. Every other
tool (retrieve/generate/merge/score/finalize) already exists in agents.py -
executor.execute_decision calls those directly. This one turns a
scored_candidates list into an objective sufficiency signal, so the
supervisor doesn't have to "guess" whether results are good enough.
"""


def evaluate_candidate_quality(
    scored_candidates: list[dict],
    required_count: int = 5,
    minimum_top_score: float = 0.65,
    minimum_average_top_k: float = 0.50,
) -> dict:
    if not scored_candidates:
        return {
            "sufficient": False,
            "confidence": 0.0,
            "top_score": 0.0,
            "average_top_k": 0.0,
            "candidate_count": 0,
        }

    ranked = sorted(scored_candidates, key=lambda item: item["final_score"], reverse=True)
    top_items = ranked[:required_count]
    scores = [item["final_score"] for item in top_items]

    top_score = scores[0] if scores else 0.0
    average_score = sum(scores) / len(scores) if scores else 0.0

    sufficient = (
        len(top_items) >= required_count
        and top_score >= minimum_top_score
        and average_score >= minimum_average_top_k
    )

    confidence = min(1.0, 0.6 * top_score + 0.4 * average_score)

    return {
        "sufficient": sufficient,
        "confidence": confidence,
        "top_score": top_score,
        "average_top_k": average_score,
        "candidate_count": len(top_items),
    }
