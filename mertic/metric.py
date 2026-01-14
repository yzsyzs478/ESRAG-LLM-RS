import json
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np


# =========================
# Argument parsing (same style as your previous scripts)
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Precision/HR/NDCG@K from per-user recommendation ratings stored in a JSON file."
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default="llm_esrag_11.json",
        help="Path to the input JSON file containing user entries and recommendation lists."
    )
    parser.add_argument(
        "--rec_key",
        type=str,
        default="recommendations",
        help="Key in each user entry that stores the recommendation list."
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[10, 5, 3],
        help="A list of K values for computing metrics (e.g., 10 5 3)."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=4.0,
        help="Relevance threshold: ratings >= threshold are treated as hits."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="llm_metrics_h.json",
        help="Path to save per-user metrics in JSON format."
    )
    parser.add_argument(
        "--split_token",
        type=str,
        default=": ",
        help="Token used to split 'Title: Rating' strings."
    )
    parser.add_argument(
        "--user_id_key",
        type=str,
        default="user_id",
        help="Key in each entry indicating the user id."
    )

    return parser.parse_args()


# =========================
# Metric computation
# =========================
def calculate_metrics(
    recommendations: List[List[float]],
    k: int,
    threshold: float = 4.0,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Compute dataset-level Precision@K / HR@K / NDCG@K (mean ± std over users).

    Args:
        recommendations: list of users; each user is a list of predicted ratings (floats).
        k: top-K cutoff.
        threshold: ratings >= threshold are considered relevant (hit=1).

    Returns:
        (precision_mean, precision_std), (hr_mean, hr_std), (ndcg_mean, ndcg_std)
    """
    precision_list, hr_list, ndcg_list = [], [], []

    for recs in recommendations:
        if not recs:
            # If this user has no valid parsed ratings, treat as zeros
            precision_list.append(0.0)
            hr_list.append(0.0)
            ndcg_list.append(0.0)
            continue

        top_k_recs = recs[:k]

        hits = sum(1 for r in top_k_recs if r >= threshold)
        precision = hits / k
        hr = 1.0 if hits > 0 else 0.0

        # Binary gain: relevant -> 1, else 0
        dcg = sum(
            ((2 ** (1 if r >= threshold else 0) - 1) / np.log2(idx + 2))
            for idx, r in enumerate(top_k_recs)
        )

        ideal_recs = sorted(recs, reverse=True)[:k]
        idcg = sum(
            ((2 ** (1 if r >= threshold else 0) - 1) / np.log2(idx + 2))
            for idx, r in enumerate(ideal_recs)
        )

        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        precision_list.append(float(precision))
        hr_list.append(float(hr))
        ndcg_list.append(float(ndcg))

    precision_mean = float(np.mean(precision_list))
    precision_std = float(np.std(precision_list))
    hr_mean = float(np.mean(hr_list))
    hr_std = float(np.std(hr_list))
    ndcg_mean = float(np.mean(ndcg_list))
    ndcg_std = float(np.std(ndcg_list))

    return (precision_mean, precision_std), (hr_mean, hr_std), (ndcg_mean, ndcg_std)


# =========================
# Parsing helper
# =========================
def extract_recommendations(
    data: List[Dict[str, Any]],
    rec_key: str,
    split_token: str = ": ",
) -> List[List[float]]:
    """
    Extract per-user rating lists from data[rec_key], assuming items are strings like "Title: Rating".

    Args:
        data: list of user entries.
        rec_key: key holding the recommendation list.
        split_token: token used to split "Title: Rating".

    Returns:
        List of lists, each inner list contains float ratings for one user.
    """
    recommendations: List[List[float]] = []

    for entry in data:
        if rec_key not in entry:
            continue

        recs: List[float] = []
        for item in entry.get(rec_key, []):
            try:
                # Expecting format: "Title: Rating"
                rating_str = str(item).split(split_token, 1)[1]
                recs.append(float(rating_str))
            except (IndexError, ValueError, TypeError):
                # Skip malformed entries
                continue

        recommendations.append(recs)

    return recommendations


# =========================
# Main
# =========================
def main():
    args = parse_args()

    # Load input JSON
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Dataset-level metrics
    recommendations = extract_recommendations(
        data=data,
        rec_key=args.rec_key,
        split_token=args.split_token,
    )

    metrics = {
        f"top{k}": calculate_metrics(recommendations, k, threshold=args.threshold)
        for k in args.ks
    }

    # Per-user metrics
    user_metrics: Dict[str, Dict[str, Any]] = {args.rec_key: {}}

    for entry in data:
        user_id = entry.get(args.user_id_key, None)
        if user_id is None:
            continue
        if args.rec_key not in entry:
            continue

        recs: List[float] = []
        for item in entry.get(args.rec_key, []):
            try:
                rating_str = str(item).split(args.split_token, 1)[1]
                recs.append(float(rating_str))
            except (IndexError, ValueError, TypeError):
                continue

        user_metrics[args.rec_key][str(user_id)] = {}
        for k in args.ks:
            (p_mean, _), (hr_mean, _), (ndcg_mean, _) = calculate_metrics([recs], k, threshold=args.threshold)
            user_metrics[args.rec_key][str(user_id)][f"top{k}"] = {
                "Precision": p_mean,
                "HR": hr_mean,
                "NDCG": ndcg_mean,
            }

    # Save per-user metrics
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(user_metrics, f, indent=4, ensure_ascii=False)

    # Print dataset-level results
    print(f"rec_key = {args.rec_key}, threshold = {args.threshold:.2f}")
    for top_k, ((p_mean, p_std), (hr_mean, hr_std), (ndcg_mean, ndcg_std)) in metrics.items():
        print(f"{top_k}:")
        print(f"  Precision: {p_mean:.4f} ± {p_std:.4f}")
        print(f"  HR:        {hr_mean:.4f} ± {hr_std:.4f}")
        print(f"  NDCG:      {ndcg_mean:.4f} ± {ndcg_std:.4f}")


if __name__ == "__main__":
    main()
