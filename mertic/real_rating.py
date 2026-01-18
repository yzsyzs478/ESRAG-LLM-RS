import json
import argparse
from typing import Any, Dict, List, Optional


# =========================
# Argument parsing (same style as your previous scripts)
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter recommendations by exact title match and attach ratings from validation_set only."
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default="llm_esrag_1.json",
        help="Path to the input JSON file containing users, recommendations, validation_set."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="llm_esrag_11.json",
        help="Path to save the filtered and rated recommendations JSON output."
    )
    parser.add_argument(
        "--rec_key",
        type=str,
        default="recommendations",
        help="Key in each user entry that stores the recommendation list."
    )
    parser.add_argument(
        "--validation_key",
        type=str,
        default="validation_set",
        help="Key in each user entry that stores the validation set list."
    )
    parser.add_argument(
        "--user_id_key",
        type=str,
        default="user_id",
        help="Key in each user entry indicating the user id."
    )
    parser.add_argument(
        "--list_prefix_sep",
        type=str,
        default=". ",
        help="Separator used to parse enumerated recommendations, e.g., '1. Title(year)'."
    )
    parser.add_argument(
        "--title_rating_sep",
        type=str,
        default=": ",
        help="Separator used when writing output, e.g., '1. Title(year): 4.0'."
    )

    return parser.parse_args()


# =========================
# Normalization helper
# =========================
def norm(s: str) -> str:
    """
    Canonical form for exact matching:
    lowercase + strip leading/trailing whitespace.
    """
    return (s or "").lower().strip()


# =========================
# Core logic
# =========================
def add_ratings_and_filter(
    recommendations: List[str],
    combined_ratings: Dict[str, Any],
    list_prefix_sep: str = ". ",
    title_rating_sep: str = ": ",
) -> List[str]:
    """
    Keep only recommendations that exactly match keys in combined_ratings after normalization,
    and attach the matched rating.

    Expected input line format: "idx{list_prefix_sep}Title"
    Output line format:        "new_idx{list_prefix_sep}Title{title_rating_sep}rating"
    """
    rated_recommendations: List[str] = []
    if not recommendations:
        return rated_recommendations

    new_idx = 0
    for rec in recommendations:
        try:
            title = str(rec).split(list_prefix_sep, 1)[1].strip()
        except Exception:
            print(f"[WARN] Failed to parse recommendation line: {rec}")
            continue

        key = norm(title)

        # Exact match lookup in the validation set ratings dictionary
        if key in combined_ratings:
            new_idx += 1
            rating = combined_ratings[key]
            rated_recommendations.append(f"{new_idx}{list_prefix_sep}{title}{title_rating_sep}{rating}")

    return rated_recommendations


def main():
    args = parse_args()

    # Read input JSON file
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_data: List[Dict[str, Any]] = []

    for user in data:
        user_id = user.get(args.user_id_key)
        recommendations = user.get(args.rec_key, []) or []

        # Only create rating map from validation_set
        validation_map = {
            norm(item.get("title", "")): item.get("rating", None)
            for item in (user.get(args.validation_key, []) or [])
            if isinstance(item, dict) and item.get("title")
        }

        # Use validation set ratings for matching
        rated_recommendations = add_ratings_and_filter(
            recommendations=recommendations,
            combined_ratings=validation_map,  # Use only validation set for matching
            list_prefix_sep=args.list_prefix_sep,
            title_rating_sep=args.title_rating_sep,
        )

        output_data.append({
            args.user_id_key: user_id,
            args.rec_key: rated_recommendations,
            args.validation_key: user.get(args.validation_key, []),
        })

    # Write output JSON file
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print("[INFO] Extraction, filtering, and saving completed.")


if __name__ == "__main__":
    main()
