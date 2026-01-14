import json
import re
import argparse
from typing import Any, Dict, List, Optional


# =========================
# Argument parsing (same style as your previous scripts)
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract clean 'Title (Year)' strings from recommendations and save a simplified JSON."
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default="llm_esrag.json",
        help="Path to the original JSON file that contains per-user recommendations and metadata."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="llm_esrag_1.json",
        help="Path to save the simplified JSON file."
    )
    parser.add_argument(
        "--user_id_key",
        type=str,
        default="user_id",
        help="Key in each entry indicating the user id."
    )
    parser.add_argument(
        "--rec_key",
        type=str,
        default="recommendations",
        help="Key in each entry holding the recommendation strings."
    )
    parser.add_argument(
        "--subset_key_in",
        type=str,
        default="watched_movies_subset",
        help="Key in each entry holding the watched subset."
    )
    parser.add_argument(
        "--subset_key_out",
        type=str,
        default="recommendation_subset",
        help="Key name used in the output JSON for the watched subset."
    )
    parser.add_argument(
        "--validation_key",
        type=str,
        default="validation_set",
        help="Key in each entry holding the validation set."
    )
    parser.add_argument(
        "--title_year_regex",
        type=str,
        default=r"(\w.*? \(\d{4}\))",
        help="Regex used to extract 'Title (Year)' from each recommendation line."
    )
    parser.add_argument(
        "--enumerate_format",
        type=str,
        default="{i}. {title}",
        help="Output formatting for enumerated recommendation titles."
    )

    return parser.parse_args()


# =========================
# Helper: extract movie titles from recommendation strings
# =========================
def extract_titles(
    recommendations: List[str],
    title_year_pattern: re.Pattern,
) -> List[str]:
    """
    Extract 'Title (Year)' from free-form recommendation strings.

    Supports formats like:
      - "Title (Year)"
      - "Title: Title (Year)"
      - "1. Title (Year) - Rating: 5"
    """
    titles: List[str] = []
    for recommendation in recommendations or []:
        rec_str = str(recommendation)

        # Match the core "Title (Year)" span
        match = title_year_pattern.search(rec_str)
        if not match:
            continue

        title = match.group(1)

        # Remove leading numbering (e.g., "1. ") and "Title: "
        clean_title = re.sub(r"^\d+\.\s*", "", title)
        clean_title = re.sub(r"^Title:\s*", "", clean_title)

        titles.append(clean_title)

    return titles


# =========================
# Main
# =========================
def main():
    args = parse_args()
    title_year_pattern = re.compile(args.title_year_regex)

    # Load the original JSON file
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data_list: List[Dict[str, Any]] = []

    # Process each user's data
    for user_data in data:
        user_id = user_data.get(args.user_id_key)
        recommendation_subset = user_data.get(args.subset_key_in)
        validation_set = user_data.get(args.validation_key, None)

        # Extract titles from the recommendations list
        rec_lines = user_data.get(args.rec_key, []) or []
        recommendation_titles = extract_titles(rec_lines, title_year_pattern)

        new_user_data = {
            args.user_id_key: user_id,
            args.rec_key: [
                args.enumerate_format.format(i=i + 1, title=title)
                for i, title in enumerate(recommendation_titles)
            ],
            args.subset_key_out: recommendation_subset,
            args.validation_key: validation_set,
        }

        new_data_list.append(new_user_data)

    # Save to a new JSON file
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(new_data_list, f, indent=4, ensure_ascii=False)

    print("[INFO] Data extraction and saving completed.")


if __name__ == "__main__":
    main()
