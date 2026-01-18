# -*- coding: utf-8 -*-
import json
import os
import re
import time
import argparse
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Argument parsing (same style as the previous script)
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run per-user retrieval + LLM recommendation with a Contriever retriever and FAISS."
    )

    # ---------- Data paths ----------
    parser.add_argument(
        "--file_path",
        type=str,
        default="user_movie_history_sample",
        help="Path to the target user JSONL file (each line is a user entry with History)."
    )
    parser.add_argument(
        "--user_history_path",
        type=str,
        default="user_movie_history_database_sample.jsonl",
        help="Path to the full user history JSONL file used to build the FAISS index."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="llm_esrag.json",
        help="Path to save the output JSON file."
    )

    # ---------- Retriever ----------
    parser.add_argument(
        "--retriever_name",
        type=str,
        default="facebook/contriever",
        help="HuggingFace model name/path for the retriever encoder."
    )
    parser.add_argument(
        "--retriever_ckpt",
        type=str,
        default="esrag.pth",
        help="Path to the retriever checkpoint (.pth) to be loaded into the encoder (strict=False)."
    )
    parser.add_argument(
        "--embed_max_length",
        type=int,
        default=512,
        help="Max token length for encoding user histories and building FAISS embeddings."
    )

    # ---------- FAISS ----------
    parser.add_argument(
        "--num_similar_users",
        type=int,
        default=5,
        help="Number of similar users retrieved from FAISS for each target user."
    )

    # ---------- LLM (OpenAI-compatible client) ----------
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", ""),
        help="OpenAI-compatible API key (defaults to env OPENAI_API_KEY)."
    )
    parser.add_argument(
        "--openai_base_url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", "").strip(),
        help="OpenAI-compatible base_url (defaults to env OPENAI_BASE_URL)."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen3-8b",
        help="LLM model name used by the OpenAI client."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=6000,
        help="max_tokens for OpenAI chat.completions."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the LLM."
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=5,
        help="Maximum retry attempts for each LLM request."
    )
    parser.add_argument(
        "--retry_sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between LLM retries."
    )
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=0.0,
        help="Optional per-request timeout (seconds). 0 means not set."
    )

    # ---------- Logging ----------
    parser.add_argument(
        "--sidecar_timing_log",
        type=str,
        default="llm_call_timings_qwen.jsonl",
        help="Sidecar JSONL file path to log each LLM call timing record."
    )

    return parser.parse_args()


# =========================
# Load user data (JSONL)
# =========================
def load_user_data(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip().lstrip("\ufeff")
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                if line.endswith(","):
                    try:
                        data.append(json.loads(line[:-1].rstrip()))
                        continue
                    except json.JSONDecodeError:
                        pass
                print(f"[WARN] JSON parse error; skipping line {i}: {e}; snippet: {line[:120]} ...")
                continue
    return data


# =========================
# Convert a movie history to a single string
# =========================
def movie_history_to_string(history: List[Dict[str, Any]]) -> str:
    return " ".join([
        f"{movie.get('title','')} {movie.get('directedBy','')} {movie.get('starring','')} {movie.get('rating','')}"
        for movie in history
    ])


# =========================
# Load retriever model (Contriever + optional checkpoint)
# =========================
def load_retriever_model(retriever_model_path: str, model_name: str = "facebook/contriever"):
    retriever_model = AutoModel.from_pretrained(model_name).to(device)
    retriever_model.load_state_dict(torch.load(retriever_model_path, map_location=device), strict=False)
    retriever_model.eval()
    return retriever_model


# =========================
# Generate a single embedding
# =========================
@torch.no_grad()
def generate_embedding(text: str, tokenizer, retriever_model, max_length: int = 512) -> np.ndarray:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)
    outputs = retriever_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()


# =========================
# Build FAISS index over all users
# =========================
def build_user_faiss_index(all_users, tokenizer, retriever_model, max_length: int = 512):
    embeddings = np.array([
        generate_embedding(movie_history_to_string(user["History"]), tokenizer, retriever_model, max_length=max_length)
        for user in all_users
    ], dtype=np.float32)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


# =========================
# Retrieve similar users from FAISS
# =========================
def find_similar_users(user_embedding: np.ndarray, user_index, num_similar_users: int = 10):
    _, I = user_index.search(np.array([user_embedding], dtype=np.float32), num_similar_users)
    return I[0]


# ==========================
# Extract only title(year) from model output, restricted to candidate set
# (Do not change prompt; only change parsing)
# ==========================
_YEAR_RE = re.compile(r"\((18\d{2}|19\d{2}|20\d{2})\)")
_LINE_MOVIE_RE = re.compile(
    r"^\s*\d+\s*[\.\)]\s*(?:\*\*)?(?P<title>.+?\((?:18\d{2}|19\d{2}|20\d{2})\))(?:\*\*)?(?:\s*[–—-].*)?$"
)


def _norm(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("**", "").replace("__", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_title_year_only(text: str, candidate_movies, k: int = 11):
    cand_set = {_norm(m["title"]) for m in candidate_movies if m.get("title")}
    recs = []

    for raw in (text or "").splitlines():
        line = _norm(raw)
        if not line or line in {"---", "—", "–"}:
            continue

        m = _LINE_MOVIE_RE.match(line)
        if m:
            t = _norm(m.group("title"))
        else:
            if not _YEAR_RE.search(line):
                continue
            # Trim tails like "- Rating: 5 / – Rating: 5 / — Rating: 5"
            t = re.split(r"\s+[–—-]\s+", line, maxsplit=1)[0].strip()
            t = _norm(t)

        # Must be in candidate set to prevent hallucinations and filter explanation lines
        if t in cand_set and t not in recs:
            recs.append(t)
        if len(recs) >= k:
            break

    return recs


# =========================
# Single LLM call for recommendations (qwen3-8b + OpenAI client)
# =========================
def suggest_movie(
    watched_movies_subset,
    similar_user_histories,
    candidate_movies,
    client: OpenAI,
    model_name: str = "qwen3-8b",
    max_attempts: int = 5,
    request_timeout: float = 0.0,
    max_tokens: int = 3000,
    temperature: float = 0.7,
    retry_sleep: float = 2.0,
):
    """
    Keep the prompt unchanged; only adjust parsing:
    only keep up to 11 title(year) items and require they come from the candidate set.
    """

    similar_user_histories_text = "\n\n".join([
        "\n".join([f"title: {movie['title']}, rating: {movie['rating']}" for movie in user_history])
        for user_history in similar_user_histories
    ])

    candidate_movies_text = "\n".join([f"title: {movie['title']}" for movie in candidate_movies])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a movie recommendation system. Given a set of movies (each including the title and rating) that a user "
                "watched, the user preference is based on the ratings that user gives to these movies. Now, please recommend 11 movies "
                "from the candidate movie set based on the user's preferences and a similar user's movie history. The recommended years "
                "for the movie should be from 1880 to 2021. The format for the recommended movies should be title(year)."
            ),
        },
        {
            "role": "user",
            "content": (
                "Here are the movies that user watched and rated:\n" +
                "\n".join([f"title: {movie['title']}, rating: {movie['rating']}" for movie in watched_movies_subset]) +
                "\n\nHere is the movie history of a similar user:\n" +
                similar_user_histories_text +
                "\n\nHere are the candidate movies:\n" +
                candidate_movies_text +
                "\nPlease recommend 11 movies from the candidate movies based on user preferences and similar user's history and rank "
                "them according to how much the user might like them. The ranking should be based on a rating scale from 1 to 5, "
                "where 5 means I like them the most and 1 means I don't like them at all."
            ),
        },
    ]

    last_error = None
    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            kwargs = dict(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={"enable_thinking": False},
            )
            if request_timeout and request_timeout > 0:
                kwargs["timeout"] = request_timeout

            resp = client.chat.completions.create(**kwargs)

            latency_sec = time.time() - t0
            ended_at = datetime.now(timezone.utc).isoformat()

            text = (resp.choices[0].message.content or "").strip()

            # Key: extract only title(year) from candidate set, up to 11 items
            recommendations = extract_title_year_only(text, candidate_movies, k=11)

            return {
                "status": "ok",
                "attempt": attempt,
                "latency_sec": latency_sec,
                "started_at": started_at,
                "ended_at": ended_at,
                "recommendations": recommendations,
            }

        except Exception as e:
            latency_sec = time.time() - t0
            ended_at = datetime.now(timezone.utc).isoformat()
            last_error = str(e)

            if attempt < max_attempts:
                time.sleep(retry_sleep)
            else:
                return {
                    "status": "failed",
                    "attempt": attempt,
                    "latency_sec": latency_sec,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "error": last_error,
                    "recommendations": [],
                }


# =========================
# Sidecar timing log: append one JSONL line per LLM call
# =========================
def append_timing_log(sidecar_path: str, log_obj: Dict[str, Any]):
    with open(sidecar_path, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(log_obj, ensure_ascii=False) + "\n")


# =========================
# Main processing function
# =========================
def process_batch_with_individual_retrieval(
    file_path: str,
    output_path: str,
    user_history_path: str,
    tokenizer,
    retriever_model,
    client: OpenAI,
    model_name: str,
    sidecar_timing_log: str,
    num_similar_users: int,
    embed_max_length: int,
    max_attempts: int,
    request_timeout: float,
    max_tokens: int,
    temperature: float,
    retry_sleep: float,
):
    users = load_user_data(file_path)
    all_users = load_user_data(user_history_path)

    print("[INFO] Building FAISS index...")
    user_index = build_user_faiss_index(all_users, tokenizer, retriever_model, max_length=embed_max_length)
    print("[INFO] FAISS index built.")

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as infile:
            existing_data = json.load(infile)
    else:
        existing_data = []

    processed_user_ids = {entry["user_id"] for entry in existing_data}
    output_data = existing_data

    for user in users:
        if user["user_id"] in processed_user_ids:
            print(f"[INFO] Skipping already processed user: {user['user_id']}")
            continue

        history_len = len(user["History"])
        split_idx = max(1, history_len // 3)

        watched_movies_subset = [
            {
                "title": movie["title"],
                "rating": movie["rating"],
                "directedBy": movie.get("directedBy", ""),
                "starring": movie.get("starring", ""),
            }
            for movie in user["History"][:split_idx]
        ]

        candidate_movies = [{"title": movie["title"]} for movie in user["History"][split_idx:]]
        validation_set = [
            {"title": movie["title"], "rating": movie["rating"]}
            for movie in user["History"][split_idx:]
        ]

        user_embedding = generate_embedding(
            movie_history_to_string(user["History"]),
            tokenizer,
            retriever_model,
            max_length=embed_max_length,
        )
        similar_indices = find_similar_users(user_embedding, user_index, num_similar_users=num_similar_users)

        all_recommendations = []
        llm_call_timings = []

        for round_id, idx in enumerate(similar_indices, start=1):
            sim_user = all_users[int(idx)]
            sim_user_history = [
                {"title": movie["title"], "rating": movie["rating"]}
                for movie in sim_user["History"]
            ]

            result = suggest_movie(
                watched_movies_subset=watched_movies_subset,
                similar_user_histories=[sim_user_history],
                candidate_movies=candidate_movies,
                client=client,
                model_name=model_name,
                max_attempts=max_attempts,
                request_timeout=request_timeout,
                max_tokens=max_tokens,
                temperature=temperature,
                retry_sleep=retry_sleep,
            )

            timing_record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "target_user_id": user["user_id"],
                "round": round_id,
                "similar_user_id": sim_user.get("user_id", int(idx)),
                "status": result["status"],
                "attempt": result["attempt"],
                "latency_sec": result["latency_sec"],
                "started_at": result["started_at"],
                "ended_at": result["ended_at"],
            }
            if result["status"] == "failed":
                timing_record["error"] = result.get("error", "")

            append_timing_log(sidecar_timing_log, timing_record)
            llm_call_timings.append(timing_record)

            # Each round takes up to 11 items (already filtered to title(year))
            recs = result.get("recommendations", [])[:11]
            all_recommendations.extend(recs)

        # Majority voting by frequency, take Top-11
        movie_counts = Counter(all_recommendations)
        sorted_recommendations = [movie for movie, _ in movie_counts.most_common(11)]

        output_data.append({
            "user_id": user["user_id"],
            "watched_movies_subset": watched_movies_subset,
            "validation_set": validation_set,
            "recommendations": sorted_recommendations,
            "llm_call_timings": llm_call_timings,
        })

        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(output_data, outfile, ensure_ascii=False, indent=4)

        print(f"[INFO] Finished processing user: {user['user_id']}")


# =========================
# Entry point
# =========================
def main():
    args = parse_args()

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.retriever_name)
    retriever_model = load_retriever_model(args.retriever_ckpt, model_name=args.retriever_name)

    if not args.openai_api_key:
        raise ValueError(
            "Missing OPENAI_API_KEY. Please set env OPENAI_API_KEY or pass --openai_api_key. "
            "Do NOT hardcode API keys in source code."
        )

    client = OpenAI(
        api_key=args.openai_api_key,
        base_url=args.openai_base_url,
    )

    process_batch_with_individual_retrieval(
        file_path=args.file_path,
        output_path=args.output_path,
        user_history_path=args.user_history_path,
        tokenizer=tokenizer,
        retriever_model=retriever_model,
        client=client,
        model_name=args.model_name,
        sidecar_timing_log=args.sidecar_timing_log,
        num_similar_users=args.num_similar_users,
        embed_max_length=args.embed_max_length,
        max_attempts=args.max_attempts,
        request_timeout=args.request_timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        retry_sleep=args.retry_sleep,
    )


if __name__ == "__main__":
    main()
