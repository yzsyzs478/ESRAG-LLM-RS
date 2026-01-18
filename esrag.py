import os
import re
import gc
import json
import argparse
import difflib
import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import faiss

from transformers import AutoTokenizer, AutoModel
from openai import OpenAI


# =========================
# Argument parsing (same style as the previous script)
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Contriever retriever with KL(P_R || Q) using epsilon-greedy sampling and a Similarity MLP."
    )

    # ---------- Data & paths ----------
    parser.add_argument(
        "--movie_db_path",
        type=str,
        default="movie.json",
        help="Path to the movie metadata JSON (title -> {directedBy, starring, ...})."
    )
    parser.add_argument(
        "--user_data_path",
        type=str,
        default="esrag_data_sample.jsonl",
        help="Path to target users JSONL for training (each line is a user entry with History)."
    )
    parser.add_argument(
        "--movie_database_path",
        type=str,
        default="user_movie_history_database_sample.jsonl",
        help="Path to the full user history JSONL (kept for compatibility; not directly used when FAISS is fixed)."
    )

    # ---------- OpenAI SDK compatible-mode (replaces requests API) ----------
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
        "--llm_model",
        type=str,
        default="",
        help="Model name used by the OpenAI client (e.g., qwen3-8b)."
    )
    parser.add_argument(
        "--llm_temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the LLM."
    )
    parser.add_argument(
        "--llm_max_tokens",
        type=int,
        default=6000,
        help="max_tokens for OpenAI chat.completions."
    )
    parser.add_argument(
        "--llm_max_attempts",
        type=int,
        default=5,
        help="Maximum retry attempts for a single LLM call."
    )
    parser.add_argument(
        "--llm_retry_sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between LLM retries."
    )
    parser.add_argument(
        "--llm_timeout",
        type=float,
        default=0.0,
        help="Optional per-request timeout (seconds). 0 means not set."
    )

    # ---------- Retriever / Critic ----------
    parser.add_argument(
        "--retriever_name",
        type=str,
        default="facebook/contriever",
        help="HuggingFace model name/path for the Contriever encoder."
    )
    parser.add_argument(
        "--critic_ckpt",
        type=str,
        default="critic.pth",
        help="Path to the critic checkpoint (RatingPredictor) used for rating inference."
    )
    parser.add_argument(
        "--critic_hidden_size",
        type=int,
        default=1024,
        help="Hidden layer size of the critic MLP (must match the checkpoint)."
    )
    parser.add_argument(
        "--critic_num_classes",
        type=int,
        default=10,
        help="Number of discrete rating classes (must match the checkpoint)."
    )
    parser.add_argument(
        "--max_history_length",
        type=int,
        default=20,
        help="Maximum number of movies in user history used by critic features (pad/truncate to this length)."
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=768,
        help="Embedding dimension for Contriever/Critic (Contriever is typically 768)."
    )

    # ---------- Embedding cache / encoding ----------
    parser.add_argument(
        "--embed_batch_size",
        type=int,
        default=32,
        help="Batch size for cached embedding generation (generate_embeddings_cached)."
    )
    parser.add_argument(
        "--embed_max_length",
        type=int,
        default=512,
        help="Max token length for embedding generation (generate_embeddings_cached)."
    )
    parser.add_argument(
        "--train_maxlen",
        type=int,
        default=256,
        help="Max token length for training-time encoding (encode_texts_torch) with gradients."
    )

    # ---------- Fixed FAISS index ----------
    parser.add_argument(
        "--faiss_maxlen",
        type=int,
        default=512,
        help="Max token length used when building the fixed FAISS index (encode_texts_torch no-grad)."
    )
    parser.add_argument(
        "--faiss_topk",
        type=int,
        default=10,
        help="Top-K nearest neighbors retrieved from FAISS (before epsilon-greedy sampling)."
    )
    parser.add_argument(
        "--sample_k",
        type=int,
        default=10,
        help="Number of similar users sampled per target user for KL training."
    )

    # ---------- Training hyperparameters ----------
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Temperature for the P_R softmax over similarity_mlp logits."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Temperature for the Q softmax derived from predicted_scores."
    )
    parser.add_argument(
        "--initial_epsilon",
        type=float,
        default=0.7,
        help="Initial epsilon for epsilon-greedy sampling (higher means more exploration)."
    )
    parser.add_argument(
        "--final_epsilon",
        type=float,
        default=0.01,
        help="Final epsilon after linear decay."
    )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm."
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable AMP mixed precision training (CUDA only)."
    )

    # ---------- Optimizer ----------
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for the joint optimizer (retriever + similarity_mlp)."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for the joint optimizer."
    )

    # ---------- Checkpoint / outputs ----------
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoint_movies_simmlp1.pth",
        help="Path to the combined checkpoint (retriever + similarity_mlp + optimizer)."
    )
    parser.add_argument(
        "--save_prefix",
        type=str,
        default="epoch1",
        help="Prefix for per-epoch saved weights, e.g., retrievermlp_{save_prefix}_{epoch}.pth."
    )
    parser.add_argument(
        "--best_retriever_path",
        type=str,
        default="esrag.pth",
        help="Path to save the best retriever weights (lowest avg KL loss)."
    )
    parser.add_argument(
        "--best_simmlp_path",
        type=str,
        default="mlp.pth",
        help="Path to save the best similarity MLP weights (lowest avg KL loss)."
    )

    # ---------- System ----------
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for numpy/torch."
    )
    parser.add_argument(
        "--disable_tf32",
        action="store_true",
        help="Disable TF32 for matmul/cudnn (enabled by default in this script)."
    )

    return parser.parse_args()


# =========================
# Utilities
# =========================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Load JSONL user data
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
                print(f"⚠️ JSON parse error; skipping line {i}: {e}; snippet: {line[:120]} ...")
                continue
    return data


# =========================
# Movie metadata (title -> directedBy, starring)
# =========================
def build_movie_details_dict(movie_db_path: str) -> Dict[str, Dict[str, Any]]:
    with open(movie_db_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_movie_details(title: str, movie_details_dict: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    if title in movie_details_dict:
        return (
            movie_details_dict[title].get("directedBy", "Unknown"),
            movie_details_dict[title].get("starring", "Unknown"),
        )
    possible_titles = movie_details_dict.keys()
    matched_title = difflib.get_close_matches(title, possible_titles, n=1, cutoff=0.8)
    if matched_title:
        t = matched_title[0]
        return (
            movie_details_dict[t].get("directedBy", "Unknown"),
            movie_details_dict[t].get("starring", "Unknown"),
        )
    return "Unknown", "Unknown"


# =========================
# Critic (inference-only)
# =========================
class RatingPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return self.fc3(x)


def map_predicted_class_to_rating(predicted_class: int) -> float:
    mapping = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.0, 6: 3.5, 7: 4.0, 8: 4.5, 9: 5.0}
    return float(mapping.get(int(predicted_class), 3.0))


# =========================
# Similarity MLP (for P_R)
# =========================
class SimilarityMLP(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, e_i: torch.Tensor, e_s: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([e_i, e_s, torch.abs(e_i - e_s), e_i * e_s], dim=-1)
        return self.net(feat).squeeze(-1)


# =========================
# Embedding cache + batching
# =========================
class EmbeddingCache:
    def __init__(self):
        self.cache: Dict[str, np.ndarray] = {}

    def get(self, key: str):
        return self.cache.get(key, None)

    def put(self, key: str, value: np.ndarray):
        self.cache[key] = value


@torch.no_grad()
def generate_embeddings_cached(
    texts: List[str],
    tokenizer,
    retriever_model,
    device: torch.device,
    cache: EmbeddingCache,
    batch_size: int = 32,
    max_length: int = 512,
) -> List[np.ndarray]:
    results: List[Optional[np.ndarray]] = [None] * len(texts)
    uncached_texts, uncached_indices = [], []

    for i, text in enumerate(texts):
        v = cache.get(text)
        if v is not None:
            results[i] = v
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)

    if uncached_texts:
        for i in range(0, len(uncached_texts), batch_size):
            batch_texts = uncached_texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            outputs = retriever_model(**inputs)
            new_embs = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)

            for j, idx in enumerate(uncached_indices[i:i + batch_size]):
                text_key = uncached_texts[i + j]
                cache.put(text_key, new_embs[j])
                results[idx] = new_embs[j]

    return [r if r is not None else np.zeros(retriever_model.config.hidden_size, dtype=np.float32) for r in results]


def encode_texts_torch(
    texts: List[str],
    tokenizer,
    model,
    max_length: int = 512,
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    all_embs = []
    get_emb = model.get_input_embeddings()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)

        input_embeds = get_emb(inputs["input_ids"])
        input_embeds.requires_grad_()

        outputs = model(
            inputs_embeds=input_embeds,
            attention_mask=inputs.get("attention_mask", None),
            token_type_ids=inputs.get("token_type_ids", None),
        )
        cls = outputs.last_hidden_state[:, 0, :]
        cls = F.normalize(cls, dim=-1)
        all_embs.append(cls)

    return torch.cat(all_embs, dim=0) if all_embs else torch.zeros((0, model.config.hidden_size), device=device)


# =========================
# Title parsing (kept consistent with your earlier parsing)
# =========================
def extract_title_with_year(text: str) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    text = re.sub(r"^\d+\.\s*", "", text).strip()
    text = re.sub(r",\s*rating:\s*\d+(\.\d+)?", "", text).strip()
    text = re.sub(r"\(a\.k\.a\..*?\)", "", text).strip()
    match = re.search(r"(.+?)\s*\((\d{4})\)", text)
    if match:
        return f"{match.group(1).strip()} ({match.group(2).strip()})"
    return None


# =========================
# LLM recommendation (OpenAI client)
# =========================
def generate_recommendations(
    watched_movies: List[Dict[str, Any]],
    similar_user_history: List[Dict[str, Any]],
    *,
    client: OpenAI,
    llm_model: str,
    temperature: float,
    max_tokens: int,
    max_attempts: int,
    retry_sleep: float,
    timeout: float = 0.0,
) -> List[str]:
    combined = watched_movies + similar_user_history
    similar_user_history_text = "\n".join([
        f"title: {extract_title_with_year(m.get('title',''))}, rating: {m.get('rating','')}"
        for m in combined
        if isinstance(m, dict) and extract_title_with_year(m.get("title", ""))
    ])

    template = "{title} ({year})"
    messages = [
        {
            "role": "system",
            "content": (
                "You are a movie recommendation system. Given a set of movies "
                "(each including the title and rating) that a user watched, the user preference "
                "is based on the ratings that user gives to these movies. Now, please recommend "
                f"10 movies based on the user's preferences and a similar user's movie history, "
                "with the recommended years for the movie ranging from 1880 to 2021. "
                f"The format for the recommended movies should be {template}."
            ),
        },
        {
            "role": "user",
            "content": (
                "Here are the movies that user watched and rated:\n"
                + "\n".join([f"title: {m.get('title','')}, rating: {m.get('rating','')}" for m in watched_movies])
                + "\n\nHere are the movie histories of similar users:\n"
                + similar_user_history_text
                + "\nPlease recommend 10 movies for the user based on their preferences, while also incorporating "
                "the viewing history of similar users as auxiliary information. Rank the recommended movies "
                "according to the predicted level of user preference, using a rating scale from 1 to 5, "
                "where 5 indicates strong liking and 1 indicates strong disinterest. Only include movies with "
                "predicted ratings of 4 or higher."
            ),
        },
    ]

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            kwargs = dict(
                model=llm_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={"enable_thinking": False},
            )
            if timeout and timeout > 0:
                kwargs["timeout"] = timeout

            resp = client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()

            recs = text.split("\n")
            parsed = [extract_title_with_year(line.strip()) for line in recs if extract_title_with_year(line.strip())]
            return parsed

        except Exception as e:
            last_error = str(e)
            if attempt < max_attempts:
                time.sleep(retry_sleep)
            else:
                print(f"[LLM Failed] attempts={max_attempts}, last_error={last_error}")
                return []


# =========================
# Critic rating prediction (unchanged logic)
# =========================
def predict_rating(
    user_history: Any,
    predicted_movie: Dict[str, Any],
    *,
    tokenizer,
    retriever_model,
    device: torch.device,
    cache: EmbeddingCache,
    critic_model: nn.Module,
    movie_details_dict: Dict[str, Dict[str, Any]],
    max_history_length: int,
    embedding_dim: int,
    embed_batch_size: int,
    embed_max_length: int,
) -> Optional[float]:
    if isinstance(user_history, dict) and "History" in user_history:
        user_history = user_history["History"]
    elif not isinstance(user_history, list):
        return None

    history_texts = [
        f"{m['title']} directed by {m.get('directedBy','Unknown')} starring {m.get('starring','Unknown')} rating {m.get('rating','Unknown')}"
        for m in user_history
        if isinstance(m, dict) and "title" in m
    ][:max_history_length]

    if not history_texts:
        return None

    hist_embs = generate_embeddings_cached(
        history_texts,
        tokenizer=tokenizer,
        retriever_model=retriever_model,
        device=device,
        cache=cache,
        batch_size=embed_batch_size,
        max_length=embed_max_length,
    )

    if len(hist_embs) < max_history_length:
        hist_embs.extend([np.zeros(embedding_dim, dtype=np.float32) for _ in range(max_history_length - len(hist_embs))])

    d, s = get_movie_details(predicted_movie.get("title", ""), movie_details_dict)
    predicted_movie["directedBy"], predicted_movie["starring"] = d, s

    movie_text = f"{predicted_movie.get('title','')} directed by {predicted_movie.get('directedBy','Unknown')} starring {predicted_movie.get('starring','Unknown')}"
    movie_emb = generate_embeddings_cached(
        [movie_text],
        tokenizer=tokenizer,
        retriever_model=retriever_model,
        device=device,
        cache=cache,
        batch_size=embed_batch_size,
        max_length=embed_max_length,
    )[0]

    feats = np.array(hist_embs + [movie_emb], dtype=np.float32).flatten()
    x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)

    x = (x - x.mean()) / (x.std() + 1e-6)

    with torch.no_grad():
        logits = critic_model(x)
        cls = int(torch.argmax(logits, dim=1).item())
        return map_predicted_class_to_rating(cls)


# =========================
# Numerically robust helpers (unchanged)
# =========================
def _ensure_finite_tensor(x: torch.Tensor, name: str):
    if not torch.isfinite(x).all():
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        print(f"[WARN] Non-finite detected in {name}; replaced with zeros.")
    return torch.clamp(x, -1e6, 1e6)


def safe_softmax(logits: torch.Tensor, temp: float, dim: int = 0, eps: float = 1e-8):
    logits = _ensure_finite_tensor(logits, "logits_before_softmax")
    log_p = F.log_softmax(logits / max(temp, eps), dim=dim)
    p = torch.exp(log_p)
    p = p / (p.sum(dim=dim, keepdim=True) + eps)
    p = torch.clamp(p, eps, 1.0)
    p = p / p.sum(dim=dim, keepdim=True)
    return p, log_p


def safe_softmax_from_scores(scores: List[float], temp: float, device: torch.device, eps: float = 1e-8):
    v = torch.tensor(scores, dtype=torch.float32, device=device) / max(temp, eps)
    v = _ensure_finite_tensor(v, "q_raw")
    q = F.softmax(v, dim=0).detach()
    q = q / (q.sum() + eps)
    q = torch.clamp(q, eps, 1.0)
    q = (q / q.sum()).detach()
    return q


def safe_kl_from_log(logP: torch.Tensor, Q: torch.Tensor, eps: float = 1e-8):
    Q = _ensure_finite_tensor(Q, "Q_distribution")
    Q = torch.clamp(Q, eps, 1.0)
    Q = Q / Q.sum()
    return torch.sum(torch.exp(logP) * (logP - torch.log(Q)))


def _entropy(p: torch.Tensor):
    p = torch.clamp(p, 1e-8, 1.0)
    return float(-(p * p.log()).sum().item())


# =========================
# Checkpoint (combined retriever + MLP)
# =========================
def save_checkpoint_combined(retriever_model, similarity_mlp, optimizer, epoch, loss, best_avg_loss, filename):
    ckpt = {
        "epoch": epoch,
        "retriever_state": retriever_model.state_dict(),
        "mlp_state": similarity_mlp.state_dict(),
        "opt_state": optimizer.state_dict(),
        "loss": float(loss),
        "best_avg_loss": float(best_avg_loss),
    }
    torch.save(ckpt, filename)
    print(f"[CKPT] Saved checkpoint at epoch {epoch} -> {filename}")


def load_checkpoint_combined(retriever_model, similarity_mlp, optimizer, filename, device: torch.device):
    ckpt = torch.load(filename, map_location=device)
    retriever_model.load_state_dict(ckpt["retriever_state"])
    similarity_mlp.load_state_dict(ckpt["mlp_state"])
    optimizer.load_state_dict(ckpt["opt_state"])
    epoch = int(ckpt["epoch"])
    loss = float(ckpt.get("loss", float("inf")))
    best_avg_loss = float(ckpt.get("best_avg_loss", float("inf")))
    print(f"[CKPT] Loaded {filename}: epoch={epoch}, loss={loss:.4f}, best_avg_loss={best_avg_loss:.4f}")
    return epoch, loss, best_avg_loss


# =========================
# Build a fixed FAISS index (one-time)
# =========================
def build_faiss_fixed(
    user_data: List[Dict[str, Any]],
    tokenizer,
    retriever_model,
    device: torch.device,
    faiss_maxlen: int = 512,
) -> Tuple[faiss.Index, np.ndarray, List[int], Dict[int, int]]:
    texts_all, val_idx_all = [], []
    for idx, u in enumerate(user_data):
        hist = u.get("History", [])
        if isinstance(hist, list) and len(hist) > 0:
            titles_all = [m["title"] for m in hist if isinstance(m, dict) and "title" in m]
            texts_all.append(" ".join(titles_all))
            val_idx_all.append(idx)

    if not val_idx_all:
        raise RuntimeError("No valid users with history found for building FAISS.")

    with torch.no_grad():
        embs = encode_texts_torch(
            texts_all,
            tokenizer=tokenizer,
            model=retriever_model,
            max_length=faiss_maxlen,
            batch_size=16,
            device=device,
        )
    embs_np = embs.detach().cpu().numpy().astype("float32")
    embs_np /= (np.linalg.norm(embs_np, axis=1, keepdims=True) + 1e-12)

    index = faiss.IndexFlatIP(embs_np.shape[1])
    index.add(embs_np)

    pos = {u_idx: i for i, u_idx in enumerate(val_idx_all)}
    return index, embs_np, val_idx_all, pos


# =========================
# Training (fixed FAISS; retriever still receives gradients)
# =========================
def retrain_contriever_with_kl_epsilon(
    *,
    user_data: List[Dict[str, Any]],
    tokenizer,
    retriever_model,
    similarity_mlp,
    optimizer,
    critic_model,
    movie_details_dict: Dict[str, Dict[str, Any]],
    # OpenAI client
    client: OpenAI,
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_max_attempts: int,
    llm_retry_sleep: float,
    llm_timeout: float,
    # training hyper-params
    gamma: float,
    beta: float,
    epochs: int,
    checkpoint_path: str,
    initial_epsilon: float,
    final_epsilon: float,
    train_maxlen: int,
    accum_steps: int,
    max_grad_norm: float,
    use_amp: bool,
    # faiss
    faiss_pack: Tuple[faiss.Index, np.ndarray, List[int], Dict[int, int]],
    faiss_topk: int,
    sample_k: int,
    # critic/embedding
    max_history_length: int,
    embedding_dim: int,
    embed_batch_size: int,
    embed_max_length: int,
    # outputs
    save_prefix: str,
    best_retriever_path: str,
    best_simmlp_path: str,
    device: torch.device,
):
    faiss_index, embs_np, val_idx_all, pos = faiss_pack

    retriever_model.train()
    similarity_mlp.train()

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    embed_cache = EmbeddingCache()

    if os.path.exists(checkpoint_path):
        start_epoch, _, best_avg_loss = load_checkpoint_combined(
            retriever_model, similarity_mlp, optimizer, checkpoint_path, device=device
        )
    else:
        start_epoch, best_avg_loss = 0, float("inf")

    for epoch in range(start_epoch, epochs):
        # Linear epsilon decay
        if epochs <= 1:
            epsilon = float(final_epsilon)
        else:
            t = epoch / (epochs - 1)
            epsilon = initial_epsilon + (final_epsilon - initial_epsilon) * t
        epsilon = float(np.clip(epsilon, final_epsilon, initial_epsilon))
        print(f"\nEpoch [{epoch+1}/{epochs}] - epsilon={epsilon:.4f}")

        total_loss = 0.0
        accum_counter = 0
        optimizer.zero_grad(set_to_none=True)

        for k, user_idx in enumerate(val_idx_all, start=1):
            target_hist = user_data[user_idx].get("History", [])
            if not target_hist:
                continue

            try:
                u_i_text = " ".join([m["title"] for m in target_hist if isinstance(m, dict) and "title" in m])

                local_i = pos[user_idx]
                _, I = faiss_index.search(np.array([embs_np[local_i]], dtype=np.float32), faiss_topk)
                topk = [int(i) for i in dict.fromkeys(I[0]) if int(i) != local_i]

                non_top_pool = sorted(set(range(len(embs_np))) - set(topk) - {local_i})

                # epsilon-greedy sampling
                selected_ids = []
                rng = np.random
                while len(selected_ids) < sample_k:
                    if rng.rand() < epsilon:
                        pool = [i for i in non_top_pool if i not in selected_ids]
                        if pool:
                            selected_ids.append(int(rng.choice(pool)))
                        else:
                            all_pool = [i for i in range(len(embs_np)) if i not in selected_ids and i != local_i]
                            if all_pool:
                                selected_ids.append(int(rng.choice(all_pool)))
                            else:
                                break
                    else:
                        remain = [i for i in topk if i not in selected_ids]
                        if remain:
                            selected_ids.append(int(rng.choice(remain)))
                        else:
                            all_pool = [i for i in range(len(embs_np)) if i not in selected_ids and i != local_i]
                            if all_pool:
                                selected_ids.append(int(rng.choice(all_pool)))
                            else:
                                break

                if len(selected_ids) == 0:
                    continue

                u_i_emb = encode_texts_torch([u_i_text], tokenizer, retriever_model, max_length=train_maxlen, device=device)
                u_i_emb = u_i_emb.repeat(len(selected_ids), 1)

                sim_texts, sim_users = [], []
                for sid in selected_ids:
                    entry = user_data[val_idx_all[sid]]
                    sim_users.append(entry)
                    h = entry.get("History", [])
                    sim_texts.append(" ".join([m["title"] for m in h if isinstance(m, dict) and "title" in m]) if h else "")

                u_s_emb = encode_texts_torch(sim_texts, tokenizer, retriever_model, max_length=train_maxlen, device=device)

                logits = similarity_mlp(u_i_emb, u_s_emb)
                P_R, logP_R = safe_softmax(logits, temp=gamma, dim=0)

                # Q: similar user -> LLM -> Critic
                predicted_scores = []
                for u_s_entry in sim_users:
                    recs = generate_recommendations(
                        watched_movies=target_hist,
                        similar_user_history=u_s_entry.get("History", []),
                        client=client,
                        llm_model=llm_model,
                        temperature=llm_temperature,
                        max_tokens=llm_max_tokens,
                        max_attempts=llm_max_attempts,
                        retry_sleep=llm_retry_sleep,
                        timeout=llm_timeout,
                    )

                    scores = []
                    for title in recs:
                        score = predict_rating(
                            target_hist,
                            {
                                "title": title,
                                "directedBy": movie_details_dict.get(title, {}).get("directedBy", "Unknown"),
                                "starring": movie_details_dict.get(title, {}).get("starring", "Unknown"),
                            },
                            tokenizer=tokenizer,
                            retriever_model=retriever_model,
                            device=device,
                            cache=embed_cache,
                            critic_model=critic_model,
                            movie_details_dict=movie_details_dict,
                            max_history_length=max_history_length,
                            embedding_dim=embedding_dim,
                            embed_batch_size=embed_batch_size,
                            embed_max_length=embed_max_length,
                        )
                        if score is not None and np.isfinite(score):
                            scores.append(float(score))

                    if scores:
                        w = np.array(scores, dtype=np.float32)
                        w = w / (np.sum(w) + 1e-8)
                        val = float(np.dot(scores, w))
                    else:
                        val = 3.0

                    val = float(np.clip(val, 0.5, 5.0))
                    predicted_scores.append(val)

                if len(predicted_scores) == 0:
                    continue

                Q = safe_softmax_from_scores(predicted_scores, temp=beta, device=device)
                kl_loss = safe_kl_from_log(logP_R, Q)
                kl_loss = torch.clamp(kl_loss, 0.0, 10.0)

            except Exception as e:
                print(f"[SKIP] Exception at user k={k}: {e}")
                continue

            if not torch.isfinite(kl_loss):
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                continue

            loss_scaled = kl_loss / max(1, accum_steps)

            if amp_enabled:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            accum_counter += 1
            total_loss += float(kl_loss.item())

            if accum_counter % accum_steps == 0:
                if amp_enabled:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    list(retriever_model.parameters()) + list(similarity_mlp.parameters()),
                    max_grad_norm,
                )

                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0

            if k % 50 == 0:
                try:
                    with torch.no_grad():
                        print(
                            f"[Dbg] k={k} H(P_R)={_entropy(P_R):.3f} H(Q)={_entropy(Q):.3f} "
                            f"maxP={float(P_R.max()):.3f} maxQ={float(Q.max()):.3f}"
                        )
                except Exception:
                    pass

            if k % 10 == 0:
                print(f"User [{k}/{len(val_idx_all)}], eps={epsilon:.4f}, KL(avg so far)={total_loss/k:.4f}")

        if accum_counter > 0:
            if amp_enabled:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(retriever_model.parameters()) + list(similarity_mlp.parameters()),
                max_grad_norm,
            )
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss / max(1, len(val_idx_all))
        print(f"[DONE] Epoch [{epoch+1}] completed — Avg KL Loss: {avg_loss:.4f}")

        best_avg_loss = min(best_avg_loss, avg_loss)
        save_checkpoint_combined(
            retriever_model, similarity_mlp, optimizer,
            epoch + 1, avg_loss, best_avg_loss,
            checkpoint_path
        )

        torch.save(retriever_model.state_dict(), f"retrievermlp_{save_prefix}_{epoch+1}.pth")
        torch.save(similarity_mlp.state_dict(), f"simmlp_{save_prefix}_{epoch+1}.pth")

        if avg_loss <= best_avg_loss + 1e-9:
            torch.save(retriever_model.state_dict(), best_retriever_path)
            torch.save(similarity_mlp.state_dict(), best_simmlp_path)
            print(f"[BEST] Saved best models with Avg KL Loss: {best_avg_loss:.4f}")

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


# =========================
# main
# =========================
def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if args.disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # OpenAI-compatible client
    if not args.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Please set env OPENAI_API_KEY or pass --openai_api_key. "
            "Do NOT hardcode your API key in the source code."
        )

    client = OpenAI(
        api_key=args.openai_api_key,
        base_url=args.openai_base_url,
    )

    # Movie metadata
    movie_details_dict = build_movie_details_dict(args.movie_db_path)

    # Data
    user_data = load_user_data(args.user_data_path)
    _ = load_user_data(args.movie_database_path)  # kept for compatibility, not directly used

    # Retriever
    tokenizer = AutoTokenizer.from_pretrained(args.retriever_name)
    retriever_model = AutoModel.from_pretrained(args.retriever_name, gradient_checkpointing=True).to(device)

    if hasattr(retriever_model, "config"):
        retriever_model.config.use_cache = False
    if hasattr(retriever_model, "gradient_checkpointing_enable"):
        retriever_model.gradient_checkpointing_enable()
    if hasattr(retriever_model, "enable_input_require_grads"):
        retriever_model.enable_input_require_grads()

    # Critic (inference-only)
    critic_input_size = args.max_history_length * args.embedding_dim + args.embedding_dim
    critic_model = RatingPredictor(
        input_size=critic_input_size,
        hidden_size=args.critic_hidden_size,
        num_classes=args.critic_num_classes,
    ).to(device)
    critic_model.load_state_dict(torch.load(args.critic_ckpt, map_location=device))
    critic_model.eval()

    # Similarity MLP
    similarity_mlp = SimilarityMLP(dim=args.embedding_dim).to(device)

    # Joint optimizer
    optimizer = optim.Adam(
        list(retriever_model.parameters()) + list(similarity_mlp.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Build fixed FAISS (one-time)
    faiss_index_fixed, embs_np_fixed, val_idx_all_fixed, pos_fixed = build_faiss_fixed(
        user_data=user_data,
        tokenizer=tokenizer,
        retriever_model=retriever_model,
        device=device,
        faiss_maxlen=args.faiss_maxlen,
    )
    faiss_pack = (faiss_index_fixed, embs_np_fixed, val_idx_all_fixed, pos_fixed)

    # Train
    retrain_contriever_with_kl_epsilon(
        user_data=user_data,
        tokenizer=tokenizer,
        retriever_model=retriever_model,
        similarity_mlp=similarity_mlp,
        optimizer=optimizer,
        critic_model=critic_model,
        movie_details_dict=movie_details_dict,
        client=client,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        llm_max_attempts=args.llm_max_attempts,
        llm_retry_sleep=args.llm_retry_sleep,
        llm_timeout=args.llm_timeout,
        gamma=args.gamma,
        beta=args.beta,
        epochs=args.epochs,
        checkpoint_path=args.checkpoint_path,
        initial_epsilon=args.initial_epsilon,
        final_epsilon=args.final_epsilon,
        train_maxlen=args.train_maxlen,
        accum_steps=args.accum_steps,
        max_grad_norm=args.max_grad_norm,
        use_amp=args.use_amp,
        faiss_pack=faiss_pack,
        faiss_topk=args.faiss_topk,
        sample_k=args.sample_k,
        max_history_length=args.max_history_length,
        embedding_dim=args.embedding_dim,
        embed_batch_size=args.embed_batch_size,
        embed_max_length=args.embed_max_length,
        save_prefix=args.save_prefix,
        best_retriever_path=args.best_retriever_path,
        best_simmlp_path=args.best_simmlp_path,
        device=device,
    )


if __name__ == "__main__":
    main()
