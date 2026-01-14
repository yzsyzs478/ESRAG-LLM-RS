import json
import re
import string
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = None
bert_model = None
emb_cache = {}


def normalize_title(t: str) -> str:
    if not isinstance(t, str):
        return ""
    s = t.strip().lower()
    s = re.sub(r"\(\s*\d{4}\s*\)\s*$", "", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    return s


@torch.no_grad()
def generate_embedding(text: str, max_seq_length: int, device) -> np.ndarray:
    if text in emb_cache:
        return emb_cache[text]
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length
    ).to(device)
    outputs = bert_model(**inputs)
    vec = outputs.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy()
    emb_cache[text] = vec
    return vec


def build_movie_index(movie_data):
    movie_index_raw = {m.get("title", ""): m for m in movie_data if "title" in m}
    movie_index_norm = {normalize_title(m.get("title", "")): m for m in movie_data if "title" in m}
    return movie_index_raw, movie_index_norm


def lookup_movie(title: str, movie_index_raw, movie_index_norm):
    m = movie_index_raw.get(title)
    if m is None:
        m = movie_index_norm.get(normalize_title(title))
    if m is None:
        return {"directedBy": "unknown", "starring": "unknown"}
    return {
      "directedBy": m.get("directedBy", "unknown"),
      "starring": m.get("starring", "unknown"),
    }


class RatingPredictor(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=10):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


def process_history(history, max_history_length: int, embedding_size: int, max_seq_length: int, device):
    texts = []
    for mv in (history or []):
        title = mv.get("title", "")
        directed_by = mv.get("directedBy", "")
        starring = mv.get("starring", "")
        texts.append(f"{title} directed by {directed_by} starring {starring}")
    history_embeddings = [generate_embedding(t, max_seq_length, device) for t in texts]
    if len(history_embeddings) < max_history_length:
        history_embeddings += [np.zeros(embedding_size, dtype=np.float32)] * (max_history_length - len(history_embeddings))
    else:
        history_embeddings = history_embeddings[:max_history_length]
    return np.array(history_embeddings, dtype=np.float32).flatten()


def parse_recommendation_title(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    m = re.match(r"^\s*\d+\s*[\.\)\-]\s*(.+)$", s)
    if m:
        return m.group(1).strip()
    return s


def main():
    global tokenizer, bert_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--max_history_len", type=int, default=20)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--ckpt_path", default="critic.pth")
    parser.add_argument("--movie_file", default="movie.json")
    parser.add_argument("--critic_file", default="llm_esrag_1.json")
    parser.add_argument("--output_file", default="predict_rating_llm_esrag_1.json")
    parser.add_argument("--max_seq_length", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    bert_model = AutoModel.from_pretrained(args.model_name).to(device)
    bert_model.eval()
    embedding_size = bert_model.config.hidden_size

    with open(args.movie_file, "r", encoding="utf-8") as file:
        movie_data = json.load(file)
    movie_index_raw, movie_index_norm = build_movie_index(movie_data)

    with open(args.critic_file, "r", encoding="utf-8") as file:
        critic_data = json.load(file)

    input_size = args.max_history_len * embedding_size + embedding_size
    model = RatingPredictor(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_classes=args.num_classes
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.eval()

    results = {}
    for user in critic_data:
        user_id = user.get("user_id")
        history_embeddings = process_history(
            user.get("recommendation_subset", []),
            max_history_length=args.max_history_len,
            embedding_size=embedding_size,
            max_seq_length=args.max_seq_length,
            device=device
        )
        results[user_id] = {}

        for rec_type in [
            "adjusted_recommendations",
            "second_adjusted_recommendations",
            "third_adjusted_recommendations"
        ]:
            if rec_type not in user:
                continue
            results[user_id][rec_type] = {}
            for recommendation in user[rec_type]:
                title = parse_recommendation_title(recommendation)
                movie_info = lookup_movie(title, movie_index_raw, movie_index_norm)
                text = f"{title} directed by {movie_info['directedBy']} starring {movie_info['starring']}"
                embedding = generate_embedding(text, args.max_seq_length, device)
                feature = np.hstack((history_embeddings, embedding)).astype(np.float32)
                feature_tensor = torch.tensor(
                    feature,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    logits = model(feature_tensor)
                    pred_class = torch.argmax(logits, dim=1).item()
                rating = pred_class * 0.5 + 0.5
                results[user_id][rec_type][recommendation] = float(rating)

    with open(args.output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print(f"[DONE] Predicted ratings saved to '{args.output_file}'.")


if __name__ == "__main__":
    main()
