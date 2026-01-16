import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import GradScaler, autocast

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = None
bert_model = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a BERT-based Critic rating predictor for movie recommendations."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="critic_data_sample.json",
        help="Path to the training data JSON file (list of user entries with History / PredictedMovie / PredictedRating)."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Name or path of the Transformer encoder used to generate text embeddings (e.g., bert-base-uncased)."
    )

    parser.add_argument(
        "--max_history_length",
        type=int,
        default=20,
        help="Maximum number of historical movies per user to encode; extra items are truncated, fewer are zero-padded."
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="Number of discrete rating classes (e.g., 10 for ratings discretized into 0.5-step bins from 0.5 to 5.0)."
    )

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=1024,
        help="Hidden layer size of the MLP rating predictor."
    )

    parser.add_argument(
        "--outer_batch_size",
        type=int,
        default=1000,
        help="Number of samples processed per outer batch (data is chunked to control memory when encoding with BERT)."
    )
    parser.add_argument(
        "--inner_batch_size",
        type=int,
        default=16,
        help="Mini-batch size used by the DataLoader when training the MLP within each outer batch."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs over all outer batches."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate for the AdamW optimizer."
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of each outer batch used as a validation split (0–1)."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum token length for the Transformer tokenizer when encoding movie texts."
    )

    parser.add_argument(
        "--best_model_path",
        type=str,
        default="critic_b.pth",
        help="Path to save the best model checkpoint (selected by highest validation micro-precision)."
    )
    parser.add_argument(
        "--final_model_path",
        type=str,
        default="critic.pth",
        help="Path to save the final model checkpoint at the end of training."
    )

    return parser.parse_args()



@torch.no_grad()
def generate_embedding(text: str, max_seq_length: int = 512) -> np.ndarray:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length
    ).to(device)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy()


def process_data(
    data,
    max_history_length: int,
    embedding_size: int,
    max_seq_length: int
) -> tuple[np.ndarray, np.ndarray]:
    all_embeddings, all_labels = [], []

    for entry in data:
        history_texts = []
        for mv in (entry.get('History', []) or [])[:max_history_length]:
            history_texts.append(
                f"{mv.get('title', '')} directed by {mv.get('directedBy', '')} "
                f"starring {mv.get('starring', '')} rating {mv.get('rating', '')}"
            )

        history_embeddings = [
            generate_embedding(t, max_seq_length=max_seq_length)
            for t in history_texts
        ]

        if len(history_embeddings) < max_history_length:
            history_embeddings += [
                np.zeros(embedding_size, dtype=np.float32)
            ] * (max_history_length - len(history_embeddings))
        else:
            history_embeddings = history_embeddings[:max_history_length]

        pm = entry.get('PredictedMovie', {}) or {}
        predicted_movie_text = (
            f"{pm.get('title', '')} directed by {pm.get('directedBy', '')} "
            f"starring {pm.get('starring', '')}"
        )
        predicted_movie_embedding = generate_embedding(
            predicted_movie_text,
            max_seq_length=max_seq_length
        )

        combined_embeddings = np.array(
            history_embeddings + [predicted_movie_embedding],
            dtype=np.float32
        ).flatten()

        rating_label = int((float(entry['PredictedRating']) - 0.5) / 0.5)
        all_embeddings.append(combined_embeddings)
        all_labels.append(rating_label)

    return np.asarray(all_embeddings, dtype=np.float32), np.asarray(all_labels, dtype=np.int64)


def compute_micro_pr(y_true, y_pred, num_classes=10, eps=1e-12):
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    TP = np.diag(cm).astype(np.float64)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP

    TP_sum = TP.sum()
    FP_sum = FP.sum()
    FN_sum = FN.sum()

    micro_p = TP_sum / (TP_sum + FP_sum + eps)
    micro_r = TP_sum / (TP_sum + FN_sum + eps)
    return micro_p, micro_r


def compute_micro_f1(micro_p, micro_r, eps=1e-12):
    return 2 * micro_p * micro_r / (micro_p + micro_r + eps)


class RatingPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=10):
        super(RatingPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


def train_model_in_batches(
    data,
    outer_batch_size: int,
    model: nn.Module,
    criterion,
    optimizer,
    scaler: GradScaler,
    epochs: int,
    max_history_length: int,
    embedding_size: int,
    num_classes: int,
    inner_batch_size: int,
    val_ratio: float,
    max_seq_length: int,
    best_model_path: str,
    final_model_path: str,
):
    num_batches = len(data) // outer_batch_size + (1 if len(data) % outer_batch_size != 0 else 0)
    best_val_score = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_steps = 0
        epoch_val_labels, epoch_val_preds = [], []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * outer_batch_size
            end_idx = min((batch_idx + 1) * outer_batch_size, len(data))
            batch_data = data[start_idx:end_idx]

            features_np, labels_np = process_data(
                batch_data,
                max_history_length=max_history_length,
                embedding_size=embedding_size,
                max_seq_length=max_seq_length
            )

            unique_labels, counts = np.unique(labels_np, return_counts=True)
            min_count = counts.min()
            use_stratify = (len(labels_np) >= 2) and (min_count >= 2)

            if not use_stratify:
                print(
                    f"[Warning] Epoch {epoch + 1}, Batch {batch_idx}: "
                    f"some class has < 2 samples (min_count={min_count}), "
                    f"fall back to non-stratified split."
                )
                stratify_arg = None
            else:
                stratify_arg = labels_np

            X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
                features_np,
                labels_np,
                test_size=val_ratio,
                random_state=42,
                stratify=stratify_arg
            )

            X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
            y_train = torch.tensor(y_train_np, dtype=torch.long, device=device)
            X_val = torch.tensor(X_val_np, dtype=torch.float32, device=device)
            y_val = torch.tensor(y_val_np, dtype=torch.long, device=device)

            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            train_loader = DataLoader(
                train_dataset,
                batch_size=inner_batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=inner_batch_size
            )

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                total_steps += 1

            model.eval()
            with torch.no_grad():
                for X_val_batch, y_val_batch in val_loader:
                    val_outputs = model(X_val_batch)
                    _, val_pred = torch.max(val_outputs, 1)
                    epoch_val_labels.extend(y_val_batch.cpu().numpy().tolist())
                    epoch_val_preds.extend(val_pred.cpu().numpy().tolist())
            model.train()

        model.eval()
        micro_p, micro_r = 0.0, 0.0
        if len(epoch_val_labels):
            micro_p, micro_r = compute_micro_pr(
                epoch_val_labels,
                epoch_val_preds,
                num_classes=num_classes
            )

        avg_loss = total_loss / max(total_steps, 1)
        print(
            f"[Epoch {epoch + 1:03d}] "
            f"Loss: {avg_loss:.4f} | Val micro-P: {micro_p:.4f} | Val micro-R: {micro_r:.4f}"
        )

        curr_score = micro_p
        if curr_score > best_val_score:
            best_val_score = curr_score
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Best fine-tuning model updated at epoch {epoch + 1} "
                f"with Val micro-P: {micro_p:.4f}"
            )

    torch.save(model.state_dict(), final_model_path)
    print(f"Final fine-tuning model saved to {final_model_path}")


def test_model(
    data,
    outer_batch_size: int,
    model: nn.Module,
    model_path: str,
    label: str,
    max_history_length: int,
    embedding_size: int,
    num_classes: int,
    inner_batch_size: int,
    max_seq_length: int,
):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"\nTesting with {label}...")

    num_batches = len(data) // outer_batch_size + (1 if len(data) % outer_batch_size != 0 else 0)
    test_labels, test_preds = [], []

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * outer_batch_size
            end_idx = min((batch_idx + 1) * outer_batch_size, len(data))
            batch_data = data[start_idx:end_idx]

            features_np, labels_np = process_data(
                batch_data,
                max_history_length=max_history_length,
                embedding_size=embedding_size,
                max_seq_length=max_seq_length
            )

            features = torch.tensor(features_np, dtype=torch.float32, device=device)
            labels = torch.tensor(labels_np, dtype=torch.long, device=device)

            test_dataset = TensorDataset(features, labels)
            test_loader = DataLoader(
                test_dataset,
                batch_size=inner_batch_size,
                shuffle=False
            )

            for X_test_batch, y_test_batch in test_loader:
                test_outputs = model(X_test_batch)
                _, test_pred = torch.max(test_outputs, 1)
                test_labels.extend(y_test_batch.cpu().numpy().tolist())
                test_preds.extend(test_pred.cpu().numpy().tolist())

    micro_p, micro_r = (0.0, 0.0)
    if len(test_labels):
        micro_p, micro_r = compute_micro_pr(
            test_labels,
            test_preds,
            num_classes=num_classes
        )
    micro_f1 = compute_micro_f1(micro_p, micro_r)

    print(
        f"{label} - Test micro-P: {micro_p:.4f}, "
        f"Test micro-R: {micro_r:.4f}, "
        f"micro-F1: {micro_f1:.4f}"
    )


def main():
    global tokenizer, bert_model

    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    bert_model = AutoModel.from_pretrained(args.model_name).to(device)
    bert_model.eval()
    embedding_size = bert_model.config.hidden_size

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    input_size = args.max_history_length * embedding_size + embedding_size
    model = RatingPredictor(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_classes=args.num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    train_model_in_batches(
        data=data,
        outer_batch_size=args.outer_batch_size,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        epochs=args.epochs,
        max_history_length=args.max_history_length,
        embedding_size=embedding_size,
        num_classes=args.num_classes,
        inner_batch_size=args.inner_batch_size,
        val_ratio=args.val_ratio,
        max_seq_length=args.max_seq_length,
        best_model_path=args.best_model_path,
        final_model_path=args.final_model_path,
    )

    test_model(
        data=data,
        outer_batch_size=args.outer_batch_size,
        model=model,
        model_path=args.best_model_path,
        label="Best Model",
        max_history_length=args.max_history_length,
        embedding_size=embedding_size,
        num_classes=args.num_classes,
        inner_batch_size=args.inner_batch_size,
        max_seq_length=args.max_seq_length,
    )
    test_model(
        data=data,
        outer_batch_size=args.outer_batch_size,
        model=model,
        model_path=args.final_model_path,
        label="Final Model",
        max_history_length=args.max_history_length,
        embedding_size=embedding_size,
        num_classes=args.num_classes,
        inner_batch_size=args.inner_batch_size,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
