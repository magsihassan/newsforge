"""
NewsForge — Model Evaluation Script
=====================================
Evaluates the fine-tuned DistilBERT model on the held-out test set.
Prints accuracy, per-class precision/recall/F1, macro F1, and confusion matrix.

Usage:
    python evaluate.py

Prerequisite:
    Run train.py first to generate the saved model in model/saved_model/
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")
BATCH_SIZE = 16
MAX_LENGTH = 256


class BiasDataset(Dataset):
    """PyTorch Dataset for tokenized bias classification data."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def evaluate():
    """
    Load the saved model and evaluate it on the test set.
    Prints detailed classification metrics and confusion matrix.
    """
    print("=" * 60)
    print("NEWSFORGE MODEL EVALUATION")
    print("=" * 60)

    # ─────────────────────────────────────────
    # Check that saved model exists
    # ─────────────────────────────────────────
    if not os.path.exists(SAVE_DIR):
        print(f"\n  ERROR: No saved model found at {SAVE_DIR}")
        print(f"  Run 'python train.py' first to train and save the model.")
        return

    # ─────────────────────────────────────────
    # Load metadata and test data
    # ─────────────────────────────────────────
    print("\n[1/4] Loading model metadata...")
    metadata_path = os.path.join(SAVE_DIR, "newsforge_metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    label_names = metadata["label_names"]
    print(f"       Labels: {label_names}")
    print(f"       Dataset: {metadata['dataset']} ({metadata['dataset_size']} samples)")
    print(f"       Training best epoch: {metadata['best_epoch']}")

    print("\n[2/4] Loading test data...")
    test_data_path = os.path.join(SAVE_DIR, "test_data.json")
    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    test_texts = test_data["texts"]
    test_labels = test_data["labels"]
    print(f"       Test samples: {len(test_texts)}")

    # ─────────────────────────────────────────
    # Load model and tokenizer
    # ─────────────────────────────────────────
    print("\n[3/4] Loading fine-tuned DistilBERT model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"       Device: {device}")

    tokenizer = DistilBertTokenizer.from_pretrained(SAVE_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(SAVE_DIR)
    model.to(device)
    model.eval()

    # ─────────────────────────────────────────
    # Run inference on test set
    # ─────────────────────────────────────────
    print("\n[4/4] Running evaluation on test set...")

    test_dataset = BiasDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    # ─────────────────────────────────────────
    # Print Results
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Overall accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"\n  Overall Accuracy:  {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"  Macro F1 Score:    {macro_f1:.4f}")
    print(f"  Weighted F1 Score: {weighted_f1:.4f}")

    # Per-class classification report
    print(f"\n{'─' * 60}")
    print("  Per-Class Classification Report:")
    print(f"{'─' * 60}")

    # Filter to only classes present in test data
    present_labels = sorted(set(all_labels + all_preds))
    present_names = [label_names[i] for i in present_labels]

    report = classification_report(
        all_labels,
        all_preds,
        labels=present_labels,
        target_names=present_names,
        digits=4,
        zero_division=0,
    )
    print(report)

    # Confusion Matrix
    print(f"{'─' * 60}")
    print("  Confusion Matrix:")
    print(f"{'─' * 60}")

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(label_names))))

    # Header
    header = "  Predicted →  " + "  ".join(f"{name[:7]:>7s}" for name in label_names)
    print(f"\n{header}")
    print("  " + "─" * (15 + 9 * len(label_names)))

    # Rows
    for i, name in enumerate(label_names):
        row_values = "  ".join(f"{cm[i][j]:7d}" for j in range(len(label_names)))
        print(f"  {name:>13s} │ {row_values}")

    print(f"\n{'─' * 60}")

    # Average confidence analysis
    print("\n  Average Confidence per Predicted Class:")
    print(f"{'─' * 60}")
    for i, name in enumerate(label_names):
        class_probs = [
            all_probs[j][i]
            for j in range(len(all_preds))
            if all_preds[j] == i
        ]
        if class_probs:
            avg_conf = np.mean(class_probs)
            print(f"    {name:>13s}: {avg_conf:.4f} ({avg_conf * 100:.1f}%)")
        else:
            print(f"    {name:>13s}: N/A (no predictions)")

    print(f"\n{'=' * 60}")
    print("  Evaluation complete ✓")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    evaluate()
