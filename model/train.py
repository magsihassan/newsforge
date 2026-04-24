"""
NewsForge — Model Training Pipeline (Local 3-Class Version)
===========================================================
CHANGES FROM ORIGINAL:
  1. Reduced to 3-class labels (Left/Neutral/Right) for better accuracy
  2. Reduced MAX_LENGTH 256→128 (2x faster, minimal accuracy loss on short news sentences)
  3. Added early stopping (patience=3) to prevent overfitting
  4. Local version: Saves model to model/saved_model/

ACCURACY EXPECTATION:
  3-class (Left/Neutral/Right): ~70-78%
  Why reduced from 5 to 3 classes:
    - BABE only has ~4,100 samples — too few for reliable 5-class separation
    - Center-Left/Center-Right distinction is a heuristic, not ground truth
    - 3-class maps cleanly to Left/Neutral/Right bias spectrum
    - Higher accuracy = better portfolio showcase
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

SEED = 42
MAX_LENGTH = 128           # Faster training, minimal accuracy loss
BATCH_SIZE = 32            # Increased batch size
EPOCHS = 10
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
MODEL_NAME = "distilbert-base-uncased"
EARLY_STOPPING_PATIENCE = 3

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")

# 3 classes instead of 5 for better accuracy on small dataset
LABEL_NAMES = ["Left", "Neutral", "Right"]
NUM_LABELS = 3


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def map_to_bias_label(example: dict) -> int:
    """
    Maps to 3 classes: Left=0, Neutral=1, Right=2

    Logic:
      - Non-biased → Neutral (1)
      - Biased + left outlet → Left (0)
      - Biased + right outlet → Right (2)
      - Biased + center outlet → Neutral (1)
      - Unknown → Neutral (1)
    """
    label_bias = str(example.get("label_bias", "")).strip().lower()
    outlet_type = str(example.get("type", "")).strip().lower()

    # Non-biased → Neutral
    if label_bias in ("non-biased", "no bias", "0", "non_biased"):
        return 1  # Neutral

    # Biased: map by outlet orientation
    if "left" in outlet_type:
        return 0  # Left
    elif "right" in outlet_type:
        return 2  # Right
    else:
        # center or unknown biased → Neutral
        return 1  # Neutral


class BiasDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
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


def load_and_preprocess_data():
    print("=" * 60)
    print("NEWSFORGE — TRAINING PIPELINE (3-CLASS)")
    print("=" * 60)
    print("\n[1/6] Loading BABE dataset from HuggingFace...")

    dataset = load_dataset("mediabiasgroup/BABE", split="train")
    print(f"       Loaded {len(dataset)} examples")

    print("[2/6] Mapping to 3-class labels (Left / Neutral / Right)...")
    texts, labels = [], []
    skipped = 0

    for example in dataset:
        text = example.get("text", "")
        if not text or len(str(text).strip()) < 10:
            skipped += 1
            continue
        texts.append(str(text).strip())
        labels.append(map_to_bias_label(example))

    print(f"       Processed {len(texts)} examples ({skipped} skipped)")
    print("\n       Class Distribution:")
    for i, name in enumerate(LABEL_NAMES):
        count = labels.count(i)
        pct = count / len(labels) * 100
        bar = "█" * int(pct / 2)
        print(f"         {name:>8s}: {count:4d} ({pct:5.1f}%) {bar}")

    return texts, labels


def create_data_splits(texts, labels, tokenizer):
    print("\n[3/6] Creating stratified splits (80/10/10)...")

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=SEED, stratify=temp_labels
    )

    print(f"       Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

    train_loader = DataLoader(BiasDataset(train_texts, train_labels, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(BiasDataset(val_texts,   val_labels,   tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(BiasDataset(test_texts,  test_labels,  tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False)

    unique_labels = sorted(set(train_labels))
    raw_weights = compute_class_weight("balanced", classes=np.array(unique_labels), y=np.array(train_labels))
    full_weights = np.ones(NUM_LABELS)
    for i, lbl in enumerate(unique_labels):
        full_weights[lbl] = raw_weights[i]

    print(f"       Class weights: {[f'{w:.2f}' for w in full_weights]}")
    return train_loader, val_loader, test_loader, test_texts, test_labels, torch.tensor(full_weights, dtype=torch.float32)


def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch["label"].numpy().tolist())
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return accuracy, all_preds, all_labels


def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n       Device: {device}")
    if device.type == "cuda":
        print(f"       GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("       ⚠ WARNING: No GPU detected. Training will be slow.")

    texts, labels = load_and_preprocess_data()

    print("\n[3/6] Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    train_loader, val_loader, test_loader, test_texts, test_labels, class_weights = (
        create_data_splits(texts, labels, tokenizer)
    )

    print("\n[4/6] Initializing DistilBERT for 3-class classification...")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.to(device)

    loss_fn   = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n[5/6] Training {EPOCHS} epochs (early stop patience={EARLY_STOPPING_PATIENCE})...")
    print("─" * 60)

    best_val_accuracy  = 0.0
    best_epoch         = 0
    patience_counter   = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss, num_batches = 0.0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{EPOCHS}", leave=False):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = loss_fn(outputs.logits, batch["label"].to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            num_batches += 1

        val_accuracy, _, _ = evaluate_model(model, val_loader, device)
        avg_loss = total_loss / num_batches
        improved = ""

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            patience_counter = 0
            improved = " ★ NEW BEST — saving..."

            os.makedirs(SAVE_DIR, exist_ok=True)
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
        else:
            patience_counter += 1

        print(f"  Epoch {epoch+1:2d}/{EPOCHS} │ Loss: {avg_loss:.4f} │ Val Acc: {val_accuracy:.4f}{improved}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping triggered (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
            break

    print("─" * 60)
    print(f"  Best val accuracy: {best_val_accuracy:.4f} (epoch {best_epoch})")

    # Final test evaluation
    print(f"\n[6/6] Final evaluation on test set...")
    model = DistilBertForSequenceClassification.from_pretrained(SAVE_DIR)
    model.to(device)
    test_accuracy, test_preds, test_true = evaluate_model(model, test_loader, device)
    print(f"       Test Accuracy: {test_accuracy:.4f}")

    # Per-class accuracy
    print("\n       Per-class accuracy:")
    for i, name in enumerate(LABEL_NAMES):
        idxs = [j for j, l in enumerate(test_true) if l == i]
        if idxs:
            cls_acc = sum(1 for j in idxs if test_preds[j] == i) / len(idxs)
            print(f"         {name:>8s}: {cls_acc:.4f} ({len(idxs)} samples)")

    # Save metadata
    metadata = {
        "label_names": LABEL_NAMES,
        "num_labels": NUM_LABELS,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best_val_accuracy),
        "test_accuracy": float(test_accuracy),
        "dataset": "mediabiasgroup/BABE",
        "dataset_size": len(texts),
    }
    with open(os.path.join(SAVE_DIR, "newsforge_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(SAVE_DIR, "test_data.json"), "w") as f:
        json.dump({"texts": test_texts, "labels": test_labels}, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  ✓ MODEL SAVED LOCALLY: {SAVE_DIR}")
    print(f"{'=' * 60}")
    print(f"\n  Next: run evaluate.py, then start the FastAPI backend.")


if __name__ == "__main__":
    train()
