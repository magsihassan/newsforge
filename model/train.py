"""
NewsForge — Model Training Pipeline
=====================================
Fine-tunes DistilBERT (distilbert-base-uncased) on the BABE (Bias Annotations By Experts)
dataset for 5-class political bias classification.

Dataset: mediabiasgroup/BABE on HuggingFace
    - ~4,121 news sentences from US news outlets
    - Binary bias labels (Biased / Non-biased)
    - Outlet political orientation (left / right / center)

Label Mapping Strategy:
    We combine the binary bias label with outlet political orientation to create
    5 composite classes:
        0 = Left       — biased sentence from left-leaning outlet
        1 = Center-Left — biased sentence from center outlet (opinion-leaning)
        2 = Neutral     — non-biased sentence (any outlet)
        3 = Center-Right — biased sentence from center outlet (factual-leaning)
        4 = Right       — biased sentence from right-leaning outlet

Usage:
    python train.py

Output:
    model/saved_model/  — contains the fine-tuned model, tokenizer, and label mapping
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

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
MODEL_NAME = "distilbert-base-uncased"
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")

# Label definitions — the 5 bias categories
LABEL_NAMES = ["Left", "Center-Left", "Neutral", "Center-Right", "Right"]
NUM_LABELS = len(LABEL_NAMES)


def set_seed(seed: int):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def map_to_bias_label(example: dict) -> int:
    """
    Map a BABE dataset example to one of 5 bias classes.

    Strategy:
        - Non-biased sentences → Neutral (2)
        - Biased + left outlet → Left (0)
        - Biased + right outlet → Right (4)
        - Biased + center outlet → Center-Left (1) or Center-Right (3)
          depending on the opinion label intensity

    Args:
        example: A single row from the BABE dataset with keys:
                 'label_bias', 'type', 'label_opinion'

    Returns:
        Integer label (0-4)
    """
    label_bias = str(example.get("label_bias", "")).strip().lower()
    outlet_type = str(example.get("type", "")).strip().lower()
    label_opinion = str(example.get("label_opinion", "")).strip().lower()

    # Non-biased sentences are always Neutral
    if label_bias in ("non-biased", "no bias", "0", "non_biased"):
        return 2  # Neutral

    # Biased sentences: map based on outlet political orientation
    if "left" in outlet_type:
        return 0  # Left
    elif "right" in outlet_type:
        return 4  # Right
    elif "center" in outlet_type or "centre" in outlet_type:
        # For center outlets, use opinion label to split into Center-Left / Center-Right
        # More opinionated → Center-Left (progressive editorial tendency)
        # More factual but still biased → Center-Right (establishment framing)
        if "opinion" in label_opinion or "expresses" in label_opinion:
            return 1  # Center-Left
        else:
            return 3  # Center-Right
    else:
        # Unknown outlet type — default to Neutral
        return 2  # Neutral


class BiasDataset(Dataset):
    """
    PyTorch Dataset wrapper for tokenized bias classification data.

    Each item returns:
        - input_ids: Tokenized input tensor
        - attention_mask: Attention mask tensor
        - label: Integer class label (0-4)
    """

    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_and_preprocess_data():
    """
    Download the BABE dataset from HuggingFace and preprocess it.

    Returns:
        texts: List of news sentence strings
        labels: List of integer labels (0-4)
    """
    print("=" * 60)
    print("NEWSFORGE MODEL TRAINING PIPELINE")
    print("=" * 60)
    print("\n[1/6] Loading BABE dataset from HuggingFace...")

    # Load the dataset — it has a single 'train' split
    dataset = load_dataset("mediabiasgroup/BABE", split="train")
    print(f"       Loaded {len(dataset)} examples")

    # Map each example to our 5-class label system
    print("[2/6] Mapping labels to 5-class bias system...")
    texts = []
    labels = []
    skipped = 0

    for example in dataset:
        text = example.get("text", "")
        if not text or len(str(text).strip()) < 10:
            skipped += 1
            continue

        label = map_to_bias_label(example)
        texts.append(str(text).strip())
        labels.append(label)

    print(f"       Processed {len(texts)} examples ({skipped} skipped)")

    # Print class distribution
    print("\n       Class Distribution:")
    for i, name in enumerate(LABEL_NAMES):
        count = labels.count(i)
        pct = count / len(labels) * 100
        bar = "█" * int(pct / 2)
        print(f"         {name:>13s}: {count:4d} ({pct:5.1f}%) {bar}")

    return texts, labels


def create_data_splits(texts, labels, tokenizer):
    """
    Create stratified train/validation/test splits (80/10/10).

    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoaders
        test_texts, test_labels: Raw test data for evaluation
        class_weights: Tensor of class weights for weighted loss
    """
    print("\n[3/6] Creating stratified data splits (80/10/10)...")

    # First split: 80% train, 20% temp
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    # Second split: 50/50 of temp → 10% val, 10% test
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=SEED, stratify=temp_labels
    )

    print(f"       Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

    # Create PyTorch datasets
    train_dataset = BiasDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = BiasDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    test_dataset = BiasDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Compute class weights to handle imbalance
    unique_labels = sorted(set(train_labels))
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(unique_labels),
        y=np.array(train_labels),
    )

    # Ensure we have weights for all 5 classes (some might be missing)
    full_weights = np.ones(NUM_LABELS)
    for i, label in enumerate(unique_labels):
        full_weights[label] = class_weights[i]

    print(f"       Class weights: {[f'{w:.2f}' for w in full_weights]}")

    return train_loader, val_loader, test_loader, test_texts, test_labels, torch.tensor(full_weights, dtype=torch.float32)


def evaluate_model(model, data_loader, device):
    """
    Evaluate model on a data loader and return accuracy.

    Returns:
        accuracy: float
        all_preds: list of predicted labels
        all_labels: list of true labels
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    accuracy = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
    return accuracy, all_preds, all_labels


def train():
    """
    Main training function. Downloads data, fine-tunes DistilBERT, and saves the model.
    """
    set_seed(SEED)

    # Detect device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n       Using device: {device}")
    if device.type == "cuda":
        print(f"       GPU: {torch.cuda.get_device_name(0)}")

    # Load and preprocess data
    texts, labels = load_and_preprocess_data()

    # Initialize tokenizer
    print("\n[3/6] Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    # Create data splits
    train_loader, val_loader, test_loader, test_texts, test_labels, class_weights = (
        create_data_splits(texts, labels, tokenizer)
    )

    # Initialize model
    print("\n[4/6] Initializing DistilBERT for 5-class classification...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    model.to(device)

    # Set up weighted loss function to handle class imbalance
    class_weights = class_weights.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Learning rate scheduler with linear warmup
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ─────────────────────────────────────────
    # Training Loop
    # ─────────────────────────────────────────
    print(f"\n[5/6] Training for {EPOCHS} epochs...")
    print(f"       Total steps: {total_steps} | Warmup: {warmup_steps}")
    print("─" * 60)

    best_val_accuracy = 0.0
    best_epoch = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["label"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        # Evaluate on validation set
        avg_loss = total_loss / num_batches
        val_accuracy, _, _ = evaluate_model(model, val_loader, device)

        # Save best model
        improved = ""
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            improved = " ★ NEW BEST"

            # Save the best model checkpoint
            os.makedirs(SAVE_DIR, exist_ok=True)
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)

        print(
            f"  Epoch {epoch + 1:2d}/{EPOCHS} │ "
            f"Loss: {avg_loss:.4f} │ "
            f"Val Acc: {val_accuracy:.4f}{improved}"
        )

    print("─" * 60)
    print(f"  Best validation accuracy: {best_val_accuracy:.4f} (epoch {best_epoch})")

    # ─────────────────────────────────────────
    # Final Evaluation on Test Set
    # ─────────────────────────────────────────
    print(f"\n[6/6] Evaluating best model on test set...")

    # Reload best model for final evaluation
    model = DistilBertForSequenceClassification.from_pretrained(SAVE_DIR)
    model.to(device)

    test_accuracy, test_preds, test_true = evaluate_model(model, test_loader, device)
    print(f"       Test Accuracy: {test_accuracy:.4f}")

    # Save label mapping and metadata
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

    metadata_path = os.path.join(SAVE_DIR, "newsforge_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save test data for evaluate.py
    test_data = {
        "texts": test_texts,
        "labels": test_labels,
    }
    test_data_path = os.path.join(SAVE_DIR, "test_data.json")
    with open(test_data_path, "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  MODEL SAVED TO: {SAVE_DIR}")
    print(f"  Files: config.json, model.safetensors, tokenizer files,")
    print(f"         newsforge_metadata.json, test_data.json")
    print(f"{'=' * 60}")
    print(f"\n  Next steps:")
    print(f"    1. Run 'python evaluate.py' for detailed metrics")
    print(f"    2. Start the backend: cd ../backend && uvicorn main:app --reload")
    print(f"\n  Done! ✓")


if __name__ == "__main__":
    train()
