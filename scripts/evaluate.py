import os
import sys
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# --------------------------------------------------
# Project root
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)


# --------------------------------------------------
# Imports
# --------------------------------------------------

from model.codex_model import CODEXSLM
from scripts.create_training_dataset import CODEXDataset


# --------------------------------------------------
# Configuration
# --------------------------------------------------

VOCAB_SIZE = 8000
CONTEXT_LENGTH = 512

HIDDEN_SIZE = 256
NUM_HEADS = 4
NUM_LAYERS = 4
INTERMEDIATE_SIZE = 1024

BATCH_SIZE = 2

DEVICE = torch.device("cpu")


# --------------------------------------------------
# Paths
# --------------------------------------------------

TOKENIZER_PATH = os.path.join(
    PROJECT_ROOT,
    "tokenizer",
    "codex_tokenizer_v2.json"
)

VALIDATION_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "cleaned",
    "validation.jsonl"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "codex_slm_epoch_1.pt"
)


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("CODEX-SLM MODEL EVALUATION")
    print("=" * 60)

    print("\nDevice:")
    print(DEVICE)

    # --------------------------------------------------
    # Validation dataset
    # --------------------------------------------------

    print("\nLoading validation dataset...")

    dataset = CODEXDataset(
        VALIDATION_DATA_PATH,
        TOKENIZER_PATH,
        CONTEXT_LENGTH
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(
        f"Validation samples: {len(dataset)}"
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print("\nCreating model...")

    model = CODEXSLM(
        vocab_size=VOCAB_SIZE,
        max_seq_len=CONTEXT_LENGTH,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        intermediate_size=INTERMEDIATE_SIZE
    )

    model = model.to(DEVICE)

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------

    print("\nLoading trained checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("Checkpoint loaded successfully.")

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    model.eval()

    loss_function = nn.CrossEntropyLoss(
        ignore_index=0
    )

    total_loss = 0.0
    total_batches = 0

    print("\nEvaluating...")

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(input_ids)

            loss = loss_function(
                logits.reshape(-1, VOCAB_SIZE),
                labels.reshape(-1)
            )

            total_loss += loss.item()
            total_batches += 1

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    validation_loss = (
        total_loss / total_batches
    )

    perplexity = math.exp(
        min(validation_loss, 20)
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"\nValidation Loss: "
        f"{validation_loss:.4f}"
    )

    print(
        f"Perplexity: "
        f"{perplexity:.4f}"
    )

    print("\nEvaluation completed successfully.")
