import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

from model.codex_model import CODEXSLM
from scripts.create_training_dataset import CODEXDataset


# ============================================================
# PATHS
# ============================================================

TOKENIZER_PATH = os.path.join(
    PROJECT_ROOT,
    "tokenizer",
    "codex_tokenizer_v2.json"
)

VALIDATION_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "cleaned",
    "validation.jsonl"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "best_model_v3.pt"
)


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cpu")

VOCAB_SIZE = 8000
MAX_SEQ_LEN = 512
HIDDEN_SIZE = 256
NUM_HEADS = 4
NUM_LAYERS = 4
INTERMEDIATE_SIZE = 1024

BATCH_SIZE = 2


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 60)
print("CODEX-SLM V3 BASE MODEL EVALUATION")
print("=" * 60)

print()
print("Device:")
print(DEVICE)


# ============================================================
# CHECK FILES
# ============================================================

print()
print("=" * 60)
print("CHECKING FILES")
print("=" * 60)

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(
        f"Tokenizer not found:\n{TOKENIZER_PATH}"
    )

if not os.path.exists(VALIDATION_FILE):
    raise FileNotFoundError(
        f"Validation dataset not found:\n{VALIDATION_FILE}"
    )

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"V3 checkpoint not found:\n{CHECKPOINT_PATH}"
    )

print()
print("Tokenizer V2: OK")
print("Validation dataset: OK")
print("V3 checkpoint: OK")

print()
print("Tokenizer:")
print(TOKENIZER_PATH)

print()
print("Checkpoint:")
print(CHECKPOINT_PATH)


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("=" * 60)
print("LOADING VALIDATION DATASET")
print("=" * 60)

validation_dataset = CODEXDataset(
    VALIDATION_FILE,
    TOKENIZER_PATH,
    max_length=MAX_SEQ_LEN
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print()
print("Validation samples:")
print(len(validation_dataset))

print()
print("Validation batches:")
print(len(validation_loader))


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("=" * 60)
print("CREATING CODEX-SLM V3")
print("=" * 60)

model = CODEXSLM(
    vocab_size=VOCAB_SIZE,
    max_seq_len=MAX_SEQ_LEN,
    hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    intermediate_size=INTERMEDIATE_SIZE
)

model = model.to(DEVICE)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print()
print("Model created successfully.")

print()
print("Total parameters:")
print(f"{total_parameters:,}")


# ============================================================
# LOAD V3 CHECKPOINT
# ============================================================

print()
print("=" * 60)
print("LOADING V3 CHECKPOINT")
print("=" * 60)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    state_dict = checkpoint[
        "model_state_dict"
    ]

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict,
    strict=True
)

model.eval()

print()
print("V3 checkpoint loaded successfully.")

print()
print("Checkpoint:")
print(CHECKPOINT_PATH)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=-100
)


# ============================================================
# EVALUATION
# ============================================================

print()
print("=" * 60)
print("EVALUATING V3 MODEL")
print("=" * 60)

total_loss = 0.0

total_valid_tokens = 0

total_correct = 0

total_predictions = 0


with torch.no_grad():

    for batch_index, batch in enumerate(
        validation_loader
    ):

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        labels = batch[
            "labels"
        ].to(DEVICE)

        logits = model(
            input_ids
        )

        loss = criterion(
            logits.reshape(
                -1,
                VOCAB_SIZE
            ),
            labels.reshape(-1)
        )

        valid_mask = (
            labels != -100
        )

        valid_tokens = (
            valid_mask.sum().item()
        )

        if valid_tokens > 0:

            predictions = (
                logits.argmax(
                    dim=-1
                )
            )

            correct = (
                (
                    predictions[valid_mask]
                    ==
                    labels[valid_mask]
                )
                .sum()
                .item()
            )

            total_correct += correct

            total_predictions += (
                valid_tokens
            )

            total_loss += (
                loss.item()
                *
                valid_tokens
            )

            total_valid_tokens += (
                valid_tokens
            )

        if (
            (batch_index + 1) % 100 == 0
            or
            (batch_index + 1)
            == len(validation_loader)
        ):

            if total_valid_tokens > 0:

                running_loss = (
                    total_loss
                    /
                    total_valid_tokens
                )

            else:

                running_loss = 0.0

            if total_predictions > 0:

                running_accuracy = (
                    total_correct
                    /
                    total_predictions
                ) * 100.0

            else:

                running_accuracy = 0.0

            print(
                f"Batch "
                f"{batch_index + 1}/"
                f"{len(validation_loader)} | "
                f"Loss: "
                f"{running_loss:.6f} | "
                f"Accuracy: "
                f"{running_accuracy:.4f}%"
            )


# ============================================================
# FINAL RESULTS
# ============================================================

if total_valid_tokens > 0:

    average_loss = (
        total_loss
        /
        total_valid_tokens
    )

else:

    average_loss = 0.0


if total_predictions > 0:

    accuracy = (
        total_correct
        /
        total_predictions
    ) * 100.0

else:

    accuracy = 0.0


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("V3 EVALUATION RESULTS")
print("=" * 60)

print()
print("Validation samples:")
print(len(validation_dataset))

print()
print("Valid tokens:")
print(total_valid_tokens)

print()
print("Correct predictions:")
print(total_correct)

print()
print("Average loss:")
print(f"{average_loss:.6f}")

print()
print("Token accuracy:")
print(f"{accuracy:.4f}%")

print()
print("Checkpoint:")
print("best_model_v3.pt")

print()
print("Tokenizer:")
print("codex_tokenizer_v2.json")

print()
print("=" * 60)
print("V3 EVALUATION COMPLETED")
print("=" * 60)