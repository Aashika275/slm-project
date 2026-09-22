import os
import sys
import math
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

TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "cleaned",
    "train.jsonl"
)

VALIDATION_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "cleaned",
    "validation.jsonl"
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "checkpoints"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

BASE_CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "codex_slm_base_v3.pt"
)

BEST_CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model_v3.pt"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# MODEL CONFIGURATION
# ============================================================

VOCAB_SIZE = 8000
MAX_SEQ_LEN = 512
HIDDEN_SIZE = 256
NUM_HEADS = 4
NUM_LAYERS = 4
INTERMEDIATE_SIZE = 1024


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 2

LEARNING_RATE = 0.0003

EPOCHS = 1

WEIGHT_DECAY = 0.01

GRADIENT_CLIP = 1.0

NUM_WORKERS = 0


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 60)
print("CODEX-SLM V3 BASE MODEL TRAINING")
print("=" * 60)

print()
print("Training from scratch.")
print("No pretrained model is being used.")

print()
print("Device:")
print(DEVICE)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

print()
print("=" * 60)
print("CHECKING REQUIRED FILES")
print("=" * 60)

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(
        f"Tokenizer not found:\n{TOKENIZER_PATH}"
    )

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(
        f"Training dataset not found:\n{TRAIN_FILE}"
    )

if not os.path.exists(VALIDATION_FILE):
    raise FileNotFoundError(
        f"Validation dataset not found:\n{VALIDATION_FILE}"
    )

print()
print("Tokenizer V2: OK")
print("Training dataset: OK")
print("Validation dataset: OK")

print()
print("Tokenizer:")
print(TOKENIZER_PATH)

print()
print("Training file:")
print(TRAIN_FILE)

print()
print("Validation file:")
print(VALIDATION_FILE)


# ============================================================
# LOAD TRAINING DATASET
# ============================================================

print()
print("=" * 60)
print("LOADING TRAINING DATASET")
print("=" * 60)

train_dataset = CODEXDataset(
    TRAIN_FILE,
    TOKENIZER_PATH,
    max_length=MAX_SEQ_LEN
)

print()
print("Training examples:")
print(len(train_dataset))


# ============================================================
# LOAD VALIDATION DATASET
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

print()
print("Validation examples:")
print(len(validation_dataset))


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print()
print("Training batches:")
print(len(train_loader))

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


# ============================================================
# PARAMETER COUNT
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print()
print("Model created successfully.")

print()
print("Total parameters:")
print(f"{total_parameters:,}")

print()
print("Trainable parameters:")
print(f"{trainable_parameters:,}")


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=-100
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

print()
print("=" * 60)
print("TRAINING CONFIGURATION")
print("=" * 60)

print()
print("Tokenizer:")
print("codex_tokenizer_v2.json")

print()
print("Vocabulary size:")
print(VOCAB_SIZE)

print()
print("Context length:")
print(MAX_SEQ_LEN)

print()
print("Hidden size:")
print(HIDDEN_SIZE)

print()
print("Attention heads:")
print(NUM_HEADS)

print()
print("Transformer layers:")
print(NUM_LAYERS)

print()
print("Intermediate size:")
print(INTERMEDIATE_SIZE)

print()
print("Batch size:")
print(BATCH_SIZE)

print()
print("Learning rate:")
print(LEARNING_RATE)

print()
print("Epochs:")
print(EPOCHS)

print()
print("Weight decay:")
print(WEIGHT_DECAY)

print()
print("Gradient clipping:")
print(GRADIENT_CLIP)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    epoch
):

    model.train()

    total_loss = 0.0

    total_valid_tokens = 0

    total_correct = 0

    total_predictions = 0

    print()
    print("=" * 60)

    print(
        f"STARTING TRAINING EPOCH "
        f"{epoch}/{EPOCHS}"
    )

    print("=" * 60)

    for batch_index, batch in enumerate(loader):

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        labels = batch[
            "labels"
        ].to(DEVICE)

        optimizer.zero_grad()

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

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP
        )

        optimizer.step()

        loss_value = loss.item()

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

            total_predictions += valid_tokens

            total_loss += (
                loss_value
                *
                valid_tokens
            )

            total_valid_tokens += (
                valid_tokens
            )

        if (
            (batch_index + 1) % 100 == 0
            or
            (batch_index + 1) == len(loader)
        ):

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

            print(
                f"Epoch "
                f"{epoch}/{EPOCHS} | "
                f"Batch "
                f"{batch_index + 1}/"
                f"{len(loader)} | "
                f"Loss: "
                f"{loss_value:.6f} | "
                f"Avg Loss: "
                f"{average_loss:.6f} | "
                f"Accuracy: "
                f"{accuracy:.2f}%"
            )

    if total_valid_tokens > 0:

        epoch_loss = (
            total_loss
            /
            total_valid_tokens
        )

    else:

        epoch_loss = 0.0

    if total_predictions > 0:

        epoch_accuracy = (
            total_correct
            /
            total_predictions
        ) * 100.0

    else:

        epoch_accuracy = 0.0

    print()
    print(
        f"Epoch {epoch} completed."
    )

    print(
        f"Training Loss: "
        f"{epoch_loss:.6f}"
    )

    print(
        f"Training Token Accuracy: "
        f"{epoch_accuracy:.4f}%"
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    total_valid_tokens = 0

    total_correct = 0

    total_predictions = 0

    with torch.no_grad():

        for batch in loader:

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

    return (
        average_loss,
        accuracy,
        total_correct,
        total_predictions
    )


# ============================================================
# TRAINING LOOP
# ============================================================

best_validation_loss = float(
    "inf"
)

print()
print("=" * 60)
print("STARTING CODEX-SLM V3 TRAINING")
print("=" * 60)

for epoch in range(
    1,
    EPOCHS + 1
):

    train_loss, train_accuracy = (
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            epoch
        )
    )

    print()
    print("=" * 60)
    print(
        f"VALIDATING AFTER EPOCH {epoch}"
    )
    print("=" * 60)

    (
        validation_loss,
        validation_accuracy,
        validation_correct,
        validation_tokens
    ) = evaluate(
        model,
        validation_loader,
        criterion
    )

    print()
    print(
        f"Validation Loss: "
        f"{validation_loss:.6f}"
    )

    print(
        f"Validation Token Accuracy: "
        f"{validation_accuracy:.4f}%"
    )

    print(
        f"Validation Correct Tokens: "
        f"{validation_correct}"
    )

    print(
        f"Validation Total Tokens: "
        f"{validation_tokens}"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if validation_loss < best_validation_loss:

        best_validation_loss = (
            validation_loss
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "model_config": {
                    "vocab_size":
                        VOCAB_SIZE,

                    "max_seq_len":
                        MAX_SEQ_LEN,

                    "hidden_size":
                        HIDDEN_SIZE,

                    "num_heads":
                        NUM_HEADS,

                    "num_layers":
                        NUM_LAYERS,

                    "intermediate_size":
                        INTERMEDIATE_SIZE
                },

                "tokenizer":
                    "codex_tokenizer_v2.json",

                "training_type":
                    "from_scratch",

                "epoch":
                    epoch,

                "validation_loss":
                    validation_loss,

                "validation_accuracy":
                    validation_accuracy,

                "total_parameters":
                    total_parameters
            },
            BEST_CHECKPOINT_PATH
        )

        print()
        print(
            "New best model saved:"
        )

        print(
            BEST_CHECKPOINT_PATH
        )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print()
print("=" * 60)
print("SAVING FINAL V3 MODEL")
print("=" * 60)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "model_config": {
            "vocab_size":
                VOCAB_SIZE,

            "max_seq_len":
                MAX_SEQ_LEN,

            "hidden_size":
                HIDDEN_SIZE,

            "num_heads":
                NUM_HEADS,

            "num_layers":
                NUM_LAYERS,

            "intermediate_size":
                INTERMEDIATE_SIZE
        },

        "tokenizer":
            "codex_tokenizer_v2.json",

        "training_type":
            "from_scratch",

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "batch_size":
            BATCH_SIZE,

        "best_validation_loss":
            best_validation_loss,

        "total_parameters":
            total_parameters
    },
    BASE_CHECKPOINT_PATH
)


# ============================================================
# CHECKPOINT VERIFICATION
# ============================================================

if not os.path.exists(
    BASE_CHECKPOINT_PATH
):

    raise RuntimeError(
        "Final V3 checkpoint was not created."
    )

if not os.path.exists(
    BEST_CHECKPOINT_PATH
):

    raise RuntimeError(
        "Best V3 checkpoint was not created."
    )


base_size = os.path.getsize(
    BASE_CHECKPOINT_PATH
)

best_size = os.path.getsize(
    BEST_CHECKPOINT_PATH
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("CODEX-SLM V3 TRAINING COMPLETED")
print("=" * 60)

print()
print("Tokenizer:")
print("codex_tokenizer_v2.json")

print()
print("Training examples:")
print(len(train_dataset))

print()
print("Validation examples:")
print(len(validation_dataset))

print()
print("Total parameters:")
print(f"{total_parameters:,}")

print()
print("Best validation loss:")
print(
    f"{best_validation_loss:.6f}"
)

print()
print("Final checkpoint:")
print(BASE_CHECKPOINT_PATH)

print(
    f"Size: "
    f"{base_size / (1024 * 1024):.2f} MB"
)

print()
print("Best checkpoint:")
print(BEST_CHECKPOINT_PATH)

print(
    f"Size: "
    f"{best_size / (1024 * 1024):.2f} MB"
)

print()
print("Old V2 checkpoint was NOT overwritten:")
print(
    "checkpoints/best_model_v2.pt"
)

print()
print("=" * 60)
print("READY FOR V3 EVALUATION")
print("=" * 60)