import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from model.codex_model import CODEXSLM
from scripts.instruction_dataset import InstructionDataset


# ============================================================
# PATHS
# ============================================================

TOKENIZER_PATH = os.path.join(
    PROJECT_ROOT,
    "tokenizer",
    "codex_tokenizer_v2.json"
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "instruction_tuning",
    "train_large.jsonl"
)


# ============================================================
# CONFIGURATION
# ============================================================

device = torch.device("cpu")

BATCH_SIZE = 2
MAX_LENGTH = 512

# Only a tiny dataset for debugging
NUM_EXAMPLES = 4

EPOCHS = 10

LEARNING_RATE = 0.0001


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("CODEX-SLM OVERFITTING TEST")
print("=" * 60)

print()
print("Loading dataset...")

dataset = InstructionDataset(
    DATA_PATH,
    TOKENIZER_PATH,
    max_length=MAX_LENGTH
)

print(
    "Full dataset:",
    len(dataset)
)


# ============================================================
# SMALL SUBSET
# ============================================================

subset = Subset(
    dataset,
    range(
        min(NUM_EXAMPLES, len(dataset))
    )
)

loader = DataLoader(
    subset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(
    "Overfit examples:",
    len(subset)
)

print(
    "Epochs:",
    EPOCHS
)


# ============================================================
# CREATE MODEL
# ============================================================

model = CODEXSLM(
    vocab_size=8000,
    max_seq_len=MAX_LENGTH,
    hidden_size=256,
    num_heads=4,
    num_layers=4,
    intermediate_size=1024
)

model.to(device)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=-100
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAIN
# ============================================================

model.train()

for epoch in range(EPOCHS):

    total_loss = 0.0

    for batch in loader:

        input_ids = batch["input_ids"].to(device)

        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids
        )

        loss = criterion(
            logits.reshape(
                -1,
                logits.size(-1)
            ),
            labels.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss / len(loader)
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"Loss: {average_loss:.6f}"
    )


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 60)
print("OVERFITTING TEST COMPLETED")
print("=" * 60)

print()
print(
    "Initial loss should decrease substantially."
)

print(
    "Final loss:",
    f"{average_loss:.6f}"
)
