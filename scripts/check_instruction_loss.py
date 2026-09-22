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

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "best_model_v2.pt"
)

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print()
print("=" * 60)
print("CODEX-SLM INSTRUCTION LOSS DIAGNOSTIC")
print("=" * 60)

print()
print("Device:", device)

# ============================================================
# DATASET
# ============================================================

dataset = InstructionDataset(
    DATA_PATH,
    TOKENIZER_PATH,
    max_length=512
)

print()
print("Dataset size:", len(dataset))

# ============================================================
# DATALOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False
)

batch = next(iter(loader))

input_ids = batch["input_ids"].to(device)
labels = batch["labels"].to(device)

print()
print("Input shape:", input_ids.shape)
print("Label shape:", labels.shape)

# ============================================================
# VALID LABEL COUNT
# ============================================================

valid_count = (labels != -100).sum().item()

print()
print("Valid label tokens in batch:", valid_count)

if valid_count == 0:
    raise RuntimeError(
        "ERROR: No valid labels found."
    )

# ============================================================
# CREATE MODEL
# ============================================================

print()
print("Creating CODEX-SLM...")

model = CODEXSLM(
    vocab_size=8000,
    max_seq_len=512,
    hidden_size=256,
    num_heads=4,
    num_layers=4,
    intermediate_size=1024
)

model = model.to(device)

# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading base checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)

print("Checkpoint loaded successfully.")

# ============================================================
# EVALUATION MODE
# ============================================================

model.eval()

criterion = nn.CrossEntropyLoss(
    ignore_index=-100
)

# ============================================================
# FORWARD PASS
# ============================================================

print()
print("Running forward pass...")

with torch.no_grad():

    logits = model(input_ids)

    loss = criterion(
        logits.reshape(
            -1,
            logits.size(-1)
        ),
        labels.reshape(-1)
    )

# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("RESULT")
print("=" * 60)

print()
print(
    "Raw loss:",
    repr(loss.item())
)

print(
    "Loss with 10 decimals:",
    f"{loss.item():.10f}"
)

print(
    "Loss with 6 decimals:",
    f"{loss.item():.6f}"
)

print(
    "Loss with 4 decimals:",
    f"{loss.item():.4f}"
)

# ============================================================
# PREDICTION CHECK
# ============================================================

predictions = torch.argmax(
    logits,
    dim=-1
)

valid_mask = labels != -100

valid_predictions = predictions[
    valid_mask
]

valid_labels = labels[
    valid_mask
]

correct = (
    valid_predictions == valid_labels
).sum().item()

total = valid_labels.numel()

accuracy = (
    correct / total
    if total > 0
    else 0.0
)

print()
print("Correct predictions:", correct)
print("Total predictions:", total)
print(
    "Accuracy:",
    f"{accuracy * 100:.4f}%"
)

# ============================================================
# FIRST VALID TOKENS
# ============================================================

print()
print("First valid labels:")
print(
    valid_labels[:30].tolist()
)

print()
print("First valid predictions:")
print(
    valid_predictions[:30].tolist()
)

# ============================================================
# FINAL
# ============================================================

print()
print("=" * 60)
print("DIAGNOSTIC COMPLETED")
print("=" * 60)
