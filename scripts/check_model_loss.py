import os
import sys
import torch
import torch.nn as nn

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
# SETTINGS
# ============================================================

device = torch.device("cpu")

MAX_LENGTH = 512


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("MODEL LOSS DIAGNOSTIC")
print("=" * 60)

dataset = InstructionDataset(
    DATA_PATH,
    TOKENIZER_PATH,
    max_length=MAX_LENGTH
)

sample = dataset[0]

input_ids = sample["input_ids"].unsqueeze(0)
labels = sample["labels"].unsqueeze(0)

print()
print("Input shape:", input_ids.shape)
print("Labels shape:", labels.shape)

valid_count = (labels != -100).sum().item()

print()
print("Valid labels:", valid_count)

print(
    "Ignored labels:",
    (labels == -100).sum().item()
)


# ============================================================
# CREATE MODEL
# ============================================================

model = CODEXSLM(
    vocab_size=8000,
    max_seq_len=512,
    hidden_size=256,
    num_heads=4,
    num_layers=4,
    intermediate_size=1024
)

model.to(device)
model.train()


# ============================================================
# FORWARD PASS
# ============================================================

print()
print("Running forward pass...")

logits = model(input_ids)

print()
print("Logits shape:")
print(logits.shape)

print()
print("Logits minimum:")
print(logits.min().item())

print()
print("Logits maximum:")
print(logits.max().item())

print()
print("Logits mean:")
print(logits.mean().item())

print()
print("Logits standard deviation:")
print(logits.std().item())


# ============================================================
# CHECK FOR NaN / INF
# ============================================================

print()
print("NaN in logits:", torch.isnan(logits).any().item())
print("INF in logits:", torch.isinf(logits).any().item())


# ============================================================
# MANUAL LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=-100
)

loss = criterion(
    logits.reshape(
        -1,
        logits.size(-1)
    ),
    labels.reshape(-1)
)

print()
print("=" * 60)
print("LOSS RESULT")
print("=" * 60)

print()
print("Loss:", loss.item())

print(
    "Loss is zero:",
    loss.item() == 0.0
)


# ============================================================
# CHECK VALID TARGETS
# ============================================================

valid_labels = labels[
    labels != -100
]

print()
print("First valid labels:")
print(
    valid_labels[:20].tolist()
)


# ============================================================
# CHECK PREDICTIONS
# ============================================================

predictions = torch.argmax(
    logits,
    dim=-1
)

valid_predictions = predictions[
    labels != -100
]

print()
print("First valid predictions:")
print(
    valid_predictions[:20].tolist()
)


# ============================================================
# ACCURACY
# ============================================================

correct = (
    valid_predictions == valid_labels
).sum().item()

total = valid_labels.numel()

accuracy = correct / total

print()
print("Correct predictions:", correct)
print("Total predictions:", total)
print(
    "Accuracy:",
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("DIAGNOSTIC COMPLETED")
print("=" * 60)
