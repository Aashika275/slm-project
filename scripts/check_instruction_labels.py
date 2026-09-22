import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

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
# LOAD DATASET
# ============================================================

dataset = InstructionDataset(
    DATA_PATH,
    TOKENIZER_PATH,
    max_length=512
)


# ============================================================
# CHECK FIRST 5 EXAMPLES
# ============================================================

print()
print("=" * 60)
print("INSTRUCTION LABEL DIAGNOSTIC")
print("=" * 60)

for index in range(5):

    sample = dataset[index]

    input_ids = sample["input_ids"]
    labels = sample["labels"]

    valid_labels = labels[
        labels != -100
    ]

    print()
    print("Example:", index + 1)

    print(
        "Input length:",
        len(input_ids)
    )

    print(
        "Valid label tokens:",
        len(valid_labels)
    )

    print(
        "Ignored label tokens:",
        int((labels == -100).sum())
    )

    print(
        "Has valid labels:",
        len(valid_labels) > 0
    )

    print()
    print("First 30 labels:")

    print(
        labels[:30].tolist()
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("DIAGNOSTIC COMPLETED")
print("=" * 60)
