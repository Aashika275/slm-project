import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from model.codex_model import CODEXSLM


print("=" * 60)
print("CODEX-SLM MODEL TEST")
print("=" * 60)


# Model configuration

VOCAB_SIZE = 8000
CONTEXT_LENGTH = 512

model = CODEXSLM(
    vocab_size=VOCAB_SIZE,
    max_seq_len=CONTEXT_LENGTH,
    hidden_size=256,
    num_heads=4,
    num_layers=4,
    intermediate_size=1024
)


print("\nModel created successfully.")


# Create fake token IDs

batch_size = 2
sequence_length = 32

input_ids = torch.randint(
    0,
    VOCAB_SIZE,
    (
        batch_size,
        sequence_length
    )
)


print("\nInput shape:")
print(input_ids.shape)


# Forward pass

logits = model(
    input_ids
)


print("\nOutput shape:")
print(logits.shape)


# Parameter count

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)


print("\nTotal parameters:")
print(f"{parameter_count:,}")


print("\nExpected output shape:")
print(
    f"({batch_size}, "
    f"{sequence_length}, "
    f"{VOCAB_SIZE})"
)


print("\nModel test completed successfully.")
