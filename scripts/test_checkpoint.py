import os
import sys
import torch


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)


from model.codex_model import CODEXSLM
from scripts.checkpoint_utils import (
    save_checkpoint,
    load_checkpoint
)


DEVICE = torch.device("cpu")


CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "checkpoint_test.pt"
)


print("=" * 60)
print("CODEX-SLM CHECKPOINT TEST")
print("=" * 60)


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = CODEXSLM(
    vocab_size=8000,
    max_seq_len=512,
    hidden_size=256,
    num_heads=4,
    num_layers=4,
    intermediate_size=1024
)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)


# --------------------------------------------------
# Save
# --------------------------------------------------

save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=1,
    step=100,
    loss=2.5,
    path=CHECKPOINT_PATH
)


# --------------------------------------------------
# Create another model
# --------------------------------------------------

new_model = CODEXSLM(
    vocab_size=8000,
    max_seq_len=512,
    hidden_size=256,
    num_heads=4,
    num_layers=4,
    intermediate_size=1024
)


new_optimizer = torch.optim.AdamW(
    new_model.parameters(),
    lr=3e-4
)


# --------------------------------------------------
# Load
# --------------------------------------------------

epoch, step, loss = load_checkpoint(
    model=new_model,
    optimizer=new_optimizer,
    path=CHECKPOINT_PATH,
    device=DEVICE
)


# --------------------------------------------------
# Verify
# --------------------------------------------------

print("\nLoaded values:")

print("Epoch:", epoch)
print("Step:", step)
print("Loss:", loss)


if epoch == 1 and step == 100:

    print(
        "\nCheckpoint test passed successfully."
    )

else:

    print(
        "\nCheckpoint test failed."
    )
