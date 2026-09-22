import os
import sys
import torch
from tokenizers import Tokenizer


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.codex_model import CODEXSLM


TOKENIZER_PATH = os.path.join(
    PROJECT_ROOT,
    "tokenizer",
    "codex_tokenizer_v2.json"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "codex_slm_instruction_tuned_v3.pt"
)

DEVICE = torch.device("cpu")


print()
print("=" * 60)
print("CODEX-SLM GENERATION DIAGNOSTIC")
print("=" * 60)


# ------------------------------------------------------------
# Load tokenizer
# ------------------------------------------------------------

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)

print()
print("Tokenizer loaded.")


# ------------------------------------------------------------
# Create model
# ------------------------------------------------------------

model = CODEXSLM(
    vocab_size=8000,
    max_seq_len=512,
    hidden_size=256,
    num_heads=4,
    num_layers=4,
    intermediate_size=1024
)

model = model.to(DEVICE)


# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(
    state_dict,
    strict=True
)

model.eval()

print("Instruction-tuned model loaded.")


# ------------------------------------------------------------
# Test prompt
# ------------------------------------------------------------

prompt = (
    "<|user|>\n"
    "What is a Python list?\n"
    "<|assistant|>\n"
)

encoding = tokenizer.encode(prompt)

input_ids = encoding.ids

print()
print("Prompt:")
print(prompt)

print()
print("Input token count:", len(input_ids))

print()
print("Input IDs:")
print(input_ids)


# ------------------------------------------------------------
# Generate first 20 tokens
# ------------------------------------------------------------

generated_ids = list(input_ids)

print()
print("=" * 60)
print("NEXT TOKEN ANALYSIS")
print("=" * 60)

with torch.no_grad():

    for step in range(20):

        current_ids = generated_ids[-512:]

        input_tensor = torch.tensor(
            [current_ids],
            dtype=torch.long,
            device=DEVICE
        )

        logits = model(
            input_tensor
        )

        next_logits = logits[0, -1, :]

        probabilities = torch.softmax(
            next_logits,
            dim=-1
        )

        top_values, top_indices = torch.topk(
            probabilities,
            10
        )

        print()
        print(f"STEP {step + 1}")

        print("Top predictions:")

        for rank in range(10):

            token_id = top_indices[rank].item()
            probability = top_values[rank].item()

            token = tokenizer.decode(
                [token_id]
            )

            print(
                f"{rank + 1:2d}. "
                f"ID={token_id:4d} "
                f"Prob={probability:.6f} "
                f"Token={repr(token)}"
            )

        next_token_id = top_indices[0].item()

        generated_ids.append(
            next_token_id
        )


# ------------------------------------------------------------
# Decode
# ------------------------------------------------------------

generated_tokens = generated_ids[
    len(input_ids):
]

text = tokenizer.decode(
    generated_tokens
)

print()
print("=" * 60)
print("GREEDY OUTPUT")
print("=" * 60)

print()
print(text)

print()
print("=" * 60)
print("DIAGNOSTIC COMPLETED")
print("=" * 60)
