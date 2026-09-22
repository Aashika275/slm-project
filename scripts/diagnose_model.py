import os
import sys
import torch
from tokenizers import Tokenizer

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from model.codex_model import CODEXSLM


# ============================================================
# PATHS
# ============================================================

TOKENIZER_PATH = os.path.join(
    PROJECT_ROOT,
    "tokenizer",
    "codex_tokenizer_v2.json"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "best_model.pt"
)

INSTRUCTION_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "codex_slm_instruction_tuned_large.pt"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)


# ============================================================
# TOKEN INFORMATION
# ============================================================

print("=" * 60)
print("TOKENIZER DIAGNOSTIC")
print("=" * 60)

print()
print("PAD:", tokenizer.token_to_id("<|pad|>"))
print("UNK:", tokenizer.token_to_id("<|unk|>"))
print("BOS:", tokenizer.token_to_id("<|bos|>"))
print("EOS:", tokenizer.token_to_id("<|eos|>"))
print("USER:", tokenizer.token_to_id("<|user|>"))
print("ASSISTANT:", tokenizer.token_to_id("<|assistant|>"))
print("END:", tokenizer.token_to_id("<|end|>"))

print()
print("Vocabulary size:", tokenizer.get_vocab_size())


# ============================================================
# CREATE MODEL FUNCTION
# ============================================================

def create_model():

    model = CODEXSLM(
        vocab_size=8000,
        max_seq_len=512,
        hidden_size=256,
        num_heads=4,
        num_layers=4,
        intermediate_size=1024
    )

    return model


# ============================================================
# CHECK CHECKPOINT
# ============================================================

def inspect_checkpoint(path, name):

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print()
    print("Path:")
    print(path)

    if not os.path.exists(path):

        print("FILE DOES NOT EXIST")

        return

    size = os.path.getsize(path)

    print()
    print(
        f"File size: {size / (1024 * 1024):.2f} MB"
    )

    checkpoint = torch.load(
        path,
        map_location=device
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

    print()
    print(
        "Number of tensors:",
        len(state_dict)
    )

    total_parameters = 0

    total_nonzero = 0

    for name, tensor in state_dict.items():

        if torch.is_tensor(tensor):

            total_parameters += tensor.numel()

            total_nonzero += torch.count_nonzero(
                tensor
            ).item()

    print(
        "Total parameters:",
        total_parameters
    )

    print(
        "Non-zero parameters:",
        total_nonzero
    )

    print(
        "Non-zero percentage:",
        f"{100 * total_nonzero / total_parameters:.2f}%"
    )

    # Show several parameter statistics
    print()
    print("Parameter statistics:")

    count = 0

    for name, tensor in state_dict.items():

        if torch.is_tensor(tensor):

            print(
                name,
                "| shape:",
                tuple(tensor.shape),
                "| mean:",
                f"{tensor.float().mean().item():.6f}",
                "| std:",
                f"{tensor.float().std().item():.6f}"
            )

            count += 1

            if count >= 5:
                break


# ============================================================
# INSPECT BOTH MODELS
# ============================================================

inspect_checkpoint(
    MODEL_PATH,
    "BASE MODEL"
)

inspect_checkpoint(
    INSTRUCTION_MODEL_PATH,
    "INSTRUCTION-TUNED MODEL"
)


# ============================================================
# TEST TOKENIZATION
# ============================================================

print()
print("=" * 60)
print("TOKENIZATION TEST")
print("=" * 60)

test_text = (
    "<|user|>\n"
    "Write a Python function to reverse a string.\n"
    "<|assistant|>\n"
)

encoded = tokenizer.encode(
    test_text
)

print()
print("Text:")
print(test_text)

print()
print("Token count:")
print(len(encoded.ids))

print()
print("Token IDs:")
print(encoded.ids[:30])

print()
print("Decoded:")
print(tokenizer.decode(encoded.ids))


# ============================================================
# LOAD BASE MODEL
# ============================================================

print()
print("=" * 60)
print("MODEL FORWARD TEST")
print("=" * 60)

model = create_model()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model.to(device)
model.eval()


# ============================================================
# FORWARD PASS
# ============================================================

input_ids = torch.tensor(
    [encoded.ids],
    dtype=torch.long
)

with torch.no_grad():

    logits = model(
        input_ids
    )

print()
print("Input shape:")
print(input_ids.shape)

print()
print("Logits shape:")
print(logits.shape)

print()
print("Logits mean:")
print(logits.mean().item())

print()
print("Logits std:")
print(logits.std().item())

print()
print("Forward pass completed successfully.")


# ============================================================
# TOP PREDICTION
# ============================================================

next_logits = logits[:, -1, :]

top_values, top_ids = torch.topk(
    next_logits,
    k=10,
    dim=-1
)

print()
print("Top 10 predicted token IDs:")

for token_id, value in zip(
    top_ids[0],
    top_values[0]
):

    token_id = token_id.item()

    try:
        token_text = tokenizer.decode(
            [token_id]
        )
    except Exception:
        token_text = "<decode error>"

    print(
        token_id,
        "|",
        repr(token_text),
        "| score:",
        f"{value.item():.4f}"
    )


print()
print("=" * 60)
print("DIAGNOSTIC COMPLETED")
print("=" * 60)
