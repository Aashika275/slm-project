import os
import sys
import torch


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

from tokenizers import Tokenizer
from model.codex_model import CODEXSLM


# ============================================================
# CONFIGURATION
# ============================================================

VOCAB_SIZE = 8000
CONTEXT_LENGTH = 512

HIDDEN_SIZE = 256
NUM_HEADS = 4
NUM_LAYERS = 4
INTERMEDIATE_SIZE = 1024

DEVICE = torch.device("cpu")


# ============================================================
# PATHS
# ============================================================

TOKENIZER_PATH = os.path.join(
    PROJECT_ROOT,
    "tokenizer",
    "codex_tokenizer_v2.json"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "best_model.pt"
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)


# ============================================================
# CREATE MODEL
# ============================================================

print("Creating CODEX-SLM...")

model = CODEXSLM(
    vocab_size=VOCAB_SIZE,
    max_seq_len=CONTEXT_LENGTH,
    hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    intermediate_size=INTERMEDIATE_SIZE
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading best checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

print("Model ready.")


# ============================================================
# GENERATION
# ============================================================

def generate_text(
    prompt,
    max_new_tokens=100,
    temperature=0.8
):

    # --------------------------------------------------------
    # Format prompt
    # --------------------------------------------------------

    formatted_prompt = (
        "<|user|>\n"
        + prompt
        + "\n<|assistant|>\n"
    )

    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------

    encoded = tokenizer.encode(
        formatted_prompt
    )

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=DEVICE
    )

    generated_ids = input_ids.clone()

    # --------------------------------------------------------
    # Special token
    # --------------------------------------------------------

    end_token_id = tokenizer.token_to_id(
        "<|end|>"
    )

    # --------------------------------------------------------
    # Generate tokens
    # --------------------------------------------------------

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Keep context within model limit

            if (
                generated_ids.shape[1]
                > CONTEXT_LENGTH
            ):

                generated_ids = generated_ids[
                    :,
                    -CONTEXT_LENGTH:
                ]

            # Model prediction

            logits = model(
                generated_ids
            )

            # Last-token prediction

            next_token_logits = logits[
                :,
                -1,
                :
            ]

            # Temperature

            if temperature > 0:

                next_token_logits = (
                    next_token_logits
                    / temperature
                )

                probabilities = torch.softmax(
                    next_token_logits,
                    dim=-1
                )

                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1
                )

            else:

                next_token = torch.argmax(
                    next_token_logits,
                    dim=-1,
                    keepdim=True
                )

            # Add token

            generated_ids = torch.cat(
                [
                    generated_ids,
                    next_token
                ],
                dim=1
            )

            # Stop token

            if (
                end_token_id is not None
                and next_token.item()
                == end_token_id
            ):

                break

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    generated_text = tokenizer.decode(
        generated_ids[0].tolist()
    )

    # --------------------------------------------------------
    # Remove prompt section
    # --------------------------------------------------------

    assistant_marker = (
        "<|assistant|>"
    )

    if assistant_marker in generated_text:

        generated_text = (
            generated_text
            .split(
                assistant_marker,
                1
            )[1]
        )

    # Remove end token

    generated_text = (
        generated_text
        .replace(
            "<|end|>",
            ""
        )
        .strip()
    )

    return generated_text


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("CODEX-SLM TEXT GENERATION")
    print("=" * 60)

    prompt = (
        "Write a Python function that "
        "checks whether a number is prime."
    )

    print("\nPrompt:")
    print(prompt)

    print("\nGenerating...\n")

    response = generate_text(
        prompt=prompt,
        max_new_tokens=100,
        temperature=0.8
    )

    print("=" * 60)
    print("CODEX-SLM RESPONSE")
    print("=" * 60)

    print(response)

    print("\n" + "=" * 60)
    print("GENERATION COMPLETED")
    print("=" * 60)
