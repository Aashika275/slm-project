import os
import torch
from tokenizers import Tokenizer

from model.codex_model import CODEXSLM


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOKENIZER_PATH = os.path.join(
    BASE_DIR,
    "tokenizer",
    "codex_tokenizer_v2.json"
)

V3_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "best_model_v3.pt"
)

V4_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "codex_slm_instruction_tuned_v4.pt"
)


# ============================================================
# CONFIG
# ============================================================

VOCAB_SIZE = 8000
MAX_SEQ_LEN = 512
HIDDEN_SIZE = 256
NUM_HEADS = 4
NUM_LAYERS = 4
INTERMEDIATE_SIZE = 1024

DEVICE = torch.device("cpu")

TEST_PROMPT = "What is a Python list?"

MAX_NEW_TOKENS = 30


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("=" * 70)
print("CODEX-SLM V4 GENERATION DIAGNOSTIC")
print("=" * 70)

print()
print("Loading tokenizer...")

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)

print("Tokenizer loaded.")


# ============================================================
# TOKENIZER TEST
# ============================================================

print()
print("=" * 70)
print("TOKENIZER TEST")
print("=" * 70)

formatted_prompt = (
    "<|user|> "
    + TEST_PROMPT
    + " "
    + "<|assistant|> "
)

encoding = tokenizer.encode(formatted_prompt)

print()
print("Prompt:")
print(formatted_prompt)

print()
print("Token IDs:")
print(encoding.ids)

print()
print("Tokens:")
print(encoding.tokens)

print()
print("Decoded:")
print(tokenizer.decode(encoding.ids))


# ============================================================
# MODEL CREATION
# ============================================================

def create_model():

    model = CODEXSLM(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        intermediate_size=INTERMEDIATE_SIZE
    )

    return model.to(DEVICE)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_model(path):

    model = create_model()

    print()
    print(f"Loading:")
    print(path)

    state_dict = torch.load(
        path,
        map_location=DEVICE
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    print("Loaded successfully.")

    return model


# ============================================================
# GREEDY GENERATION
# ============================================================

def greedy_generate(model, prompt):

    formatted = (
        "<|user|> "
        + prompt
        + " "
        + "<|assistant|> "
    )

    encoded = tokenizer.encode(formatted)

    input_ids = encoded.ids

    generated_ids = list(input_ids)

    print()
    print("Prompt:")
    print(prompt)

    print()
    print("Generated token IDs:")

    with torch.no_grad():

        for step in range(MAX_NEW_TOKENS):

            context_ids = generated_ids[-MAX_SEQ_LEN:]

            input_tensor = torch.tensor(
                [context_ids],
                dtype=torch.long,
                device=DEVICE
            )

            logits = model(input_tensor)

            next_logits = logits[0, -1, :]

            # Pure greedy decoding.
            next_token = torch.argmax(
                next_logits
            ).item()

            generated_ids.append(next_token)

            token_text = tokenizer.decode(
                [next_token]
            )

            print(
                f"Step {step + 1:02d}: "
                f"ID={next_token:<5} "
                f"Token={repr(token_text)}"
            )

            # Stop at end tokens.
            if next_token in [3, 6]:
                break

    response_ids = generated_ids[len(input_ids):]

    response = tokenizer.decode(
        response_ids
    )

    print()
    print("Final generated IDs:")
    print(response_ids)

    print()
    print("Final decoded response:")
    print(repr(response))

    return response


# ============================================================
# TOP TOKEN DIAGNOSTIC
# ============================================================

def inspect_next_token(model, prompt):

    formatted = (
        "<|user|> "
        + prompt
        + " "
        + "<|assistant|> "
    )

    encoded = tokenizer.encode(formatted)

    input_tensor = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():

        logits = model(input_tensor)

        next_logits = logits[0, -1, :]

        probabilities = torch.softmax(
            next_logits,
            dim=-1
        )

        values, indices = torch.topk(
            probabilities,
            10
        )

    print()
    print("Top 10 next-token predictions:")

    for rank, (value, index) in enumerate(
        zip(values.tolist(), indices.tolist()),
        start=1
    ):

        token_text = tokenizer.decode(
            [index]
        )

        print(
            f"{rank:02d}. "
            f"ID={index:<5} "
            f"Probability={value * 100:.4f}% "
            f"Token={repr(token_text)}"
        )


# ============================================================
# LOAD V3
# ============================================================

print()
print("=" * 70)
print("V3 BASE MODEL DIAGNOSTIC")
print("=" * 70)

v3_model = load_model(V3_PATH)

inspect_next_token(
    v3_model,
    TEST_PROMPT
)

greedy_generate(
    v3_model,
    TEST_PROMPT
)


# ============================================================
# LOAD V4
# ============================================================

print()
print("=" * 70)
print("V4 INSTRUCTION-TUNED MODEL DIAGNOSTIC")
print("=" * 70)

v4_model = load_model(V4_PATH)

inspect_next_token(
    v4_model,
    TEST_PROMPT
)

greedy_generate(
    v4_model,
    TEST_PROMPT
)


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("DIAGNOSTIC COMPLETED")
print("=" * 70)