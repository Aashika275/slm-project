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

CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "codex_slm_instruction_tuned_v4.pt"
)


# ============================================================
# MODEL CONFIG
# ============================================================

VOCAB_SIZE = 8000
MAX_SEQ_LEN = 512
HIDDEN_SIZE = 256
NUM_HEADS = 4
NUM_LAYERS = 4
INTERMEDIATE_SIZE = 1024

DEVICE = torch.device("cpu")


# ============================================================
# SPECIAL TOKENS
# ============================================================

EOS_ID = 3
END_ID = 6


# ============================================================
# GENERATION SETTINGS
# ============================================================

MAX_NEW_TOKENS = 80

TEMPERATURE = 0.8

TOP_K = 40


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)

print("Tokenizer loaded.")


# ============================================================
# CREATE MODEL
# ============================================================

print("Creating model...")

model = CODEXSLM(
    vocab_size=VOCAB_SIZE,
    max_seq_len=MAX_SEQ_LEN,
    hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    intermediate_size=INTERMEDIATE_SIZE
)


# ============================================================
# LOAD V4 CHECKPOINT
# ============================================================

print("Loading V4 instruction-tuned checkpoint...")

if not os.path.exists(CHECKPOINT_PATH):

    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT_PATH}"
    )


state_dict = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    state_dict,
    strict=True
)

model.to(DEVICE)

model.eval()

print("V4 model loaded successfully.")


# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate(prompt):

    formatted_prompt = (
        "<|user|> "
        + prompt
        + " "
        + "<|assistant|> "
    )

    encoded = tokenizer.encode(
        formatted_prompt
    )

    input_ids = encoded.ids

    generated_ids = list(input_ids)

    print()
    print("Prompt:")
    print(prompt)

    print()
    print("Generating...")

    with torch.no_grad():

        for _ in range(MAX_NEW_TOKENS):

            # Keep context within model limit
            context_ids = generated_ids[-MAX_SEQ_LEN:]

            input_tensor = torch.tensor(
                [context_ids],
                dtype=torch.long,
                device=DEVICE
            )

            logits = model(
                input_tensor
            )

            next_token_logits = logits[0, -1, :]

            # Temperature
            next_token_logits = (
                next_token_logits / TEMPERATURE
            )

            # Top-K filtering
            top_values, top_indices = torch.topk(
                next_token_logits,
                TOP_K
            )

            probabilities = torch.softmax(
                top_values,
                dim=-1
            )

            next_index = torch.multinomial(
                probabilities,
                num_samples=1
            )

            next_token = top_indices[
                next_index
            ].item()

            generated_ids.append(
                next_token
            )

            # Stop tokens
            if next_token == EOS_ID:
                break

            if next_token == END_ID:
                break

    # Remove prompt tokens
    response_ids = generated_ids[
        len(input_ids):
    ]

    response = tokenizer.decode(
        response_ids
    )

    return response.strip()


# ============================================================
# TEST PROMPTS
# ============================================================

TEST_PROMPTS = [

    "What is a Python list?",

    "Write a Python function to check whether a number is prime.",

    "How do I create a REST API using Node.js?",

    "Write a SQL query to find duplicate records.",

    "What is the difference between Git merge and Git rebase?"
]


# ============================================================
# RUN TESTS
# ============================================================

print()
print("=" * 70)
print("CODEX-SLM V4 INFERENCE TEST")
print("=" * 70)

for number, prompt in enumerate(
    TEST_PROMPTS,
    start=1
):

    print()
    print("=" * 70)
    print(f"TEST {number}")
    print("=" * 70)

    response = generate(prompt)

    print()
    print("Response:")
    print(response)

print()
print("=" * 70)
print("V4 INFERENCE TEST COMPLETED")
print("=" * 70)