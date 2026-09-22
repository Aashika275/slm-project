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


# ============================================================
# IMPORT MODEL
# ============================================================

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
    "codex_slm_instruction_tuned_large.pt"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print()
print("=" * 60)
print("CODEX-SLM LARGE INSTRUCTION MODEL TEST")
print("=" * 60)

print()
print("Device:", device)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(
        f"Tokenizer not found:\n{TOKENIZER_PATH}"
    )

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Instruction model not found:\n{MODEL_PATH}"
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)


# ============================================================
# SPECIAL TOKENS
# ============================================================

END_ID = tokenizer.token_to_id(
    "<|end|>"
)

PAD_ID = tokenizer.token_to_id(
    "<|pad|>"
)

print("END ID:", END_ID)
print("PAD ID:", PAD_ID)


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


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

print()
print("Large instruction-tuned model loaded successfully.")


# ============================================================
# GENERATION
# ============================================================

def generate(
    prompt,
    max_new_tokens=80,
    temperature=0.7
):

    formatted_prompt = (
        "<|user|>\n"
        + prompt
        + "\n"
        + "<|assistant|>\n"
    )

    encoded = tokenizer.encode(
        formatted_prompt
    )

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device
    )

    generated_tokens = []

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Keep context inside model limit
            input_ids = input_ids[:, -512:]

            logits = model(input_ids)

            next_token_logits = logits[:, -1, :]

            # Temperature
            next_token_logits = (
                next_token_logits / temperature
            )

            # Never generate padding
            next_token_logits[:, PAD_ID] = -float("inf")

            probabilities = torch.softmax(
                next_token_logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            token_id = next_token.item()

            # Stop at END token
            if token_id == END_ID:
                break

            generated_tokens.append(
                token_id
            )

            input_ids = torch.cat(
                [
                    input_ids,
                    next_token
                ],
                dim=1
            )

    response = tokenizer.decode(
        generated_tokens
    )

    return response.strip()


# ============================================================
# TEST PROMPTS
# ============================================================

prompts = [

    "Write a Python function to reverse a string.",

    "Write a SQL query to find users older than 18.",

    "Explain what a REST API is.",

    "Write a JavaScript function to calculate the sum of an array.",

    "What is authentication?",

    "Write a Python function to check whether a number is prime.",

    "Explain the purpose of unit testing.",

    "Write an Express.js route that returns a JSON response.",

    "Explain database indexing.",

    "Write a Dockerfile for a Node.js application."

]


# ============================================================
# RUN TESTS
# ============================================================

print()
print("=" * 60)
print("INSTRUCTION TEST RESULTS")
print("=" * 60)


for i, prompt in enumerate(prompts, start=1):

    print()
    print(f"TEST {i}")
    print("-" * 60)

    print("Prompt:")
    print(prompt)

    print()
    print("Response:")

    response = generate(
        prompt,
        max_new_tokens=80,
        temperature=0.7
    )

    print(response)

    print()
    print("-" * 60)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("MODEL TEST COMPLETED")
print("=" * 60)
