import os
import sys
import torch
from tokenizers import Tokenizer

# --------------------------------------------------
# Project root
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from model.codex_model import CODEXSLM


# --------------------------------------------------
# Paths
# --------------------------------------------------

TOKENIZER_PATH = os.path.join(
    PROJECT_ROOT,
    "tokenizer",
    "codex_tokenizer_v2.json"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "codex_slm_instruction_tuned.pt"
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cpu")


# --------------------------------------------------
# Tokenizer
# --------------------------------------------------

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)


# --------------------------------------------------
# Special token IDs
# --------------------------------------------------

END_ID = tokenizer.token_to_id("<|end|>")
PAD_ID = tokenizer.token_to_id("<|pad|>")


print("END ID:", END_ID)
print("PAD ID:", PAD_ID)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = CODEXSLM(
    vocab_size=8000,
    max_seq_len=512,
    hidden_size=256,
    num_heads=4,
    num_layers=4,
    intermediate_size=1024
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

print("Instruction-tuned model loaded successfully.")


# --------------------------------------------------
# Generation
# --------------------------------------------------

def generate(
    prompt,
    max_new_tokens=60,
    temperature=0.8
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

            # Keep maximum context length
            input_ids = input_ids[:, -512:]

            logits = model(input_ids)

            next_token_logits = logits[:, -1, :]

            # Temperature
            next_token_logits = (
                next_token_logits / temperature
            )

            # Prevent PAD from being generated
            next_token_logits[:, PAD_ID] = -float("inf")

            # Probability distribution
            probabilities = torch.softmax(
                next_token_logits,
                dim=-1
            )

            # Sample
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            token_id = next_token.item()

            # Stop at END token
            if token_id == END_ID:
                break

            generated_tokens.append(token_id)

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1
            )

    # Decode ONLY newly generated tokens
    response = tokenizer.decode(
        generated_tokens
    )

    return response.strip()


# --------------------------------------------------
# Tests
# --------------------------------------------------

prompts = [
    "Write a Python function to reverse a string.",
    "Write a SQL query to find users older than 18.",
    "Explain what a REST API is.",
    "Write a JavaScript function to calculate the sum of an array.",
    "What is authentication?"
]


print()
print("=" * 60)
print("CODEX-SLM INSTRUCTION TEST")
print("=" * 60)


for i, prompt in enumerate(prompts, start=1):

    print()
    print(f"TEST {i}")
    print("Prompt:", prompt)

    response = generate(
        prompt,
        max_new_tokens=60,
        temperature=0.8
    )

    print("Response:")
    print(response)

    print("-" * 60)
