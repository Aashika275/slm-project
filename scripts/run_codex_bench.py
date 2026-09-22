import os
import sys
import json
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
from scripts.config_loader import load_config


# ============================================================
# CONFIGURATION
# ============================================================

config = load_config()

model_config = config["model"]

VOCAB_SIZE = model_config["vocab_size"]
CONTEXT_LENGTH = model_config["context_length"]

HIDDEN_SIZE = model_config["hidden_size"]
NUM_HEADS = model_config["num_heads"]
NUM_LAYERS = model_config["num_layers"]
INTERMEDIATE_SIZE = model_config["intermediate_size"]

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

BENCHMARK_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "evaluation",
    "codex_bench.json"
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

print("Creating model...")

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

def generate(
    prompt,
    max_new_tokens=80
):

    formatted_prompt = (
        "<|user|>\n"
        + prompt
        + "\n<|assistant|>\n"
    )

    encoded = tokenizer.encode(
        formatted_prompt
    )

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=DEVICE
    )

    generated_ids = input_ids.clone()

    end_token_id = tokenizer.token_to_id(
        "<|end|>"
    )

    with torch.no_grad():

        for _ in range(max_new_tokens):

            if (
                generated_ids.shape[1]
                > CONTEXT_LENGTH
            ):

                generated_ids = generated_ids[
                    :,
                    -CONTEXT_LENGTH:
                ]

            logits = model(
                generated_ids
            )

            next_token_logits = logits[
                :,
                -1,
                :
            ]

            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True
            )

            generated_ids = torch.cat(
                [
                    generated_ids,
                    next_token
                ],
                dim=1
            )

            if (
                end_token_id is not None
                and next_token.item()
                == end_token_id
            ):

                break

    text = tokenizer.decode(
        generated_ids[0].tolist()
    )

    if "<|assistant|>" in text:

        text = text.split(
            "<|assistant|>",
            1
        )[1]

    text = text.replace(
        "<|end|>",
        ""
    )

    return text.strip()


# ============================================================
# LOAD BENCHMARK
# ============================================================

print("\nLoading CODEX-Bench...")

with open(
    BENCHMARK_PATH,
    "r",
    encoding="utf-8"
) as file:

    benchmark = json.load(file)


print(
    f"Benchmark tasks: {len(benchmark)}"
)


# ============================================================
# RUN BENCHMARK
# ============================================================

results = []

total_score = 0

print("\n" + "=" * 60)
print("CODEX-BENCH")
print("=" * 60)


for task in benchmark:

    task_id = task["id"]
    category = task["category"]
    prompt = task["prompt"]

    expected_keywords = [
        keyword.lower()
        for keyword in task[
            "expected_keywords"
        ]
    ]

    print("\n" + "-" * 60)

    print(
        f"Task: {task_id}"
    )

    print(
        f"Category: {category}"
    )

    print(
        f"Prompt: {prompt}"
    )

    response = generate(
        prompt
    )

    response_lower = response.lower()

    matched = 0

    for keyword in expected_keywords:

        if keyword in response_lower:

            matched += 1

    if len(expected_keywords) > 0:

        score = (
            matched
            / len(expected_keywords)
        )

    else:

        score = 0.0

    total_score += score

    print(
        f"\nResponse:\n{response}"
    )

    print(
        f"\nKeyword score: "
        f"{score * 100:.2f}%"
    )

    results.append(
        {
            "id": task_id,
            "category": category,
            "prompt": prompt,
            "response": response,
            "score": score
        }
    )


# ============================================================
# FINAL SCORE
# ============================================================

overall_score = (
    total_score
    / len(benchmark)
) * 100


print("\n" + "=" * 60)
print("CODEX-BENCH RESULTS")
print("=" * 60)

print(
    f"\nOverall keyword score: "
    f"{overall_score:.2f}%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

RESULT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "evaluation",
    "codex_bench_results.json"
)

with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "overall_score":
                overall_score,
            "results":
                results
        },
        file,
        indent=2
    )


print(
    f"\nResults saved to:\n"
    f"{RESULT_PATH}"
)

print("\nCODEX-Bench completed.")
