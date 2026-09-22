import os
import sys
import json
import random


# --------------------------------------------------
# Project root
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

SOURCE_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "cleaned",
    "train.jsonl"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "instruction_tuning",
    "train_large.jsonl"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MAX_EXAMPLES = 9000

random.seed(42)


# --------------------------------------------------
# Check source
# --------------------------------------------------

if not os.path.exists(SOURCE_PATH):
    raise FileNotFoundError(
        f"Source dataset not found: {SOURCE_PATH}"
    )


# --------------------------------------------------
# Read dataset
# --------------------------------------------------

records = []

with open(
    SOURCE_PATH,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        record = json.loads(line)

        if (
            "instruction" in record
            and "response" in record
        ):
            records.append({
                "instruction": record["instruction"],
                "response": record["response"]
            })


# --------------------------------------------------
# Shuffle
# --------------------------------------------------

random.shuffle(records)


# --------------------------------------------------
# Limit examples
# --------------------------------------------------

records = records[:MAX_EXAMPLES]


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)


# --------------------------------------------------
# Save instruction dataset
# --------------------------------------------------

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    for record in records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()
print("=" * 60)
print("INSTRUCTION DATASET CREATED")
print("=" * 60)

print("Source:", SOURCE_PATH)
print("Output:", OUTPUT_PATH)
print("Examples:", len(records))

print()
print("First example:")
print(records[0]["instruction"])

print()
print("Dataset preparation completed successfully.")
