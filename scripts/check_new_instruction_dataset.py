import os
import sys
import json
import torch
import torch.nn as nn
from tokenizers import Tokenizer


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
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

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "instruction_tuning",
    "codex_instruction_train.jsonl"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "codex_slm_instruction_tuned_v3.pt"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 60)
print("CODEX-SLM NEW INSTRUCTION DATASET DIAGNOSTIC")
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

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Instruction dataset not found:\n{DATA_PATH}"
    )

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT_PATH}"
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

print()
print("Loading tokenizer...")

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)

print("Tokenizer loaded.")


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("Loading instruction dataset...")

examples = []

with open(
    DATA_PATH,
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
            examples.append(record)


print(
    "Loaded examples:",
    len(examples)
)


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("Creating CODEX-SLM...")

model = CODEXSLM(
    vocab_size=8000,
    max_seq_len=512,
    hidden_size=256,
    num_heads=4,
    num_layers=4,
    intermediate_size=1024
)

model = model.to(device)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print()
print("Loading best_model_v2.pt...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
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

model.load_state_dict(
    state_dict,
    strict=True
)

print("Checkpoint loaded successfully.")


# ============================================================
# EVALUATION MODE
# ============================================================

model.eval()


# ============================================================
# SPECIAL TOKENS
# ============================================================

PAD_ID = tokenizer.token_to_id(
    "<|pad|>"
)

USER_ID = tokenizer.token_to_id(
    "<|user|>"
)

ASSISTANT_ID = tokenizer.token_to_id(
    "<|assistant|>"
)

END_ID = tokenizer.token_to_id(
    "<|end|>"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=-100
)


# ============================================================
# EVALUATE
# ============================================================

total_loss = 0.0
total_tokens = 0
correct_tokens = 0


print()
print("=" * 60)
print("EVALUATING 100 INSTRUCTION EXAMPLES")
print("=" * 60)


with torch.no_grad():

    for index, record in enumerate(examples):

        instruction = record[
            "instruction"
        ]

        response = record[
            "response"
        ]


        # ----------------------------------------------------
        # BUILD CONVERSATION
        # ----------------------------------------------------

        user_text = (
            "<|user|>\n"
            + instruction
            + "\n"
            + "<|assistant|>\n"
        )

        assistant_text = (
            response
            + "\n"
            + "<|end|>"
        )


        # ----------------------------------------------------
        # TOKENIZE
        # ----------------------------------------------------

        user_tokens = tokenizer.encode(
            user_text
        ).ids

        assistant_tokens = tokenizer.encode(
            assistant_text
        ).ids


        # ----------------------------------------------------
        # COMBINE
        # ----------------------------------------------------

        token_ids = (
            user_tokens
            + assistant_tokens
        )

        labels = (
            [-100] * len(user_tokens)
            + assistant_tokens
        )


        # ----------------------------------------------------
        # TRUNCATE
        # ----------------------------------------------------

        token_ids = token_ids[:513]

        labels = labels[:513]


        # ----------------------------------------------------
        # SHIFT
        # ----------------------------------------------------

        input_ids = token_ids[:-1]

        shifted_labels = labels[1:]


        # ----------------------------------------------------
        # PAD
        # ----------------------------------------------------

        padding = (
            512 - len(input_ids)
        )

        if padding > 0:

            input_ids += (
                [PAD_ID] * padding
            )

            shifted_labels += (
                [-100] * padding
            )


        # ----------------------------------------------------
        # FINAL LENGTH
        # ----------------------------------------------------

        input_ids = input_ids[:512]

        shifted_labels = shifted_labels[:512]


        # ----------------------------------------------------
        # TENSORS
        # ----------------------------------------------------

        input_tensor = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=device
        )

        label_tensor = torch.tensor(
            [shifted_labels],
            dtype=torch.long,
            device=device
        )


        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        logits = model(
            input_tensor
        )


        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        loss = criterion(
            logits.reshape(
                -1,
                logits.size(-1)
            ),
            label_tensor.reshape(-1)
        )


        # ----------------------------------------------------
        # VALID TOKENS
        # ----------------------------------------------------

        valid_mask = (
            label_tensor != -100
        )

        valid_logits = logits[
            valid_mask
        ]

        valid_labels = label_tensor[
            valid_mask
        ]

        count = valid_labels.numel()


        if count > 0:

            predictions = torch.argmax(
                valid_logits,
                dim=-1
            )

            correct = (
                predictions == valid_labels
            ).sum().item()

            correct_tokens += correct

            total_tokens += count

            total_loss += (
                loss.item() * count
            )


# ============================================================
# FINAL METRICS
# ============================================================

average_loss = (
    total_loss / total_tokens
    if total_tokens > 0
    else 0.0
)

accuracy = (
    correct_tokens / total_tokens
    if total_tokens > 0
    else 0.0
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print()
print(
    "Examples:",
    len(examples)
)

print(
    "Valid response tokens:",
    total_tokens
)

print(
    "Correct predictions:",
    correct_tokens
)

print(
    "Average loss:",
    f"{average_loss:.10f}"
)

print(
    "Token accuracy:",
    f"{accuracy * 100:.4f}%"
)


# ============================================================
# INTERPRETATION
# ============================================================

print()
print("=" * 60)
print("INTERPRETATION")
print("=" * 60)

if accuracy >= 0.95:

    print()
    print(
        "The base model already predicts most "
        "instruction response tokens correctly."
    )

    print(
        "Instruction tuning may provide limited "
        "benefit from this dataset."
    )

elif accuracy >= 0.70:

    print()
    print(
        "The base model has partial knowledge of "
        "the instruction responses."
    )

    print(
        "Instruction tuning should provide useful "
        "additional adaptation."
    )

else:

    print()
    print(
        "The base model does not yet predict these "
        "instruction responses well."
    )

    print(
        "This dataset is suitable for instruction "
        "tuning experiments."
    )


print()
print("=" * 60)
print("DIAGNOSTIC COMPLETED")
print("=" * 60)
