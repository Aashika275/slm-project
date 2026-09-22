import os
import sys
import torch


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

from model.codex_model import CODEXSLM
from tokenizers import Tokenizer


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
    "codex_slm_instruction_tuned_v3.pt"
)


# ============================================================
# CONFIGURATION
# ============================================================

VOCAB_SIZE = 8000
MAX_SEQ_LEN = 512
HIDDEN_SIZE = 256
NUM_HEADS = 4
NUM_LAYERS = 4
INTERMEDIATE_SIZE = 1024

DEVICE = torch.device("cpu")

MAX_NEW_TOKENS = 100


# ============================================================
# CHECK FILES
# ============================================================

print()
print("=" * 60)
print("CODEX-SLM INFERENCE")
print("=" * 60)

print()
print("Device:", DEVICE)

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(
        f"Tokenizer not found:\n{TOKENIZER_PATH}"
    )

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Instruction-tuned checkpoint not found:\n"
        f"{CHECKPOINT_PATH}"
    )

print()
print("Tokenizer: OK")
print("Instruction-tuned checkpoint: OK")


# ============================================================
# LOAD TOKENIZER
# ============================================================

print()
print("Loading tokenizer...")

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)

print("Tokenizer loaded successfully.")


# ============================================================
# SPECIAL TOKENS
# ============================================================

PAD_ID = tokenizer.token_to_id("<|pad|>")
USER_ID = tokenizer.token_to_id("<|user|>")
ASSISTANT_ID = tokenizer.token_to_id("<|assistant|>")
END_ID = tokenizer.token_to_id("<|end|>")

print()
print("Special token IDs:")
print("PAD:", PAD_ID)
print("USER:", USER_ID)
print("ASSISTANT:", ASSISTANT_ID)
print("END:", END_ID)


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("Creating CODEX-SLM model...")

model = CODEXSLM(
    vocab_size=VOCAB_SIZE,
    max_seq_len=MAX_SEQ_LEN,
    hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    intermediate_size=INTERMEDIATE_SIZE
)

model = model.to(DEVICE)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print()
print("Loading instruction-tuned model...")

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

print("Instruction-tuned model loaded successfully.")


# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_response(
    user_text,
    max_new_tokens=MAX_NEW_TOKENS
):

    # --------------------------------------------------------
    # Build instruction prompt
    # --------------------------------------------------------

    prompt = (
        "<|user|>\n"
        + user_text
        + "\n"
        + "<|assistant|>\n"
    )

    # --------------------------------------------------------
    # Tokenize prompt
    # --------------------------------------------------------

    encoding = tokenizer.encode(prompt)

    input_ids = encoding.ids

    # --------------------------------------------------------
    # Limit prompt length
    # --------------------------------------------------------

    if len(input_ids) >= MAX_SEQ_LEN:

        input_ids = input_ids[
            -(MAX_SEQ_LEN - 1):
        ]

    generated_ids = list(input_ids)

    # --------------------------------------------------------
    # Generate tokens
    # --------------------------------------------------------

    with torch.no_grad():

        for _ in range(max_new_tokens):

            current_ids = generated_ids[
                -MAX_SEQ_LEN:
            ]

            input_tensor = torch.tensor(
                [current_ids],
                dtype=torch.long,
                device=DEVICE
            )

            logits = model(
                input_tensor
            )

            next_token_logits = logits[
                0,
                -1,
                :
            ]

            # ------------------------------------------------
            # Greedy decoding
            # ------------------------------------------------

            next_token_id = torch.argmax(
                next_token_logits
            ).item()

            generated_ids.append(
                next_token_id
            )

            # ------------------------------------------------
            # Stop when END token is generated
            # ------------------------------------------------

            if next_token_id == END_ID:
                break

    # --------------------------------------------------------
    # Extract generated response
    # --------------------------------------------------------

    generated_part = generated_ids[
        len(input_ids):
    ]

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    response = tokenizer.decode(
        generated_part
    )

    # Remove special markers if present

    response = response.replace(
        "<|end|>",
        ""
    )

    response = response.replace(
        "<|assistant|>",
        ""
    )

    return response.strip()


# ============================================================
# INTERACTIVE CHAT
# ============================================================

print()
print("=" * 60)
print("CODEX-SLM READY")
print("=" * 60)

print()
print("Type a programming question.")
print("Type 'exit' to stop.")
print()

while True:

    try:

        user_input = input(
            "You: "
        ).strip()

    except KeyboardInterrupt:

        print()
        print("Exiting CODEX-SLM.")
        break

    if user_input.lower() == "exit":

        print()
        print("CODEX-SLM stopped.")
        break

    if not user_input:

        continue

    print()
    print("CODEX-SLM:")

    try:

        response = generate_response(
            user_input
        )

        print(response)

    except Exception as error:

        print(
            "Generation error:",
            error
        )

    print()
