import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from model.codex_model import CODEXSLM


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "instruction_tuning",
    "codex_instruction_train.jsonl"
)

TOKENIZER_PATH = os.path.join(
    BASE_DIR,
    "tokenizer",
    "codex_tokenizer_v2.json"
)

# IMPORTANT:
# Start instruction tuning from the newly trained V3 base model.
CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "best_model_v3.pt"
)

# IMPORTANT:
# Save instruction-tuned model as a NEW checkpoint.
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "codex_slm_instruction_tuned_v4.pt"
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

BATCH_SIZE = 2
LEARNING_RATE = 5e-5
EPOCHS = 3
GRAD_CLIP = 1.0

DEVICE = torch.device("cpu")


# ============================================================
# SPECIAL TOKENS
# ============================================================

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
USER_ID = 4
ASSISTANT_ID = 5
END_ID = 6


# ============================================================
# DATASET
# ============================================================

class InstructionDataset(Dataset):

    def __init__(
        self,
        data_path,
        tokenizer_path,
        max_length=512
    ):
        self.max_length = max_length

        print("Loading tokenizer...")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        print(f"Loading instruction dataset...")
        print(f"Dataset path: {data_path}")

        self.examples = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                self.examples.append(json.loads(line))

        print(f"Loaded examples: {len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def encode_text(self, text):

        encoding = self.tokenizer.encode(text)

        return encoding.ids

    def __getitem__(self, idx):

        example = self.examples[idx]

        instruction = example.get("instruction", "")
        response = example.get("response", "")

        # ----------------------------------------------------
        # Construct instruction format
        # ----------------------------------------------------

        prompt = (
            "<|user|> "
            + instruction
            + " "
            + "<|assistant|> "
        )

        full_text = prompt + response + " <|end|>"

        full_ids = self.encode_text(full_text)

        prompt_ids = self.encode_text(prompt)

        # ----------------------------------------------------
        # Truncate
        # ----------------------------------------------------

        full_ids = full_ids[:self.max_length]

        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        # ----------------------------------------------------
        # Assistant-only loss
        # ----------------------------------------------------

        prompt_length = max(0, len(prompt_ids) - 1)

        for i in range(min(prompt_length, len(labels))):
            labels[i] = -100

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        target_length = self.max_length - 1

        input_ids = input_ids[:target_length]
        labels = labels[:target_length]

        padding_length = target_length - len(input_ids)

        if padding_length > 0:

            input_ids += [PAD_ID] * padding_length
            labels += [-100] * padding_length

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long
            )
        }


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("CODEX-SLM INSTRUCTION TUNING V4")
print("=" * 70)

print()
print("Configuration:")
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Epochs: {EPOCHS}")
print(f"Maximum sequence length: {MAX_SEQ_LEN}")

print()
print("Tokenizer:")
print(TOKENIZER_PATH)

print()
print("Base checkpoint:")
print(CHECKPOINT_PATH)

print()
print("Output checkpoint:")
print(OUTPUT_PATH)

print()


dataset = InstructionDataset(
    data_path=DATA_PATH,
    tokenizer_path=TOKENIZER_PATH,
    max_length=MAX_SEQ_LEN
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print()
print(f"Training examples: {len(dataset)}")
print()


# ============================================================
# CREATE MODEL
# ============================================================

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
# LOAD V3 BASE CHECKPOINT
# ============================================================

print()
print("Loading V3 base checkpoint...")

if not os.path.exists(CHECKPOINT_PATH):

    raise FileNotFoundError(
        f"V3 checkpoint not found:\n{CHECKPOINT_PATH}"
    )


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)


# Handle both plain state_dict and checkpoint dictionaries.

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict,
    strict=True
)

print("V3 base checkpoint loaded successfully.")


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=-100
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("STARTING INSTRUCTION TUNING")
print("=" * 70)
print()

model.train()

for epoch in range(EPOCHS):

    total_loss = 0.0
    batches = 0

    for batch_idx, batch in enumerate(dataloader):

        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits = model(input_ids)

        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            labels.reshape(-1)
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        total_loss += loss.item()
        batches += 1

        if (batch_idx + 1) % 100 == 0:

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {loss.item():.6f}"
            )

    average_loss = total_loss / max(batches, 1)

    print()
    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"Average Loss: {average_loss:.6f}"
    )
    print()


# ============================================================
# SAVE MODEL
# ============================================================

print("=" * 70)
print("SAVING INSTRUCTION-TUNED MODEL")
print("=" * 70)

torch.save(
    model.state_dict(),
    OUTPUT_PATH
)

print()
print("Instruction tuning completed successfully.")
print()
print("Base model:")
print("best_model_v3.pt")
print()
print("Instruction-tuned model:")
print("codex_slm_instruction_tuned_v4.pt")
print()
print(f"Saved to:")
print(OUTPUT_PATH)
print()


# ============================================================
# FILE SIZE
# ============================================================

if os.path.exists(OUTPUT_PATH):

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)

    print(f"Checkpoint size: {size_mb:.2f} MB")


print()
print("=" * 70)
print("V4 INSTRUCTION TUNING FINISHED")
print("=" * 70)