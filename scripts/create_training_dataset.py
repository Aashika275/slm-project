import os
import sys
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


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
# DEFAULT PATHS
# ============================================================

TOKENIZER_PATH = os.path.join(
    PROJECT_ROOT,
    "tokenizer",
    "codex_tokenizer_v2.json"
)

TRAIN_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "cleaned",
    "train.jsonl"
)

MAX_LENGTH = 512


# ============================================================
# DATASET
# ============================================================

class CODEXDataset(Dataset):

    def __init__(
        self,
        data_path,
        tokenizer_path,
        max_length=512
    ):

        self.max_length = max_length

        # ----------------------------------------------------
        # Convert paths to absolute paths
        # ----------------------------------------------------

        data_path = os.path.abspath(data_path)
        tokenizer_path = os.path.abspath(tokenizer_path)

        print("\nDataset path:")
        print(data_path)

        print("\nTokenizer path:")
        print(tokenizer_path)

        # ----------------------------------------------------
        # Check dataset file
        # ----------------------------------------------------

        if not os.path.exists(data_path):

            raise FileNotFoundError(
                f"Dataset file not found:\n{data_path}"
            )

        # ----------------------------------------------------
        # Check tokenizer file
        # ----------------------------------------------------

        if not os.path.exists(tokenizer_path):

            raise FileNotFoundError(
                f"Tokenizer file not found:\n{tokenizer_path}"
            )

        print("\nDataset exists: True")
        print("Tokenizer exists: True")

        # ----------------------------------------------------
        # Load tokenizer
        # ----------------------------------------------------

        print("\nLoading tokenizer...")

        self.tokenizer = Tokenizer.from_file(
            tokenizer_path
        )

        print("Tokenizer loaded successfully.")

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        print("\nLoading dataset...")

        self.records = []

        with open(
            data_path,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                # ------------------------------------------------
                # Validate record
                # ------------------------------------------------

                if "instruction" not in record:
                    continue

                if "response" not in record:
                    continue

                # ------------------------------------------------
                # Create training text
                # ------------------------------------------------

                text = (
                    "<|user|>\n"
                    + str(record["instruction"])
                    + "\n<|assistant|>\n"
                    + str(record["response"])
                    + "\n<|end|>"
                )

                self.records.append(text)

        print(
            f"Loaded {len(self.records)} records."
        )

        if len(self.records) == 0:

            raise ValueError(
                "No valid records found in dataset."
            )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(self.records)

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        text = self.records[index]

        # ----------------------------------------------------
        # Tokenize
        # ----------------------------------------------------

        encoded = self.tokenizer.encode(
            text
        )

        token_ids = encoded.ids

        # ----------------------------------------------------
        # Padding token
        # ----------------------------------------------------

        pad_token_id = (
            self.tokenizer.token_to_id(
                "<|pad|>"
            )
        )

        if pad_token_id is None:

            pad_token_id = 0

        # ----------------------------------------------------
        # Truncate
        # ----------------------------------------------------

        if len(token_ids) > self.max_length:

            token_ids = token_ids[
                :self.max_length
            ]

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        else:

            padding_length = (
                self.max_length
                - len(token_ids)
            )

            token_ids = (
                token_ids
                + [pad_token_id]
                * padding_length
            )

        # ----------------------------------------------------
        # Create input and target
        # ----------------------------------------------------

        input_ids = torch.tensor(
            token_ids[:-1],
            dtype=torch.long
        )

        labels = torch.tensor(
            token_ids[1:],
            dtype=torch.long
        )

        return {
            "input_ids": input_ids,
            "labels": labels
        }


# ============================================================
# DATASET TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CODEX-SLM TRAINING DATASET TEST")
    print("=" * 60)

    dataset = CODEXDataset(
        data_path=TRAIN_DATA_PATH,
        tokenizer_path=TOKENIZER_PATH,
        max_length=MAX_LENGTH
    )

    print("\nDataset size:")
    print(len(dataset))

    # --------------------------------------------------------
    # Test first sample
    # --------------------------------------------------------

    sample = dataset[0]

    print("\nSample input shape:")
    print(sample["input_ids"].shape)

    print("\nSample label shape:")
    print(sample["labels"].shape)

    print("\nFirst 20 input token IDs:")
    print(sample["input_ids"][:20])

    print("\nFirst 20 label token IDs:")
    print(sample["labels"][:20])

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )

    batch = next(
        iter(dataloader)
    )

    print("\nBatch input shape:")
    print(batch["input_ids"].shape)

    print("\nBatch label shape:")
    print(batch["labels"].shape)

    print("\n" + "=" * 60)
    print("DATASET TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
