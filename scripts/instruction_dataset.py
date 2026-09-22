import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# ============================================================
# DEFAULT PATHS
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
    "train_large.jsonl"
)


# ============================================================
# INSTRUCTION DATASET
# ============================================================

class InstructionDataset(Dataset):

    def __init__(
        self,
        data_path,
        tokenizer_path,
        max_length=512
    ):

        self.data_path = os.path.abspath(
            data_path
        )

        self.tokenizer_path = os.path.abspath(
            tokenizer_path
        )

        self.max_length = max_length

        print(
            "Dataset path:",
            self.data_path
        )

        print(
            "Tokenizer path:",
            self.tokenizer_path
        )

        # ----------------------------------------------------
        # Check files
        # ----------------------------------------------------

        if not os.path.exists(
            self.data_path
        ):
            raise FileNotFoundError(
                f"Dataset not found:\n{self.data_path}"
            )

        if not os.path.exists(
            self.tokenizer_path
        ):
            raise FileNotFoundError(
                f"Tokenizer not found:\n{self.tokenizer_path}"
            )

        # ----------------------------------------------------
        # Load tokenizer
        # ----------------------------------------------------

        self.tokenizer = Tokenizer.from_file(
            self.tokenizer_path
        )

        # ----------------------------------------------------
        # Special token IDs
        # ----------------------------------------------------

        self.pad_id = self.tokenizer.token_to_id(
            "<|pad|>"
        )

        self.user_id = self.tokenizer.token_to_id(
            "<|user|>"
        )

        self.assistant_id = self.tokenizer.token_to_id(
            "<|assistant|>"
        )

        self.end_id = self.tokenizer.token_to_id(
            "<|end|>"
        )

        print(
            "PAD ID:",
            self.pad_id
        )

        print(
            "USER ID:",
            self.user_id
        )

        print(
            "ASSISTANT ID:",
            self.assistant_id
        )

        print(
            "END ID:",
            self.end_id
        )

        # ----------------------------------------------------
        # Load JSONL
        # ----------------------------------------------------

        self.data = []

        with open(
            self.data_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                if (
                    "instruction" not in record
                    or "response" not in record
                ):
                    continue

                self.data.append(
                    {
                        "instruction": record[
                            "instruction"
                        ],
                        "response": record[
                            "response"
                        ]
                    }
                )

        print(
            "Loaded examples:",
            len(self.data)
        )


    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(self.data)


    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        record = self.data[index]

        instruction = record[
            "instruction"
        ]

        response = record[
            "response"
        ]

        # ----------------------------------------------------
        # Format conversation
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
        # Tokenize separately
        # ----------------------------------------------------

        user_tokens = self.tokenizer.encode(
            user_text
        ).ids

        assistant_tokens = self.tokenizer.encode(
            assistant_text
        ).ids

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        token_ids = (
            user_tokens
            + assistant_tokens
        )

        # ----------------------------------------------------
        # Create labels for the FULL sequence
        #
        # User tokens:
        #     -100
        #
        # Assistant tokens:
        #     actual token ID
        # ----------------------------------------------------

        full_labels = (
            [-100] * len(user_tokens)
            + assistant_tokens
        )

        # ----------------------------------------------------
        # We need ONE EXTRA token because we shift:
        #
        # input:
        #     token[0] ... token[n-1]
        #
        # label:
        #     token[1] ... token[n]
        #
        # Therefore max_length + 1 is temporarily needed.
        # ----------------------------------------------------

        token_ids = token_ids[
            :self.max_length + 1
        ]

        full_labels = full_labels[
            :self.max_length + 1
        ]

        # ----------------------------------------------------
        # NEXT-TOKEN SHIFT
        #
        # This is the critical correction.
        # ----------------------------------------------------

        input_ids = token_ids[:-1]

        labels = full_labels[1:]

        # ----------------------------------------------------
        # Pad to exactly max_length
        # ----------------------------------------------------

        input_padding = (
            self.max_length
            - len(input_ids)
        )

        label_padding = (
            self.max_length
            - len(labels)
        )

        if input_padding > 0:

            input_ids += (
                [self.pad_id]
                * input_padding
            )

        if label_padding > 0:

            labels += (
                [-100]
                * label_padding
            )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        input_ids = input_ids[
            :self.max_length
        ]

        labels = labels[
            :self.max_length
        ]

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        input_ids = torch.tensor(
            input_ids,
            dtype=torch.long
        )

        labels = torch.tensor(
            labels,
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

    dataset = InstructionDataset(
        DATA_PATH,
        TOKENIZER_PATH,
        max_length=512
    )

    print()
    print("=" * 60)
    print("INSTRUCTION DATASET TEST")
    print("=" * 60)

    print()
    print(
        "Dataset size:",
        len(dataset)
    )

    sample = dataset[0]

    print()
    print(
        "Input shape:",
        sample["input_ids"].shape
    )

    print(
        "Label shape:",
        sample["labels"].shape
    )

    # --------------------------------------------------------
    # Valid labels
    # --------------------------------------------------------

    valid_labels = sample["labels"][
        sample["labels"] != -100
    ]

    print()
    print(
        "Valid label tokens:",
        len(valid_labels)
    )

    print(
        "Ignored label tokens:",
        int(
            (
                sample["labels"] == -100
            ).sum()
        )
    )

    print()
    print("First 30 input IDs:")
    print(
        sample["input_ids"][:30].tolist()
    )

    print()
    print("First 30 labels:")
    print(
        sample["labels"][:30].tolist()
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )

    batch = next(
        iter(loader)
    )

    print()
    print(
        "Batch input shape:",
        batch["input_ids"].shape
    )

    print(
        "Batch label shape:",
        batch["labels"].shape
    )

    print()
    print(
        "Instruction dataset test "
        "completed successfully."
    )
