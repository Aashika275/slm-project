import json
from pathlib import Path


CLEANED_DIR = Path("data/cleaned")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


FILES = [
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
]


OUTPUT_FILE = PROCESSED_DIR / "tokenizer_corpus.txt"


def main():

    total_records = 0
    total_characters = 0

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output:

        for filename in FILES:

            input_path = CLEANED_DIR / filename

            print(f"Reading: {input_path}")

            if not input_path.exists():
                print(
                    f"WARNING: {input_path} does not exist."
                )
                continue

            with open(
                input_path,
                "r",
                encoding="utf-8"
            ) as input_file:

                for line in input_file:

                    line = line.strip()

                    if not line:
                        continue

                    record = json.loads(line)

                    instruction = record.get(
                        "instruction",
                        ""
                    )

                    response = record.get(
                        "response",
                        ""
                    )

                    text = (
                        "<|user|>\n"
                        + instruction
                        + "\n"
                        + "<|assistant|>\n"
                        + response
                        + "\n"
                        + "<|end|>\n"
                    )

                    output.write(text)

                    total_records += 1
                    total_characters += len(text)

    print("\n" + "=" * 60)
    print("TOKENIZER DATA PREPARATION COMPLETE")
    print("=" * 60)

    print("Records:", total_records)
    print("Characters:", total_characters)
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
