import json
import re
from pathlib import Path


RAW_DIR = Path("data/raw")
CLEANED_DIR = Path("data/cleaned")

CLEANED_DIR.mkdir(parents=True, exist_ok=True)


FILES = [
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
]

REQUIRED_FIELDS = [
    "id",
    "task",
    "domain",
    "instruction",
    "response",
]


def clean_text(text):
    """Normalize unnecessary whitespace."""

    text = str(text)

    # Replace Windows-style line endings
    text = text.replace("\r\n", "\n")

    # Remove trailing spaces from lines
    text = "\n".join(
        line.rstrip()
        for line in text.split("\n")
    )

    # Replace multiple spaces with one
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_record(record):
    """Clean and normalize one dataset record."""

    cleaned = {}

    for field in REQUIRED_FIELDS:
        value = record.get(field, "")

        if isinstance(value, str):
            value = clean_text(value)

        cleaned[field] = value

    # Preserve source if available
    if "source" in record:
        cleaned["source"] = clean_text(record["source"])

    return cleaned


def is_valid_record(record):
    """Check whether a record contains usable information."""

    for field in REQUIRED_FIELDS:

        if field not in record:
            return False

        value = record[field]

        if value is None:
            return False

        if isinstance(value, str) and not value.strip():
            return False

    return True


def process_file(filename):

    input_path = RAW_DIR / filename
    output_path = CLEANED_DIR / filename

    print("\n" + "=" * 60)
    print(f"Processing: {filename}")
    print("=" * 60)

    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return

    total = 0
    valid = 0
    invalid = 0
    duplicate_ids = 0

    seen_ids = set()

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as input_file, open(
        output_path,
        "w",
        encoding="utf-8"
    ) as output_file:

        for line_number, line in enumerate(
            input_file,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            total += 1

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                invalid += 1
                print(
                    f"Invalid JSON at line {line_number}"
                )
                continue

            if not is_valid_record(record):
                invalid += 1
                continue

            cleaned = clean_record(record)

            record_id = cleaned["id"]

            if record_id in seen_ids:
                duplicate_ids += 1
                continue

            seen_ids.add(record_id)

            output_file.write(
                json.dumps(
                    cleaned,
                    ensure_ascii=False
                )
                + "\n"
            )

            valid += 1

    print(f"Original records : {total}")
    print(f"Valid records    : {valid}")
    print(f"Invalid records  : {invalid}")
    print(f"Duplicate IDs    : {duplicate_ids}")
    print(f"Output file      : {output_path}")


def main():

    print("=" * 60)
    print("CODEX-SLM DATASET CLEANING")
    print("=" * 60)

    for filename in FILES:
        process_file(filename)

    print("\n" + "=" * 60)
    print("CLEANING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
