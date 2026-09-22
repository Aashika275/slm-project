import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data/raw")

FILES = {
    "train": DATA_DIR / "train.jsonl",
    "validation": DATA_DIR / "validation.jsonl",
    "test": DATA_DIR / "test.jsonl",
}


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                print(
                    f"Invalid JSON: {path}, "
                    f"line {line_number}: {error}"
                )

    return records


def inspect_dataset(name, path):
    print("\n" + "=" * 60)
    print(name.upper())
    print("=" * 60)

    if not path.exists():
        print("FILE NOT FOUND:", path)
        return []

    records = load_jsonl(path)

    print("Records:", len(records))

    if not records:
        return []

    required_fields = [
        "id",
        "task",
        "domain",
        "instruction",
        "response",
    ]

    missing_records = 0

    for record in records:
        missing = [
            field for field in required_fields
            if field not in record
        ]

        if missing:
            missing_records += 1

    print("Records with missing fields:", missing_records)

    ids = [
        record.get("id")
        for record in records
        if record.get("id") is not None
    ]

    duplicate_ids = len(ids) - len(set(ids))

    print("Duplicate IDs:", duplicate_ids)

    tasks = Counter(
        record.get("task", "UNKNOWN")
        for record in records
    )

    domains = Counter(
        record.get("domain", "UNKNOWN")
        for record in records
    )

    print("\nTasks:")
    for task, count in tasks.items():
        print(f"  {task}: {count}")

    print("\nDomains:")
    for domain, count in domains.items():
        print(f"  {domain}: {count}")

    instruction_lengths = [
        len(str(record.get("instruction", "")))
        for record in records
    ]

    response_lengths = [
        len(str(record.get("response", "")))
        for record in records
    ]

    print("\nInstruction length:")
    print("  Minimum:", min(instruction_lengths))
    print("  Maximum:", max(instruction_lengths))
    print(
        "  Average:",
        round(
            sum(instruction_lengths)
            / len(instruction_lengths),
            2
        )
    )

    print("\nResponse length:")
    print("  Minimum:", min(response_lengths))
    print("  Maximum:", max(response_lengths))
    print(
        "  Average:",
        round(
            sum(response_lengths)
            / len(response_lengths),
            2
        )
    )

    print("\nFirst record:")
    print(json.dumps(
        records[0],
        indent=2,
        ensure_ascii=False
    ))

    return records


def main():
    print("=" * 60)
    print("CODEX-SLM DATASET INSPECTION")
    print("=" * 60)

    datasets = {}

    for name, path in FILES.items():
        datasets[name] = inspect_dataset(name, path)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total = 0

    for name, records in datasets.items():
        count = len(records)
        total += count
        print(f"{name}: {count}")

    print("Total:", total)
    print("\nInspection completed.")


if __name__ == "__main__":
    main()
