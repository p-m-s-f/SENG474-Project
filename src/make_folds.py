import csv
from pathlib import Path
import random

project_root = Path(__file__).resolve().parent.parent
split_dir = project_root / "splits"

records = []

for name in ("train.csv", "val.csv"):
    with (split_dir / name).open(newline="", encoding="utf-8") as file:
        records.extend(csv.DictReader(file))

print(f"Development images: {len(records)}")

records_by_label = {}

for record in records:
    label = record["label"]
    if label not in records_by_label:
        records_by_label[label] = []
    records_by_label[label].append(record)

for label, group in records_by_label.items():
    print(f"{label}: {len(group)}")

rng = random.Random(0)
folds = [[], [], []]

for label in sorted(records_by_label):
    group = sorted(records_by_label[label], key=lambda row: row["filename"])
    rng.shuffle(group)

    for index, record in enumerate(group):
        folds[index % 3].append(record)

for number, fold in enumerate(folds, start=1):
    print(f"Fold {number}: {len(fold)} images")

for number, validation in enumerate(folds, start=1):
    training = [
        record
        for other_number, fold in enumerate(folds, start=1)
        if other_number != number
        for record in fold
    ]

    fold_dir = split_dir / "cv" / f"fold_{number}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for filename, rows in [("train.csv", training), ("val.csv", validation)]:
        with (fold_dir / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["filename", "label"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved fold {number}: {len(training)} train, {len(validation)} val")