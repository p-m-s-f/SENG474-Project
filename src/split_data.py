import csv
import random
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
label_file = project_root / "image_labels.csv"

with label_file.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    images = list(reader)

print(f"Images available: {len(images)}")

categories = {
    "no_fire": [],
    "fire_low": [],
    "fire_medium": [],
    "fire_high": [],
}

for image in images:
    categories[image["label"]].append(image)


rng = random.Random(0)

train = []
val = []
test = []

for label, records in categories.items():
    records = sorted(records, key=lambda record: record["filename"])
    rng.shuffle(records)

    train_end = int(len(records) * 0.70)
    val_end = train_end + int(len(records) * 0.15)

    train.extend(records[:train_end])
    val.extend(records[train_end:val_end])
    test.extend(records[val_end:])

print(f"Training: {len(train)}")
print(f"Validation: {len(val)}")
print(f"Test: {len(test)}")

train_names = {record["filename"] for record in train}
val_names = {record["filename"] for record in val}
test_names = {record["filename"] for record in test}

assert train_names.isdisjoint(val_names)
assert train_names.isdisjoint(test_names)
assert val_names.isdisjoint(test_names)

assert len(train_names | val_names | test_names) == len(images)

print("Split checks passed")

split_dir = project_root / "splits"
split_dir.mkdir(exist_ok=True)

for name, records in [("train", train), ("val", val), ("test", test)]:
    output = split_dir / f"{name}.csv"

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "label"])
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved {output.name}: {len(records)} images")