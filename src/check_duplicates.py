import csv
import hashlib
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
label_file = project_root / "image_labels.csv"

with label_file.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    images = list(reader)

print(len(images))

fingerprints = {}

for record in images:
    image_path = project_root / "data" / record["label"] / record["filename"]

    with image_path.open("rb") as file:
        fingerprint = hashlib.file_digest(file, "sha256").hexdigest()

    if fingerprint not in fingerprints:
        fingerprints[fingerprint] = []

    fingerprints[fingerprint].append(image_path)


duplicate_groups = 0

for paths in fingerprints.values():
    if len(paths) > 1:
        duplicate_groups += 1
        print("Duplicate group:")
        for path in paths:
            print(path)

print(f"Exact duplicate groups: {duplicate_groups}")