import csv
import hashlib
from pathlib import Path
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
label_file = project_root / "image_labels.csv"

def visual_hash(image_path):
    with Image.open(image_path) as image:
        small_image = image.convert("L").resize((9, 8))
        pixels = list(small_image.get_flattened_data())

    return tuple(
        pixels[row * 9 + column] > pixels[row * 9 + column + 1]
        for row in range(8)
        for column in range(8)
    )

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

visual_hashes = []

for record in images:
    image_path = project_root / "data" / record["label"] / record["filename"]
    image_hash = visual_hash(image_path)

    visual_hashes.append((image_path, image_hash))

print(f"Visual hashes calculated: {len(visual_hashes)}")

similar_pairs = []

for i in range(len(visual_hashes)):
    path_a, hash_a = visual_hashes[i]

    for j in range(i + 1, len(visual_hashes)):
        path_b, hash_b = visual_hashes[j]

        difference = sum(a != b for a, b in zip(hash_a, hash_b))

        if difference <= 5:
            similar_pairs.append((path_a, path_b, difference))

print(f"Possible similar pairs: {len(similar_pairs)}")

for path_a, path_b, difference in similar_pairs:
    print(f"\nHash difference: {difference}")
    print(path_a)
    print(path_b)