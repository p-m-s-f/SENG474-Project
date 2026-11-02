import argparse
import csv
import hashlib
import shutil
from pathlib import Path


def file_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    label_file = project_root / "image_labels.csv"
    data_dir = project_root / "data"
    source_dir = Path(args.source).expanduser().resolve()

    categories = {"no_fire", "fire_low", "fire_medium", "fire_high"}

    if not source_dir.is_dir():
        raise ValueError(f"Source folder not found: {source_dir}")

    #keep the original download separate from the prepared data.
    if (
        source_dir == data_dir.resolve()
        or source_dir.is_relative_to(data_dir.resolve())
        or data_dir.resolve().is_relative_to(source_dir)
    ):
        raise ValueError(
            "Choose a downloaded dataset folder separate from "
            "the project's data folder."
        )

    with label_file.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if not {"filename", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain filename and label columns.")

        records = list(reader)

    if not records:
        raise ValueError("The label CSV is empty.")

    filenames = set()

    for record in records:
        filename = record["filename"]
        label = record["label"]

        if label not in categories:
            raise ValueError(f"Unknown category: {label}")

        if (
            not filename
            or filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
        ):
            raise ValueError(f"Invalid filename: {filename}")

        if filename in filenames:
            raise ValueError(f"Repeated CSV filename: {filename}")

        filenames.add(filename)

    print("Searching the downloaded dataset...")

    matches = {}

    for path in source_dir.rglob("*"):
        if path.is_file() and path.name in filenames:
            matches.setdefault(path.name, []).append(path)

    problems = []
    copies = []
    skipped = 0

    #check every record before copying anything.
    for record in records:
        filename = record["filename"]
        candidates = matches.get(filename, [])

        if not candidates:
            problems.append(f"Missing: {filename}")
            continue

        if len(candidates) > 1:
            problems.append(
                f"Multiple source files named {filename}:\n"
                + "\n".join(f"  {path}" for path in candidates)
            )
            continue

        source = candidates[0]
        destination = data_dir / record["label"] / filename

        if destination.exists():
            if (
                destination.is_file()
                and source.stat().st_size == destination.stat().st_size
                and file_hash(source) == file_hash(destination)
            ):
                skipped += 1
            else:
                problems.append(f"Existing file differs: {destination}")
        else:
            copies.append((source, destination))

    if problems:
        for problem in problems:
            print(problem)

        raise SystemExit(
            f"\nFound {len(problems)} problems. No files were copied."
        )

    print(f"Images listed in CSV: {len(records)}")
    print(f"Already prepared: {skipped}")
    print(f"Images to copy: {len(copies)}")

    if args.dry_run:
        print("Dry run complete. No files were copied.")
        return

    for index, (source, destination) in enumerate(copies, start=1):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

        if index % 100 == 0 or index == len(copies):
            print(f"Copied {index}/{len(copies)}")

    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()

#test
#python src/prepare_data.py --source "/path/to/extracted/kaggle-dataset" --dry-run
#run
#python src/prepare_data.py --source "/path/to/extracted/kaggle-dataset"