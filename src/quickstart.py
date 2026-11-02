import csv
import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent

    #required packages load check
    try:
        import torch
        import torchvision
    except ImportError as error:
        raise SystemExit(
            f"Missing dependency: {error}\n"
            "Follow the README installation instructions first."
        )

    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"Torchvision: {torchvision.__version__}", flush=True)

    #prediction and evaluation use the saved trained models.
    for stage in ["detector", "risk"]:
        checkpoint = project_root / "checkpoints" / f"{stage}_best.pt"

        if not checkpoint.is_file():
            raise SystemExit(
                f"Missing checkpoint: {checkpoint}\n"
                "Follow the README instructions to download "
                "the checkpoints or train both models."
            )

    # Check that all validation images are available.
    validation_file = project_root / "splits" / "val.csv"

    if not validation_file.is_file():
        raise SystemExit(f"Missing validation CSV: {validation_file}")

    with validation_file.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        if not {"filename", "label"}.issubset(reader.fieldnames or []):
            raise SystemExit("Validation CSV needs filename and label columns.")

        records = list(reader)

    if not records:
        raise SystemExit("The validation CSV is empty.")

    image_paths = [
        project_root / "data" / record["label"] / record["filename"]
        for record in records
    ]

    missing = [path for path in image_paths if not path.is_file()]

    if missing:
        raise SystemExit(
            f"Missing {len(missing)} validation images.\n"
            f"First missing image: {missing[0]}\n"
            "Follow the README dataset preparation instructions first."
        )

    commands = [
        (
            "Run one sample prediction",
            ["predict", "--image", str(image_paths[0])],
        ),
        (
            "Evaluate the validation split",
            ["evaluate", "--split", "val"],
        ),
    ]

    for description, arguments in commands:
        print(f"\n{description}", flush=True)

        result = subprocess.run([
            sys.executable,
            str(project_root / "src" / "main.py"),
            *arguments,
        ])

        if result.returncode != 0:
            raise SystemExit(result.returncode)

    print("\nQuickstart completed successfully.", flush=True)
    print("Prediction and validation evaluation both ran successfully.")
    print(f"Results: {project_root / 'outputs' / 'val'}")
    print("Existing validation output files were overwritten.")
    print("No models were retrained and the test set was not evaluated.")


if __name__ == "__main__":
    main()