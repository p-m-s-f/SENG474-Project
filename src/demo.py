import argparse
import csv
import random
from pathlib import Path
from PIL import Image
from predict import get_device, load_models, predict_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predicted-category",
        required=True,
        choices=["no_fire", "fire_low", "fire_medium", "fire_high"],
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    validation_file = project_root / "splits" / "val.csv"

    with validation_file.open("r", newline="", encoding="utf-8") as file:
        records = list(csv.DictReader(file))

    device = get_device()
    detector, risk_model = load_models(project_root, device)
    matches = []

    print(f"Device: {device}")
    print(f"Scanning {len(records)} validation images...")

    for index, record in enumerate(records, start=1):
        image_path = (
            project_root / "data"
            / record["label"]
            / record["filename"]
        )

        prediction = predict_image(
            image_path, detector, risk_model, device
        )

        if prediction == args.predicted_category:
            matches.append(record)

        if index % 25 == 0 or index == len(records):
            print(f"Checked {index}/{len(records)}")

    print(f"Matching predictions: {len(matches)}")

    if not matches:
        print("No validation images received that prediction.")
        return

    rng = random.Random(args.seed)
    selected = rng.choice(matches)

    image_path = (
        project_root / "data"
        / selected["label"]
        / selected["filename"]
    )

    print("\nRandom validation example filtered by prediction")
    print(f"Image: {selected['filename']}")
    print(f"Assigned label: {selected['label']}")
    print(f"Prediction: {args.predicted_category}")
    print(f"Matches assigned label: {selected['label'] == args.predicted_category}")

    with Image.open(image_path) as image:
        display_image = image.convert("RGB")

    display_image.thumbnail((1200, 1200))
    display_image.show(
        title=f"Predicted: {args.predicted_category}"
    )


if __name__ == "__main__":
    main()