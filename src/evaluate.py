import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from predict import get_device, load_models, predict_image


def file_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def divide(numerator, denominator):
    # Undefined precision/recall is reported as zero.
    return numerator / denominator if denominator else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    split_file = project_root / "splits" / f"{args.split}.csv"

    class_names = ["no_fire", "fire_low", "fire_medium", "fire_high"]
    class_indices = {
        name: index for index, name in enumerate(class_names)
    }

    with split_file.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if not {"filename", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("Split CSV must contain filename and label columns.")

        records = list(reader)

    if not records:
        raise ValueError("The selected evaluation split is empty.")

    filenames = set()

    for record in records:
        if record["label"] not in class_indices:
            raise ValueError(f"Unknown label: {record['label']}")

        if record["filename"] in filenames:
            raise ValueError(f"Repeated filename: {record['filename']}")

        filenames.add(record["filename"])

    print(f"Evaluation split: {args.split}")
    print(f"Images: {len(records)}")

    device = get_device()
    detector, risk_model = load_models(project_root, device)
    print(f"Device: {device}")

    # Record exactly which saved models were evaluated.
    checkpoint_info = {}

    for stage in ["detector", "risk"]:
        path = project_root / "checkpoints" / f"{stage}_best.pt"
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        checkpoint_info[stage] = {
            "filename": path.name,
            "sha256": file_hash(path),
            "epoch": checkpoint["epoch"],
            "validation_loss": checkpoint["val_loss"],
            "class_names": checkpoint["class_names"],
        }

    results = []

    for index, record in enumerate(records, start=1):
        image_path = (
            project_root / "data"
            / record["label"]
            / record["filename"]
        )

        prediction = predict_image(
            image_path, detector, risk_model, device
        )

        results.append({
            "filename": record["filename"],
            "actual": record["label"],
            "predicted": prediction,
        })

        if index % 25 == 0 or index == len(records):
            print(f"Evaluated {index}/{len(records)}")

    # Rows are reference labels; columns are predictions.
    matrix = [[0 for _ in class_names] for _ in class_names]

    for result in results:
        actual = class_indices[result["actual"]]
        predicted = class_indices[result["predicted"]]
        matrix[actual][predicted] += 1

    correct = sum(matrix[i][i] for i in range(len(class_names)))
    accuracy = correct / len(results)

    print(f"\nCascade accuracy: {accuracy:.2%}")

    print("\nConfusion matrix: rows = actual, columns = predicted")
    print(f"{'':>14}" + "".join(f"{name:>14}" for name in class_names))

    for name, row in zip(class_names, matrix):
        print(f"{name:>14}" + "".join(f"{count:>14}" for count in row))

    per_class = {}

    print("\nPer-class metrics")
    print(f"{'Category':>14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Images':>8}")

    for index, name in enumerate(class_names):
        true_positive = matrix[index][index]
        actual_count = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)

        precision = divide(true_positive, predicted_count)
        recall = divide(true_positive, actual_count)
        f1 = divide(2 * precision * recall, precision + recall)

        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual_count,
            "predicted_count": predicted_count,
        }

        print(
            f"{name:>14} {precision:>10.2%} {recall:>10.2%} "
            f"{f1:>10.2%} {actual_count:>8}"
        )

    macro = {
        metric: sum(values[metric] for values in per_class.values())
        / len(class_names)
        for metric in ["precision", "recall", "f1"]
    }

    print(f"\nMacro-F1: {macro['f1']:.2%}")

    # Collapse all three risk categories into "fire".
    true_negative = matrix[0][0]
    false_positive = sum(matrix[0][1:])
    false_negative = sum(matrix[i][0] for i in range(1, 4))
    true_positive = sum(
        matrix[i][j]
        for i in range(1, 4)
        for j in range(1, 4)
    )

    fire_precision = divide(
        true_positive, true_positive + false_positive
    )
    fire_recall = divide(
        true_positive, true_positive + false_negative
    )

    detection = {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_alarms": false_positive,
        "missed_fires": false_negative,
        "false_alarm_rate": divide(
            false_positive, false_positive + true_negative
        ),
        "missed_fire_rate": divide(
            false_negative, false_negative + true_positive
        ),
        "fire_precision": fire_precision,
        "fire_recall": fire_recall,
        "fire_f1": divide(
            2 * fire_precision * fire_recall,
            fire_precision + fire_recall,
        ),
    }

    print(f"\nFalse alarms: {false_positive}")
    print(f"Missed fires: {false_negative}")
    print(f"Fire recall: {fire_recall:.2%}")

    output_dir = project_root / "outputs" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["filename", "actual", "predicted"],
        )
        writer.writeheader()
        writer.writerows(results)

    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *class_names])

        for name, row in zip(class_names, matrix):
            writer.writerow([name, *row])

    metrics = {
        "split": args.split,
        "split_sha256": file_hash(split_file),
        "images": len(results),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "accuracy": accuracy,
        "per_class": per_class,
        "macro_average": macro,
        "binary_detection": detection,
        "class_order": class_names,
        "confusion_matrix": matrix,
        "checkpoints": checkpoint_info,
        "zero_division_policy": "Undefined metrics reported as 0.",
    }

    with (output_dir / "metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print("Existing results for this split are overwritten.")


if __name__ == "__main__":
    main()