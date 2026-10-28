import csv
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"
output = project_root / "image_labels.csv"

categories = ["no_fire", "fire_low", "fire_medium", "fire_high"]
image_types = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main():
    labels = []
    filenames = set()

    for category in categories:
        folder = data_dir / category

        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")

        count = 0

        for image in sorted(folder.iterdir()):
            if not image.is_file() or image.suffix.lower() not in image_types:
                continue

            if image.name in filenames:
                raise ValueError(f"Image listed twice: {image.name}")

            filenames.add(image.name)
            labels.append([image.name, category])
            count += 1

        print(f"{category}: {count} images")

    if not labels:
        raise ValueError("No images found")

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "label"])
        writer.writerows(labels)

    print(f"Exported {len(labels)} labels to {output.name}")


if __name__ == "__main__":
    main()