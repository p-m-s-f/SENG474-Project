import csv
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
label_file = project_root / "image_labels.csv"

with label_file.open("r", newline="", encoding="utf-8") as file: 
    line = csv.DictReader(file)
    first_image = next(line)
    print(first_image)

