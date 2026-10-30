import csv
from pathlib import Path
from PIL import Image
from torchvision import transforms

project_root = Path(__file__).resolve().parent.parent
label_file = project_root / "image_labels.csv"

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

with label_file.open("r", newline="", encoding="utf-8") as file: 
    line = csv.DictReader(file)
    first_image = next(line)
    #print(first_image)

    image_path = project_root/ "data" / first_image["label"] / first_image["filename"]
    print(image_path)
    print(image_path.is_file())

    with Image.open(image_path) as image:
        image_tensor = transform(image.convert("RGB"))
        print(image_tensor.shape)

