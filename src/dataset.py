import csv
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

project_root = Path(__file__).resolve().parent.parent
label_file = project_root / "image_labels.csv"

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


label_numbers = {
    "no_fire": 0,
    "fire_low": 1,
    "fire_medium": 2,
    "fire_high": 3,
}

class WildfireDataset(Dataset):

    def __init__(self):
        with label_file.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            self.images = list(reader)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        record = self.images[index]
        image_path = project_root/ "data" / record["label"] / record["filename"]
        #print(image_path)
        #print(image_path.is_file())

        with Image.open(image_path) as image:
            image_tensor = transform(image.convert("RGB"))
            #print(image_tensor.shape)
        label = label_numbers[record["label"]]
        return image_tensor, label


if __name__ == "__main__":
    dataset = WildfireDataset()
    print(f"Dataset size: {len(dataset)}")

    image_tensor, label = dataset[700]
    print(image_tensor.shape)
    print(label)