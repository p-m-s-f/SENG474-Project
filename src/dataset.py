import csv
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

project_root = Path(__file__).resolve().parent.parent
label_file = project_root / "image_labels.csv"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


detector_labels = {
    "no_fire": 0,
    "fire_low": 1,
    "fire_medium": 1,
    "fire_high": 1,
}

risk_labels = {
    "fire_low": 0,
    "fire_medium": 1,
    "fire_high": 2,
}

class WildfireDataset(Dataset):

    def __init__(self, csv_file=label_file, stage="detector"):
        if stage not in ("detector", "risk"):
            raise ValueError("stage must be 'detector' or  'risk'")

        self.stage = stage
        
        with Path(csv_file).open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            self.images = list(reader)

        if self.stage == "risk":
            self.images = [
                record for record in self.images
                if record["label"] != "no_fire"
            ]

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
        if self.stage == "detector":
            label = detector_labels[record["label"]]
        else:
            label = risk_labels[record["label"]]
        return image_tensor, label


if __name__ == "__main__":
    dataset = WildfireDataset(project_root / "splits" / "train.csv")
    print(f"Dataset size: {len(dataset)}")

    image_tensor, label = dataset[700]
    print(image_tensor.shape)
    print(label)

    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    batch_images, batch_labels = next(iter(loader))

    print(batch_images.shape)
    print(batch_labels)

    for split in ["train", "val", "test"]:
        split_dataset = WildfireDataset(project_root / "splits" / f"{split}.csv")
        image_tensor, label = split_dataset[0]
        print(f"{split}: {len(split_dataset)} images, sample shape: {image_tensor.shape}")

    risk_dataset = WildfireDataset(
        project_root / "splits" / "train.csv",
        stage="risk",
    )

    print(f"Risk training images: {len(risk_dataset)}")
    print(f"Risk labels: {sorted({
        risk_labels[record['label']]
        for record in risk_dataset.images
    })}")

    risk_loader = DataLoader(risk_dataset, batch_size=4, shuffle=False)
    risk_images, risk_targets = next(iter(risk_loader))

    print(risk_images.shape)
    print(risk_targets)