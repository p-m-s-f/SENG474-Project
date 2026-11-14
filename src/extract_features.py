import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from pathlib import Path
from dataset import WildfireDataset
from torch.utils.data import ConcatDataset, DataLoader
from predict import get_device


model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Identity()
model.eval()

project_root = Path(__file__).resolve().parent.parent

train_dataset = WildfireDataset(project_root / "splits" / "train.csv")
val_dataset = WildfireDataset(project_root / "splits" / "val.csv")
dataset = ConcatDataset([train_dataset, val_dataset])
records = train_dataset.images + val_dataset.images

device = get_device()
model = model.to(device)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

batches = []
with torch.inference_mode():
    for batch_number, (images, _) in enumerate(loader, start=1):
        batches.append(model(images.to(device)).cpu())
        if batch_number % 10 == 0:
            print(f"Processed {batch_number}/{len(loader)} batches")

features = torch.cat(batches, dim=0)
print(features.shape)

output_path = project_root / "outputs" / "development_features.pt"
output_path.parent.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "features": features,
        "filenames": [record["filename"] for record in records],
        "labels": [record["label"] for record in records],
    },
    output_path,
)

print(f"Saved features to {output_path}")
