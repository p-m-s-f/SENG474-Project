import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import WildfireDataset


def build_model(num_classes):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    train_csv = project_root / "splits" / "train.csv"

    for stage, num_classes in [("detector", 2), ("risk", 3)]:
        dataset = WildfireDataset(train_csv, stage=stage)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        images, labels = next(iter(loader))

        model = build_model(num_classes)
        model.eval()

        with torch.no_grad():
            scores = model(images)

        print(f"{stage}: {scores.shape}")