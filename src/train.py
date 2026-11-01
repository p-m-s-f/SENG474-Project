import random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import WildfireDataset
from model import build_model


def main():
    random.seed(0)
    torch.manual_seed(0)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Training device: {device}")


    project_root = Path(__file__).resolve().parent.parent

    train_dataset = WildfireDataset(
        project_root / "splits" / "train.csv",
        stage="detector",
    )
    val_dataset = WildfireDataset(
        project_root / "splits" / "val.csv",
        stage="detector",
    )

    generator = torch.Generator().manual_seed(0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    model = build_model(2).to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
    )

    print("Detector ready for training")

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_images = 0

    for batch_number, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        scores = model(images)
        loss = loss_function(scores, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = scores.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_images += images.size(0)

        if batch_number % 10 == 0:
            print(f"Completed batch {batch_number}/{len(train_loader)}")

    print(f"Training loss: {total_loss / total_images:.4f}")
    print(f"Training accuracy: {total_correct / total_images:.2%}")

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_images = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            scores = model(images)
            loss = loss_function(scores, labels)

            val_loss += loss.item() * images.size(0)
            predictions = scores.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()
            val_images += images.size(0)

    print(f"Validation loss: {val_loss / val_images:.4f}")
    print(f"Validation accuracy: {val_correct / val_images:.2%}")

if __name__ == "__main__":
    main()



