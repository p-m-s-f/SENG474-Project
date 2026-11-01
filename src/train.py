import random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import WildfireDataset
from model import build_model
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["detector", "risk"],
        default="detector",
    )
    args = parser.parse_args()

    stage = args.stage

    if stage == "detector":
        class_names = ["no_fire", "fire"]
    else:
        class_names = ["fire_low", "fire_medium", "fire_high"]

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
        stage=stage,
    )
    val_dataset = WildfireDataset(
        project_root / "splits" / "val.csv",
        stage=stage,
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

    model = build_model(len(class_names)).to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
    )

    print(f"{stage.capitalize()} ready for training")

    epochs = 5
    best_val_loss = float("inf")

    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

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

        average_val_loss = val_loss / val_images

        print(f"Validation loss: {average_val_loss:.4f}")
        print(f"Validation accuracy: {val_correct / val_images:.2%}")

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss

            torch.save(
                {
                    "model_state_dict": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                    "stage": stage,
                    "num_classes": len(class_names),
                    "class_names": class_names,
                    "epoch": epoch + 1,
                    "val_loss": average_val_loss,
                    "seed": 0,
                },
                checkpoint_dir / f"{stage}_best.pt",
            )
            print(f"Saved best {stage} checkpoint")

if __name__ == "__main__":
    main()



