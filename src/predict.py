import argparse
from pathlib import Path
import torch
from PIL import Image
from dataset import transform
from model import build_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models(project_root, device):
    models = {}

    for stage, num_classes in [("detector", 2), ("risk", 3)]:
        checkpoint = torch.load(
            project_root / "checkpoints" / f"{stage}_best.pt",
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint["stage"] != stage:
            raise ValueError(f"Wrong checkpoint for {stage}")

        model = build_model(num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        models[stage] = model

    return models["detector"], models["risk"]


def predict_image(image_path, detector, risk_model, device):
    with Image.open(image_path) as image:
        image_tensor = transform(image.convert("RGB"))

    image_batch = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        detector_prediction = detector(image_batch).argmax(dim=1).item()

        if detector_prediction == 0:
            return "no_fire"

        risk_prediction = risk_model(image_batch).argmax(dim=1).item()
        risk_names = ["fire_low", "fire_medium", "fire_high"]

        return risk_names[risk_prediction]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    device = get_device()
    image_path = Path(args.image)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    detector, risk_model = load_models(project_root, device)
    prediction = predict_image(image_path, detector, risk_model, device)

    print(f"Device: {device}")
    print(f"Image: {image_path}")
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()