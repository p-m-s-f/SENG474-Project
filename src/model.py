import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(num_classes):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


if __name__ == "__main__":
    detector = build_model(2)
    risk_model = build_model(3)

    print("Detector:", detector.fc)
    print("Risk model:", risk_model.fc)