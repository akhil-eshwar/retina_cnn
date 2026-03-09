import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def get_model():

    model = efficientnet_b0(
        weights=EfficientNet_B0_Weights.DEFAULT
    )

    for param in model.features.parameters():
        param.requires_grad = False


    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        5
    )

    return model