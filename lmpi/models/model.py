import torch.nn as nn
import torchvision.models as models

from .layers import Flatten


def encoder():
    enc = models.resnet34(pretrained=False, progress=True)
    return nn.Sequential(
        *list(enc.children())[:-1],
    )


def decoder(output: int):
    return nn.Sequential(Flatten(), nn.Linear(512, output, bias=True))
