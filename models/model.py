import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def encoder():
    enc = models.resnet34(pretrained=False, progress=True)
    return nn.Sequential(
        *list(enc.children())[:-1],
    )


def decoder(output: int):
    return nn.Sequential(Flatten(), nn.Linear(512, output, bias=True))


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
