import torch.nn as nn
from torch import Tensor


class Flatten(nn.Module):
    def forward(self, x: Tensor):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
