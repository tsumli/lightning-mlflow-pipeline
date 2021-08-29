import os

from torch.utils.data import Subset
from torchvision.datasets import MNIST


def TrialMNISTDatasets():
    dataset_train = MNIST(root=os.path.abspath("."), train=True, download=True)

    train, val = Subset(dataset_train, range(100)), Subset(dataset_train, range(100, 200))

    test = MNIST(root=os.path.abspath("."), train=False, download=True)
    test = Subset(test, range(100))
    return {
        "train": train,
        "val": val,
        "test": test
    }
