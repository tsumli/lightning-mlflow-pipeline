import os
from typing import Tuple

import torch
from dataset.dataloader import decorator_dataloader
from torch.utils.data import Dataset, random_split
from torchvision.datasets import MNIST
from transforms.train_test_transform import TestTransform, TrainTransform


class MNISTDatasetTransformed(Dataset):
    def __init__(self, dataset: Dataset, train: bool) -> None:
        self.images = [i[0] for i in dataset]
        self.labels = [i[1] for i in dataset]
        self.trans = TrainTransform() if train else TestTransform()

    def __getitem__(self, idx) -> Tuple[torch.Tensor, float]:
        out_image = self.trans(self.images[idx])
        out_label = self.labels[idx]
        return out_image, out_label

    def __len__(self) -> int:
        return len(self.labels)


@decorator_dataloader
def MNIST_datasets(train_ratio: float = 0.7) -> Tuple[Dataset, ...]:
    """MNIST dataset (train, val, test) transformed by
    TrainTransform, TestTransform

    Parameters
    ----------
    train_ratio : float
        Ratio to split data to train-val.
        train = data * `train_ratio`

    Returns
    ----------
    datasets : tuple[torch.utils.data.Dataset]
        torch.utils.data.Dataset for the phases (train, val, test)

    """
    dataset_train = MNIST(root=os.path.abspath("."), train=True, download=True)
    # uncomment if you use full MNIST dataset
    dataset_train, _ = random_split(dataset_train, [int(len(dataset_train) * 0.01), len(dataset_train) - int(len(dataset_train) * 0.01)])
    len_train = int(len(dataset_train) * train_ratio)
    len_val = len(dataset_train) - len_train
    train, val = random_split(dataset_train, [len_train, len_val])

    test = MNIST(root=os.path.abspath("."), train=False, download=True)

    datasets = (
        MNISTDatasetTransformed(train, True),
        MNISTDatasetTransformed(val, False),
        MNISTDatasetTransformed(test, False),
    )
    return datasets
