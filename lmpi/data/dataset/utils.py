from typing import Dict, Union, Optional

import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision import datasets as D
from .. import dataset as implemented_dataset


def get_datasets(
    name: str,
    root: str,
    test_size: float = 0.3,
    random_state: int = 42,
    transform: Optional = None,
    target_transform: Optional = None,
    test_transform: Optional = None,
    test_target_transform: Optional = None,
) -> Dict[str, Union[torch.utils.data.Dataset]]:
    """
    Returns
    ----------
    datasets : Dict[str, torch.utils.data.Dataset]
        torch.utils.data.Dataset for the phases (train, val, test)

    """

    if hasattr(implemented_dataset, name):
        dataset = getattr(implemented_dataset, name)
    elif hasattr(D, name):
        dataset = getattr(D, name)
    else:
        raise KeyError(f"dataset: {name} is not implemented")

    dataset_train = dataset(
        root=root,
        train=True,
        download=True,
        transform=transform,
        target_transform=target_transform
    )

    idx_train, idx_val = train_test_split(
        list(range(len(dataset_train))),
        test_size=test_size,
        random_state=random_state
    )

    dataset_val = dataset(
        root=root,
        train=True,
        download=False,
        transform=test_transform,
        target_transform=test_target_transform
    )

    dataset_train = Subset(dataset_train, idx_train)
    dataset_val = Subset(dataset_val, idx_val)

    dataset_test = dataset(
        root=root,
        train=False,
        download=True,
        transform=test_transform,
        target_transform=test_target_transform
    )

    return {
        "train": dataset_train,
        "val": dataset_val,
        "test": dataset_test,
    }
