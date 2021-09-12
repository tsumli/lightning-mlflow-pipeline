from torch.utils.data import DataLoader, Subset
from omegaconf import DictConfig

from lmpi.data.dataset.utils import get_datasets
from lmpi.transform.utils import get_transform


def make_dataloader(config: DictConfig):
    datasets = get_datasets(
        name=config.dataset.name,
        root=config.dataset.root,
        test_size=config.dataset.test_size,
        random_state=config.dataset.random_state,
        **get_transform(),
    )

    train_dataloader = DataLoader(
        Subset(datasets["train"], range(20)),
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        Subset(datasets["val"], range(20)),
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        num_workers=4,
    )
    test_dataloader = DataLoader(
        Subset(datasets["test"], range(20)),
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        num_workers=4,
    )

    return {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
    }
