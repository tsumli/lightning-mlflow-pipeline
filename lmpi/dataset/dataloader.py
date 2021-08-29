from typing import Any

from torch.utils.data import DataLoader


def decorator_dataloader(func):
    def wrapper(*args, **kwargs) -> Any:
        ret = [
            DataLoader(
                dataset, batch_size=16, shuffle=True if i == 0 else False, num_workers=4
            )
            for i, dataset in enumerate(list(func(*args, **kwargs)))
        ]
        return ret

    return wrapper
