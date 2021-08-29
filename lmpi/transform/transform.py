import torch
from PIL import Image
from torchvision import transforms


class TrainTransform:
    def __init__(self) -> None:
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, (1.0, 1.0), (1.0, 1.0)),
                transforms.RandomResizedCrop(224),
                transforms.ColorJitter(0.5, 0.5, 0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __call__(self, img: Image) -> torch.Tensor:
        img = img.convert("RGB")
        transformed_img: torch.Tensor = self.transform(img)
        return transformed_img


class TestTransform:
    def __init__(self) -> None:
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, (1.0, 1.0), (1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __call__(self, img: Image) -> torch.Tensor:
        img = img.convert("RGB")
        transformed_img: torch.Tensor = self.transform(img)
        return transformed_img
