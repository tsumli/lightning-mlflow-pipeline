import pytest
import torch
from PIL import Image

from lmpi.transform import transform as T


@pytest.mark.parametrize("transform", ["TrainTransform", "TestTransform"])
def test_transform(transform):
    image = Image.new("RGB", (1000, 1000))
    t = getattr(T, transform)()
    output = t(image)
    assert isinstance(output, torch.Tensor)
    assert output.size() == (3, 224, 224)
