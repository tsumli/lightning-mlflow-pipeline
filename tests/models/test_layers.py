import torch
from lmpi.models import layers


class TestLayers:
    def test_flatten(self):
        input_tensor = torch.tensor(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]]
            ]
        )
        output_tensor = layers.Flatten()(input_tensor)
        assert output_tensor.size() == input_tensor.view(2, -1).size()
        assert torch.equal(
            output_tensor,
            torch.tensor([
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12]
            ])
        )

        input_tensor = torch.randn(10, 100, 10)
        output_tensor = layers.Flatten()(input_tensor)
        assert torch.equal(output_tensor, input_tensor.view(10, -1))
