from .transform import TestTransform, TrainTransform


def get_transform():
    return {
        "transform": TrainTransform(),
        "target_transform": None,
        "test_transform": TestTransform(),
        "test_target_transform": None,
    }
