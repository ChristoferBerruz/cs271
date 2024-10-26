import pytest

from mlproject.models import CNN2D


class TestCNN2DModel:

    def test_train(self, training_dataset_img2d, testing_dataset_img2d):
        cnn1d = CNN2D(input_dim=100, output_dim=2)
        cnn1d.train(
            training_dataset_img2d,
            testing_dataset_img2d
        )
