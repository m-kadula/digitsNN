from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from mnist import MNIST


def _expand_digit_to_list(i: int) -> list:
    """Converts digit to one-hot encoded list"""
    arr = [0.] * 10
    arr[i] = 1.
    return arr


def array_to_digit(arr: NDArray) -> int:
    """Converts one-hot encoded list to digit"""
    assert arr.shape == (10,)
    max_index = -1
    max_val = float('-inf')
    for i in range(arr.shape[0]):
        if arr[i] > max_val:
            max_index = i
            max_val = arr[i]
    return max_index


class TrainingData(NamedTuple):
    """Container for training and testing data from MNIST dataset"""
    training_images: NDArray
    training_labels: NDArray
    training_labels_digits: NDArray
    testing_images: NDArray
    testing_labels: NDArray
    testing_labels_digits: NDArray


def load_mnist(directory: Path) -> TrainingData:
    """Loads MNIST dataset from directory"""
    mnist_data = MNIST(directory)
    mnist_data.gz = True
    training_images, training_labels = mnist_data.load_training()
    testing_images, testing_labels = mnist_data.load_testing()

    return TrainingData(
        training_images=np.array(training_images) / 255.,
        training_labels=np.array(list(map(_expand_digit_to_list, training_labels))),
        training_labels_digits=np.array(training_labels),
        testing_images=np.array(testing_images) / 255.,
        testing_labels=np.array(list(map(_expand_digit_to_list, testing_labels))),
        testing_labels_digits=np.array(testing_labels)
    )


def plot_images(images: NDArray, labels: Optional[list[str]] = None, shape: tuple[int, int] = (28, 28)):
    """Plot images from array of images with optional labels"""
    if len(images.shape) == 1:
        images = images[np.newaxis, :]
    fig, axs = plt.subplots(1, images.shape[0], figsize=(20, 20))
    for i in range(images.shape[0]):
        axs[i].imshow(images[i, :].reshape(shape), cmap='grey', vmin=0, vmax=1)
        if labels is not None:
            axs[i].set_title(labels[i])
    plt.show()
