"""Loading and indexing CIFAR-10-C corruption files."""

import numpy as np

from src.config import IMAGES_PER_LEVEL


def load_corruption(path):
    """Load a CIFAR-10-C corruption file, e.g. 'data/defocus_blur.npy'."""
    return np.load(path)


def load_labels(path):
    """Load the CIFAR-10 ground-truth labels, e.g. 'data/labels.npy'."""
    return np.load(path)


def indices_for_label(labels, label_int):
    """Indices (into the base 10,000 test images) whose ground-truth label matches."""
    return np.where(labels == label_int)[0]


def get_image(corruption_array, base_image_index, severity):
    """Return the image array for a given base image at a given severity (1-5).

    CIFAR-10-C stores 50,000 images per corruption file: the same 10,000 base
    images repeated at 5 severity levels, offset by (severity - 1) * IMAGES_PER_LEVEL.
    """
    index = (severity - 1) * IMAGES_PER_LEVEL + base_image_index
    return corruption_array[index]
