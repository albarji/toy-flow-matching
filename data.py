"""Module for generating toy data for flow matching experiments."""

import numpy as np
from datasets import load_dataset
from sklearn.datasets import load_digits as sklearn_load_digits, make_moons, make_swiss_roll

def generate_two_gaussians(n=1000, supervised=False):
    """Generates a toy dataset consisting of two well-separated Gaussian clusters.

    Arguments:
        n: total number of data points to generate (default: 1000). The dataset will consist of n/2 points from each Gaussian cluster.
        supervised: whether to generate labels for the Gaussian clusters (default: False).

    Returns:
        If supervised is False, a numpy array of shape (n, 2) containing the generated data points.
        If supervised is True, a tuple (data, labels) where data is a numpy array of shape (n, 2) and labels is a numpy array of shape (n,) containing the class labels.
    """
    data = np.vstack([
        np.random.multivariate_normal([5, 5], np.array([[1, -0.75], [-0.75, 1]]), n // 2),
        np.random.multivariate_normal([-5, -5], np.array([[1, -0.75], [-0.75, 1]]), n // 2)
    ])
    perm = np.random.permutation(n)
    data = data[perm]
    if supervised:
        labels = np.array([0] * (n // 2) + [1] * (n // 2))
        labels = labels[perm]
        return data, labels
    return data

def generate_swiss_roll(n=1000):
    """Generates a toy dataset in the shape of a 2D Swiss roll.

    Arguments:
        n: number of data points to generate (default: 1000).

    Returns:
        A numpy array of shape (n, 2) containing the generated data points.
    """
    x, _ = make_swiss_roll(n_samples=n, noise=0.5)
    # Make two-dimensional
    x = x[:, [0, 2]]

    x = (x - x.mean()) / x.std()
    return x

def generate_two_moons(n=1000):
    """Generates a toy dataset in the shape of two interleaving moons.

    Arguments:
        n: number of data points to generate (default: 1000).
    Returns:
        A numpy array of shape (n, 2) containing the generated data points.
    """
    x, _ = make_moons(n_samples=n, noise=0.1)
    x = (x - x.mean()) / x.std()
    return x

def generate_toy_data(dataset_type, n=1000):
    """Generates toy datasets for flow matching experiments.

    Arguments:
        n: number of data points to generate (default: 1000).
        dataset_type: type of dataset to generate. Can be "two_gaussians" for two well-separated Gaussian clusters, "swiss_roll" for a 2D Swiss roll shape, or "two_moons" for two interleaving moons.

    Returns:
        A numpy array of shape (n, 2) containing the generated data points.
    """
    if dataset_type == "two_gaussians":
        return generate_two_gaussians(n)
    elif dataset_type == "two_gaussians_supervised":
        return generate_two_gaussians(n, supervised=True)
    elif dataset_type == "swiss_roll":
        return generate_swiss_roll(n)
    elif dataset_type == "two_moons":
        return generate_two_moons(n)
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

def load_banana():
    """Loads the banana-shaped dataset from a CSV file.

    Returns:
        A numpy array of shape (n, 2) containing the data points from the banana dataset.
        A numpy array of shape (n,) containing the class labels for the banana dataset.
    """
    data = np.loadtxt("datasets/banana.csv", delimiter=",")
    return data[:, :2], data[:, 2].astype(int)

def load_digits():
    """Loads the digits dataset from sklearn.

    Returns:
        A numpy array of shape (n, 8, 8) containing the digit images.
        A numpy array of shape (n,) containing the class labels for the digits.
    """
    data = sklearn_load_digits()
    target_data = data.images
    target_data /= target_data.max()  # Normalize pixel values to [0, 1]
    target_labels = data.target
    # Shuffle data and labels together
    perm = np.random.permutation(len(target_data))
    return target_data[perm], target_labels[perm]

def load_mnist():
    """Loads the MNIST dataset from Hugging Face datasets.

    Returns:
        A numpy array of shape (n, 28, 28) containing the MNIST images.
        A numpy array of shape (n,) containing the class labels for the MNIST dataset.
    """
    ds = load_dataset("ylecun/mnist")
    train_ds = ds["train"].with_format("numpy").map(lambda x: {"image": x["image"].astype("float32") / 255.0, "label": x["label"]}, batched=True)
    return train_ds[:]["image"], train_ds[:]["label"]
