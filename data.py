"""Module for generating toy data for flow matching experiments."""

import numpy as np
import torch
from sklearn.datasets import load_digits as sklearn_load_digits, make_moons, make_swiss_roll
from torch.utils.data import Dataset
from torchvision import datasets, transforms

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
    target_data = target_data[perm]
    target_labels = target_labels[perm]
    return [(torch.tensor(target_data[i], dtype=torch.float32), target_labels[i]) for i in range(len(target_data))], set(target_labels)

def load_mnist():
    """Loads the MNIST dataset from torchvision.

    Returns:
        - A pytorch Dataset containing the MNIST images and labels.
            Images are 28x28 with pixel values normalized to [0, 1] and labels are integers from 0 to 9.
        - The number of classes in the dataset (10 for MNIST).
    """
    dataset = datasets.MNIST(
        root="/tmp/mnist_data",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    labels_set = set(range(10))  # MNIST has 10 classes (digits 0-9)
    return dataset, labels_set

class AbstractCouplingsDataset(Dataset):
    """Abstract PyTorch Dataset that wraps a list of couplings between source and target data distributions.

    Couplings might not be stored in memory, but generated on-the-fly to allow for larger-than-memory datasets.
    Each coupling is a tuple (src_point, tgt_point) or (src_point, tgt_point, tgt_label) if using supervised labels.

    Inheriting classes must implement the __getitem__ method to retrieve a coupling for a given index.
    Optionally they can implement the __getitems__ method to retrieve multiple couplings for a given list of indices, which can be more efficient than calling __getitem__ multiple times.
    """

    def __init__(self, shape, num_couplings, labels_set=None):
        """Initializes the AbstractCouplingsDataset.

        Arguments:
            shape: shape of the source and target data points.
            num_couplings: number of couplings to generate.
            labels_set: set of labels in the dataset (default: None, unsupervised dataset).
        """
        self.num_couplings = num_couplings
        self.shape = shape
        self.labels_set = labels_set

    def __len__(self):
        return self.num_couplings
    
    @property
    def is_supervised(self):
        """Returns True if the dataset is supervised (i.e., has labels), False otherwise."""
        return self.labels_set is not None
    
    @property
    def num_classes(self):
        """Returns the number of classes in the dataset, or None if the dataset is unsupervised."""
        return len(self.labels_set) if self.is_supervised else None

class FixedCouplingsDataset(AbstractCouplingsDataset):
    """A PyTorch Dataset that wraps a fixed list of couplings between source and target data distributions."""

    def __init__(self, couplings):
        """Initializes the FixedCouplingsDataset.

        Arguments:
            couplings: a list of tuples (src_point, tgt_point) representing the known couplings between source and target points,
                or a list of tuples (src_point, tgt_point, tgt_label) if using supervised labels.
        """
        supervised = any(len(coupling) == 3 for coupling in couplings)
        labels_set = set([coupling[2] for coupling in couplings]) if supervised else None

        super().__init__(
            num_couplings=len(couplings), 
            shape=couplings[0][0].shape,
            labels_set=labels_set
        )

        # Pre-Transform given couplings into PyTorch float32 tensors
        self.src_tensor = torch.tensor(np.array([coupling[0] for coupling in couplings], dtype=np.float32))
        self.tgt_tensor = torch.tensor(np.array([coupling[1] for coupling in couplings], dtype=np.float32))
        if supervised:
            self.labels = np.array([coupling[2] for coupling in couplings])
    
    def __getitem__(self, idx):
        """Retrieves a coupling (src_point, tgt_point) or (src_point, tgt_point, tgt_label) for a given index.

        Arguments:
            idx: index of the coupling to retrieve.
        """
        return self.__getitems__([idx])[0]  # Return the first element of the tuple returned by __getitems__
    
    def __getitems__(self, indices):
        """Optimized retrieval of multiple couplings for a given list of indices.

        Arguments:
            indices: a list of indices for which to retrieve the couplings.
        """
        src_point = self.src_tensor[indices]
        tgt_point = self.tgt_tensor[indices]
        if self.is_supervised:
            tgt_label = self.labels[indices]
            return src_point, tgt_point, tgt_label
        return src_point, tgt_point

class IndependentDistributionsCouplingsDataset(AbstractCouplingsDataset):
    """A PyTorch Dataset that generates independent couplings between a given target dataset and random samples.

    Querying the same index multiple times will yield different couplings.

    Attributes:
        target_data: Dataset containing the target data points.
        target_labels: optional numpy array of shape (n,) containing the class labels for the target data.
        num_couplings: size of the dataset, i.e., the number of independent couplings to generate (default: 10000).
    """

    def __init__(self, target_dataset, num_couplings, labels_set=None, source_generator=None):
        """Initializes the IndependentDistributionsCouplingsDataset.

        Arguments:
            target_dataset: Dataset containing the target data points.
            labels_set: optional set containing the class labels for the target data.
            num_couplings: number of independent couplings to generate.
            source_generator: optional function to generate source data points. If not provided, use a standard normal distribution generator.
        """
        self.target_dataset = target_dataset
        self.labels_set = labels_set
        self.shape = target_dataset[0][0].shape if isinstance(target_dataset[0], tuple) else target_dataset[0].shape
        self.source_generator = source_generator if source_generator is not None else lambda shape: torch.randn(*shape)

        super().__init__(
            self.shape,
            num_couplings,
            labels_set=labels_set
        )

    def __getitem__(self, _idx):
        """Retrieves a coupling (src_point, tgt_point) or (src_point, tgt_point, tgt_label) for a given index.

        Arguments:
            _idx: index of the coupling to retrieve. Ignored since couplings are generated independently and randomly.
        """
        source = self.source_generator(self.shape)
        target_idx = np.random.choice(len(self.target_dataset))
        target = self.target_dataset[target_idx][0]
        if self.is_supervised:
            target_label = self.target_dataset[target_idx][1]
            return source, target, target_label
        return source, target

def couplings_collate_fn(batch):
    """Collate function for batching couplings in a PyTorch DataLoader.

    Arguments:
        batch: a list of couplings, where each coupling is a tuple (src_point, tgt_point) or (src_point, tgt_point, tgt_label).

    Returns:
        A tuple of batched source points, target points, and optionally target labels if the dataset is supervised.
    """
    # If the batch is not a list (e.g., the Dataset already returns a batch), return it as is
    if not isinstance(batch, list):
        return batch
    
    src_points = torch.stack([item[0] for item in batch])
    tgt_points = torch.stack([item[1] for item in batch])
    if len(batch[0]) == 3:  # Supervised case
        tgt_labels = np.array([item[2] for item in batch])
        return src_points, tgt_points, tgt_labels
    return src_points, tgt_points
