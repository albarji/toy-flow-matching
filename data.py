"""Module for generating toy data for flow matching experiments."""

import numpy as np
import torch
from datasets import load_dataset
from sklearn.datasets import load_digits as sklearn_load_digits, make_moons, make_swiss_roll
from torch.utils.data import Dataset

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

def labels_dictionary(target_labels):
    """Creates a dictionary mapping each unique label in target_labels to a unique integer index.

    Arguments:
        target_labels: a list or array of labels.

    Returns:
        A dictionary mapping each unique label to a unique integer index, with None mapped to 0 for the special case of dropped labels (flow without label conditioning).
    """
    labels_dict = {None: 0}  # Add None as a special label for dropped labels (flow without label conditioning)
    labels_dict.update({label: i+1 for i, label in enumerate(sorted(set(target_labels)))})
    return labels_dict

class AbstractCouplingsDataset(Dataset):
    """Abstract PyTorch Dataset that wraps a list of couplings between source and target data distributions.

    Couplings might not be stored in memory, but generated on-the-fly to allow for larger-than-memory datasets.
    Each coupling is a tuple (src_point, tgt_point) or (src_point, tgt_point, tgt_label) if using supervised labels.

    Inheriging classes must implement the __getitem__ method to retrieve a coupling for a given index.
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
        self.labels_dict = labels_dictionary(labels_set) if labels_set is not None else None

    def __len__(self):
        return self.num_couplings
    
    @property
    def is_supervised(self):
        """Returns True if the dataset is supervised (i.e., has labels), False otherwise."""
        return self.labels_dict is not None
    
    @property
    def num_classes(self):
        """Returns the number of classes in the dataset, or None if the dataset is unsupervised."""
        return len(self.labels_dict) if self.is_supervised else None

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
            raw_labels = [coupling[2] for coupling in couplings]
            self.labels = torch.tensor([self.labels_dict[label] for label in raw_labels], dtype=torch.int)
    
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

# class IndependentDistributionsCouplingsDataset(FunctionalCouplingsDataset):
#     """A PyTorch Dataset that generates independent couplings between given source and target data distributions.

#     A given index is guaranteed to always return the same coupling.

#     Attributes:
#         source_data: numpy array of shape (n, d) containing the source data points.
#         target_data: numpy array of shape (n, d) containing the target data points.
#         target_labels: optional numpy array of shape (n,) containing the class labels for the target data.
#         num_couplings: size of the dataset, i.e., the number of independent couplings to generate (default: 10000).
#     """

#     def __init__(self, source_data, target_data, target_labels=None, num_couplings=10000):
#         """Initializes the IndependentDistributionsCouplingsDataset.

#         Arguments:
#             source_data: numpy array of shape (n, d) containing the source data points.
#             target_data: numpy array of shape (n, d) containing the target data points.
#             target_labels: optional numpy array of shape (n,) containing the class labels for the target data.
#             num_couplings: number of independent couplings to generate (default: 10000).
#             num_classes: number of classes in the dataset (default: None, unsupervised dataset).
#         """
#         self.source_data = source_data
#         self.target_data = target_data
#         self.target_labels = target_labels
#         self.source_indexes = np.random.randint(0, self.source_data.shape[0], size=num_couplings)
#         self.target_indexes = np.random.randint(0, self.target_data.shape[0], size=num_couplings)
#         self.labels_indices = np.random.randint(0, self.target_labels.shape[0], size=num_couplings) if target_labels is not None else None

#         super().__init__(
#             coupling_generator=self._coupling_generator, 
#             num_couplings=num_couplings, 
#             shape=source_data.shape[1:],
#             labels_set=set(target_labels) if target_labels is not None else None,
#         )

#     def _coupling_generator(self, idx):
#         """Generates a coupling (src_point, tgt_point) or (src_point, tgt_point, tgt_label) for a given index.

#         Arguments:
#             idx: index of the coupling to generate.
#         """
#         src_idx = self.source_indexes[idx]
#         tgt_idx = self.target_indexes[idx]
#         coupling = (self.source_data[src_idx], self.target_data[tgt_idx])
#         if self.is_supervised:
#             tgt_label = self.target_labels[tgt_idx]
#             coupling = (coupling[0], coupling[1], tgt_label)
#         return coupling

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
        tgt_labels = torch.tensor([item[2] for item in batch], dtype=torch.int)
        return src_points, tgt_points, tgt_labels
    return src_points, tgt_points
