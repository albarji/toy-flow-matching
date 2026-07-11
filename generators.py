"""Module for Datasets that are generated on the fly using a flow model."""

import numpy as np
import torch

from data import AbstractCouplingsDataset
from models import compute_trajectories


class FlowCouplingsDataset(AbstractCouplingsDataset):
    """A PyTorch Dataset that generates couplings between a source distribution generator samples produced by using a flow model."""

    def __init__(self, flow_model, shape, num_couplings, labels_set=None, source_generator=None, simulation_arguments=None):
        """Initializes the FlowCouplingsDataset.

        Arguments:
            flow_model: a trained flow model that can generate samples from the source distribution.
            shape: shape of the source and target data points.
            num_couplings: number of couplings to generate.
            labels_set: optional set containing the class labels for the target data.
            source_generator: optional function to generate source data points. If not provided, use a standard normal distribution generator.
            simulation_arguments: optional dictionary containing additional arguments for the simulation of the flow model.
        """
        self.flow_model = flow_model
        self.shape = shape
        self.labels_set = labels_set
        self.source_generator = source_generator if source_generator is not None else lambda shape: torch.randn(*shape)
        self.simulation_arguments = simulation_arguments if simulation_arguments is not None else {}

        super().__init__(
            self.shape,
            num_couplings,
            labels_set=labels_set
        )

    def __getitem__(self, idx):
        """Retrieves a coupling (src_point, tgt_point) or (src_point, tgt_point, tgt_label) for a given index.

        Arguments:
            idx: index of the coupling to retrieve.
        """
        return [e[0] for e in self.__getitems__([idx])]
    
    def __getitems__(self, indices):
        """Optimized retrieval of multiple couplings for a given list of indices.

        Arguments:
            indices: a list of indices for which to retrieve the couplings.
        """
        n = len(indices)
        src_samples = self.source_generator((n,) + self.shape)
        simulation_args = self.simulation_arguments.copy()
        if self.is_supervised:
            # If the dataset is supervised, we need to generate labels for the source samples
            src_labels = np.random.choice(list(self.labels_set), size=n)
            simulation_args['labels'] = src_labels
        trajectories = compute_trajectories(self.flow_model, src_samples, **simulation_args)
        tgt_samples = torch.tensor(np.array([traj[-1][1] for traj in trajectories]))  # Get the final points in the trajectories
        if self.is_supervised:
            return src_samples, tgt_samples, torch.tensor(src_labels, dtype=torch.int)
        return src_samples, tgt_samples
        
class FlowOutputsDataset(FlowCouplingsDataset):
    """A PyTorch Dataset that generates outputs from a flow model given source inputs."""

    def __getitems__(self, indices):
        """Optimized retrieval of multiple couplings for a given list of indices.

        Arguments:
            indices: a list of indices for which to retrieve the couplings.
        """
        generated = super().__getitems__(indices)
        if self.is_supervised:
            return generated[1], generated[2]
        else:
            return generated[1]
