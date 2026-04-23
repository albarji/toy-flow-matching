"""Module with functions for embedding high-dimensional data into 2D for visualization."""

import numpy as np

from sklearn.manifold import TSNE


def _check_dimensions(*datasets):
    """Checks that all input datasets have the same feature dimensions.
    
    Arguments:
        datasets: variable number of data elements, which can be:
            - A numpy array of shape (*, d1, d2, ...) containing the data points to be embedded.
            - A list of trajectories, where each trajectory is a list of (t, point) tuples representing the path of a point from t=0 to t=1 under the flow model.
                Each point in the trajectory should be a numpy array of shape (d1, d2, ...), and all points across all trajectories must have the same feature dimensions as the datasets.
            All dimensions but the first (n) are considered feature dimensions and will be flattened, and must be equal in all datasets.

    Returns:
        The common feature shape of the input datasets for reference.
    """
    feature_shapes = []
    for dataset in datasets:
        if isinstance(dataset, list):  # Check if it's a list of trajectories
            for traj in dataset:
                for _, point in traj:
                    feature_shapes.append(point.shape)
        else:  # It's a regular dataset
            feature_shapes.append(dataset.shape[1:])
    if len(set(feature_shapes)) > 1:
        raise ValueError("All input datasets must have the same feature dimensions, but got shapes: {}".format(feature_shapes))    
    
    return feature_shapes[0]  # Return the common feature shape for reference
        
def _merge_datasets(*datasets):
    """Merges into a single numpy array all the data points from the input datasets and trajectory points, flattening their feature dimensions.
    
    Arguments:
        datasets: variable number of data elements, which can be:
            - A numpy array of shape (*, d1, d2, ...) containing the data points to be embedded.
            - A list of trajectories, where each trajectory is a list of (t, point) tuples representing the path of a point from t=0 to t=1 under the flow model.
                Each point in the trajectory should be a numpy array of shape (d1, d2, ...), and all points across all trajectories must have the same feature dimensions as the datasets.
            All dimensions but the first (n) are considered feature dimensions and will be flattened, and must be equal in all datasets.

    Returns:
        A single numpy array of shape (total_points, d_flat) containing all the flattened data points from the input datasets and trajectories.
    """
    flattened_datasets = []
    for dataset in datasets:
        if isinstance(dataset, list):  # Check if it's a list of trajectories
            flattened_datasets += [point.reshape(-1) for traj in dataset for _, point in traj]
        else:  # It's a regular dataset
            flattened_datasets += [dataset.reshape(dataset.shape[0], -1)]

    return np.vstack(flattened_datasets)

def _split_merged_points(merged_points, *datasets):
    """Splits the merged and flattened data back into a format compatible with their original separate datasets.
    
    Arguments:
        merged_points: A numpy array of shape (total_points, d_flat) containing all the flattened data points from the input datasets and trajectories.
        datasets: The original input datasets, which can be numpy arrays or lists of trajectories.

    Returns:
        A list of datasets in the same format as the input, with points reshaped to their original dimensions.
    """
    index = 0
    split_datasets = []
    for dataset in datasets:
        if isinstance(dataset, list):  # Check if it's a list of trajectories
            new_trajectories = []
            for traj in dataset:
                new_trajectory = []
                for t, _ in traj:
                    new_trajectory.append((t, merged_points[index]))
                    index += 1
                new_trajectories.append(new_trajectory)
            split_datasets.append(new_trajectories)
        else:  # It's a regular dataset
            num_points = dataset.shape[0]
            split_datasets.append(merged_points[index:index+num_points])
            index += num_points

    return split_datasets

def embed_data(*datasets, random_state=42, perplexity=30):
    """Embeds high-dimensional datasets into 2D using t-SNE for visualization.

    Arguments:
        datasets: a list of data elements, which can be:
            - A numpy array of shape (*, d1, d2, ...) containing the data points to be embedded.
            - A list of trajectories, where each trajectory is a list of (t, point) tuples representing the path of a point from t=0 to t=1 under the flow model.
                Each point in the trajectory should be a numpy array of shape (d1, d2, ...), and all points across all trajectories must have the same feature dimensions as the datasets.
            All dimensions but the first (n) are considered feature dimensions and will be flattened, and must be equal in all datasets.
        random_state: random seed for reproducibility (default: 42).
        perplexity: perplexity parameter for t-SNE (default: 30).
    Returns:
        A list of numpy arrays, each of shape (*, 2) containing the 2D embeddings of the input data points, in the same order as the input datasets.
        If trajectories are provided, also returns a list of trajectories with points transformed into 2D.
    """
    _check_dimensions(*datasets)
    all_points = _merge_datasets(*datasets)

    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity
    )

    combined_2d = tsne.fit_transform(all_points)

    return _split_merged_points(combined_2d, *datasets)
