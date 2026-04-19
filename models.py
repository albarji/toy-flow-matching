"""Module defining the flow model architecture, training loop and evaluation functions."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class FlowMLP(nn.Module):
    """A simple MLP architecture for modeling the velocity field in flow matching, with optional embedding layer."""
    def __init__(self, input_output_dim: int, hidden_dim: int, num_blocks: int, embedding_size: int = None, num_embeddings: int = None):
        """Initializes the FlowMLP model.

        Args:
            input_output_dim: The dimensionality of the input and output.
            hidden_dim: The number of hidden units in each layer.
            num_blocks: The number of hidden layers (blocks) in the network.
            embedding_size: Size of the embedding vector (if using embedding layer).
            num_embeddings: Number of distinct elements for the embedding layer. If None, no embedding layer is used.
        """
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")

        self.has_embedding = embedding_size is not None and num_embeddings is not None
        self.embedding_size = embedding_size if self.has_embedding else 0

        if self.has_embedding:
            self.embedding = nn.Embedding(num_embeddings, embedding_size)
        else:
            self.embedding = None

        layers = []
        in_dim = input_output_dim + (self.embedding_size if self.has_embedding else 0)

        for _ in range(num_blocks):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, input_output_dim))
        self.net = nn.Sequential(*layers)

        self.labels_dict = None  # Will be set during training if using supervised labels

    def forward(self, x: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the model.

        Args:
            x: Input tensor of shape (N, input_output_dim)
            label: Optional tensor of shape (N,) with integer class indices, or None.
        """
        if self.has_embedding:
            if label is None:
                label = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)  # Use 0 as special label for no label
            emb = self.embedding(label)
            x = torch.cat([x, emb], dim=1)
        return self.net(x)
    
def labels_dictionary(target_labels):
    labels_dict = {None: 0}  # Add None as a special label for dropped labels (flow without label conditioning)
    labels_dict.update({label: i+1 for i, label in enumerate(sorted(set(target_labels)))})
    return labels_dict
    
def train_flow_model(couplings, num_epochs=200, batch_size=2048, learning_rate=1e-3, embedding_size=64, labels_drop_rate=0.1, verbose=False):
    """Trains a FlowMLP model to learn the velocity field that transforms source_data to target_data.
    
    Arguments:
        couplings: a list of tuples (src_point, tgt_point) representing the known couplings between source and target points,
            or a list of tuples (src_point, tgt_point, tgt_label) if using supervised labels.
        num_epochs: the number of training epochs over the couplings to perform.
        batch_size: the number of point pairs to use in each training update.
        learning_rate: the learning rate for the optimizer.
        embedding_size: the size of the embedding vector. Ignored if couplings do not contain labels.
        labels_drop_rate: if labels are provided, the probability of dropping them during training to allow learning both a general flow and a label-conditioned flow. Should be between 0 and 1.
        verbose: whether to print training progress every 100 updates.

    Returns: the trained FlowMLP model.
    """
    # Prepare training tensors from existing couplings
    src_tensor = torch.tensor(np.array([coupling[0] for coupling in couplings], dtype=np.float32))
    tgt_tensor = torch.tensor(np.array([coupling[1] for coupling in couplings], dtype=np.float32))
    supervised = any(len(coupling) == 3 for coupling in couplings)
    if supervised:
        raw_labels = [coupling[2] for coupling in couplings]
        labels_dict = labels_dictionary(raw_labels)
        num_labels = len(labels_dict)
        labels = torch.tensor([labels_dict[label] for label in raw_labels], dtype=torch.int)

    # Instantiate model for 2D data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowMLP(
        input_output_dim=src_tensor.shape[1],
        hidden_dim=128,
        num_blocks=3,
        embedding_size=embedding_size if supervised else None,
        num_embeddings=num_labels if supervised else None
    ).to(device)
    src_tensor = src_tensor.to(device)
    tgt_tensor = tgt_tensor.to(device)
    if supervised:
        labels = labels.to(device)
        model.labels_dict = labels_dict  # Store the labels dictionary in the model for later use

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=num_epochs)
    criterion = nn.MSELoss()

    # Train
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for i in range(0, src_tensor.shape[0], batch_size):
            src_batch = src_tensor[i:i + batch_size]
            tgt_batch = tgt_tensor[i:i + batch_size]
            labels_batch = labels[i:i + batch_size] if supervised else None

            # Drop labels with probability labels_drop_rate
            if supervised and labels_drop_rate > 0:
                mask = torch.rand(labels_batch.shape[0], device=labels_batch.device) < labels_drop_rate
                labels_batch[mask] = 0  # Use 0 special label 0 to indicate dropped labels

            # Sample t ~ Uniform[0, 1] for each pair in minibatch
            t = torch.rand(src_batch.shape[0], 1).to(device)

            # Linear interpolation x_t = (1-t) * src + t * tgt
            x_t = (1.0 - t) * src_batch + t * tgt_batch

            # Target velocity vector: tgt - src
            v_target = tgt_batch - src_batch

            # Predict and optimize
            model_inputs = (x_t, labels_batch) if supervised else (x_t,)
            v_pred = model(*model_inputs)
            loss = criterion(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * src_batch.shape[0]

        scheduler.step()
        epoch_loss /= src_tensor.shape[0]
        if verbose:
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    return model

def sample_independent_couplings(source_data, target_data, num_couplings, target_labels=None):
    """Samples random independent couplings between source and target data points.
    
    Arguments:
        source_data: numpy array of shape (N, d) representing the source distribution points.
        target_data: numpy array of shape (M, d) representing the target distribution points.
        num_couplings: the number of random couplings to sample.
        target_labels: optional numpy array of shape (M,) containing class labels for the target data points.
    Returns: a list of tuples (src_point, tgt_point) representing the sampled couplings, or a list of tuples (src_point, tgt_point, tgt_label) if target_labels are provided.
    """
    src_indices = np.random.choice(source_data.shape[0], size=num_couplings, replace=True)
    tgt_indices = np.random.choice(target_data.shape[0], size=num_couplings, replace=True)
    if target_labels is None:
        return [(source_data[src_idx], target_data[tgt_idx]) for src_idx, tgt_idx in zip(src_indices, tgt_indices)]
    else:
        return [(source_data[src_idx], target_data[tgt_idx], target_labels[tgt_idx]) for src_idx, tgt_idx in zip(src_indices, tgt_indices)]

def estimate_velocities(model, points, label=None):
    """Helper function to estimate velocity vectors at given points using the trained model.
    
    Arguments:
        model: an already trained flow model.
        points: numpy array of shape (N, d) representing the points at which to estimate velocities.
        label: optional integer representing the label for which to estimate velocities (if the model is conditional).

    Returns: numpy array of shape (N, d) representing the estimated velocity vectors.
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        modelargs = (torch.tensor(points, dtype=torch.float32, device=device),)
        if label is not None:
            modelargs += (torch.full((points.shape[0],), model.labels_dict[label], dtype=torch.long, device=device),)
        return model(*modelargs).cpu().numpy()

def euler_integrate(initial_points, velocity_fn, n_steps):
    """
    Euler integration from t=0 to t=1.

    Arguments:
        initial_points: array-like, shape (N, d), initial points at t=0
        velocity_fn: callable(points) -> velocities, function that takes in an array of shape (N, d) and returns an array of shape (N, d) representing the velocity vectors
        n_steps: int, number of integration steps
    
    Returns:
        List of (t+1, points) tuples, including t=0 and t=1.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    x = np.asarray(initial_points, dtype=np.float32).copy()
    dt = 1.0 / n_steps

    path = [(0.0, x.copy())]

    for k in range(n_steps):
        v = velocity_fn(x)
        v = np.asarray(v, dtype=np.float32)
        x = x + dt * v
        path.append(((k + 1) * dt, x.copy()))

    return path

def compute_trajectories(model, source_data, n_steps=50, batch_size=128, reverse=False, label=None):
    """Computes trajectories of points from the source distribution under the learned flow model.

    Arguments:
        model: the trained flow model that takes in a tensor of shape (N, d) and returns a tensor of shape (N, d) representing the velocity vectors.
        source_data: numpy array of shape (N, d) representing the source distribution points
        n_steps: the number of integration steps to use for computing trajectories.
            Fewer steps means faster but less accurate trajectories.
        batch_size: the number of points to process in each batch when estimating velocities (for GPU efficiency).
        reverse: if True, integrates backward from target to source instead of forward from source to target.
        label: optional integer representing the label for which to estimate velocities (if the model is conditional).

    Returns:
        A list of trajectories, where each trajectory is a list of (t, point) tuples representing the path of a point from t=0 to t=1 under the flow model.
    """
    trajectories = []
    for i in range(0, source_data.shape[0], batch_size):
        batch_points = source_data[i:i + batch_size]
        model_kwargs = {}
        if label is not None:
            model_kwargs['label'] = label
        if reverse:
            trajectories_batch = euler_integrate(batch_points, lambda x: -estimate_velocities(model, x, **model_kwargs), n_steps)
        else:
            trajectories_batch = euler_integrate(batch_points, lambda x: estimate_velocities(model, x, **model_kwargs), n_steps)
        ts, points = zip(*trajectories_batch)
        for data_idx in range(len(batch_points)):
            trajectories.append([(ts[t], points[t][data_idx]) for t in range(n_steps + 1)])

    return trajectories
