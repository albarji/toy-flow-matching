"""Module defining the flow model architecture, training loop and evaluation functions."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class FlowMLP(nn.Module):
    """A simple MLP architecture for modeling the velocity field in flow matching."""
    def __init__(self, input_output_dim: int, hidden_dim: int, num_blocks: int):
        """Initializes the FlowMLP model.

        Args:
            input_output_dim: The dimensionality of the input and output.
            hidden_dim: The number of hidden units in each layer.
            num_blocks: The number of hidden layers (blocks) in the network.
        """
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")

        layers = []
        in_dim = input_output_dim

        for _ in range(num_blocks):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, input_output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
def train_flow_model(couplings, num_epochs=200, batch_size=2048, learning_rate=1e-3, verbose=False):
    """Trains a FlowMLP model to learn the velocity field that transforms source_data to target_data.
    
    Arguments:
        couplings: a list of tuples (src_point, tgt_point) representing the known couplings between source and target points.
        num_updates: the number of training updates to perform.
        batch_size: the number of point pairs to use in each training update.
        learning_rate: the learning rate for the optimizer.
        verbose: whether to print training progress every 100 updates.

    Returns: the trained FlowMLP model.
    """
    # Prepare training tensors from existing couplings
    src_tensor = torch.tensor(np.array([src for src, _ in couplings], dtype=np.float32))
    tgt_tensor = torch.tensor(np.array([tgt for _, tgt in couplings], dtype=np.float32))

    # Instantiate model for 2D data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowMLP(input_output_dim=src_tensor.shape[1], hidden_dim=128, num_blocks=3).to(device)
    src_tensor = src_tensor.to(device)
    tgt_tensor = tgt_tensor.to(device)

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

            # Sample t ~ Uniform[0, 1] for each pair in minibatch
            t = torch.rand(src_batch.shape[0], 1).to(device)

            # Linear interpolation x_t = (1-t) * src + t * tgt
            x_t = (1.0 - t) * src_batch + t * tgt_batch

            # Target velocity vector: tgt - src
            v_target = tgt_batch - src_batch

            # Predict and optimize
            v_pred = model(x_t)
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

def sample_independent_couplings(source_data, target_data, num_couplings):
    """Samples random independent couplings between source and target data points.
    
    Arguments:
        source_data: numpy array of shape (N, d) representing the source distribution points.
        target_data: numpy array of shape (M, d) representing the target distribution points.
        num_couplings: the number of random couplings to sample.
    Returns: a list of tuples (src_point, tgt_point) representing the sampled couplings.
    """
    src_indices = np.random.choice(source_data.shape[0], size=num_couplings, replace=True)
    tgt_indices = np.random.choice(target_data.shape[0], size=num_couplings, replace=True)
    return [(source_data[src_idx], target_data[tgt_idx]) for src_idx, tgt_idx in zip(src_indices, tgt_indices)]

def estimate_velocities(model, points):
    """Helper function to estimate velocity vectors at given points using the trained model.
    
    Arguments:
        model: an already trained flow model.
        points: numpy array of shape (N, d) representing the points at which to estimate velocities.

    Returns: numpy array of shape (N, d) representing the estimated velocity vectors.
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        return model(torch.tensor(points, dtype=torch.float32, device=device)).cpu().numpy()

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

def compute_trajectories(model, source_data, n_steps=50, batch_size=128):
    """Computes trajectories of points from the source distribution under the learned flow model.

    Arguments:
        model: the trained flow model that takes in a tensor of shape (N, d) and returns a tensor of shape (N, d) representing the velocity vectors.
        source_data: numpy array of shape (N, d) representing the source distribution points
        n_steps: the number of integration steps to use for computing trajectories.
            Fewer steps means faster but less accurate trajectories.
        batch_size: the number of points to process in each batch when estimating velocities (for GPU efficiency).

    Returns:
        A list of trajectories, where each trajectory is a list of (t, point) tuples representing the path of a point from t=0 to t=1 under the flow model.
    """
    trajectories = []
    for i in range(0, source_data.shape[0], batch_size):
        batch_points = source_data[i:i + batch_size]
        trajectories_batch = euler_integrate(batch_points, lambda x: estimate_velocities(model, x), n_steps)
        ts, points = zip(*trajectories_batch)
        for data_idx in range(len(batch_points)):
            trajectories.append([(ts[t], points[t][data_idx]) for t in range(n_steps + 1)])

    return trajectories
