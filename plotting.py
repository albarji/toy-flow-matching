"""Module with fucntions for creating visualizations and animations"""

import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go

from models import estimate_velocities

def data_ranges(*datasets, padding=1.0):
    """Helper function to compute the overall data ranges for multiple datasets, with optional padding."""
    all_data = np.vstack(datasets)
    x_min, y_min = all_data.min(axis=0) - padding
    x_max, y_max = all_data.max(axis=0) + padding
    return (x_min, x_max), (y_min, y_max)

def plot_distributions(source_data, target_data, couplings=None, max_points=1000):
    """Creates a scatter plot comparing the source and target distributions.
    
    Arguments:
        source_data: numpy array of shape (N, 2) representing the source distribution points
        target_data: numpy array of shape (N, 2) representing the target distribution points
        couplings: optional list of tuples (numpy array, numpy array) representing the couplings between source and target points.
            If provided, the couplings will be visualized as lines connecting the corresponding source and target points.
        max_points: maximum number of points to display from each dataset (for performance reasons)
    Returns:
        A Plotly Figure object visualizing the source and target distributions.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=source_data[:max_points, 0],
            y=source_data[:max_points, 1],
            mode="markers",
            name="Source Data",
            marker=dict(color="blue", size=6, opacity=0.7),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=target_data[:max_points, 0],
            y=target_data[:max_points, 1],
            mode="markers",
            name="Target Data",
            marker=dict(color="orange", size=6, opacity=0.7),
        )
    )

    if couplings is not None:
        x_lines, y_lines = [], []
        for src_point, tgt_point in couplings:
            x_lines.extend([src_point[0], tgt_point[0], None])
            y_lines.extend([src_point[1], tgt_point[1], None])
        fig.add_trace(
            go.Scatter(
                x=x_lines,
                y=y_lines,
                mode="lines",
                name="Couplings",
                line=dict(color="rgba(120,120,120,0.25)", width=1),
                hoverinfo="skip",
            )
        )

    x_range, y_range = data_ranges(source_data, target_data)
    fig.update_layout(
        title="Source vs Target Distributions" + (f" (with couplings)" if couplings is not None else ""),
        width=600,
        height=600,
        xaxis=dict(title="X1", range=x_range, constrain="domain", fixedrange=True),
        yaxis=dict(title="X2", range=y_range, scaleanchor="x"),
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=30, b=0, pad=0),
        template="plotly_white",
    )

    return fig

def plot_velocity_field(model, source_data, target_data, grid_size=25, max_points=1000):
    """Visualizes the velocity field of the flow model as a quiver plot.
    
    Arguments:
        model: the trained flow model with a .forward() method that takes in a tensor of shape (N, 2) and returns a tensor of shape (N, 2) representing the velocity vectors.
        source_data: numpy array of shape (N, 2) representing the source distribution points
        target_data: numpy array of shape (N, 2) representing the target distribution points
        grid_size: the number of points along each axis to create the grid for visualization.
        max_points: maximum number of points to display from each dataset (for performance reasons)
    Returns:
        A Plotly Figure object visualizing the velocity field of the flow model.
    """
    # Evaluate learned velocity field on a 2D grid and visualize vectors
    n_grid = grid_size
    scale = 0.05  # arrow length scaling for visualization

    x_range, y_range = data_ranges(source_data, target_data)
    x_min, x_max = x_range
    y_min, y_max = y_range

    xg = np.linspace(x_min, x_max, n_grid)
    yg = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(xg, yg)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
    v_grid = estimate_velocities(model, grid_points)

    # Build line segments for arrows: (x, y) -> (x + scale*vx, y + scale*vy)
    x_lines_field, y_lines_field = [], []
    for (x0, y0), (vx, vy) in zip(grid_points, v_grid):
        x_lines_field.extend([x0, x0 + scale * vx, None])
        y_lines_field.extend([y0, y0 + scale * vy, None])

    fig_field = go.Figure()

    quiver = ff.create_quiver(
        grid_points[:, 0],
        grid_points[:, 1],
        scale * v_grid[:, 0],
        scale * v_grid[:, 1],
        scale=1.0,
        arrow_scale=0.35,  # pointy arrow heads
        line=dict(color="rgba(20,20,20,0.55)", width=1),
        name="Predicted velocity vectors",
    )

    fig_field.add_trace(
        go.Scatter(
            x=source_data[:max_points, 0],
            y=source_data[:max_points, 1],
            mode="markers",
            name="Source Data",
            marker=dict(color="blue", size=6, opacity=0.7),
        )
    )

    fig_field.add_trace(
        go.Scatter(
            x=target_data[:max_points, 0],
            y=target_data[:max_points, 1],
            mode="markers",
            name="Target Data",
            marker=dict(color="orange", size=6, opacity=0.7),
        )
    )

    for trace in quiver.data:
        fig_field.add_trace(trace)

    fig_field.update_layout(
        title="Learned Velocity Field",
        width=600,
        height=600,
        xaxis=dict(title="X1", range=x_range, constrain="domain", fixedrange=True),
        yaxis=dict(title="X2", range=y_range, scaleanchor="x"),
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=30, b=0, pad=0),
        template="plotly_white",
    )

    return fig_field

def plot_trajectories(trajectories, show_origins=False, target_data=None, max_points=1000, max_trajectories=1000):
    """Visualizes the trajectories induced by the flow model as a line plot.
    
    Arguments:
        trajectories: a list of trajectories, where each trajectory is a list of (t, point) tuples representing the path of a point from t=0 to t=1 under the flow model
        show_origins: if True, the original source points (t=0) will be highlighted as a scatter plot in the background.
        target_data: numpy array of shape (N, 2) representing the target distribution points the trajectories should aim to match (optional).
            If provided, the target points will be visualized as a scatter plot in the background for comparison.
        max_points: maximum number of points to display from the target dataset (for performance reasons)
        max_trajectories: maximum number of trajectories to visualize (for performance reasons)
        
    Returns:
        A Plotly Figure object visualizing the trajectories of points under the flow model.
    """
    # Convert trajectories to Plotly line format
    x_traj_lines, y_traj_lines = [], []
    for traj in trajectories[:max_trajectories]:
        for _, p in traj:
            x_traj_lines.append(p[0])
            y_traj_lines.append(p[1])
        x_traj_lines.append(None)
        y_traj_lines.append(None)

    # Collect trajectory end points
    end_points = np.stack([traj[-1][1] for traj in trajectories[:max_trajectories]])

    # Plot: original points + trajectories + end points
    fig_traj = go.Figure()

    if target_data is not None:
        fig_traj.add_trace(
            go.Scatter(
                x=target_data[:max_points, 0],
                y=target_data[:max_points, 1],
                mode="markers",
                name="Target Data",
                marker=dict(color="orange", size=6, opacity=0.5),
            )
        )

    if show_origins:
        fig_traj.add_trace(
            go.Scatter(
                x=[traj[0][1][0] for traj in trajectories[:max_trajectories]],
                y=[traj[0][1][1] for traj in trajectories[:max_trajectories]],
                mode="markers",
                name="Original (source) points",
            marker=dict(color="blue", size=5, opacity=0.3),
        )
    )

    fig_traj.add_trace(
        go.Scatter(
            x=x_traj_lines,
            y=y_traj_lines,
            mode="lines",
            name="Induced trajectories",
            line=dict(color="rgba(80,80,80,0.25)", width=1),
            hoverinfo="skip",
        )
    )

    fig_traj.add_trace(
        go.Scatter(
            x=end_points[:, 0],
            y=end_points[:, 1],
            mode="markers",
            name="Trajectory end points",
            marker=dict(color="red", size=5, opacity=0.8),
        )
    )

    x_range, y_range = data_ranges(*[step[1] for traj in trajectories for step in traj])
    fig_traj.update_layout(
        title="Induced Trajectories from Source Points",
        width=600,
        height=600,
        xaxis=dict(title="X1", range=x_range, constrain="domain", fixedrange=True),
        yaxis=dict(title="X2", range=y_range, scaleanchor="x"),
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=35, b=0, pad=0),
        template="plotly_white",
    )

    return fig_traj

def animate_trajectories(trajectories, show_origins=False, target_data=None, max_points=1000,max_trajectories=1000):
    """Generates an animation of the trajectories induced by the flow model as a line plot.
    
    Arguments:
        trajectories: a list of trajectories, where each trajectory is a list of (t, point) tuples representing the path of a point from t=0 to t=1 under the flow model
        show_origins: if True, the original source points (t=0) will be highlighted as a scatter plot in the background.
        target_data: numpy array of shape (N, 2) representing the target distribution points the trajectories should aim to match (optional).
            If provided, the target points will be visualized as a scatter plot in the background for comparison.
        max_points: maximum number of points to display from the target dataset (for performance reasons)
        max_trajectories: maximum number of trajectories to visualize (for performance reasons)
        
    Returns:
        A Plotly Figure object visualizing the trajectories of points under the flow model.
    """
    # Build trajectory array: shape (n_points, n_time, 2)
    traj_array = np.stack([[p for _, p in traj] for traj in trajectories[:max_trajectories]])
    n_points, n_time, _ = traj_array.shape

    def build_history_lines(step_idx):
        x_hist, y_hist = [], []
        for i in range(n_points):
            x_hist.extend(traj_array[i, : step_idx + 1, 0].tolist())
            y_hist.extend(traj_array[i, : step_idx + 1, 1].tolist())
            x_hist.append(None)
            y_hist.append(None)
        return x_hist, y_hist

    # Initial frame (t=0): only original points, no path yet
    x0 = traj_array[:, 0, 0]
    y0 = traj_array[:, 0, 1]

    figure_data = []
    if target_data is not None:
        target_scatter = go.Scatter(
            x=target_data[:max_points, 0],
            y=target_data[:max_points, 1],
            mode="markers",
            name="Target Data",
            marker=dict(color="orange", size=6, opacity=0.5),
        )
        figure_data.append(target_scatter)

    figure_data.extend(
        [
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name="Trajectory",
                line=dict(color="rgba(80,80,80,0.25)", width=1),
                hoverinfo="skip",
            ),
            go.Scatter(
                x=x0,
                y=y0,
                mode="markers",
                name="Generated points",
                marker=dict(color="red", size=5, opacity=0.8),
            ),
        ]
    )

    if show_origins:
        origins_scatter = go.Scatter(
            x=[traj[0][1][0] for traj in trajectories[:max_trajectories]],
            y=[traj[0][1][1] for traj in trajectories[:max_trajectories]],
            mode="markers",
            name="Original (source) points",
            marker=dict(color="blue", size=5, opacity=0.3),
        )
        figure_data.append(origins_scatter) 

    fig_anim = go.Figure(data=figure_data)

    # Frames for t=1..end
    frames = []
    for k in range(1, n_time):
        x_hist, y_hist = build_history_lines(k)
        figure_data = []
        if target_data is not None:
            figure_data.append(target_scatter)
        figure_data.extend([
            go.Scatter(x=x_hist, y=y_hist),
            go.Scatter(x=traj_array[:, k, 0], y=traj_array[:, k, 1]),
        ])
        if show_origins:
            figure_data.append(origins_scatter)

        frames.append(go.Frame(name=str(k), data=figure_data))

    fig_anim.frames = frames

    # Slider + play controls
    slider_steps = [
        dict(
            method="animate",
            args=[
                [str(k)],
                dict(mode="immediate", frame=dict(duration=50, redraw=True), transition=dict(duration=0)),
            ],
            label=str(k),
        )
        for k in range(1, n_time)
    ]

    x_range, y_range = data_ranges(*[step[1] for traj in trajectories for step in traj])
    fig_anim.update_layout(
        title="Animated Induced Trajectories",
        width=600,
        height=600,
        xaxis=dict(title="X1", range=x_range, constrain="domain", fixedrange=True),
        yaxis=dict(title="X2", range=y_range, scaleanchor="x"),
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=50, b=20, pad=0),
        template="plotly_white",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.01,
                y=0.95,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=10, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    )
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Step: "),
                pad=dict(t=45),
                steps=slider_steps,
            )
        ],
    )

    return fig_anim


def plot_generated_data_comparison(target_data, trajectories, max_points=1000, max_trajectories=1000):
    """Compares the original target data points with the end points of the trajectories induced by the flow model.
    
    Arguments:
        target_data: numpy array of shape (N, 2) representing the original target distribution points
        trajectories: a list of trajectories, where each trajectory is a list of (t, point) tuples representing the path of a point from t=0 to t=1 under the flow model
        max_points: maximum number of points to display from the target dataset (for performance reasons)
        max_trajectories: maximum number of trajectories to visualize (for performance reasons)
        
    Returns:
        A Plotly Figure object visualizing the comparison between original target points and generated end points.
    """
    end_points = np.stack([traj[-1][1] for traj in trajectories[:max_trajectories]])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=target_data[:max_points, 0],
            y=target_data[:max_points, 1],
            mode="markers",
            name="Original (target) points",
            marker=dict(color="blue", size=6, opacity=0.7),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=end_points[:max_trajectories, 0],
            y=end_points[:max_trajectories, 1],
            mode="markers",
            name="Generated (end) points",
            marker=dict(color="red", size=6, opacity=0.7),
        )
    )

    x_range, y_range = data_ranges(target_data, *[step[1] for traj in trajectories[:max_trajectories] for step in traj])
    fig.update_layout(
        title="Comparison of Original Data Points and Generated Points",
        width=600,
        height=600,
        xaxis=dict(title="X1", range=x_range, constrain="domain", fixedrange=True),
        yaxis=dict(title="X2", range=y_range, scaleanchor="x"),
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=30, b=0, pad=0),
        template="plotly_white",
    )

    return fig