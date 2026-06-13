"""Module with fucntions for creating visualizations and animations"""

import os
import tempfile


import imageio.v2 as imageio
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import torch

from plotly.subplots import make_subplots
from torchview import draw_graph

import distances
import models
from models import estimate_velocities
from scipy.stats import multivariate_normal


LABEL_COLORS = ["orange", "green", "red", "purple", "brown", "pink", "cyan", "magenta", "yellow", "lime"]

def data_ranges(*datasets, padding=0.1):
    """Helper function to compute the overall data ranges for multiple datasets, with optional padding.
    
    Arguments:
        datasets: variable number of numpy arrays of shape (N, 2) representing different datasets to be visualized together.
        padding: fraction of the data range to add as padding on each side (default: 0.1 for 10% padding).
    """
    all_data = np.vstack(datasets)
    x_min, y_min = all_data.min(axis=0)
    x_max, y_max = all_data.max(axis=0)
    x_len = x_max - x_min
    y_len = y_max - y_min
    # Add extra padding to make the plot look nicer
    x_min -= padding * x_len
    x_max += padding * x_len
    y_min -= padding * y_len
    y_max += padding * y_len
    return (x_min, x_max), (y_min, y_max)

def plot_distributions(source_data, target_data, target_labels=None, couplings=None, max_points=1000, max_couplings=1000):
    """Creates a scatter plot comparing the source and target distributions.
    
    Arguments:
        source_data: numpy array of shape (N, 2) representing the source distribution points
        target_data: numpy array of shape (N, 2) representing the target distribution points
        target_labels: optional numpy array of shape (N,) with integer class labels for the target data points.
            If provided, target points will be colored by class instead of using a single color.
        couplings: optional list of tuples (numpy array, numpy array) representing the couplings between source and target points.
            If provided, the couplings will be visualized as lines connecting the corresponding source and target points.
            If target_labels are provided, couplings should be a list of tuples (src_point, tgt_point, tgt_label) where tgt_label
            is the class label of the target point for coloring the coupling lines.
        max_points: maximum number of points to display from each dataset (for performance reasons)
        max_couplings: maximum number of couplings to visualize (for performance reasons)
    Returns:
        A Plotly Figure object visualizing the source and target distributions.
    """
    fig = go.Figure()

    if couplings is not None:
        source_data = np.vstack([np.array([coupling[0] for coupling in couplings]), source_data])
        target_data = np.vstack([np.array([coupling[1] for coupling in couplings]), target_data])
        if target_labels is not None:
            target_labels = np.hstack([np.array([coupling[2] for coupling in couplings]), target_labels])

    source_data = source_data[:max_points]
    target_data = target_data[:max_points]
    if target_labels is None:
        target_labels = np.zeros(target_data.shape[0], dtype=int)
    else:
        target_labels = target_labels[:max_points]

    fig.add_trace(
        go.Scatter(
            x=source_data[:, 0],
            y=source_data[:, 1],
            mode="markers",
            name="Source Data",
            marker=dict(color="blue", size=6, opacity=0.7),
        )
    )

    unique_labels = sorted(np.unique(target_labels))
    for i, label in enumerate(unique_labels):
        mask = target_labels == label
        name = "Target Data" if len(unique_labels) == 1 else f"Target Class {label}"
        fig.add_trace(
            go.Scatter(
                x=target_data[mask, 0],
                y=target_data[mask, 1],
                mode="markers",
                name=name,
                marker=dict(color=LABEL_COLORS[i % len(LABEL_COLORS)], size=6, opacity=0.7),
            )
        )

    if couplings is not None:
        x_lines, y_lines = [], []
        for coupling in couplings[:max_couplings]:
            src_point, tgt_point = coupling[:2]
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

def plot_velocity_field(model, source_data, target_data, target_labels=None, field_label=None, grid_size=25, max_points=1000):
    """Visualizes the velocity field of the flow model as a quiver plot.
    
    Arguments:
        model: the trained flow model with a .forward() method that takes in a tensor of shape (N, 2) and returns a tensor of shape (N, 2) representing the velocity vectors.
        source_data: numpy array of shape (N, 2) representing the source distribution points
        target_data: numpy array of shape (N, 2) representing the target distribution points
        target_labels: optional numpy array of shape (N,) representing the labels for the target data points.
        field_label: None to visualize the unconditional velocity field, or an integer label to visualize the velocity field conditioned on that label.
        grid_size: the number of points along each axis to create the grid for visualization.
        max_points: maximum number of points to display from each dataset (for performance reasons)
    Returns:
        A Plotly Figure object visualizing the velocity field of the flow model.
    """
    source_data = source_data[:max_points]
    target_data = target_data[:max_points]
    if target_labels is None:
        target_labels = np.zeros(target_data.shape[0], dtype=int)
    else:
        target_labels = target_labels[:max_points]

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
    v_grid = estimate_velocities(model, grid_points, labels=np.full(grid_points.shape[0], field_label) if field_label is not None else None)

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
            x=source_data[:, 0],
            y=source_data[:, 1],
            mode="markers",
            name="Source Data",
            marker=dict(color="blue", size=6, opacity=0.7),
        )
    )

    unique_labels = sorted(np.unique(target_labels))
    for i, label in enumerate(unique_labels):
        mask = target_labels == label
        name = "Target Data" if len(unique_labels) == 1 else f"Target Class {label}"
        fig_field.add_trace(
            go.Scatter(
                x=target_data[mask, 0],
                y=target_data[mask, 1],
                mode="markers",
                name=name,
                marker=dict(color=LABEL_COLORS[i % len(LABEL_COLORS)], size=6, opacity=0.7),
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

def plot_class_conditional_velocity_fields(model, source_data, target_data, target_labels=None, grid_size=25, max_points=1000):
    """Generates subplots visualizing the velocity fields of the flow model conditioned on each class label.
    
    Arguments:
        model: the trained flow model with a .forward() method that takes in a tensor of shape (N, 2) and returns a tensor of shape (N, 2) representing the velocity vectors.
        source_data: numpy array of shape (N, 2) representing the source distribution points
        target_data: numpy array of shape (N, 2) representing the target distribution points
        target_labels: optional numpy array of shape (N,) representing the labels for the target data points.
        field_label: None to visualize the unconditional velocity field, or an integer label to visualize the velocity field conditioned on that label.
        grid_size: the number of points along each axis to create the grid for visualization.
        max_points: maximum number of points to display from each dataset (for performance reasons)
    Returns:
        A Plotly Figure object visualizing the velocity fields of the flow model conditioned on each class label.
    """
    unique_labels = list(model.labels_dict.keys())
    n_labels = len(unique_labels)
    x_range, y_range = data_ranges(source_data[:max_points], target_data[:max_points])

    subplot_titles = ["Unconditional velocity field"] + [f"Velocity field conditioned on label {label}" for label in unique_labels if label is not None]
    fig = make_subplots(rows=1, cols=n_labels, subplot_titles=subplot_titles)

    seen_legend_names = set()

    for i, label in enumerate(unique_labels, start=1):
        fig_field = plot_velocity_field(model, source_data, target_data, target_labels=target_labels, field_label=label, grid_size=grid_size, max_points=max_points)
        for trace in fig_field.data:
            trace_name = getattr(trace, "name", None)
            if trace_name is not None:
                trace.legendgroup = trace_name
                trace.showlegend = trace_name not in seen_legend_names
                seen_legend_names.add(trace_name)
            else:
                trace.showlegend = False
            fig.add_trace(trace, row=1, col=i)

    fig.update_layout(
        width=450 * n_labels,
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=30, b=0, pad=0),
        template="plotly_white",
    )

    for col in range(1, n_labels + 1):
        fig.update_xaxes(title_text="X1", range=x_range, constrain="domain", fixedrange=True, row=1, col=col)
        fig.update_yaxes(title_text="X2", range=y_range, scaleanchor=f"x{col if col > 1 else ''}", row=1, col=col)

    return fig

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
    target_data = target_data[:max_points]
    trajectories = trajectories[:max_trajectories]

    # Convert trajectories to Plotly line format
    x_traj_lines, y_traj_lines = [], []
    for traj in trajectories:
        for _, p in traj:
            x_traj_lines.append(p[0])
            y_traj_lines.append(p[1])
        x_traj_lines.append(None)
        y_traj_lines.append(None)

    # Collect trajectory end points
    end_points = np.stack([traj[-1][1] for traj in trajectories])

    # Plot: original points + trajectories + end points
    fig_traj = go.Figure()

    if target_data is not None:
        fig_traj.add_trace(
            go.Scatter(
                x=target_data[:, 0],
                y=target_data[:, 1],
                mode="markers",
                name="Target Data",
                marker=dict(color="orange", size=6, opacity=0.5),
            )
        )

    if show_origins:
        fig_traj.add_trace(
            go.Scatter(
                x=[traj[0][1][0] for traj in trajectories],
                y=[traj[0][1][1] for traj in trajectories],
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

def animate_trajectories(trajectories, labels=None, target_data=None, max_points=1000, max_trajectories=1000, max_trajectory_steps=50, 
                         draw_controls=True, title="Animated Trajectories", draw_axes=True, draw_legend=True):
    """Generates an animation of trajectories induced by the flow model.

    Supports both unconditional and class-conditional visualizations from a
    single list of trajectories. When labels are provided, each class is drawn
    in its own colour from LABEL_COLORS.

    Arguments:
        trajectories: list of trajectories. Each trajectory is a list of
            (t, point) tuples representing the path of a point from t=0 to t=1.
        labels: optional array-like of shape (len(trajectories),) with class
            labels for each trajectory. If provided, trajectories are grouped by
            label for class-conditional visualization.
        target_data: optional numpy array of shape (N, 2) with target points to
            display in the background for comparison.
        max_points: maximum number of target points to display.
        max_trajectories: maximum number of trajectories to visualize.
        max_trajectory_steps: maximum number of time steps to visualize along each trajectory (for performance reasons).
            If trajectories have more steps, they will be subsampled to this number of steps.
        draw_controls: if True, adds play button and slider controls to the animation.
        title: title of the plot.
        draw_axes: if True, draws x and y axes with labels; if False, hides axes for a cleaner look.
        draw_legend: if True, displays the legend; if False, hides the legend.

    Returns:
        A Plotly Figure object with the animated trajectories.
    """
    trajectories = trajectories[:max_trajectories]
    if not trajectories:
        raise ValueError("trajectories must contain at least one trajectory")

    # Always normalize to {label: trajectories} for internal processing.
    if labels is None:
        class_trajectories = {None: trajectories}
        is_class_conditional = False
    else:
        labels = np.asarray(labels)[:max_trajectories]
        if labels.shape[0] != len(trajectories):
            raise ValueError("labels must have the same length as trajectories")
        class_trajectories = {}
        for traj, label in zip(trajectories, labels):
            class_trajectories.setdefault(label, []).append(traj)
        is_class_conditional = True

    # Subsample trajectories to max_trajectory_steps if needed
    for label, trajs in class_trajectories.items():
        for i, traj in enumerate(trajs):
            if len(traj) > max_trajectory_steps:
                indices = np.linspace(0, len(traj) - 1, max_trajectory_steps, dtype=int)
                class_trajectories[label][i] = [traj[idx] for idx in indices]

    if target_data is not None:
        target_data = target_data[:max_points]

    unique_labels = sorted(class_trajectories.keys(), key=lambda x: (x is None, x))

    # Build trajectory arrays per class: shape (n_points, n_time, 2)
    class_traj_arrays = {}
    for label in unique_labels:
        class_traj_arrays[label] = np.stack(
            [[p for _, p in traj] for traj in class_trajectories[label]]
        )

    n_time = class_traj_arrays[unique_labels[0]].shape[1]

    def build_history_lines(traj_array, step_idx):
        n_pts = traj_array.shape[0]
        x_hist, y_hist = [], []
        for i in range(n_pts):
            x_hist.extend(traj_array[i, : step_idx + 1, 0].tolist())
            y_hist.extend(traj_array[i, : step_idx + 1, 1].tolist())
            x_hist.append(None)
            y_hist.append(None)
        return x_hist, y_hist

    # --- initial figure data ---
    figure_data = []

    if target_data is not None:
        target_color = "gray" if is_class_conditional else "orange"
        target_opacity = 0.3 if is_class_conditional else 0.5
        target_scatter = go.Scatter(
            x=target_data[:, 0],
            y=target_data[:, 1],
            mode="markers",
            name="Target Data",
            marker=dict(color=target_color, size=6, opacity=target_opacity),
        )
        figure_data.append(target_scatter)

    for idx, label in enumerate(unique_labels):
        traj_array = class_traj_arrays[label]
        x0 = traj_array[:, 0, 0]
        y0 = traj_array[:, 0, 1]

        if is_class_conditional:
            color = LABEL_COLORS[idx % len(LABEL_COLORS)]
            traj_name = f"Trajectory (class {label})"
            pts_name = f"Generated points (class {label})"
            line_style = dict(color=color, width=1)
            traj_opacity = 0.4
            marker_style = dict(color=color, size=5, opacity=0.8)
        else:
            traj_name = "Trajectory"
            pts_name = "Generated points"
            line_style = dict(color="rgba(80,80,80,0.25)", width=1)
            traj_opacity = 1.0
            marker_style = dict(color="red", size=5, opacity=0.8)

        figure_data.append(
            go.Scatter(
                x=[], y=[],
                mode="lines",
                name=traj_name,
                line=line_style,
                opacity=traj_opacity,
                hoverinfo="skip",
            )
        )
        figure_data.append(
            go.Scatter(
                x=x0, y=y0,
                mode="markers",
                name=pts_name,
                marker=marker_style,
            )
        )

    fig_anim = go.Figure(data=figure_data)

    # --- animation frames ---
    frames = []
    for k in range(0, n_time):
        frame_data = []
        if target_data is not None:
            frame_data.append(target_scatter)
        for label in unique_labels:
            traj_array = class_traj_arrays[label]
            x_hist, y_hist = build_history_lines(traj_array, k)
            frame_data.append(go.Scatter(x=x_hist, y=y_hist))
            frame_data.append(go.Scatter(x=traj_array[:, k, 0], y=traj_array[:, k, 1]))
        frames.append(go.Frame(name=str(k), data=frame_data))

    fig_anim.frames = frames

    # --- data ranges ---
    ranges_inputs = []
    for label in unique_labels:
        for traj in class_trajectories[label]:
            for _, p in traj:
                ranges_inputs.append(p)
    if target_data is not None:
        ranges_inputs.append(target_data)
    x_range, y_range = data_ranges(*ranges_inputs)

    # --- slider + play controls ---

    if draw_controls:
        buttons = [
            dict(
                type="buttons",
                showactive=False,
                x=0.01,
                y=0.90,
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
        ]
        slider_steps = [
            dict(
                method="animate",
                args=[
                    [str(k)],
                    dict(mode="immediate", frame=dict(duration=50, redraw=True), transition=dict(duration=0)),
                ],
                label=str(k),
            )
            for k in range(0, n_time)
        ]
        sliders = [
            dict(
                active=0,
                currentvalue=dict(prefix="Step: "),
                pad=dict(t=45),
                steps=slider_steps,
            )
        ]

    # Margins adjustment
    margin = dict(l=0, r=0, t=0, b=0, pad=0)
    if title != "":
        margin['t'] += 30
    if draw_controls:
        margin['t'] += 20
        margin['b'] += 20

    fig_anim.update_layout(
        title=title,
        width=600,
        height=600,
        xaxis=dict(title="X1", visible=draw_axes, range=x_range, constrain="domain", fixedrange=True),
        yaxis=dict(title="X2", visible=draw_axes, range=y_range, scaleanchor="x"),
        legend=dict(visible=draw_legend, orientation="h", yanchor="top", y=1.02, xanchor="center", x=0.5),
        margin=margin,
        template="plotly_white",
        updatemenus=buttons if draw_controls else [],
        sliders=sliders if draw_controls else [],
    )

    return fig_anim

def plot_generated_data_comparison(target_data, trajectories_or_generated_data, max_points=1000):
    """Compares the original target data points with the end points of the trajectories induced by the flow model.
    
    Arguments:
        target_data: numpy array of shape (N, 2) representing the original target distribution points
        trajectories_or_generated_data: a list of trajectories, where each trajectory is a list of (t, point) tuples representing the path of a point from t=0 to t=1 under the flow model, 
            or a numpy array of generated data points.
        max_points: maximum number of points to display from the target dataset (for performance reasons)
        
    Returns:
        A Plotly Figure object visualizing the comparison between original target points and generated end points.
    """
    target_data = target_data[:max_points]
    if isinstance(trajectories_or_generated_data, list):
        generated_data = np.stack([traj[-1][1] for traj in trajectories_or_generated_data])
    else:
        generated_data = trajectories_or_generated_data
    generated_data = generated_data[:max_points]
    x_range, y_range = data_ranges(target_data, generated_data)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=target_data[:, 0],
            y=target_data[:, 1],
            mode="markers",
            name="Original (target) points",
            marker=dict(color="blue", size=6, opacity=0.7),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=generated_data[:, 0],
            y=generated_data[:, 1],
            mode="markers",
            name="Generated (end) points",
            marker=dict(color="red", size=6, opacity=0.7),
        )
    )

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

def plot_class_conditioned_generated_data_comparison(target_data, trajectories, trajectories_labels, target_labels, max_points=1000, max_trajectories=1000):
    """Compares the original target data points with the end points of the trajectories induced by the flow model conditioned on class labels.
    
    Arguments:
        target_data: numpy array of shape (N, 2) representing the original target distribution points
        trajectories: list of trajectories.
            Each trajectory is a list of (t, point) tuples representing the path of a point from t=0 to t=1.
        trajectories_labels: array-like of shape (len(trajectories),) with class labels for each trajectory.
        target_labels: numpy array of shape (N,) representing the class labels of the target data points
        max_points: maximum number of points to display from the target dataset (for performance reasons)
        max_trajectories: maximum number of trajectories to visualize (for performance reasons)
        
    Returns:
        A Plotly Figure object visualizing the comparison between original target points and generated end points for each of the classes.
    """
    target_data = target_data[:max_points]
    trajectories = trajectories[:max_trajectories]
    trajectories_labels = np.asarray(trajectories_labels)[:max_trajectories]
    if len(trajectories) == 0:
        raise ValueError("trajectories must contain at least one trajectory")
    if trajectories_labels.shape[0] != len(trajectories):
        raise ValueError("trajectories_labels must have the same length as trajectories")

    class_trajectories = {}
    for traj, label in zip(trajectories, trajectories_labels):
        class_trajectories.setdefault(label, []).append(traj)

    end_points = {label: np.stack([traj[-1][1] for traj in trajs]) for label, trajs in class_trajectories.items()}
    target_labels = target_labels[:max_points]
    data_ranges_inputs = [target_data] + list(end_points.values())
    x_range, y_range = data_ranges(*data_ranges_inputs)

    unique_labels = sorted(np.unique(target_labels))
    n_labels = len(unique_labels)
    subplot_titles = [f"Class {label}" for label in unique_labels]

    fig = make_subplots(rows=1, cols=n_labels, subplot_titles=subplot_titles)

    for i, label in enumerate(unique_labels, start=0):
        mask = target_labels == label
        trace = go.Scatter(
            x=target_data[mask, 0],
            y=target_data[mask, 1],
            mode="markers",
            name=f"Original (target) points",
            marker=dict(color="gray", size=6, opacity=0.7),
            showlegend=(i == 0)
        )
        fig.add_trace(trace, row=1, col=i+1)

        trace = go.Scatter(
            x=end_points[label][:, 0],
            y=end_points[label][:, 1],
            mode="markers",
            name=f"Generated (end) points for class {label}",
            marker=dict(color=LABEL_COLORS[i % len(LABEL_COLORS)], size=6, opacity=0.7),
        )
        fig.add_trace(trace, row=1, col=i+1)


    fig.update_layout(
        width=500 * n_labels,
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=30, b=0, pad=0),
        template="plotly_white",
    )

    for col in range(1, n_labels + 1):
        fig.update_xaxes(title_text="X1", range=x_range, constrain="domain", fixedrange=True, row=1, col=col)
        fig.update_yaxes(title_text="X2", range=y_range, scaleanchor=f"x{col if col > 1 else ''}", row=1, col=col)

    return fig

def mesh_from_data(data, grid_size=25, max_points=1000):
    """Helper function to create a grid of points covering the range of the data for visualization purposes.
    
    Arguments:
        data: numpy array of shape (N, 2) representing the data points to determine the range for the grid.
        grid_size: the number of points along each axis to create the grid for visualization.
        max_points: maximum number of points to consider from the data for determining the range (for performance reasons)

    Returns:
        A numpy array of shape (grid_size*grid_size, 2) representing the grid points covering the range of the data.
    """
    data = data[:max_points]
    x_range, y_range = data_ranges(data)
    x_min, x_max = x_range
    y_min, y_max = y_range
    xg = np.linspace(x_min, x_max, grid_size)
    yg = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(xg, yg)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
    return grid_points

def plot_density_map(reversed_mesh_trajectories, target_data, source_pdf=None, max_points=1000):
    """Visualizes a map of region, coloured by the degree of anomaly as estimated by a reversed flow model.

    Arguments:
        model: the trained flow model.
        target_data: numpy array of shape (N, 2) representing the original target distribution points
        source_pdf: callable(points) -> pdf_values, function that takes in an array of shape (N, 2) and returns an array of shape (N,) representing the probability density under the source distribution at those points.
        grid_size: the number of points along each axis to create the grid for visualization.
        max_points: maximum number of points to display from the source dataset (for performance reasons)            

    Returns:
        A Plotly Figure object visualizing the outlier map.
    """
    target_data = target_data[:max_points]
    x_range, y_range = data_ranges(target_data)
    x_min, x_max = x_range
    y_min, y_max = y_range

    startpoints = np.stack([traj[0][1] for traj in reversed_mesh_trajectories])
    endpoints = np.stack([traj[-1][1] for traj in reversed_mesh_trajectories])
    if source_pdf is None:
        source_pdf = lambda points: [multivariate_normal.pdf(point, mean=[0, 0], cov=np.eye(2)) for point in points]
    pdf_estimate = source_pdf(endpoints)

    xg = np.unique(startpoints[:, 0])
    yg = np.unique(startpoints[:, 1])
    Z = np.array(pdf_estimate).reshape(len(yg), len(xg))

    fig = go.Figure(data=go.Contour(z=Z, x=xg, y=yg, colorscale=[ [0, "black"], [1, "white"] ], contours=dict(showlabels=True)))

    fig.add_trace(
        go.Scatter(
            x=target_data[:, 0],
            y=target_data[:, 1],
            mode="markers",
            name="Target Data",
            marker=dict(color="orange", size=6, opacity=0.7),
        )
    )

    fig.update_layout(
        title="Outlier Map (density estimate from reversed flow)",
        width=600,
        height=600,
        xaxis=dict(title="X1", range=(x_min, x_max), constrain="domain", fixedrange=True),
        yaxis=dict(title="X2", range=(y_min, y_max), scaleanchor="x"),
        margin=dict(l=0, r=0, t=30, b=0, pad=0),
        template="plotly_white",
    )

    return fig

def plot_image_grid(data, labels, samples_per_label=10):
    """Plots a grid of images from the data, grouped by their labels.  Assumes data is of shape (N, H, W) and labels is of shape (N,).

    Aruments:
        data: numpy array of shape (N, H, W) representing the image data points
        labels: numpy array of shape (N,) representing the class labels for each image
        samples_per_label: number of images to display for each label (default: 10)

    Returns:
        A Plotly Figure object visualizing the image grid.
    """

    import plotly.graph_objects as go

    unique_labels = np.unique(labels)
    n_cols = 10
    n_rows = len(unique_labels)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        horizontal_spacing=0.005,
        vertical_spacing=0.02
    )

    for r, label in enumerate(unique_labels, start=1):
        sample_idx = np.where(labels == label)[0][:samples_per_label]  # 10 different samples for this label
        for c, idx in enumerate(sample_idx, start=1):
            fig.add_trace(
                go.Heatmap(
                    z=data[idx],
                    colorscale="gray",
                    showscale=False,
                    hovertemplate=f"label={label}<br>sample_index={idx}<extra></extra>",
                ),
                row=r,
                col=c,
            )
            fig.update_xaxes(showticklabels=False, row=r, col=c)
            fig.update_yaxes(showticklabels=False, autorange="reversed", row=r, col=c)

        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=-0.02,
            y=1 - (r - 0.5) / n_rows,
            text=f"Digit {label}",
            showarrow=False,
            xanchor="right",
            font=dict(size=12),
        )

    fig.update_layout(
        title="MNIST-like digits: 10 samples per label",
        width=1100,
        height=1100,
        margin=dict(l=90, r=20, t=60, b=20),
    )

    return fig


def plot_euler_steps_vs_wasserstein_distance(networks, source_data, target_data, min_steps=1, max_steps=50, num_steps=25):
    """Plots the Wasserstein distance between the original target data and the generated data obtained by applying the flow model with different numbers of Euler steps.
    
    Arguments:
        networks: either a single network representing the flow model, or a dictionary of {label: network} to compare in the plot.
        source_data: numpy array of shape representing the source distribution points
        target_data: numpy array of shape representing the target distribution points
        min_steps: minimum number of Euler steps (default: 1)
        max_steps: maximum number of Euler steps (default: 50)
        num_steps: number of steps to evaluate between min_steps and max_steps (default: 25)

    Returns:
        A Plotly Figure object visualizing the relationship between the number of Euler steps and the Wasserstein distance to the target data.
    """
    if not isinstance(networks, dict):
        networks = {"": networks}

    euler_steps = np.logspace(np.log10(min_steps), np.log10(max_steps), num=num_steps, dtype=int)
    wasserstein_distances = {network_name: [] for network_name in networks.keys()}

    for steps in euler_steps:
        for network_name, network in networks.items():
            trajectories = models.compute_trajectories(network, source_data, n_steps=steps, batch_size=2048)
            generated_data = np.array([traj[-1][1] for traj in trajectories])
            distance = distances.wasserstein_distance(target_data, generated_data)
            wasserstein_distances[network_name].append(distance)

    fig = go.Figure()
    for network_name, network_distances in wasserstein_distances.items():
        fig.add_trace(
            go.Scatter(
                x=euler_steps,
                y=network_distances,
                # mode="lines+markers+text",
                mode="lines+markers",
                name=network_name,
            text=[str(steps) for steps in euler_steps],
            textposition="top right",
            textfont=dict(size=9)
        )
    )
    fig.update_layout(
        title="Wasserstein Distance vs Number of Euler Steps",
        xaxis_title="Number of Euler Steps",
        xaxis=dict(
            tickmode="array",
            tickvals=euler_steps,
            ticktext=[str(steps) for steps in euler_steps],
        ),
        yaxis_title="Wasserstein Distance",
        yaxis_type="log",
        width=800,
        height=400,
        template="plotly_white",
        margin=dict(l=0, r=0, t=30, b=0, pad=0),
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="center", x=0.5),
    )

    return fig


def plotly_animation_to_mp4(animation, output_path="animation.mp4", fps=20, width=900, height=700):
    """
    Export a Plotly animated figure (go.Figure with .frames) to MP4.
    The interactive animation controls (Play button and slider) are removed
    from the exported video frames.
    """
    if not animation.frames:
        raise ValueError("The figure has no frames. 'animation.frames' is empty.")

    # Build a base figure from the original one (without dynamic frame state)
    base = go.Figure(animation)

    # Create temporary directory for frame images
    with tempfile.TemporaryDirectory() as tmpdir:
        png_paths = []

        # Render each frame
        for i, fr in enumerate(animation.frames):
            fig_i = go.Figure(base)

            # Apply frame data if present
            if hasattr(fr, "data") and fr.data is not None:
                fig_i.update(data=fr.data)

            # Apply frame layout updates if present
            if hasattr(fr, "layout") and fr.layout is not None:
                fig_i.update_layout(fr.layout)

            png_path = os.path.join(tmpdir, f"frame_{i:05d}.png")
            fig_i.write_image(png_path, format="png", width=width, height=height, scale=2)
            png_paths.append(png_path)

        # Write MP4
        with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8) as writer:
            for p in png_paths:
                writer.append_data(imageio.imread(p))

    print(f"Saved MP4 to: {output_path}")


def plot_network(network, coupling_sample, save_filename=None):
    """Plots the architecture of a PyTorch network using torchview.

    Arguments:
        network: a PyTorch nn.Module representing the network to visualize.
        coupling_sample: a sample input tuple to pass through the network for visualization.

    Returns:
        A torchview Graph object representing the network architecture.
    """

    sample_input = torch.tensor([coupling_sample[0][0]], dtype=torch.float, requires_grad=False).to("cuda")
    if len(coupling_sample[0]) > 2:
        sample_input = (sample_input, torch.tensor([coupling_sample[0][2]], dtype=torch.long, requires_grad=False).to("cuda"))

    model_graph = draw_graph(
        network,
        input_data=sample_input,
        depth=2,                       # controls module-hierarchy detail
        graph_dir="TB",                # top-to-bottom layout
        expand_nested=False,
        hide_inner_tensors=True,
        hide_module_functions=True,
        show_shapes=True,
        save_graph=save_filename is not None,
        filename=save_filename,
    )

    return model_graph.visual_graph
