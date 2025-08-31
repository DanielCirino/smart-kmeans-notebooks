"""
Visualization utilities for the smart K-means research project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yaml
import os


def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def setup_plotting_style():
    """Set up plotting style based on configuration."""
    config = load_config()
    style = config['plots']['style']

    plt.style.use(style)
    sns.set_palette("husl")

    # Set default figure parameters
    plt.rcParams['figure.figsize'] = config['plots']['figsize']
    plt.rcParams['figure.dpi'] = config['plots']['dpi']
    plt.rcParams['savefig.dpi'] = config['plots']['dpi']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def plot_elbow_curve(k_values, inertias, optimal_k=None, save_path=None):
    """
    Plot elbow curve for K-means clustering.

    Parameters:
    -----------
    k_values : list
        Range of k values tested
    inertias : list
        Inertia values for each k
    optimal_k : int, optional
        Optimal k value to highlight
    save_path : str, optional
        Path to save the plot

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    setup_plotting_style()

    fig, ax = plt.subplots()
    ax.plot(k_values, inertias, marker='o', linewidth=2, markersize=8)

    if optimal_k is not None:
        optimal_idx = k_values.index(optimal_k)
        ax.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k = {optimal_k}')
        ax.plot(optimal_k, inertias[optimal_idx], marker='o', markersize=12, color='red')
        ax.legend()

    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    ax.set_title('Elbow Method for Optimal k')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return fig


def plot_silhouette_analysis(X, k_range, random_state=42, save_path=None):
    """
    Plot silhouette analysis for different k values.

    Parameters:
    -----------
    X : array-like
        Feature matrix
    k_range : list or range
        Range of k values to analyze
    random_state : int
        Random seed
    save_path : str, optional
        Path to save the plot

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, silhouette_samples

    setup_plotting_style()

    n_clusters_range = len(k_range)
    fig, axes = plt.subplots(2, (n_clusters_range + 1) // 2, figsize=(15, 8))
    axes = axes.flatten() if n_clusters_range > 1 else [axes]

    silhouette_scores = []

    for idx, n_clusters in enumerate(k_range):
        if n_clusters < 2:
            continue

        ax = axes[idx]

        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # Calculate silhouette scores
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        # Plot silhouette plot
        y_lower = 10
        for i in range(n_clusters):
            cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            cluster_silhouette_values.sort()

            size_cluster_i = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            y_lower = y_upper + 10

        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax.set_xlabel('Silhouette Coefficient Values')
        ax.set_ylabel('Cluster Label')
        ax.set_title(f'k={n_clusters}, Avg Score: {silhouette_avg:.3f}')

    # Remove empty subplots
    for idx in range(len(k_range), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return fig


def plot_clusters_2d(X, labels, centers=None, feature_names=None, save_path=None):
    """
    Plot 2D visualization of clusters.

    Parameters:
    -----------
    X : array-like
        Feature matrix (will use first 2 features if > 2D)
    labels : array-like
        Cluster labels
    centers : array-like, optional
        Cluster centers
    feature_names : list, optional
        Names of features for axis labels
    save_path : str, optional
        Path to save the plot

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    setup_plotting_style()

    # Use first two features for 2D plot
    X_plot = X[:, :2] if X.shape[1] >= 2 else X

    fig, ax = plt.subplots()

    # Plot data points
    scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)

    # Plot cluster centers if provided
    if centers is not None:
        centers_plot = centers[:, :2] if centers.shape[1] >= 2 else centers
        ax.scatter(centers_plot[:, 0], centers_plot[:, 1], c='red', marker='x',
                  s=200, linewidths=3, label='Centroids')
        ax.legend()

    # Set labels
    if feature_names and len(feature_names) >= 2:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    else:
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

    ax.set_title('Cluster Visualization (2D)')
    plt.colorbar(scatter, label='Cluster')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return fig


def plot_cluster_comparison(results_dict, save_path=None):
    """
    Plot comparison of different clustering results.

    Parameters:
    -----------
    results_dict : dict
        Dictionary with method names as keys and results as values
    save_path : str, optional
        Path to save the plot

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    setup_plotting_style()

    methods = list(results_dict.keys())
    metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [results_dict[method].get(metric, 0) for method in methods]

        bars = ax.bar(methods, values, alpha=0.7)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_ylabel('Score')

        # Rotate x-axis labels if needed
        if len(max(methods, key=len)) > 8:
            ax.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return fig


def save_figure(fig, filename, figures_path=None):
    """
    Save figure to the results directory.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Output filename
    figures_path : str, optional
        Path to figures directory
    """
    if figures_path is None:
        config = load_config()
        figures_path = config['output']['figures_path']

    os.makedirs(figures_path, exist_ok=True)
    filepath = os.path.join(figures_path, filename)

    config = load_config()
    save_format = config['plots']['save_format']
    dpi = config['plots']['dpi']

    fig.savefig(filepath, format=save_format, dpi=dpi, bbox_inches='tight')


def create_interactive_scatter(X, labels, feature_names=None):
    """
    Create interactive scatter plot using Plotly.

    Parameters:
    -----------
    X : array-like
        Feature matrix
    labels : array-like
        Cluster labels
    feature_names : list, optional
        Names of features

    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive figure
    """
    # Create DataFrame for easier handling
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

    df = pd.DataFrame(X, columns=feature_names[:X.shape[1]])
    df['Cluster'] = labels

    # Create scatter plot
    if X.shape[1] >= 3:
        fig = px.scatter_3d(df, x=feature_names[0], y=feature_names[1], z=feature_names[2],
                           color='Cluster', title='Interactive 3D Cluster Visualization')
    else:
        fig = px.scatter(df, x=feature_names[0], y=feature_names[1],
                        color='Cluster', title='Interactive 2D Cluster Visualization')

    return fig
