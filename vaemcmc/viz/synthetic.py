from typing import Optional, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde


def plot_kde(
    sample: torch.Tensor,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    n_grid: int = 100,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Figure:
    if ax is None:
        fig = fig or plt.figure()
        ax = fig.add_subplot()

    sample = sample.reshape(-1, sample.shape[-1])
    kernel = gaussian_kde(sample.T)
    x = np.linspace(*xlim, n_grid)
    y = np.linspace(*ylim, n_grid)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    kde = np.reshape(kernel(positions).T, X.shape)
    ax.contour(X, Y, kde, colors="midnightblue", linewidths=1, **kwargs)
    # ax.set_title(f"KDE")

    return fig


def plot_chain(
    chain: torch.Tensor,
    # xlim: Tuple[float, float],
    # ylim: Tuple[float, float],
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Figure:
    if ax is None:
        fig = fig or plt.figure()
        ax = fig.add_subplot()

    ax.plot(
        *chain.T,
        "-",
        alpha=min(0.6, 1000.0 / chain.shape[0]),
        c="coral",
        linewidth=1,
        marker="o",
        markersize=1,
        mec="black",
        **kwargs,
    )  # , c='k')
    # ax.set_title(f"Trajectory of single chain")

    return fig


def scatter(
    pts: torch.Tensor,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    c: str = "coral",
    **kwargs,
) -> plt.Figure:
    if ax is None:
        fig = fig or plt.figure()
        ax = fig.add_subplot()
    ax.scatter(
        *pts.T,
        s=30,
        c=c,
        edgecolors="black",
        linewidth=0.5,
        alpha=0.5,
        marker="o",
        **kwargs,
    )
    return fig
