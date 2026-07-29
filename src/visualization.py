"""
visualization.py

Reusable plotting helpers for OCT images, boundaries, and masks.

Design rule for this file: every function takes already-sliced 2D arrays
(a single B-scan, a single boundary set for one scan, a single mask) —
never a full (H, W, num_scans) volume, and never a Subject. That keeps
these functions simple and testable, and it mirrors how masks.py works:
this file doesn't know what a "Subject" is, it just knows how to draw
arrays. The notebook is responsible for slicing out the scan it wants
before calling these.
"""

import matplotlib.pyplot as plt
import numpy as np


def show_oct(image: np.ndarray, title: str = "OCT B-scan", figsize=(12, 6)):
    """
    Display a single OCT B-scan in grayscale.

    Parameters
    ----------
    image : np.ndarray
        Shape (height, width). A single 2D B-scan, e.g. images[:, :, scan].
    """
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_boundaries(
    image: np.ndarray,
    boundaries: np.ndarray,
    title: str = "OCT with Manual Boundaries",
    figsize=(12, 6),
):
    """
    Display a B-scan with each retinal boundary overlaid as a line.

    Parameters
    ----------
    image : np.ndarray
        Shape (height, width).
    boundaries : np.ndarray
        Shape (num_boundaries, width) — boundaries for ONE scan, e.g.
        layers[:, :, scan]. NaN entries are simply not plotted (matplotlib
        skips them automatically, which is why the earlier all-NaN scans
        just showed nothing rather than crashing).
    """
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap="gray")

    x = np.arange(boundaries.shape[1])
    for i in range(boundaries.shape[0]):
        plt.plot(x, boundaries[i], label=f"Boundary {i + 1}", linewidth=1)

    plt.title(title)
    plt.legend(loc="upper right", fontsize=8)
    plt.axis("off")
    plt.show()


def show_mask(mask: np.ndarray, title: str = "Segmentation Mask", figsize=(12, 6)):
    """
    Display a multi-class segmentation mask with a colorbar, so you can
    visually confirm how many distinct classes are present.

    Parameters
    ----------
    mask : np.ndarray
        Shape (height, width), integer class labels (0 = background).
    """
    plt.figure(figsize=figsize)
    im = plt.imshow(mask)
    plt.colorbar(im, label="Class")
    plt.title(title)
    plt.axis("off")
    plt.show()


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
    title: str = "OCT with Mask Overlay",
    figsize=(12, 6),
):
    """
    Display a B-scan with its segmentation mask overlaid semi-transparently.
    This is the sanity-check view: if the colored regions don't line up
    with the visible retinal layers in the grayscale image, something in
    the mask generation is wrong.

    Parameters
    ----------
    image : np.ndarray
        Shape (height, width).
    mask : np.ndarray
        Shape (height, width), same shape as image.
    alpha : float
        Transparency of the mask overlay (0 = invisible, 1 = opaque).
    """
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap="gray")
    plt.imshow(mask, alpha=alpha)
    plt.title(title)
    plt.axis("off")
    plt.show()