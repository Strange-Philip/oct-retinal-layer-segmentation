"""
io.py

Responsible for one thing: turning a raw .mat file on disk into a clean,
predictable Python object. Nothing in this file knows about masks, U-Nets,
or training — it only knows how to read the Chiu-format .mat files.

Keeping this isolated means that if you ever get data in a different format
(e.g. a .h5 export, or a different research group's .mat layout), you only
need to change this one file. Everything downstream (masks.py, dataset.py)
keeps working because it only talks to the `Subject` object, never to
scipy.io directly.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as sio


@dataclass
class Subject:
    """
    A single subject's OCT data, in a clean, predictable shape.

    Attributes
    ----------
    subject_id : str
        Derived from the filename, e.g. "Subject_01".
    images : np.ndarray
        Shape (height, width, num_scans) — e.g. (496, 768, 61).
    manual_layers_1 : np.ndarray
        Shape (num_boundaries, width, num_scans) — e.g. (8, 768, 61).
        Boundary coordinates from grader 1. NaN where not annotated.
    manual_layers_2 : Optional[np.ndarray]
        Same shape as manual_layers_1, from grader 2, if present.
    manual_fluid_1 : Optional[np.ndarray]
        Fluid annotation mask from grader 1, if present in the file.
    manual_fluid_2 : Optional[np.ndarray]
        Fluid annotation mask from grader 2, if present in the file.
    raw : dict
        The untouched dict from loadmat, kept around as an escape hatch
        in case you need a field this class doesn't expose yet.
    """

    subject_id: str
    images: np.ndarray
    manual_layers_1: np.ndarray
    manual_layers_2: Optional[np.ndarray]
    manual_fluid_1: Optional[np.ndarray]
    manual_fluid_2: Optional[np.ndarray]
    raw: dict

    @property
    def num_scans(self) -> int:
        return self.images.shape[-1]

    @property
    def image_shape(self) -> tuple:
        """(height, width) of a single B-scan."""
        return self.images.shape[:2]

    def annotated_scans(self, grader: int = 1) -> list[int]:
        """
        Return the indices of B-scans that actually have manual boundary
        annotations for the given grader.

        This matters because — as you discovered — most scans in a subject
        are NOT manually annotated. Only a handful (e.g. scan 10, 15, 20...)
        have real boundary values; the rest are all-NaN. Anything that loops
        over "every scan" for training data needs this list, not range(61).
        """
        layers = self.manual_layers_1 if grader == 1 else self.manual_layers_2
        if layers is None:
            return []

        valid = []
        for scan in range(layers.shape[-1]):
            if np.any(~np.isnan(layers[:, :, scan])):
                valid.append(scan)
        return valid


def load_subject(filepath: str | Path) -> Subject:
    """
    Load a single Chiu-format .mat file into a Subject.

    Parameters
    ----------
    filepath : str or Path
        Path to a Subject_XX.mat file.

    Returns
    -------
    Subject
    """
    filepath = Path(filepath)
    data = sio.loadmat(str(filepath))

    if "images" not in data:
        raise KeyError(
            f"{filepath.name} has no 'images' key. "
            f"Available keys: {[k for k in data if not k.startswith('__')]}"
        )
    if "manualLayers1" not in data:
        raise KeyError(
            f"{filepath.name} has no 'manualLayers1' key. "
            f"Available keys: {[k for k in data if not k.startswith('__')]}"
        )

    return Subject(
        subject_id=filepath.stem,
        images=data["images"],
        manual_layers_1=data["manualLayers1"],
        manual_layers_2=data.get("manualLayers2"),
        manual_fluid_1=data.get("manualFluid1"),
        manual_fluid_2=data.get("manualFluid2"),
        raw=data,
    )


def load_dataset(directory: str | Path, pattern: str = "*.mat") -> list[Subject]:
    """
    Load every .mat file in a directory as a list of Subjects.

    Useful once you move from "one subject in a notebook" to "loop over
    all 10 subjects to build a training set" in dataset.py.
    """
    directory = Path(directory)
    filepaths = sorted(directory.glob(pattern))

    if not filepaths:
        raise FileNotFoundError(f"No files matching '{pattern}' in {directory}")

    return [load_subject(fp) for fp in filepaths]