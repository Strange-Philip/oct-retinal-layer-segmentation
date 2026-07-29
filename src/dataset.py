"""
dataset.py

PyTorch Dataset that ties together io.py (loading subjects) and masks.py
(generating segmentation targets). This is the first file in the project
that knows about PyTorch — everything below it (io, masks, visualization)
stays framework-agnostic on purpose, so those modules stay reusable even
if you ever swap out PyTorch for something else.
"""

from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from src.masks import generate_mask


class OCTLayerDataset(Dataset):
    """
    A PyTorch Dataset over annotated OCT B-scans and their derived
    segmentation masks.

    Parameters
    ----------
    subjects : list[Subject]
        Subjects loaded via `src.io.load_dataset` (or a hand-picked
        list of `load_subject` results, e.g. for a train/val split).
    grader : {1, 2}
        Which grader's manual boundary annotations to use for generating
        masks. Chosen once per Dataset instance rather than per-sample,
        since a training run should be consistent about which grader's
        "ground truth" it's learning from.

    Notes
    -----
    The index this Dataset exposes (0, 1, 2, ...) does NOT correspond to
    scan numbers. Most scans per subject have no manual annotation at all
    (see Subject.annotated_scans), so this class builds a flat index of
    only the (subject, scan) pairs that actually have ground truth, across
    every subject. That's what makes `len(dataset)` meaningful and lets
    a DataLoader shuffle across subjects transparently.
    """

    def __init__(self, subjects: list, grader: Literal[1, 2] = 1):
        if grader not in (1, 2):
            raise ValueError(f"grader must be 1 or 2, got {grader}")

        self.subjects = subjects
        self.grader = grader

        # Flat index: index -> (subject_index, scan_index)
        # Built once at construction time so __getitem__ stays O(1),
        # not re-scanning for NaNs on every single training step.
        self._index: list[tuple[int, int]] = []

        for subj_idx, subject in enumerate(subjects):
            scans = subject.annotated_scans(grader=grader)
            if not scans:
                print(
                    f"[WARN] {subject.subject_id} has no annotated scans "
                    f"for grader {grader}; skipping."
                )
            for scan in scans:
                self._index.append((subj_idx, scan))

        if not self._index:
            raise ValueError(
                f"No annotated scans found for grader {grader} "
                f"across {len(subjects)} subjects."
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        subj_idx, scan = self._index[idx]
        subject = self.subjects[subj_idx]

        image = subject.images[:, :, scan]

        layers = (
            subject.manual_layers_1 if self.grader == 1 else subject.manual_layers_2
        )
        boundaries = layers[:, :, scan]

        mask = generate_mask(boundaries, subject.image_shape)

        # Normalize image to [0, 1] float32. Raw OCT images are typically
        # uint8 (0-255); neural nets train far more reliably on small,
        # zero-centered-ish ranges than on raw 0-255 integers.
        image = image.astype(np.float32) / 255.0

        # Add a channel dimension: (H, W) -> (1, H, W). PyTorch conv layers
        # expect (channels, height, width), and OCT is single-channel
        # grayscale, so channels=1.
        image_tensor = torch.from_numpy(image).unsqueeze(0)

        # Mask stays as integer class labels (H, W), no channel dimension —
        # this is the shape torch.nn.CrossEntropyLoss expects for targets.
        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor

    def subject_for_index(self, idx: int):
        """
        Debugging helper: given a flat dataset index, tell you which
        subject and scan it actually came from. Useful when a training
        loop reports a weird loss on batch N and you want to go inspect
        the actual image.
        """
        subj_idx, scan = self._index[idx]
        return self.subjects[subj_idx].subject_id, scan