"""
losses.py

Loss functions for OCT retinal layer segmentation.

We'll begin with standard CrossEntropyLoss since our masks are
integer class labels. Later we'll add Dice loss and combine the two.
"""

import torch.nn as nn


def create_cross_entropy_loss():
    """
    Standard multi-class segmentation loss.

    Expects:
        prediction: (B, C, H, W)
        target:     (B, H, W)
    """

    return nn.CrossEntropyLoss()