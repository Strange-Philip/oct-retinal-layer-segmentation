import numpy as np


def generate_mask(boundaries, image_shape):
    """
    Convert retinal layer boundaries into
    a pixel-wise segmentation mask.

    Parameters
    ----------
    boundaries : np.ndarray
        Shape (8, width)

    image_shape : tuple
        (height, width)

    Returns
    -------
    np.ndarray
        Integer segmentation mask
    """

    height, width = image_shape

    mask = np.zeros((height, width), dtype=np.uint8)

    for x in range(width):

        for region in range(7):

            top = boundaries[region, x]
            bottom = boundaries[region + 1, x]

            if np.isnan(top) or np.isnan(bottom):
                continue

            # Clip to valid image rows. Without this, a negative
            # coordinate (e.g. -2) would wrap around via Python's
            # negative indexing and silently fill the WRONG rows
            # at the bottom of the image instead of being ignored.
            top = max(0, int(round(top)))
            bottom = min(height, int(round(bottom)))

            # If the boundaries are out of order or collapsed onto
            # each other, there's nothing valid to fill for this column.
            if top >= bottom:
                continue

            mask[top:bottom, x] = region + 1

    return mask