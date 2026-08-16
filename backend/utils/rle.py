"""COCO-style uncompressed run-length encoding for binary masks.

Masks travel to the client and into the export as run lengths rather than
pixels. The encoding is the COCO uncompressed variant: column-major (Fortran)
runs that always begin with a run of zeros, possibly of length zero. Consumers
that already read COCO need no special case for this export.
"""
from __future__ import annotations
import numpy as np


def mask_to_rle(mask: np.ndarray) -> dict:
    """Encode a binary mask as column-major run lengths.

    Args:
        mask: HxW array of 0/1 (any integer or boolean dtype).

    Returns:
        ``{"counts": [int, ...], "size": [height, width]}``. The first run is
        always a run of zeros, so a mask whose top-left pixel is set gets a
        leading ``0`` count.
    """
    h, w = mask.shape
    flat = np.asarray(mask, dtype=np.uint8).ravel(order="F")
    if flat.size == 0:
        return {"counts": [], "size": [int(h), int(w)]}

    # Indices where the value changes, plus the implicit boundaries.
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    runs = np.diff(bounds)

    counts = runs.tolist()
    if flat[0] == 1:
        # COCO requires the first run to be zeros.
        counts = [0] + counts
    return {"counts": [int(c) for c in counts], "size": [int(h), int(w)]}


def rle_to_mask(rle: dict) -> np.ndarray:
    """Decode run lengths back into a mask. Inverse of :func:`mask_to_rle`.

    Args:
        rle: ``{"counts": [...], "size": [height, width]}`` as produced by
            :func:`mask_to_rle`.

    Returns:
        HxW uint8 array of 0/1.
    """
    h, w = rle["size"]
    counts = list(rle["counts"])
    flat = np.zeros(h * w, dtype=np.uint8)
    pos, value = 0, 0
    for run in counts:
        if value:
            flat[pos : pos + run] = 1
        pos += run
        value ^= 1
    return flat.reshape((h, w), order="F")


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    """Tight bounding box around the foreground.

    Args:
        mask: HxW array; any non-zero pixel counts as foreground.

    Returns:
        ``[x, y, width, height]``, or None if the mask is empty.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
