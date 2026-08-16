"""PNG encoding for frames, masks and overlays.

Every pixel the browser draws is rendered here and sent as a PNG: the base
frame, the composited layer of committed masks, and the live preview. The
client never touches mask pixels, which keeps mask geometry in exactly one
place.
"""
from __future__ import annotations
import io

import numpy as np
from PIL import Image


def _png(arr: np.ndarray, mode: str) -> bytes:
    """Encode an array as PNG bytes.

    Args:
        arr: Array whose shape matches ``mode``.
        mode: Pillow image mode, e.g. ``"L"``, ``"RGB"`` or ``"RGBA"``.

    Returns:
        The encoded PNG.
    """
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def array_to_png(arr: np.ndarray) -> bytes:
    """Encode a displayable frame.

    Args:
        arr: 8-bit HxW grayscale or HxWx3 RGB.

    Returns:
        The encoded PNG.
    """
    if arr.ndim == 2:
        return _png(arr.astype(np.uint8), "L")
    return _png(arr.astype(np.uint8), "RGB")


def mask_to_png(mask01: np.ndarray) -> bytes:
    """Encode a binary mask as a black-and-white PNG.

    Args:
        mask01: HxW mask. uint8 input is thresholded at zero; anything else is
            clipped to 0..1 and scaled.

    Returns:
        An 8-bit grayscale PNG whose pixels are 0 or 255.
    """
    m = np.asarray(mask01)
    if m.dtype != np.uint8:
        m = (np.clip(m, 0, 1) * 255).astype(np.uint8)
    else:
        m = (m > 0).astype(np.uint8) * 255
    return _png(m, "L")


def _outline(mask: np.ndarray, width: int = 1) -> np.ndarray:
    """Find the boundary pixels of a mask.

    Erodes by 4-connected shifts rather than pulling in scipy for one call.

    Args:
        mask: HxW mask.
        width: Outline thickness in pixels; each unit is one erosion step.

    Returns:
        HxW boolean array, True on the boundary.
    """
    m = mask.astype(bool)
    eroded = m.copy()
    for _ in range(max(1, width)):
        e = eroded.copy()
        e[1:, :] &= eroded[:-1, :]
        e[:-1, :] &= eroded[1:, :]
        e[:, 1:] &= eroded[:, :-1]
        e[:, :-1] &= eroded[:, 1:]
        eroded = e
    return m & ~eroded


def overlay_png(shape: tuple[int, int], items: list[dict], alpha: int = 110) -> bytes:
    """Composite many labelled masks into one RGBA layer.

    Every mask also gets a fully opaque outline, which is what keeps two
    adjacent instances of the same label -- and therefore the same colour --
    visually separable. A selected mask gets a thicker white outline.

    Args:
        shape: ``(height, width)`` of the frame. Masks of any other shape are
            skipped rather than resized.
        items: One dict per mask: ``{"mask": HxW bool/uint8, "color": (r, g, b),
            "selected": bool, "alpha": int}``. ``selected`` and ``alpha`` are
            optional. Later items paint over earlier ones.
        alpha: Default fill opacity, 0-255, for items without their own.

    Returns:
        The encoded RGBA PNG.
    """
    h, w = shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    for item in items:
        m = np.asarray(item["mask"]).astype(bool)
        if m.shape != (h, w) or not m.any():
            continue
        r, g, b = item["color"]
        a = int(item.get("alpha", alpha))

        rgba[m] = (r, g, b, a)
        edge = _outline(m, 2 if item.get("selected") else 1)
        rgba[edge] = (255, 255, 255, 255) if item.get("selected") else (r, g, b, 255)

    return _png(rgba, "RGBA")


def colored_overlay_png(mask01: np.ndarray, color=(232, 69, 60), alpha: int = 110) -> bytes:
    """Render a single mask as an RGBA overlay.

    Used for the live preview, before the mask is committed.

    Args:
        mask01: HxW mask.
        color: ``(r, g, b)`` fill colour.
        alpha: Fill opacity, 0-255.

    Returns:
        The encoded RGBA PNG.
    """
    m = np.asarray(mask01).astype(bool)
    return overlay_png(m.shape, [{"mask": m, "color": tuple(color), "alpha": alpha}], alpha)
