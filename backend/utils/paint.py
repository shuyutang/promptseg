"""Brush and eraser geometry.

A stroke is a list of pointer positions in native image pixels plus a radius.
Replaying one means stamping a disc at every position and along the segments
between them. Strokes are kept beside the prompts rather than baked into the
mask, so the committed mask is always ``model(prompts)`` with the strokes
replayed on top -- which is what lets a hand correction survive re-prompting.
"""
from __future__ import annotations

import numpy as np

MAX_RADIUS = 256
"""Largest brush radius honoured, in pixels. Clamps hostile or mistyped input."""


def _disc(radius: int) -> np.ndarray:
    """Build the brush stamp.

    Args:
        radius: Disc radius in pixels.

    Returns:
        A ``(2*radius+1, 2*radius+1)`` boolean array, True inside the circle.
    """
    r = int(radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _stamp(mask: np.ndarray, cx: int, cy: int, disc: np.ndarray, value: bool) -> None:
    """Paint one disc into a mask, in place, clipped to the image.

    Args:
        mask: HxW boolean array, modified in place.
        cx: Disc centre x, in image pixels.
        cy: Disc centre y, in image pixels.
        disc: Stamp from :func:`_disc`.
        value: True to add pixels, False to erase them.
    """
    r = disc.shape[0] // 2
    h, w = mask.shape
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    if y0 >= y1 or x0 >= x1:
        return
    sub = disc[y0 - (cy - r):y1 - (cy - r), x0 - (cx - r):x1 - (cx - r)]
    region = mask[y0:y1, x0:x1]
    if value:
        region |= sub
    else:
        region &= ~sub


def _walk(x0: int, y0: int, x1: int, y1: int, step: float) -> list[tuple[int, int]]:
    """Sample points along a segment, spaced closely enough that discs overlap.

    The browser only reports pointer positions every few pixels, so a fast drag
    would otherwise paint a dotted line instead of a stroke.

    Args:
        x0: Start x.
        y0: Start y.
        x1: End x.
        y1: End y.
        step: Maximum spacing between samples, in pixels.

    Returns:
        Integer ``(x, y)`` points including both endpoints.
    """
    dx, dy = x1 - x0, y1 - y0
    dist = float(np.hypot(dx, dy))
    n = max(1, int(dist / max(step, 0.5)))
    return [(int(round(x0 + dx * i / n)), int(round(y0 + dy * i / n))) for i in range(n + 1)]


def apply_strokes(mask: np.ndarray, strokes: list[dict]) -> np.ndarray:
    """Replay brush strokes on top of a model mask, in order.

    Args:
        mask: HxW model mask to start from. Not modified.
        strokes: Stroke dicts, each ``{"mode": "add"|"erase", "radius": int,
            "points": [[x, y], ...]}`` in absolute image coordinates. Missing
            keys fall back to add / radius 6.

    Returns:
        A new HxW boolean array: the input mask with every stroke applied.
        Erasing removes whatever is underneath, model pixels included, so
        erasing what you painted does not restore the original mask.
    """
    out = np.asarray(mask).astype(bool).copy()
    if not strokes:
        return out

    for s in strokes:
        pts = s.get("points") or []
        if not pts:
            continue
        radius = int(max(1, min(int(s.get("radius", 6)), MAX_RADIUS)))
        add = str(s.get("mode", "add")) != "erase"
        disc = _disc(radius)
        step = max(1.0, radius * 0.5)

        prev = None
        for p in pts:
            x, y = int(p[0]), int(p[1])
            if prev is None:
                _stamp(out, x, y, disc, add)
            else:
                for px, py in _walk(prev[0], prev[1], x, y, step):
                    _stamp(out, px, py, disc, add)
            prev = (x, y)

    return out
