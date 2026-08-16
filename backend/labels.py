"""Label vocabulary and colour assignment.

A label's colour is derived from its name, so the same label looks the same
across sessions and machines without anything being stored. Collisions inside
one workspace are resolved by stepping to the next palette slot that is far
enough away in CIE L*a*b*, and the resolved mapping is written into the export
so a consumer never has to re-derive it.
"""
from __future__ import annotations
import hashlib

PALETTE: list[str] = [
    "#E8453C",  # red
    "#3B82F6",  # blue
    "#22C55E",  # green
    "#F59E0B",  # amber
    "#A855F7",  # purple
    "#06B6D4",  # cyan
    "#EC4899",  # pink
    "#84CC16",  # lime
    "#F97316",  # orange
    "#14B8A6",  # teal
    "#6366F1",  # indigo
    "#EAB308",  # yellow
]
"""Saturated, mutually distinguishable hues that stay legible over grayscale
scans and colour photographs alike. Deliberately excludes near-black/near-white/
low-chroma tones, which would vanish against the image underneath."""


def canonical(label: str) -> str:
    """Normalise a label for matching.

    Labels are matched case- and whitespace-insensitively, so ``"Left Lung"``
    and ``"left  lung"`` are the same label.

    Args:
        label: Label as typed.

    Returns:
        Lowercased text with runs of whitespace collapsed to single spaces.
    """
    return " ".join(label.strip().split()).lower()


def default_color(label: str) -> str:
    """Pick a label's preferred colour.

    Args:
        label: Label as typed; normalised internally.

    Returns:
        A hex colour from :data:`PALETTE`, deterministic across sessions and
        machines.
    """
    h = hashlib.md5(canonical(label).encode("utf-8")).digest()
    return PALETTE[h[0] % len(PALETTE)]


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """Convert a hex colour to channel values.

    Args:
        color: ``"#RRGGBB"`` or ``"RRGGBB"``.

    Returns:
        ``(r, g, b)``, each 0-255.
    """
    c = color.lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


def _to_lab(color: str) -> tuple[float, float, float]:
    """Convert sRGB to CIE L*a*b* (D65), so colours can be compared the way eyes do.

    Args:
        color: Hex colour.

    Returns:
        ``(L*, a*, b*)``.
    """
    def lin(v: float) -> float:
        """Undo the sRGB transfer function.

        Args:
            v: One channel, 0-255.

        Returns:
            The linear-light value, 0-1.
        """
        v /= 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(v) for v in hex_to_rgb(color))
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 1.00000
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        """Apply the L*a*b* nonlinearity, with its linear segment near zero.

        Args:
            t: One tristimulus value, already divided by its white point.

        Returns:
            The companded value.
        """
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)

    fx, fy, fz = f(x), f(y), f(z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def color_distance(a: str, b: str) -> float:
    """Perceptual distance between two colours.

    Args:
        a: Hex colour.
        b: Hex colour.

    Returns:
        CIE76 delta-E. Roughly: below 25 reads as "the same colour" at swatch
        size.
    """
    la, lb = _to_lab(a), _to_lab(b)
    return sum((x - y) ** 2 for x, y in zip(la, lb)) ** 0.5


MIN_DISTANCE = 28.0
"""Two labels whose colours are closer than this are treated as colliding."""


class LabelRegistry:
    """Label -> colour assignment for one workspace.

    The colour is derived from the label name so the same label looks the same
    everywhere by default. If two distinct labels happen to hash to the same
    slot, the later one steps to the next free slot so the user can still tell
    them apart. The resolved mapping is written into the export.

    A registry is shared by every image in a workspace, which is what makes a
    label keep its colour across the whole folder.
    """

    def __init__(self) -> None:
        """Start with no labels assigned."""
        self._colors: dict[str, str] = {}

    def color_for(self, label: str) -> str:
        """Get a label's colour, assigning one on first use.

        Args:
            label: Label as typed; matched canonically.

        Returns:
            A hex colour from :data:`PALETTE`, stable for the lifetime of the
            registry. Falls back to the most distant remaining slot once every
            palette entry collides.
        """
        key = canonical(label)
        if key in self._colors:
            return self._colors[key]

        preferred = default_color(key)
        taken = list(self._colors.values())
        color = preferred

        def clashes(c: str) -> bool:
            """Test a candidate colour against the ones already handed out.

            Args:
                c: Hex colour to test.

            Returns:
                True if it is within :data:`MIN_DISTANCE` of any taken colour.
            """
            return any(color_distance(c, t) < MIN_DISTANCE for t in taken)

        if clashes(color):
            start = PALETTE.index(preferred)
            rotated = [PALETTE[(start + i) % len(PALETTE)] for i in range(1, len(PALETTE))]
            # Prefer a clearly distinct slot; fall back to the most distant one
            # available once the palette is exhausted.
            free = [c for c in rotated if not clashes(c)]
            if free:
                color = free[0]
            elif taken:
                color = max(rotated, key=lambda c: min(color_distance(c, t) for t in taken))

        self._colors[key] = color
        return color

    def as_dict(self) -> dict[str, str]:
        """Snapshot the assignments.

        Returns:
            A copy mapping canonical label to hex colour.
        """
        return dict(self._colors)

    def restore(self, colors: dict[str, str]) -> None:
        """Adopt a previously saved assignment.

        Reopening a saved session has to put every label back on the colour the
        user learned; re-deriving them would be almost right, and wrong exactly
        where a collision had been resolved.

        Args:
            colors: Canonical label to hex colour, as :meth:`as_dict` produced.
        """
        self._colors = {canonical(k): v for k, v in colors.items()}
