"""Mask encoding and label colours.

Two things a consumer of the export depends on: that the RLE decodes back to the
mask that was drawn, and that two labels in one image never look alike.
"""
from __future__ import annotations
import numpy as np
import pytest

from labels import LabelRegistry, canonical, default_color, hex_to_rgb
from utils.rle import mask_bbox, mask_to_rle, rle_to_mask


@pytest.mark.parametrize("mask", [
    np.zeros((8, 6), np.uint8),
    np.ones((8, 6), np.uint8),
    np.eye(9, dtype=np.uint8),
    (np.random.default_rng(0).random((32, 24)) > 0.7).astype(np.uint8),
])
def test_rle_roundtrip(mask):
    """Encoding then decoding returns the same mask, empty and full included.

    Args:
        mask: The parametrised mask under test.
    """
    assert np.array_equal(rle_to_mask(mask_to_rle(mask)), mask)


def test_rle_starts_with_zero_run():
    """COCO requires the first run to count zeros, even when the mask starts at 1."""
    m = np.ones((2, 2), np.uint8)
    rle = mask_to_rle(m)
    assert rle["counts"][0] == 0
    assert sum(rle["counts"]) == 4


def test_bbox():
    """The bounding box is [x, y, w, h], and None for an empty mask."""
    m = np.zeros((10, 10), np.uint8)
    m[3:6, 2:8] = 1
    assert mask_bbox(m) == [2, 3, 6, 3]  # x, y, w, h
    assert mask_bbox(np.zeros((4, 4), np.uint8)) is None


def test_label_color_is_deterministic_and_case_insensitive():
    """A label's colour depends only on its canonical form."""
    assert default_color("Vertebra") == default_color("  vertebra ")
    assert canonical("  Left   Kidney ") == "left kidney"
    assert hex_to_rgb("#E8453C") == (232, 69, 60)


def test_same_label_same_color_distinct_labels_differ():
    """One label keeps one colour; distinct labels do not collide."""
    reg = LabelRegistry()
    a1 = reg.color_for("liver")
    a2 = reg.color_for("LIVER")
    assert a1 == a2, "same label must always get the same colour"

    colors = {reg.color_for(n) for n in ["liver", "spleen", "kidney", "aorta", "spine"]}
    assert len(colors) == 5, "distinct labels must not collide within one image"


def test_distinct_labels_are_perceptually_separable():
    """Regression: 'lung' and 'vertebra' hashed to two near-identical oranges."""
    import itertools
    from labels import MIN_DISTANCE, color_distance

    reg = LabelRegistry()
    names = ["lung", "vertebra", "liver", "spleen", "kidney", "aorta", "cord", "disc"]
    assigned = [reg.color_for(n) for n in names]
    for (na, ca), (nb, cb) in itertools.combinations(zip(names, assigned), 2):
        d = color_distance(ca, cb)
        assert d >= MIN_DISTANCE, f"{na} ({ca}) and {nb} ({cb}) too close: dE={d:.1f}"


def test_color_assignment_is_order_independent_for_a_single_label():
    """A label's colour must not depend on what else is in the image first."""
    assert LabelRegistry().color_for("tumour") == default_color("tumour")
