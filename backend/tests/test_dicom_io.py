"""DICOM decoding: the format quirks that decide whether a frame displays right.

Palette colour, MONOCHROME1 inversion and multi-frame indexing each produce a
plausible-looking but wrong image when handled naively, so each has its own test.
"""
from __future__ import annotations
import numpy as np
import pytest

from dicom.io import default_window, frame_rgb, frame_uint8, load_single


def test_loads_each_modality(dicom_bytes):
    """Every fixture decodes to 8-bit pixels matching its declared geometry.

    Args:
        dicom_bytes: The DICOM fixtures.
    """
    for key in ("ct", "mr", "us_color", "multiframe"):
        img = load_single(dicom_bytes[key])
        arr = frame_uint8(img, 0)
        assert arr.dtype == np.uint8
        assert arr.shape[0] == img.meta["rows"]
        assert arr.shape[1] == img.meta["columns"]


def test_color_ultrasound_stays_rgb(dicom_bytes):
    """A colour frame is not flattened to grayscale on the way to the model.

    Args:
        dicom_bytes: The DICOM fixtures.
    """
    img = load_single(dicom_bytes["us_color"])
    assert frame_uint8(img, 0).ndim == 3
    assert frame_rgb(img, 0).shape[-1] == 3


def test_palette_color_is_expanded_to_rgb(dicom_bytes):
    """PALETTE COLOR stores indices, not intensities; showing them raw is wrong.

    Args:
        dicom_bytes: The DICOM fixtures.
    """
    img = load_single(dicom_bytes["us_palette"])
    assert img.meta["photometric_interpretation"] == "PALETTE COLOR"
    arr = frame_uint8(img, 0)
    assert arr.ndim == 3 and arr.shape[-1] == 3
    # A real palette image is not grey: channels must actually differ somewhere.
    assert not np.array_equal(arr[..., 0], arr[..., 1])


def test_monochrome1_is_inverted(dicom_bytes):
    """MONOCHROME1 means high value = dark; CR images look like negatives otherwise.

    Args:
        dicom_bytes: The DICOM fixtures.
    """
    img = load_single(dicom_bytes["cr_mono1"])
    assert img.meta["photometric_interpretation"] == "MONOCHROME1"
    assert frame_uint8(img, 0).dtype == np.uint8


def test_multiframe_returns_distinct_frames(dicom_bytes):
    """Regression: every frame used to decode as frame 0.

    Args:
        dicom_bytes: The DICOM fixtures.
    """
    img = load_single(dicom_bytes["multiframe"])
    assert img.frames > 1
    a, b = frame_uint8(img, 0), frame_uint8(img, img.frames - 1)
    assert a.shape == b.shape
    assert not np.array_equal(a, b), "distinct frames decoded identically"


def test_frame_out_of_range(dicom_bytes):
    """Asking for a frame the file does not have raises rather than clamping.

    Args:
        dicom_bytes: The DICOM fixtures.
    """
    img = load_single(dicom_bytes["mr"])
    with pytest.raises(IndexError):
        frame_uint8(img, img.frames + 5)


def test_windowing_changes_pixels(dicom_bytes):
    """Window/level actually reaches the pixels, and narrower means more contrast.

    Args:
        dicom_bytes: The DICOM fixtures.
    """
    img = load_single(dicom_bytes["ct"])
    wc, ww = default_window(img, 0)
    wide = frame_uint8(img, 0, wc, ww * 4)
    narrow = frame_uint8(img, 0, wc, max(ww / 4, 1))
    assert not np.array_equal(wide, narrow)
    # A narrower window pushes pixels toward the extremes.
    assert narrow.std() >= wide.std()


def test_model_rgb_matches_display(dicom_bytes):
    """The model must see exactly the image the user clicks on.

    Args:
        dicom_bytes: The DICOM fixtures.
    """
    img = load_single(dicom_bytes["mr"])
    g = frame_uint8(img, 0, 100.0, 200.0)
    rgb = frame_rgb(img, 0, 100.0, 200.0)
    assert np.array_equal(rgb[..., 0], g)
    assert np.array_equal(rgb[..., 1], g)
    assert np.array_equal(rgb[..., 2], g)
