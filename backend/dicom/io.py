"""DICOM pixel decoding, windowing and display rendering.

The chain from stored bytes to a displayable frame has several places where a
naive read shows the wrong image: palette colour stores indices rather than
intensities, MONOCHROME1 stores an inverted scale, and a multi-frame file's
pixel array has to be indexed before anything else happens. This module owns all
of it, so the rest of the app only ever sees 8-bit pixels.

The frame it produces is both what the browser draws and what the model is
given, which is what makes a click at (x, y) mean the same thing to the user and
to SAM.
"""
from __future__ import annotations
import io
from dataclasses import dataclass, field

import numpy as np
import pydicom
try:  # pydicom >= 3
    from pydicom.pixels import apply_color_lut, apply_modality_lut, apply_voi_lut
except ImportError:  # pragma: no cover - pydicom 2.x
    from pydicom.pixel_data_handlers.util import (
        apply_color_lut, apply_modality_lut, apply_voi_lut,
    )


@dataclass
class DicomImage:
    """A loaded study: either one file (possibly multi-frame) or a sorted series.

    Attributes:
        kind: ``"single"`` for one file, ``"series"`` for one dataset per frame.
        datasets: The parsed pydicom datasets.
        frames: Total frames addressable through this image.
        meta: Non-identifying technical metadata, from :func:`_meta`.
        _cache: Frame index to post-modality-LUT array. Decoding is expensive
            for compressed transfer syntaxes, and windowing is re-applied on
            top, so decoded frames are kept.
    """
    kind: str                       # "single" | "series"
    datasets: list[pydicom.Dataset]
    frames: int
    meta: dict
    _cache: dict[int, np.ndarray] = field(default_factory=dict)


def _is_color(ds: pydicom.Dataset, arr: np.ndarray) -> bool:
    """Decide whether a decoded frame is colour.

    Args:
        ds: The dataset the frame came from.
        arr: The decoded pixel array.

    Returns:
        True for RGB/YBR data or three samples per pixel.
    """
    spp = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    photo = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2") or "")
    return spp == 3 or (arr.ndim == 3 and arr.shape[-1] == 3) or photo.upper().startswith(("RGB", "YBR"))


def _first(value, default=None) -> float | None:
    """Read a tag that may be single-valued or a MultiValue.

    WindowCenter and WindowWidth are the usual offenders.

    Args:
        value: The raw tag value, or None.
        default: What to return when the value is missing or unparseable.

    Returns:
        The first value as a float, or ``default``.
    """
    if value is None:
        return default
    if isinstance(value, pydicom.multival.MultiValue):
        return float(value[0]) if len(value) else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_dataset(b: bytes) -> pydicom.Dataset:
    """Parse DICOM bytes.

    Args:
        b: Raw file bytes.

    Returns:
        The dataset. Read with ``force=True``, so headerless files parse too.
    """
    return pydicom.dcmread(io.BytesIO(b), force=True)


def load_single(b: bytes) -> DicomImage:
    """Load one DICOM file.

    Args:
        b: Raw file bytes.

    Returns:
        A :class:`DicomImage` of kind ``"single"``, with one frame or many.

    Raises:
        ValueError: If the file carries no PixelData.
    """
    ds = read_dataset(b)
    if "PixelData" not in ds:
        raise ValueError("File contains no PixelData.")
    frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    return DicomImage("single", [ds], frames, _meta(ds, frames))


def _meta(ds: pydicom.Dataset, frames: int) -> dict:
    """Collect non-identifying technical metadata, carried into the export.

    Args:
        ds: The parsed dataset.
        frames: Frame count, already resolved.

    Returns:
        Modality, geometry, UIDs, series description, pixel spacing and the
        file's own window. No patient identifiers.
    """
    return {
        "modality": str(getattr(ds, "Modality", "") or ""),
        "rows": int(getattr(ds, "Rows", 0) or 0),
        "columns": int(getattr(ds, "Columns", 0) or 0),
        "frames": frames,
        "photometric_interpretation": str(getattr(ds, "PhotometricInterpretation", "") or ""),
        "sop_class_uid": str(getattr(ds, "SOPClassUID", "") or ""),
        "series_instance_uid": str(getattr(ds, "SeriesInstanceUID", "") or ""),
        "study_instance_uid": str(getattr(ds, "StudyInstanceUID", "") or ""),
        "series_description": str(getattr(ds, "SeriesDescription", "") or ""),
        "pixel_spacing": [float(v) for v in (getattr(ds, "PixelSpacing", None) or [])],
        "window_center": _first(getattr(ds, "WindowCenter", None)),
        "window_width": _first(getattr(ds, "WindowWidth", None)),
    }


def dataset_for_frame(img: DicomImage, frame: int) -> tuple[pydicom.Dataset, int]:
    """Map a global frame index onto the dataset that holds it.

    Args:
        img: The loaded image.
        frame: Global frame index.

    Returns:
        ``(dataset, index_within_that_dataset)``.

    Raises:
        IndexError: If the frame index is out of range.
    """
    if frame < 0 or frame >= img.frames:
        raise IndexError(f"frame {frame} out of range (0..{img.frames - 1})")
    if img.kind == "series":
        return img.datasets[frame], 0
    return img.datasets[0], frame


def frame_shape(img: DicomImage, frame: int) -> tuple[int, int]:
    """Get one frame's geometry, from the header -- no pixel decode.

    A picked folder routinely holds images of different sizes, so geometry is
    per frame, not per study.

    Args:
        img: The loaded image.
        frame: Global frame index.

    Returns:
        ``(rows, columns)``.

    Raises:
        IndexError: If the frame index is out of range.
    """
    ds, _ = dataset_for_frame(img, frame)
    return int(getattr(ds, "Rows", 0) or 0), int(getattr(ds, "Columns", 0) or 0)


def _raw_frame(img: DicomImage, frame: int) -> np.ndarray:
    """Decode one frame and apply the modality LUT, caching the result.

    Args:
        img: The loaded image.
        frame: Global frame index.

    Returns:
        Grayscale float32 in modality units, or uint8 RGB for colour and
        palette images.

    Raises:
        IndexError: If the frame index is out of range.
    """
    if frame in img._cache:
        return img._cache[frame]

    ds, sub = dataset_for_frame(img, frame)
    arr = ds.pixel_array

    n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if n_frames > 1 and arr.ndim >= 3:
        arr = arr[sub]  # (F,H,W) or (F,H,W,3) -- this is the frame bug that was here

    photo = str(getattr(ds, "PhotometricInterpretation", "") or "").upper()
    if photo == "PALETTE COLOR":
        # SamplesPerPixel is 1, but the stored values are palette indices, not
        # intensities -- rendering them as grayscale shows the wrong image.
        rgb = apply_color_lut(arr, ds)
        out = rgb.astype(np.uint8) if rgb.dtype == np.uint8 else _minmax_u8(rgb.astype(np.float32))
    elif _is_color(ds, arr):
        out = arr if arr.dtype == np.uint8 else _minmax_u8(arr.astype(np.float32))
    else:
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        out = apply_modality_lut(arr, ds).astype(np.float32)

    img._cache[frame] = out
    return out


def _minmax_u8(a: np.ndarray) -> np.ndarray:
    """Scale an array to 8 bits by its own range.

    Args:
        a: Any numeric array.

    Returns:
        uint8 array of the same shape; all zeros if the input is constant.
    """
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.clip((a - lo) / (hi - lo) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def default_window(img: DicomImage, frame: int) -> tuple[float, float]:
    """Get the window the viewer opens with.

    Args:
        img: The loaded image.
        frame: Global frame index.

    Returns:
        ``(center, width)``: the file's own WindowCenter/WindowWidth when
        present, otherwise the 1st-99th percentile span of the pixels.

    Raises:
        IndexError: If the frame index is out of range.
    """
    ds, _ = dataset_for_frame(img, frame)
    wc = _first(getattr(ds, "WindowCenter", None))
    ww = _first(getattr(ds, "WindowWidth", None))
    if wc is not None and ww:
        return float(wc), float(ww)

    a = _raw_frame(img, frame)
    if a.dtype == np.uint8 and a.ndim == 3:
        return 127.5, 255.0
    lo, hi = np.percentile(a, (1, 99))
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max()) or 1.0
    return float((lo + hi) / 2.0), float(max(hi - lo, 1e-6))


def frame_uint8(img: DicomImage, frame: int,
                window_center: float | None = None,
                window_width: float | None = None) -> np.ndarray:
    """Render one frame for display.

    This is exactly what the browser shows AND what the model sees, so a click
    at (x, y) means the same thing to the user and to SAM.

    Args:
        img: The loaded image.
        frame: Global frame index.
        window_center: Explicit window centre. Omit both this and the width to
            use the file's VOI LUT, which may be a lookup table rather than a
            simple centre/width.
        window_width: Explicit window width.

    Returns:
        8-bit HxW grayscale, or HxWx3 RGB for colour images. MONOCHROME1 is
        inverted so it displays the way a reader expects.

    Raises:
        IndexError: If the frame index is out of range.
    """
    a = _raw_frame(img, frame)
    if a.ndim == 3:  # already-normalised color
        return a

    ds, _ = dataset_for_frame(img, frame)

    if window_center is None or window_width is None:
        # No explicit window: prefer the file's VOI LUT, which may be a table
        # rather than a simple centre/width.
        try:
            v = apply_voi_lut(a.astype(a.dtype), ds).astype(np.float32)
            out = _minmax_u8(v)
        except Exception:
            wc, ww = default_window(img, frame)
            out = _window_u8(a, wc, ww)
    else:
        out = _window_u8(a, float(window_center), float(window_width))

    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        out = 255 - out
    return out


def _window_u8(a: np.ndarray, wc: float, ww: float) -> np.ndarray:
    """Apply a linear window to a frame.

    Args:
        a: Frame in modality units.
        wc: Window centre.
        ww: Window width; clamped away from zero.

    Returns:
        uint8 array of the same shape.
    """
    ww = max(float(ww), 1e-6)
    lo = wc - ww / 2.0
    return np.clip((a - lo) / ww * 255.0 + 0.5, 0, 255).astype(np.uint8)


def frame_rgb(img: DicomImage, frame: int,
              window_center: float | None = None,
              window_width: float | None = None) -> np.ndarray:
    """Render one frame for the model.

    Args:
        img: The loaded image.
        frame: Global frame index.
        window_center: Explicit window centre, or None for the file's own.
        window_width: Explicit window width, or None for the file's own.

    Returns:
        HxWx3 uint8, grayscale replicated across channels.

    Raises:
        IndexError: If the frame index is out of range.
    """
    g = frame_uint8(img, frame, window_center, window_width)
    if g.ndim == 3:
        return g
    return np.stack([g, g, g], axis=-1)
