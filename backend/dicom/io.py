from __future__ import annotations
import io
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pydicom
try:  # pydicom >= 3
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
except ImportError:  # pragma: no cover - pydicom 2.x
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut


@dataclass
class DicomImage:
    """A loaded study: either one file (possibly multi-frame) or a sorted series."""
    kind: str                       # "single" | "series"
    datasets: list[pydicom.Dataset]
    frames: int
    meta: dict
    # frame index -> post-modality-LUT float array (decode is expensive for
    # compressed transfer syntaxes, and windowing is re-applied on top)
    _cache: dict[int, np.ndarray] = field(default_factory=dict)


def _is_color(ds: pydicom.Dataset, arr: np.ndarray) -> bool:
    spp = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    photo = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2") or "")
    return spp == 3 or (arr.ndim == 3 and arr.shape[-1] == 3) or photo.upper().startswith(("RGB", "YBR"))


def _first(value, default=None) -> float | None:
    """WindowCenter/Width may be single-valued or a MultiValue."""
    if value is None:
        return default
    if isinstance(value, pydicom.multival.MultiValue):
        return float(value[0]) if len(value) else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_dataset(b: bytes) -> pydicom.Dataset:
    return pydicom.dcmread(io.BytesIO(b), force=True)


def load_single(b: bytes) -> DicomImage:
    ds = read_dataset(b)
    if "PixelData" not in ds:
        raise ValueError("File contains no PixelData.")
    frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    return DicomImage("single", [ds], frames, _meta(ds, frames))


def load_zip_series(zip_bytes: bytes) -> DicomImage:
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    series: list[pydicom.Dataset] = []
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        try:
            ds = pydicom.dcmread(io.BytesIO(zf.read(name)), force=True)
        except Exception:
            continue
        if "PixelData" in ds:
            series.append(ds)
    if not series:
        raise ValueError("No readable DICOM images in zip.")

    def zval(d):
        ipp = getattr(d, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) == 3:
            return float(ipp[2])
        return float(getattr(d, "InstanceNumber", 0) or 0)

    series.sort(key=zval)
    return DicomImage("series", series, len(series), _meta(series[0], len(series)))


def _meta(ds: pydicom.Dataset, frames: int) -> dict:
    """Non-identifying technical metadata, carried into the export for traceability."""
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
    """Map a global frame index onto (dataset, index within that dataset)."""
    if frame < 0 or frame >= img.frames:
        raise IndexError(f"frame {frame} out of range (0..{img.frames - 1})")
    if img.kind == "series":
        return img.datasets[frame], 0
    return img.datasets[0], frame


def _raw_frame(img: DicomImage, frame: int) -> np.ndarray:
    """Decoded frame after the modality LUT. Grayscale float32, or uint8 RGB."""
    if frame in img._cache:
        return img._cache[frame]

    ds, sub = dataset_for_frame(img, frame)
    arr = ds.pixel_array

    n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if n_frames > 1 and arr.ndim >= 3:
        arr = arr[sub]  # (F,H,W) or (F,H,W,3) -- this is the frame bug that was here

    if _is_color(ds, arr):
        out = arr if arr.dtype == np.uint8 else _minmax_u8(arr.astype(np.float32))
    else:
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        out = apply_modality_lut(arr, ds).astype(np.float32)

    img._cache[frame] = out
    return out


def _minmax_u8(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.clip((a - lo) / (hi - lo) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def default_window(img: DicomImage, frame: int) -> tuple[float, float]:
    """The window the viewer opens with: DICOM WC/WW if present, else 1-99 pct."""
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
    """Displayable 8-bit frame. HxW for grayscale, HxWx3 for color.

    This is exactly what the browser shows AND what the model sees, so a click
    at (x, y) means the same thing to the user and to SAM.
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
    ww = max(float(ww), 1e-6)
    lo = wc - ww / 2.0
    return np.clip((a - lo) / ww * 255.0 + 0.5, 0, 255).astype(np.uint8)


def frame_rgb(img: DicomImage, frame: int,
              window_center: float | None = None,
              window_width: float | None = None) -> np.ndarray:
    """HxWx3 uint8 for the model."""
    g = frame_uint8(img, frame, window_center, window_width)
    if g.ndim == 3:
        return g
    return np.stack([g, g, g], axis=-1)
