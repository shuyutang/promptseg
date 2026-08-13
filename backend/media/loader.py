from __future__ import annotations
import io
import posixpath
import re
import zipfile

from media.base import ImageSource
from media.dicom_source import DicomSource
from media.raster_source import EXTENSIONS as RASTER_EXTENSIONS, RasterSource

DICOM_EXTENSIONS = {".dcm", ".dicom", ".ima", ".img", ""}

# Files a folder picker sweeps up that are never images.
_JUNK = re.compile(r"(^|/)(\.|__MACOSX/|Thumbs\.db$|DICOMDIR$)", re.IGNORECASE)


class LoadError(Exception):
    """One file failed; the rest of the batch still loads."""


def _ext(name: str) -> str:
    return posixpath.splitext(name.lower())[1]


def looks_like_dicom(data: bytes) -> bool:
    return len(data) > 132 and data[128:132] == b"DICM"


def load_one(filename: str, data: bytes) -> ImageSource:
    """Load a single file, choosing the reader by content first, name second."""
    if not data:
        raise LoadError(f"{filename}: empty file")

    ext = _ext(filename)
    try:
        if looks_like_dicom(data):
            return DicomSource.from_bytes(data)
        if ext in RASTER_EXTENSIONS:
            return RasterSource(data, filename)
        if ext in DICOM_EXTENSIONS:
            # Headerless DICOM (no preamble) is common in older exports.
            return DicomSource.from_bytes(data)
        # Unknown extension: let Pillow sniff it before giving up.
        return RasterSource(data, filename)
    except LoadError:
        raise
    except Exception as e:
        raise LoadError(f"{filename}: {e}") from e


def _natural(name: str) -> tuple:
    """img2 before img10, which plain string sorting gets backwards."""
    base = posixpath.basename(name).lower()
    return tuple(
        (0, int(t), "") if t.isdigit() else (1, 0, t)
        for t in re.findall(r"\d+|\D+", base)
    )


def _series_of(src: ImageSource) -> tuple[str, float | None]:
    """(series identity, slice position). Position is only meaningful within a
    series -- sorting two unrelated scans by z would interleave them."""
    if not isinstance(src, DicomSource):
        return "", None
    ds = src.dataset
    uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) == 3:
        return uid, float(ipp[2])
    n = getattr(ds, "InstanceNumber", None)
    return uid, (float(n) if n is not None else None)


def sort_loaded(items: list[tuple[str, ImageSource]]) -> list[tuple[str, ImageSource]]:
    """Order a folder the way a reader expects: folders together, each series in
    slice order, everything else naturally by filename."""
    rows = []
    for name, src in items:
        uid, z = _series_of(src)
        rows.append({"name": name, "src": src, "dir": posixpath.dirname(name).lower(),
                     "uid": uid, "z": z, "nat": _natural(name)})

    # A group sits where its first-named member would have sat, so series stay
    # contiguous without jumping to the front of the folder.
    rank: dict[tuple[str, str], tuple] = {}
    for r in rows:
        key = (r["dir"], r["uid"])
        if key not in rank or r["nat"] < rank[key]:
            rank[key] = r["nat"]

    rows.sort(key=lambda r: (
        r["dir"], rank[(r["dir"], r["uid"])], r["uid"],
        (0, r["z"]) if r["z"] is not None else (1, 0.0),
        r["nat"],
    ))
    return [(r["name"], r["src"]) for r in rows]


def expand(filename: str, data: bytes) -> list[tuple[str, bytes]]:
    """A zip becomes its members; anything else stays one file."""
    if _ext(filename) != ".zip":
        return [(filename, data)]
    out: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.is_dir() or _JUNK.search(info.filename):
                continue
            out.append((info.filename, zf.read(info)))
    if not out:
        raise LoadError(f"{filename}: zip contains no files")
    return out


def load_batch(files: list[tuple[str, bytes]]) -> tuple[list[tuple[str, ImageSource]], list[str]]:
    """Load many uploaded files into (name, source) pairs plus per-file errors.

    One unreadable file in a folder of 200 must not fail the whole upload, so
    failures are collected and reported rather than raised.
    """
    loaded: list[tuple[str, ImageSource]] = []
    errors: list[str] = []

    for filename, data in files:
        try:
            members = expand(filename, data)
        except Exception as e:
            errors.append(str(e) if isinstance(e, LoadError) else f"{filename}: {e}")
            continue
        for name, payload in members:
            if _JUNK.search(name):
                continue
            try:
                loaded.append((name, load_one(name, payload)))
            except LoadError as e:
                errors.append(str(e))

    return sort_loaded(loaded), errors
