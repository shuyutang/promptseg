"""Turning an upload into an ordered list of image sources.

Two jobs a folder pick makes necessary. First, deciding what each file is:
content before extension, because folder exports routinely have no extension at
all. Second, ordering: naturally by filename, except inside one DICOM series,
where slice position wins -- exported slices are often named in acquisition
order, which is not anatomical order.

One unreadable file in a folder of 200 must not fail the batch, so failures are
collected per file and reported alongside the images that did load.
"""
from __future__ import annotations
import io
import posixpath
import re
import zipfile

from media.base import ImageSource
from media.dicom_source import DicomSource
from media.raster_source import EXTENSIONS as RASTER_EXTENSIONS, RasterSource

DICOM_EXTENSIONS = {".dcm", ".dicom", ".ima", ".img", ""}
"""Extensions read as DICOM when the file has no ``DICM`` preamble."""

_JUNK = re.compile(r"(^|/)(\.|__MACOSX/|Thumbs\.db$|DICOMDIR$)", re.IGNORECASE)
"""Files a folder picker sweeps up that are never images."""


class LoadError(Exception):
    """One file failed; the rest of the batch still loads."""


def _ext(name: str) -> str:
    """Get a lowercased extension.

    Args:
        name: File name or path within the upload.

    Returns:
        The extension including its dot, or ``""`` if there is none.
    """
    return posixpath.splitext(name.lower())[1]


def looks_like_dicom(data: bytes) -> bool:
    """Test for the DICOM preamble.

    Args:
        data: Raw file bytes.

    Returns:
        True if bytes 128-132 are ``DICM``.
    """
    return len(data) > 132 and data[128:132] == b"DICM"


def load_one(filename: str, data: bytes) -> ImageSource:
    """Load a single file, choosing the reader by content first, name second.

    Args:
        filename: Name or path within the upload, used for the extension hint
            and for error messages.
        data: Raw file bytes.

    Returns:
        A :class:`~media.dicom_source.DicomSource` or
        :class:`~media.raster_source.RasterSource`.

    Raises:
        LoadError: If the file is empty or no reader can decode it. The message
            is prefixed with the filename, since it is shown per file.
    """
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
    """Build a natural sort key: img2 before img10, which string sorting gets backwards.

    Args:
        name: File name or path; only the basename is used.

    Returns:
        A tuple of per-token keys, comparable against other keys from this
        function.
    """
    base = posixpath.basename(name).lower()
    return tuple(
        (0, int(t), "") if t.isdigit() else (1, 0, t)
        for t in re.findall(r"\d+|\D+", base)
    )


def _series_of(src: ImageSource) -> tuple[str, float | None]:
    """Get the series identity and slice position of a source.

    Position is only meaningful within a series -- sorting two unrelated scans
    by z would interleave them.

    Args:
        src: A loaded file. Non-DICOM sources have no series.

    Returns:
        ``(series_instance_uid, position)``. Position is the z component of
        ImagePositionPatient, falling back to InstanceNumber, or None if neither
        is present.
    """
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
    """Order a folder the way a reader expects.

    Folders stay together, each series is contiguous and in slice order, and
    everything else falls back to natural filename order.

    Args:
        items: ``(name, source)`` pairs in upload order.

    Returns:
        The same pairs, reordered.
    """
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
    """Expand a zip into its members; anything else stays one file.

    Args:
        filename: Name of the uploaded file.
        data: Raw file bytes.

    Returns:
        ``(name, bytes)`` pairs. Directory entries and junk files are dropped.

    Raises:
        LoadError: If the zip holds no usable members.
    """
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
    """Load many uploaded files, expanding zips and collecting per-file errors.

    Args:
        files: ``(name, bytes)`` pairs as uploaded. A picked folder arrives as
            many pairs whose names carry the path within the folder.

    Returns:
        ``(loaded, errors)``: the readable files as ``(name, source)`` pairs in
        display order, and one message per file that failed. Failures are
        returned rather than raised so one bad file cannot fail the batch.
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
