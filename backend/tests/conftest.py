"""Shared fixtures.

Everything here is built from files that ship with pydicom or is synthesised in
process, so the suite needs no downloads, no GPU and no patient data. The whole
API is exercised against the stub model.
"""
from __future__ import annotations
import io
import os
import sys
from pathlib import Path

import pytest

# Tests exercise the full API without downloading SAM weights or needing a GPU.
os.environ.setdefault("SAM2_STUB", "1")

# Memory-only by default, so a test run never writes into the user's real data
# directory. test_persistence.py turns it back on against a tmp_path of its own.
os.environ.setdefault("SAM2_PERSIST", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydicom.data import get_testdata_file  # noqa: E402

FIXTURES = {
    "ct": "CT_small.dcm",
    "mr": "MR_small.dcm",
    "multiframe": "emri_small.dcm",
    "us_color": "US1_J2KR.dcm",
    "us_palette": "OBXXXX1A.dcm",
    "cr_mono1": "RG1_UNCR.dcm",
}
"""Public, redistributable DICOM fixtures bundled with pydicom -- no PHI, no
download, no licence friction. They cover the paths that actually differ:

    CT_small    16-bit monochrome CT with a modality LUT (rescale slope/intercept)
    MR_small    16-bit monochrome MR
    emri_small  multi-frame MR (10 frames in one file)
    US1_J2KR    colour ultrasound, JPEG2000-compressed
    OBXXXX1A    PALETTE COLOR ultrasound (indices, not intensities)
    RG1_UNCR    MONOCHROME1 chest CR (inverted display)
"""


@pytest.fixture(scope="session")
def dicom_bytes() -> dict[str, bytes]:
    """Load the DICOM fixtures.

    Returns:
        The key from :data:`FIXTURES` mapped to the file's raw bytes.
    """
    out = {}
    for key, name in FIXTURES.items():
        path = get_testdata_file(name)
        assert path, f"pydicom fixture {name} unavailable"
        out[key] = Path(path).read_bytes()
    return out


def _png(arr) -> bytes:
    """Encode an array as PNG bytes.

    Args:
        arr: Anything ``numpy.asarray`` accepts and Pillow can save.

    Returns:
        The encoded PNG.
    """
    import numpy as np
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def raster_bytes() -> dict[str, bytes]:
    """Synthesise ordinary pictures: the formats a user drags in next to DICOMs.

    Each is 96x128 and carries a bright rectangle to click on.

    Returns:
        Encoded bytes for an 8-bit grayscale PNG, an RGB PNG, a 16-bit PNG
        (where windowing still matters) and a JPEG.
    """
    import numpy as np
    from PIL import Image

    y, x = np.mgrid[0:96, 0:128]
    gray8 = ((x * 2) % 256).astype(np.uint8)
    gray8[30:70, 40:90] = 240                       # a blob to click on
    rgb = np.stack([gray8, np.roll(gray8, 10, 1), np.roll(gray8, 20, 1)], -1)
    deep = (gray8.astype(np.uint16) * 257)          # 16-bit, so windowing matters

    jpeg = io.BytesIO()
    Image.fromarray(rgb).save(jpeg, format="JPEG", quality=90)

    return {
        "png_gray": _png(gray8),
        "png_rgb": _png(rgb),
        "png16": _png(deep),
        "jpeg": jpeg.getvalue(),
    }


@pytest.fixture(scope="session")
def folder_files(dicom_bytes, raster_bytes) -> list[tuple[str, tuple[str, bytes, str]]]:
    """Build what a browser sends for a picked folder.

    Many parts named ``files``, each carrying its path within the folder. One is
    not an image, so the batch also covers the reported-not-fatal path.

    Args:
        dicom_bytes: The DICOM fixtures.
        raster_bytes: The synthesised pictures.

    Returns:
        Parts ready to pass as ``files=`` to the test client.
    """
    return [
        ("files", ("scan/img10.dcm", dicom_bytes["ct"], "application/dicom")),
        ("files", ("scan/img2.dcm", dicom_bytes["mr"], "application/dicom")),
        ("files", ("scan/photo.png", raster_bytes["png_gray"], "image/png")),
        ("files", ("scan/notes.txt", b"not an image", "text/plain")),
    ]


@pytest.fixture()
def client():
    """Start the app with a fresh client.

    The app module is imported lazily, after ``SAM2_STUB`` is set, so no weights
    are ever fetched.

    Yields:
        fastapi.testclient.TestClient: Bound to the real application, store and
        stub runner.
    """
    from fastapi.testclient import TestClient
    import app as app_module

    with TestClient(app_module.app) as c:
        yield c


@pytest.fixture()
def uploaded(client, dicom_bytes):
    """Upload one MR file, for tests that need an image but not a folder.

    Args:
        client: The test client.
        dicom_bytes: The DICOM fixtures.

    Returns:
        The upload response, including ``image_id`` and the default window.
    """
    r = client.post("/dicom/upload", files={"file": ("MR_small.dcm", dicom_bytes["mr"], "application/dicom")})
    assert r.status_code == 200, r.text
    return r.json()
