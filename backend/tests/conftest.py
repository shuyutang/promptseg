from __future__ import annotations
import io
import os
import sys
from pathlib import Path

import pytest

# Tests exercise the full API without downloading SAM weights or needing a GPU.
os.environ.setdefault("SAM2_STUB", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydicom.data import get_testdata_file  # noqa: E402

# Public, redistributable DICOM fixtures bundled with pydicom -- no PHI, no
# download, no licence friction. They cover the paths that actually differ:
#   CT_small   16-bit monochrome CT with a modality LUT (rescale slope/intercept)
#   MR_small   16-bit monochrome MR
#   emri_small multi-frame MR (10 frames in one file)
#   US1_J2KR   colour ultrasound, JPEG2000-compressed
#   OBXXXX1A   PALETTE COLOR ultrasound (indices, not intensities)
#   RG1_UNCR   MONOCHROME1 chest CR (inverted display)
FIXTURES = {
    "ct": "CT_small.dcm",
    "mr": "MR_small.dcm",
    "multiframe": "emri_small.dcm",
    "us_color": "US1_J2KR.dcm",
    "us_palette": "OBXXXX1A.dcm",
    "cr_mono1": "RG1_UNCR.dcm",
}


@pytest.fixture(scope="session")
def dicom_bytes() -> dict[str, bytes]:
    out = {}
    for key, name in FIXTURES.items():
        path = get_testdata_file(name)
        assert path, f"pydicom fixture {name} unavailable"
        out[key] = Path(path).read_bytes()
    return out


def _png(arr) -> bytes:
    import numpy as np
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def raster_bytes() -> dict[str, bytes]:
    """Ordinary pictures: the formats a user drags in next to their DICOMs."""
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
    """What a browser sends for a picked folder: many parts named `files`, each
    carrying its path within the folder."""
    return [
        ("files", ("scan/img10.dcm", dicom_bytes["ct"], "application/dicom")),
        ("files", ("scan/img2.dcm", dicom_bytes["mr"], "application/dicom")),
        ("files", ("scan/photo.png", raster_bytes["png_gray"], "image/png")),
        ("files", ("scan/notes.txt", b"not an image", "text/plain")),
    ]


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    import app as app_module

    with TestClient(app_module.app) as c:
        yield c


@pytest.fixture()
def uploaded(client, dicom_bytes):
    r = client.post("/dicom/upload", files={"file": ("MR_small.dcm", dicom_bytes["mr"], "application/dicom")})
    assert r.status_code == 200, r.text
    return r.json()
