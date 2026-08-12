from __future__ import annotations
import io
import os
import sys
import zipfile
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


@pytest.fixture(scope="session")
def series_zip(dicom_bytes) -> bytes:
    """A 3-slice 'series' built from public fixtures, for the zip path."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(3):
            zf.writestr(f"slice{i}.dcm", dicom_bytes["mr"])
    return buf.getvalue()


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
