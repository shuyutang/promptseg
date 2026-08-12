#!/usr/bin/env python
"""Collect a curated set of public sample DICOMs into ./samples/.

Every file here ships with pydicom (public, redistributable, no PHI). Run:

    .venv/bin/python scripts/make_samples.py

then drag anything from samples/ into the web UI.
"""
from __future__ import annotations
import io
import shutil
import sys
import warnings
import zipfile
from pathlib import Path

warnings.filterwarnings("ignore")

import pydicom
from pydicom.data import get_testdata_file

OUT = Path(__file__).resolve().parents[1] / "samples"

# (filename, why it is worth trying)
CURATED = [
    ("CT_small.dcm", "128x128 chest CT - small and fast, the README screenshot"),
    ("693_UNCR.dcm", "512x512 CT - realistic resolution"),
    ("eCT_Supplemental.dcm", "512x512 CT, 2 frames"),
    ("MR2_UNCR.dcm", "1024x1024 MR - biggest MR here"),
    ("MR-SIEMENS-DICOM-WithOverlays.dcm", "484x484 Siemens MR"),
    ("emri_small.dcm", "64x64 MR, 10 frames - frame scrubbing"),
    ("RG1_UNCR.dcm", "1955x1841 chest CR, MONOCHROME1 - tests inversion + window/level"),
    ("JPGLosslessP14SV1_1s_1f_8b.dcm", "768x1024 grayscale ultrasound"),
    ("US1_J2KR.dcm", "480x640 colour ultrasound, JPEG2000"),
    ("OBXXXX1A.dcm", "600x800 obstetric ultrasound, PALETTE COLOR"),
    ("color3d_jpeg_baseline.dcm", "480x640 colour US cine, 120 frames"),
]


def split_multiframe_to_zip(name: str, out: Path) -> int:
    """Explode a multi-frame file into single-frame files in a zip.

    Gives a genuine multi-slice *series* (sorted by ImagePositionPatient) to
    exercise the zip upload path, rather than duplicating one slice.
    """
    ds = pydicom.dcmread(get_testdata_file(name))
    frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    arr = ds.pixel_array

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(frames):
            s = pydicom.dcmread(get_testdata_file(name))
            s.NumberOfFrames = 1
            s.PixelData = arr[i].tobytes()
            s.InstanceNumber = i + 1
            # Fabricate a plausible slice position so the sorter has something real.
            s.ImagePositionPatient = [0.0, 0.0, float(i) * 5.0]
            s.SOPInstanceUID = pydicom.uid.generate_uid()
            b = io.BytesIO()
            s.save_as(b, enforce_file_format=True)
            zf.writestr(f"slice_{i:03d}.dcm", b.getvalue())
    out.write_bytes(buf.getvalue())
    return frames


def main() -> int:
    OUT.mkdir(exist_ok=True)
    print(f"writing to {OUT}\n")
    ok = 0
    for name, why in CURATED:
        path = get_testdata_file(name)
        if not path:
            print(f"  SKIP {name} (not available)")
            continue
        shutil.copy(path, OUT / name)
        size = (OUT / name).stat().st_size / 1024
        print(f"  {name:34} {size:7.0f} KB  {why}")
        ok += 1

    zpath = OUT / "emri_series.zip"
    n = split_multiframe_to_zip("emri_small.dcm", zpath)
    print(f"  {zpath.name:34} {zpath.stat().st_size/1024:7.0f} KB  "
          f"{n}-slice MR series (zip upload path)")

    print(f"\n{ok + 1} samples ready. Open the UI and drop one in.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
