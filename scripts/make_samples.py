#!/usr/bin/env python
"""Build ./samples/ -- ready-made folders to try the app on.

Everything here is derived from the public DICOMs bundled with pydicom
(redistributable, no PHI, no download). Run:

    .venv/bin/python scripts/make_samples.py

then press "Open folder…" in the UI and pick samples/mixed_folder.
"""
from __future__ import annotations
import io
import shutil
import sys
import warnings
import zipfile
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402
import pydicom  # noqa: E402
from PIL import Image  # noqa: E402
from pydicom.data import get_testdata_file  # noqa: E402

from media.loader import load_one  # noqa: E402  (also a smoke test of the loader)

OUT = ROOT / "samples"

# A folder that mixes modalities, sizes and file formats -- the case the file
# list and per-file geometry exist for.
MIXED = [
    ("001_ct_chest_128.dcm", "CT_small.dcm", "128x128 CT, fast to click around"),
    ("002_ct_512.dcm", "693_UNCR.dcm", "512x512 CT, realistic resolution"),
    ("003_mr_1024.dcm", "MR2_UNCR.dcm", "1024x1024 MR"),
    ("004_mr_siemens.dcm", "MR-SIEMENS-DICOM-WithOverlays.dcm", "484x484 Siemens MR"),
    ("005_us_colour.dcm", "US1_J2KR.dcm", "480x640 colour ultrasound (JPEG2000)"),
    ("006_us_palette.dcm", "OBXXXX1A.dcm", "600x800 obstetric US, PALETTE COLOR"),
    ("007_cr_chest.dcm", "RG1_UNCR.dcm", "1955x1841 chest CR, MONOCHROME1 (inverted)"),
    ("008_us_grayscale.dcm", "JPGLosslessP14SV1_1s_1f_8b.dcm", "768x1024 grayscale US"),
]

# The same pixels as ordinary pictures, so the PNG/JPEG path has something real
# in it rather than a synthetic gradient.
AS_PICTURES = [
    ("009_ct_chest.png", "CT_small.dcm", "PNG"),
    ("010_us_colour.jpg", "US1_J2KR.dcm", "JPEG"),
    ("011_mr_16bit.png", "MR2_UNCR.dcm", "PNG16"),
]

MULTIFRAME = [
    ("emri_small.dcm", "emri_small.dcm", "64x64 MR, 10 frames in one file"),
    ("us_cine.dcm", "color3d_jpeg_baseline.dcm", "480x640 colour US cine, 120 frames"),
]


def _fetch(name: str) -> Path | None:
    path = get_testdata_file(name)
    return Path(path) if path else None


def write_picture(src: Path, dest: Path, kind: str) -> None:
    """Render a DICOM frame to a normal image file, through the app's own reader."""
    source = load_one(src.name, src.read_bytes())
    if kind == "PNG16":
        # 16-bit greyscale: keeps a real intensity range, so window/level still works.
        raw = source.frame_uint8(0).astype(np.uint16) * 257
        Image.fromarray(raw).save(dest)
        return
    arr = source.frame_uint8(0)
    im = Image.fromarray(arr)
    if kind == "JPEG":
        im.convert("RGB").save(dest, quality=92)
    else:
        im.save(dest)


def explode_series(name: str, out_dir: Path) -> list[Path]:
    """Turn a multi-frame file into a folder of single-frame slices.

    Gives a genuine multi-slice series with real ImagePositionPatient values, so
    the file list is ordered by anatomy rather than by filename.
    """
    ds = pydicom.dcmread(get_testdata_file(name))
    frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    arr = ds.pixel_array
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for i in range(frames):
        s = pydicom.dcmread(get_testdata_file(name))
        s.NumberOfFrames = 1
        s.PixelData = arr[i].tobytes()
        s.InstanceNumber = i + 1
        s.ImagePositionPatient = [0.0, 0.0, float(i) * 5.0]
        s.SOPInstanceUID = pydicom.uid.generate_uid()
        # Named backwards on purpose: the loader should sort by slice position,
        # not by the name it happens to have been exported under.
        path = out_dir / f"export_{frames - i:03d}.dcm"
        s.save_as(path, enforce_file_format=True)
        written.append(path)
    return written


def zip_dir(folder: Path, dest: Path) -> None:
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(folder.iterdir()):
            zf.write(f, f"{folder.name}/{f.name}")


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    mixed = OUT / "mixed_folder"
    mixed.mkdir(parents=True)
    print(f"writing to {OUT}\n")
    print("samples/mixed_folder  -- press \"Open folder…\" and pick this one")

    rows: list[tuple[str, Path, str]] = []
    for dest_name, src_name, why in MIXED:
        src = _fetch(src_name)
        if not src:
            print(f"  SKIP {src_name} (not available)")
            continue
        shutil.copy(src, mixed / dest_name)
        rows.append((dest_name, mixed / dest_name, why))

    for dest_name, src_name, kind in AS_PICTURES:
        src = _fetch(src_name)
        if not src:
            continue
        write_picture(src, mixed / dest_name, kind)
        rows.append((dest_name, mixed / dest_name, f"{kind} rendered from {src_name}"))

    for name, path, why in rows:
        print(f"  {name:24} {path.stat().st_size / 1024:7.0f} KB  {why}")

    print("\nsamples/ct_series     -- 10 real slices, sorted by position not filename")
    slices = explode_series("emri_small.dcm", OUT / "ct_series")
    print(f"  {len(slices)} files, named export_010…export_001 but shown in slice order")

    zip_dir(OUT / "ct_series", OUT / "ct_series.zip")
    print(f"\nsamples/ct_series.zip -- the same series zipped ({(OUT / 'ct_series.zip').stat().st_size/1024:.0f} KB)")

    multi = OUT / "multiframe"
    multi.mkdir()
    print("\nsamples/multiframe    -- one file, many frames (use the frame slider)")
    for dest_name, src_name, why in MULTIFRAME:
        src = _fetch(src_name)
        if not src:
            continue
        shutil.copy(src, multi / dest_name)
        print(f"  {dest_name:24} {(multi / dest_name).stat().st_size / 1024:7.0f} KB  {why}")

    (OUT / "README.md").write_text(
        "# sample images\n\n"
        "Generated by `scripts/make_samples.py` from the public DICOMs bundled\n"
        "with pydicom. No PHI.\n\n"
        "- `mixed_folder/` — CT, MR, ultrasound, plus PNG/JPEG versions of the same\n"
        "  pixels. Load it with **Open folder…** to see the file list.\n"
        "- `ct_series/` — 10 single-frame slices whose filenames run backwards, to\n"
        "  show that ordering follows `ImagePositionPatient`.\n"
        "- `ct_series.zip` — the same series as a zip (**Add files…** accepts it).\n"
        "- `multiframe/` — single files holding 10 and 120 frames.\n"
    )

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"\nready: {sum(1 for f in OUT.rglob('*') if f.is_file())} files, {total/1024/1024:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
