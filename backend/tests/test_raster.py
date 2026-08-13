"""PNG / JPEG / TIFF alongside DICOM."""
from __future__ import annotations
import io

import numpy as np
import pytest
from PIL import Image

from media.loader import load_batch, load_one
from media.raster_source import RasterSource


def test_png_and_jpeg_load(raster_bytes):
    for key in ("png_gray", "png_rgb", "jpeg", "png16"):
        src = load_one(f"x.{'jpg' if key == 'jpeg' else 'png'}", raster_bytes[key])
        assert src.kind == "raster"
        assert src.frames == 1
        assert src.frame_shape(0) == (96, 128)
        rgb = src.frame_rgb(0)
        assert rgb.shape == (96, 128, 3) and rgb.dtype == np.uint8


def test_eight_bit_pictures_are_shown_untouched(raster_bytes):
    """Windowing an 8-bit photo can only destroy it, so it is not offered."""
    src = RasterSource(raster_bytes["png_rgb"])
    assert src.windowing is False
    a = src.frame_uint8(0, 10, 20)
    b = src.frame_uint8(0)
    assert np.array_equal(a, b)


def test_sixteen_bit_pngs_keep_a_real_window(raster_bytes):
    src = RasterSource(raster_bytes["png16"])
    assert src.windowing is True
    wc, ww = src.default_window(0)
    assert ww > 1
    narrow = src.frame_uint8(0, wc, ww / 8)
    wide = src.frame_uint8(0, wc, ww * 4)
    assert not np.array_equal(narrow, wide)


def test_multi_page_tiff_becomes_frames():
    pages = [Image.fromarray((np.full((32, 32), v, np.uint8))) for v in (10, 120, 230)]
    buf = io.BytesIO()
    pages[0].save(buf, format="TIFF", save_all=True, append_images=pages[1:])
    src = load_one("stack.tif", buf.getvalue())
    assert src.frames == 3
    assert int(src.frame_uint8(0).mean()) != int(src.frame_uint8(2).mean())


def test_content_sniffing_beats_the_extension(dicom_bytes):
    """Folder exports full of extensionless files are normal; so are .dcm files
    that are really PNGs."""
    src = load_one("IM_0001", dicom_bytes["ct"])
    assert src.kind == "dicom"


def test_unreadable_file_names_itself(raster_bytes):
    loaded, errors = load_batch([("good.png", raster_bytes["png_gray"]),
                                 ("bad.png", b"nonsense")])
    assert len(loaded) == 1
    assert len(errors) == 1 and "bad.png" in errors[0]


def test_png_segments_end_to_end(client, raster_bytes):
    d = client.post("/upload", files=[("files", ("blob.png", raster_bytes["png_gray"], "image/png"))]).json()
    iid = d["images"][0]["image_id"]
    assert d["images"][0]["windowing"] is False

    r = client.post("/annotations", json={
        "image_id": iid, "frame": 0, "label": "blob",
        "prompts": {"points": [{"x": 64, "y": 50, "label": 1}], "boxes": []},
    })
    assert r.status_code == 200, r.text
    assert r.json()["area"] > 0

    assert client.get(f"/frame.png?image_id={iid}&frame=0").status_code == 200
