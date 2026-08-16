"""The HTTP API: upload, prompt, label, edit, export.

Everything runs against the stub model, so what is under test is the API's own
behaviour -- geometry, instance numbering, colours, the export contract -- rather
than mask quality.
"""
from __future__ import annotations
import numpy as np

from utils.rle import rle_to_mask

BOX = {"points": [{"x": 32, "y": 32, "label": 1}], "boxes": []}
"""A single include point near the centre of the small fixtures."""


def _mk(client, image_id, label, x=32, y=32, frame=0):
    """Commit one annotation from a single point prompt.

    Args:
        client: The test client.
        image_id: Image to annotate.
        label: Label to use.
        x: Point x, in image pixels.
        y: Point y, in image pixels.
        frame: Frame index within the image.

    Returns:
        The created annotation as JSON.
    """
    r = client.post("/annotations", json={
        "image_id": image_id, "frame": frame, "label": label,
        "prompts": {"points": [{"x": x, "y": y, "label": 1}], "boxes": []},
    })
    assert r.status_code == 200, r.text
    return r.json()


def test_upload_reports_geometry(client, dicom_bytes):
    """An upload comes back with frame count, geometry and a usable window.

    Args:
        client: The test client.
        dicom_bytes: The DICOM fixtures.
    """
    r = client.post("/dicom/upload", files={"file": ("emri_small.dcm", dicom_bytes["multiframe"], "application/dicom")})
    d = r.json()
    assert d["frames"] > 1
    assert d["meta"]["rows"] > 0 and d["meta"]["columns"] > 0
    assert "center" in d["default_window"]


def test_mixed_size_zip_becomes_one_entry_per_file(client, dicom_bytes):
    """A zipped folder holds differently-sized images. Each becomes its own file
    entry with its own geometry -- flattening them into frames of one study made
    the viewer map clicks to the wrong coordinates.

    Args:
        client: The test client.
        dicom_bytes: The DICOM fixtures.
    """
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.dcm", dicom_bytes["ct"])   # 128x128
        zf.writestr("b.dcm", dicom_bytes["mr"])   # 64x64
    r = client.post("/upload", files=[("files", ("folder.zip", buf.getvalue(), "application/zip"))])
    d = r.json()

    assert d["added"] == 2 and not d["errors"]
    shapes = {(i["rows"], i["columns"]) for i in d["images"]}
    assert shapes == {(128, 128), (64, 64)}

    # Per-file info must match the actual rendered image.
    for item in d["images"]:
        info = client.get(f"/frame_info?image_id={item['image_id']}&frame=0").json()
        assert (info["rows"], info["columns"]) == (item["rows"], item["columns"])
        ov = client.get(f"/annotations/overlay.png?image_id={item['image_id']}&frame=0")
        assert ov.status_code == 200


def test_multiframe_file_stays_one_entry(client, dicom_bytes):
    """A multi-frame file is one row in the list, not ten.

    Args:
        client: The test client.
        dicom_bytes: The DICOM fixtures.
    """
    d = client.post("/upload", files=[
        ("files", ("emri_small.dcm", dicom_bytes["multiframe"], "application/dicom")),
    ]).json()
    assert d["added"] == 1
    assert d["images"][0]["frames"] > 1, "a multi-frame file is one entry with a frame slider"


def test_unknown_image_id_is_404(client):
    """An evicted or invented image id is a 404, not a 500.

    Args:
        client: The test client.
    """
    assert client.get("/frame.png?image_id=nope&frame=0").status_code == 404


def test_frame_png_respects_window(client, uploaded):
    """Changing the window changes the rendered pixels.

    Args:
        client: The test client.
        uploaded: An uploaded MR file.
    """
    iid = uploaded["image_id"]
    a = client.get(f"/frame.png?image_id={iid}&frame=0&wc=100&ww=50")
    b = client.get(f"/frame.png?image_id={iid}&frame=0&wc=100&ww=800")
    assert a.status_code == b.status_code == 200
    assert a.content != b.content


def test_prompt_required(client, uploaded):
    """A preview with neither prompts nor strokes is rejected.

    Args:
        client: The test client.
        uploaded: An uploaded MR file.
    """
    r = client.post("/segment/preview.png", json={
        "image_id": uploaded["image_id"], "frame": 0, "prompts": {"points": [], "boxes": []},
    })
    assert r.status_code == 400


def test_preview_returns_png_and_headers(client, uploaded):
    """The preview carries its score, area and candidate count in headers.

    Args:
        client: The test client.
        uploaded: An uploaded MR file.
    """
    r = client.post("/segment/preview.png", json={
        "image_id": uploaded["image_id"], "frame": 0, "prompts": BOX,
    })
    assert r.status_code == 200
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert float(r.headers["X-Mask-Area"]) > 0


# ---- labels & instances ------------------------------------------------

def test_multiple_instances_of_one_label_share_a_color(client, uploaded):
    """One label, many instances, one colour; a different label gets another.

    Args:
        client: The test client.
        uploaded: An uploaded MR file.
    """
    iid = uploaded["image_id"]
    a = _mk(client, iid, "vertebra", 20, 20)
    b = _mk(client, iid, "vertebra", 40, 40)
    c = _mk(client, iid, "disc", 30, 30)

    assert a["instance"] == 1 and b["instance"] == 2
    assert a["color"] == b["color"], "same label -> same colour"
    assert c["color"] != a["color"], "different label -> different colour"
    assert c["instance"] == 1, "instance numbering is per-label"


def test_label_matching_is_case_insensitive(client, uploaded):
    """'Liver' and 'liver' are one label with one colour.

    Args:
        client: The test client.
        uploaded: An uploaded MR file.
    """
    iid = uploaded["image_id"]
    a = _mk(client, iid, "Liver", 20, 20)
    b = _mk(client, iid, "liver", 40, 40)
    assert b["instance"] == 2
    assert b["color"] == a["color"]
    assert len(client.get(f"/labels?image_id={iid}").json()["labels"]) == 1


def test_list_and_filter_by_frame(client, dicom_bytes):
    """Annotations can be listed per file or narrowed to one frame.

    Args:
        client: The test client.
        dicom_bytes: The DICOM fixtures.
    """
    up = client.post("/dicom/upload", files={"file": ("emri_small.dcm", dicom_bytes["multiframe"], "application/dicom")}).json()
    iid = up["image_id"]
    _mk(client, iid, "a", frame=0)
    _mk(client, iid, "a", frame=1)
    assert len(client.get(f"/annotations?image_id={iid}").json()) == 2
    assert len(client.get(f"/annotations?image_id={iid}&frame=1").json()) == 1


# ---- editing -----------------------------------------------------------

def test_edit_prompts_changes_mask_and_keeps_identity(client, uploaded):
    """Re-prompting changes the mask but not the annotation's id or instance.

    Args:
        client: The test client.
        uploaded: An uploaded MR file.
    """
    iid = uploaded["image_id"]
    a = _mk(client, iid, "lesion", 20, 20)

    r = client.patch(f"/annotations/{a['id']}", json={
        "prompts": {"points": [{"x": 20, "y": 20, "label": 1},
                               {"x": 45, "y": 45, "label": 1}], "boxes": []},
    })
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["id"] == a["id"]
    assert b["instance"] == a["instance"]
    assert b["area"] != a["area"], "extra prompt should change the mask"
    assert len(b["prompts"]["points"]) == 2
    assert b["updated_at"] >= a["updated_at"]


def test_rename_reassigns_color_and_instance(client, uploaded):
    """Renaming moves the annotation to the new label without re-segmenting.

    Args:
        client: The test client.
        uploaded: An uploaded MR file.
    """
    iid = uploaded["image_id"]
    a = _mk(client, iid, "kidney", 20, 20)
    r = client.patch(f"/annotations/{a['id']}", json={"label": "spleen"})
    b = r.json()
    assert b["label"] == "spleen"
    assert b["color"] != a["color"]
    assert b["area"] == a["area"], "renaming must not re-run segmentation"


def test_delete(client, uploaded):
    """Deleting removes the annotation, and deleting twice is a 404.

    Args:
        client: The test client.
        uploaded: An uploaded MR file.
    """
    iid = uploaded["image_id"]
    a = _mk(client, iid, "x")
    assert client.delete(f"/annotations/{a['id']}").status_code == 200
    assert client.get(f"/annotations?image_id={iid}").json() == []
    assert client.delete(f"/annotations/{a['id']}").status_code == 404


def test_instance_numbers_are_stable_after_delete(client, uploaded):
    """Instance numbers are never recycled, so names the user learned stay put.

    Args:
        client: The test client.
        uploaded: An uploaded MR file.
    """
    iid = uploaded["image_id"]
    a = _mk(client, iid, "rib", 20, 20)
    b = _mk(client, iid, "rib", 40, 40)
    client.delete(f"/annotations/{a['id']}")
    c = _mk(client, iid, "rib", 50, 50)
    assert c["instance"] == 3, "deleting must not recycle instance numbers"
    assert b["instance"] == 2


# ---- overlay & export ---------------------------------------------------

def test_overlay_composites_all_masks(client, uploaded):
    """Every committed mask on a frame comes back in one PNG.

    Args:
        client: The test client.
        uploaded: An uploaded MR file.
    """
    iid = uploaded["image_id"]
    _mk(client, iid, "a", 20, 20)
    _mk(client, iid, "b", 45, 45)
    r = client.get(f"/annotations/overlay.png?image_id={iid}&frame=0")
    assert r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_export_json_roundtrips_masks(client, uploaded):
    """The exported RLE decodes to the reported area and fits the reported bbox.

    Args:
        client: The test client.
        uploaded: An uploaded MR file.
    """
    iid = uploaded["image_id"]
    a = _mk(client, iid, "vertebra", 20, 20)
    _mk(client, iid, "vertebra", 45, 45)

    doc = client.get(f"/export.json?image_id={iid}").json()
    assert doc["schema_version"] == "2.0"
    assert len(doc["images"]) == 1
    img = doc["images"][0]
    assert img["image_id"] == iid
    assert img["modality"]
    assert len(img["annotations"]) == 2

    labels = {l["name"]: l for l in doc["labels"]}
    assert labels["vertebra"]["count"] == 2

    e = next(x for x in img["annotations"] if x["id"] == a["id"])
    assert e["label"] == "vertebra" and e["instance"] == 1
    assert e["mask"]["format"] == "coco_rle_uncompressed"
    assert e["prompts"]["points"][0]["x"] == 20
    assert e["window"] is not None and e["threshold"] > 0

    m = rle_to_mask({"counts": e["mask"]["counts"], "size": e["mask"]["size"]})
    assert int(m.sum()) == e["area"], "exported RLE must decode to the reported area"
    x, y, w, h = e["bbox"]
    assert m[y:y + h, x:x + w].sum() == m.sum(), "bbox must contain the whole mask"


def test_export_can_omit_masks(client, uploaded):
    """A metadata-only export keeps the measurements but drops the pixels.

    Args:
        client: The test client.
        uploaded: An uploaded MR file.
    """
    iid = uploaded["image_id"]
    _mk(client, iid, "a")
    doc = client.get(f"/export.json?image_id={iid}&include_masks=false").json()
    ann = doc["images"][0]["annotations"][0]
    assert "mask" not in ann
    assert ann["area"] > 0


def test_export_is_json_serialisable_and_stable(client, uploaded):
    """Nothing in the document is a numpy type that only looks like JSON.

    Args:
        client: The test client.
        uploaded: An uploaded MR file.
    """
    import json
    iid = uploaded["image_id"]
    _mk(client, iid, "a")
    doc = client.get(f"/export.json?image_id={iid}").json()
    assert json.loads(json.dumps(doc)) == doc
