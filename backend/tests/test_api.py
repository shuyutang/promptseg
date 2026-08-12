from __future__ import annotations
import numpy as np

from utils.rle import rle_to_mask

BOX = {"points": [{"x": 32, "y": 32, "label": 1}], "boxes": []}


def _mk(client, image_id, label, x=32, y=32, frame=0):
    r = client.post("/annotations", json={
        "image_id": image_id, "frame": frame, "label": label,
        "prompts": {"points": [{"x": x, "y": y, "label": 1}], "boxes": []},
    })
    assert r.status_code == 200, r.text
    return r.json()


def test_upload_reports_geometry(client, dicom_bytes):
    r = client.post("/dicom/upload", files={"file": ("emri_small.dcm", dicom_bytes["multiframe"], "application/dicom")})
    d = r.json()
    assert d["frames"] > 1
    assert d["meta"]["rows"] > 0 and d["meta"]["columns"] > 0
    assert "center" in d["default_window"]


def test_unknown_image_id_is_404(client):
    assert client.get("/frame.png?image_id=nope&frame=0").status_code == 404


def test_frame_png_respects_window(client, uploaded):
    iid = uploaded["image_id"]
    a = client.get(f"/frame.png?image_id={iid}&frame=0&wc=100&ww=50")
    b = client.get(f"/frame.png?image_id={iid}&frame=0&wc=100&ww=800")
    assert a.status_code == b.status_code == 200
    assert a.content != b.content


def test_prompt_required(client, uploaded):
    r = client.post("/segment/preview.png", json={
        "image_id": uploaded["image_id"], "frame": 0, "prompts": {"points": [], "boxes": []},
    })
    assert r.status_code == 400


def test_preview_returns_png_and_headers(client, uploaded):
    r = client.post("/segment/preview.png", json={
        "image_id": uploaded["image_id"], "frame": 0, "prompts": BOX,
    })
    assert r.status_code == 200
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert float(r.headers["X-Mask-Area"]) > 0


# ---- labels & instances ------------------------------------------------

def test_multiple_instances_of_one_label_share_a_color(client, uploaded):
    iid = uploaded["image_id"]
    a = _mk(client, iid, "vertebra", 20, 20)
    b = _mk(client, iid, "vertebra", 40, 40)
    c = _mk(client, iid, "disc", 30, 30)

    assert a["instance"] == 1 and b["instance"] == 2
    assert a["color"] == b["color"], "same label -> same colour"
    assert c["color"] != a["color"], "different label -> different colour"
    assert c["instance"] == 1, "instance numbering is per-label"


def test_label_matching_is_case_insensitive(client, uploaded):
    iid = uploaded["image_id"]
    a = _mk(client, iid, "Liver", 20, 20)
    b = _mk(client, iid, "liver", 40, 40)
    assert b["instance"] == 2
    assert b["color"] == a["color"]
    assert len(client.get(f"/labels?image_id={iid}").json()["labels"]) == 1


def test_list_and_filter_by_frame(client, dicom_bytes):
    up = client.post("/dicom/upload", files={"file": ("emri_small.dcm", dicom_bytes["multiframe"], "application/dicom")}).json()
    iid = up["image_id"]
    _mk(client, iid, "a", frame=0)
    _mk(client, iid, "a", frame=1)
    assert len(client.get(f"/annotations?image_id={iid}").json()) == 2
    assert len(client.get(f"/annotations?image_id={iid}&frame=1").json()) == 1


# ---- editing -----------------------------------------------------------

def test_edit_prompts_changes_mask_and_keeps_identity(client, uploaded):
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
    iid = uploaded["image_id"]
    a = _mk(client, iid, "kidney", 20, 20)
    r = client.patch(f"/annotations/{a['id']}", json={"label": "spleen"})
    b = r.json()
    assert b["label"] == "spleen"
    assert b["color"] != a["color"]
    assert b["area"] == a["area"], "renaming must not re-run segmentation"


def test_delete(client, uploaded):
    iid = uploaded["image_id"]
    a = _mk(client, iid, "x")
    assert client.delete(f"/annotations/{a['id']}").status_code == 200
    assert client.get(f"/annotations?image_id={iid}").json() == []
    assert client.delete(f"/annotations/{a['id']}").status_code == 404


def test_instance_numbers_are_stable_after_delete(client, uploaded):
    iid = uploaded["image_id"]
    a = _mk(client, iid, "rib", 20, 20)
    b = _mk(client, iid, "rib", 40, 40)
    client.delete(f"/annotations/{a['id']}")
    c = _mk(client, iid, "rib", 50, 50)
    assert c["instance"] == 3, "deleting must not recycle instance numbers"
    assert b["instance"] == 2


# ---- overlay & export ---------------------------------------------------

def test_overlay_composites_all_masks(client, uploaded):
    iid = uploaded["image_id"]
    _mk(client, iid, "a", 20, 20)
    _mk(client, iid, "b", 45, 45)
    r = client.get(f"/annotations/overlay.png?image_id={iid}&frame=0")
    assert r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_export_json_roundtrips_masks(client, uploaded):
    iid = uploaded["image_id"]
    a = _mk(client, iid, "vertebra", 20, 20)
    _mk(client, iid, "vertebra", 45, 45)

    doc = client.get(f"/export.json?image_id={iid}").json()
    assert doc["schema_version"] == "1.0"
    assert doc["image"]["image_id"] == iid
    assert doc["image"]["modality"]
    assert len(doc["annotations"]) == 2

    labels = {l["name"]: l for l in doc["labels"]}
    assert labels["vertebra"]["count"] == 2

    e = next(x for x in doc["annotations"] if x["id"] == a["id"])
    assert e["label"] == "vertebra" and e["instance"] == 1
    assert e["mask"]["format"] == "coco_rle_uncompressed"
    assert e["prompts"]["points"][0]["x"] == 20
    assert e["window"] is not None and e["threshold"] > 0

    m = rle_to_mask({"counts": e["mask"]["counts"], "size": e["mask"]["size"]})
    assert int(m.sum()) == e["area"], "exported RLE must decode to the reported area"
    x, y, w, h = e["bbox"]
    assert m[y:y + h, x:x + w].sum() == m.sum(), "bbox must contain the whole mask"


def test_export_can_omit_masks(client, uploaded):
    iid = uploaded["image_id"]
    _mk(client, iid, "a")
    doc = client.get(f"/export.json?image_id={iid}&include_masks=false").json()
    assert "mask" not in doc["annotations"][0]
    assert doc["annotations"][0]["area"] > 0


def test_export_is_json_serialisable_and_stable(client, uploaded):
    import json
    iid = uploaded["image_id"]
    _mk(client, iid, "a")
    doc = client.get(f"/export.json?image_id={iid}").json()
    assert json.loads(json.dumps(doc)) == doc
