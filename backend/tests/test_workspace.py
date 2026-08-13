"""Folder loading, the file list, and one export for the whole batch."""
from __future__ import annotations
import io
import json
import zipfile


def _mk(client, image_id, label, x=32, y=32, frame=0):
    r = client.post("/annotations", json={
        "image_id": image_id, "frame": frame, "label": label,
        "prompts": {"points": [{"x": x, "y": y, "label": 1}], "boxes": []},
    })
    assert r.status_code == 200, r.text
    return r.json()


def test_folder_upload_lists_every_file_by_name(client, folder_files):
    d = client.post("/upload", files=folder_files).json()

    names = [i["filename"] for i in d["images"]]
    assert d["added"] == 3, f"expected the 3 images, got {names}"
    assert "scan/notes.txt" not in names
    assert any("notes.txt" in e for e in d["errors"]), "unreadable files are reported, not silently dropped"

    # Natural order, not lexicographic: img2 before img10.
    assert names == ["scan/img2.dcm", "scan/img10.dcm", "scan/photo.png"], names
    assert [i["index"] for i in d["images"]] == [0, 1, 2]
    assert {i["kind"] for i in d["images"]} == {"dicom", "raster"}


def test_a_real_series_sorts_by_slice_position(client, dicom_bytes):
    """Within one series, z wins over the filename -- exported slices are often
    named in acquisition order, which is not anatomical order."""
    import io as _io
    import pydicom

    files = []
    for name, z in [("z9.dcm", 9.0), ("z1.dcm", 1.0), ("z5.dcm", 5.0)]:
        ds = pydicom.dcmread(_io.BytesIO(dicom_bytes["mr"]), force=True)
        ds.ImagePositionPatient = [0.0, 0.0, z]
        buf = _io.BytesIO()
        ds.save_as(buf, enforce_file_format=True)
        files.append(("files", (name, buf.getvalue(), "application/dicom")))

    d = client.post("/upload", files=files).json()
    assert [i["filename"] for i in d["images"]] == ["z1.dcm", "z5.dcm", "z9.dcm"]


def test_workspace_listing_tracks_progress(client, folder_files):
    d = client.post("/upload", files=folder_files).json()
    wsid = d["workspace_id"]
    first = d["images"][0]["image_id"]

    _mk(client, first, "liver", 20, 20)
    client.patch(f"/images/{first}", json={"reviewed": True})

    ws = client.get(f"/workspaces/{wsid}").json()
    assert ws["image_count"] == 3
    assert ws["annotation_count"] == 1
    row = next(i for i in ws["images"] if i["image_id"] == first)
    assert row["annotation_count"] == 1 and row["reviewed"] is True
    assert row["labels"] == ["liver"]
    assert all(not i["reviewed"] for i in ws["images"] if i["image_id"] != first)


def test_one_label_keeps_one_colour_across_the_whole_folder(client, folder_files):
    """The reason colours live on the workspace: 'liver' must look the same in
    every file, or a batch is unreadable."""
    d = client.post("/upload", files=folder_files).json()
    ids = [i["image_id"] for i in d["images"]]

    a = _mk(client, ids[0], "liver", 20, 20)
    b = _mk(client, ids[1], "liver", 20, 20)
    c = _mk(client, ids[2], "kidney", 60, 50)

    assert a["color"] == b["color"]
    assert c["color"] != a["color"]
    assert a["instance"] == b["instance"] == 1, "instance numbering restarts per file"

    labels = {l["name"]: l for l in client.get(f"/labels?workspace_id={d['workspace_id']}").json()["labels"]}
    assert labels["liver"]["count"] == 2 and labels["kidney"]["count"] == 1


def test_export_covers_every_image_in_one_document(client, folder_files):
    d = client.post("/upload", files=folder_files).json()
    ids = [i["image_id"] for i in d["images"]]
    _mk(client, ids[0], "liver", 20, 20)
    _mk(client, ids[1], "liver", 20, 20)
    _mk(client, ids[2], "kidney", 60, 50)

    doc = client.get(f"/export.json?workspace_id={d['workspace_id']}").json()
    assert doc["schema_version"] == "2.0"
    assert doc["workspace"]["image_count"] == 3
    assert doc["workspace"]["annotation_count"] == 3
    assert len(doc["images"]) == 3
    assert [i["filename"] for i in doc["images"]] == [i["filename"] for i in d["images"]]

    # Images with no annotations still appear, so the export records what was
    # looked at and found empty -- not just what was marked.
    assert sum(len(i["annotations"]) for i in doc["images"]) == 3
    assert json.loads(json.dumps(doc)) == doc


def test_export_zip_carries_mask_pngs(client, folder_files):
    from PIL import Image

    d = client.post("/upload", files=folder_files).json()
    ids = [i["image_id"] for i in d["images"]]
    a = _mk(client, ids[0], "liver", 20, 20)
    _mk(client, ids[1], "liver", 20, 20)

    r = client.get(f"/export.zip?workspace_id={d['workspace_id']}")
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()

    assert "annotations.json" in names
    masks = [n for n in names if n.startswith("masks/")]
    assert len(masks) == 2, names
    assert all(n.endswith(".png") for n in masks)

    doc = json.loads(zf.read("annotations.json"))
    assert doc["workspace"]["annotation_count"] == 2

    png = next(n for n in masks if "liver_01" in n and n.startswith("masks/0000"))
    im = Image.open(io.BytesIO(zf.read(png)))
    assert im.size == (d["images"][0]["columns"], d["images"][0]["rows"])
    import numpy as np
    assert int((np.array(im) > 0).sum()) == a["area"]


def test_deleting_a_file_renumbers_the_list(client, folder_files):
    d = client.post("/upload", files=folder_files).json()
    ids = [i["image_id"] for i in d["images"]]

    assert client.delete(f"/images/{ids[0]}").status_code == 200
    ws = client.get(f"/workspaces/{d['workspace_id']}").json()
    assert [i["index"] for i in ws["images"]] == [0, 1]
    assert ids[0] not in [i["image_id"] for i in ws["images"]]
    assert client.get(f"/frame.png?image_id={ids[0]}&frame=0").status_code == 404


def test_files_can_be_added_to_an_existing_workspace(client, dicom_bytes, folder_files):
    d = client.post("/upload", files=folder_files).json()
    wsid = d["workspace_id"]

    more = client.post("/upload", files=[
        ("files", ("extra.dcm", dicom_bytes["us_color"], "application/dicom")),
    ], data={"workspace_id": wsid}).json()

    assert more["workspace_id"] == wsid
    assert more["workspace"]["image_count"] == 4


def test_unknown_workspace_is_404(client):
    assert client.get("/workspaces/nope").status_code == 404
    assert client.get("/export.json?workspace_id=nope").status_code == 404
