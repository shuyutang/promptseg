"""Hand-correcting a predicted mask.

The geometry tests pin down what a stroke means; the API tests pin down the rule
that makes brushing worth having -- strokes are replayed on top of the model
output rather than baked in, so a correction survives a later change to the
prompts.
"""
from __future__ import annotations

import numpy as np

from utils.paint import apply_strokes
from utils.rle import rle_to_mask

PROMPT = {"points": [{"x": 32, "y": 32, "label": 1}], "boxes": []}
"""A single include point near the centre of the small fixtures."""


def _mk(client, image_id, label="lesion"):
    """Commit one annotation from :data:`PROMPT`.

    Args:
        client: The test client.
        image_id: Image to annotate.
        label: Label to use.

    Returns:
        The created annotation as JSON.
    """
    r = client.post("/annotations", json={
        "image_id": image_id, "frame": 0, "label": label, "prompts": PROMPT})
    assert r.status_code == 200, r.text
    return r.json()


def _area(client, ann_id):
    """Fetch one mask as a PNG.

    Args:
        client: The test client.
        ann_id: Annotation identifier.

    Returns:
        The response, so callers can check its content type as well as its body.
    """
    m = client.get(f"/annotations/{ann_id}/mask.png")
    assert m.status_code == 200
    return m


# ---- painting ----------------------------------------------------------

def test_a_dab_paints_a_disc():
    """A single tap paints a disc of exactly the requested radius."""
    m = apply_strokes(np.zeros((50, 50), bool), [{"mode": "add", "radius": 5, "points": [[25, 25]]}])
    assert m[25, 25]
    assert m[25, 30] and not m[25, 31], "radius 5 reaches exactly 5 px"
    assert 60 < m.sum() < 100, m.sum()


def test_a_drag_paints_a_connected_line():
    """The browser samples pointer moves every few pixels; without interpolation
    a fast drag would paint a dotted line."""
    m = apply_strokes(np.zeros((40, 100), bool),
                      [{"mode": "add", "radius": 2, "points": [[5, 20], [90, 20]]}])
    assert m[20, 5:91].all(), "gap along the stroke"


def test_erase_removes_and_order_matters():
    """The eraser takes pixels out, and strokes apply in the order given."""
    full = np.ones((40, 40), bool)
    m = apply_strokes(full, [{"mode": "erase", "radius": 6, "points": [[20, 20]]}])
    assert not m[20, 20] and m[0, 0]

    # add-then-erase leaves nothing; erase-then-add puts it back
    a = apply_strokes(np.zeros((40, 40), bool), [
        {"mode": "add", "radius": 6, "points": [[20, 20]]},
        {"mode": "erase", "radius": 8, "points": [[20, 20]]}])
    assert a.sum() == 0


def test_strokes_clip_at_the_border():
    """A stroke at the edge paints what fits instead of raising."""
    m = apply_strokes(np.zeros((30, 30), bool), [{"mode": "add", "radius": 9, "points": [[0, 0], [29, 29]]}])
    assert m[0, 0] and m[29, 29]


# ---- through the API ---------------------------------------------------

def test_brush_adjusts_a_committed_mask(client, uploaded):
    """Painting grows a committed mask; erasing over it clears that band entirely.

    Args:
        client: The test client.
        uploaded: An uploaded MR file.
    """
    iid = uploaded["image_id"]
    a = _mk(client, iid)

    grown = client.patch(f"/annotations/{a['id']}", json={
        "strokes": [{"mode": "add", "radius": 4, "points": [[5, 5], [12, 5]]}]}).json()
    assert grown["area"] > a["area"]
    assert len(grown["strokes"]) == 1
    assert grown["id"] == a["id"] and grown["instance"] == a["instance"]

    assert _area(client, a["id"]).headers["content-type"] == "image/png"

    # Erasing over the same place clears it completely -- including any model
    # pixels that happened to lie under the stroke, which is what an eraser is.
    shrunk = client.patch(f"/annotations/{a['id']}", json={
        "strokes": grown["strokes"] + [{"mode": "erase", "radius": 4, "points": [[5, 5], [12, 5]]}]}).json()
    assert shrunk["area"] <= a["area"]
    mask = rle_to_mask(_rle(client, a["id"]))
    # The stroke runs x=5..12 at y=5 with radius 4, so this band is fully covered.
    assert not mask[1:10, 5:13].any(), "nothing survives inside the erased stroke"


def test_undo_is_just_a_shorter_stroke_list(client, uploaded):
    """Undo is expressed by sending fewer strokes; clearing them all restores
    the model's own mask.

    Args:
        client: The test client.
        uploaded: An uploaded MR file.
    """
    iid = uploaded["image_id"]
    a = _mk(client, iid)
    s1 = {"mode": "add", "radius": 3, "points": [[5, 5]]}
    s2 = {"mode": "add", "radius": 3, "points": [[50, 50]]}

    two = client.patch(f"/annotations/{a['id']}", json={"strokes": [s1, s2]}).json()
    one = client.patch(f"/annotations/{a['id']}", json={"strokes": [s1]}).json()
    none = client.patch(f"/annotations/{a['id']}", json={"strokes": []}).json()

    assert two["area"] > one["area"] > a["area"]
    assert none["area"] == a["area"], "clearing the strokes returns the model's own mask"


def test_hand_edits_survive_a_prompt_change(client, uploaded):
    """Strokes are replayed on top of the model output rather than baked in, so
    refining the prompts does not throw the correction away.

    Args:
        client: The test client.
        uploaded: An uploaded MR file.
    """
    iid = uploaded["image_id"]
    a = _mk(client, iid)
    stroke = {"mode": "add", "radius": 4, "points": [[5, 5]]}
    edited = client.patch(f"/annotations/{a['id']}", json={"strokes": [stroke]}).json()

    reprompted = client.patch(f"/annotations/{a['id']}", json={
        "prompts": {"points": [{"x": 32, "y": 32, "label": 1},
                               {"x": 40, "y": 40, "label": 1}], "boxes": []}}).json()

    assert len(reprompted["strokes"]) == 1
    mask = rle_to_mask(_rle(client, reprompted["id"]))
    assert mask[5, 5], "the painted pixel is still there after re-prompting"
    assert reprompted["area"] != edited["area"], "the model mask itself did change"


def _rle(client, ann_id):
    """Read an annotation's stored RLE straight from the store.

    Args:
        client: The test client, whose app module holds the store.
        ann_id: Annotation identifier.

    Returns:
        The stored ``{"counts", "size"}``, so a test can assert on individual
        pixels rather than on a PNG.
    """
    import app as app_module
    _, ann = app_module.store.find_annotation(ann_id)
    return ann.rle


def test_a_mask_can_be_drawn_with_no_model_prompt_at_all(client, uploaded):
    """Brush strokes alone make a valid annotation, with no model call and no score.

    Args:
        client: The test client.
        uploaded: An uploaded MR file.
    """
    iid = uploaded["image_id"]
    r = client.post("/annotations", json={
        "image_id": iid, "frame": 0, "label": "manual",
        "prompts": {"points": [], "boxes": []},
        "strokes": [{"mode": "add", "radius": 5, "points": [[20, 20], [30, 30]]}],
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["area"] > 0 and d["score"] is None


def test_empty_prompt_and_no_strokes_is_still_rejected(client, uploaded):
    """Nothing to act on is a 400, not an empty mask.

    Args:
        client: The test client.
        uploaded: An uploaded MR file.
    """
    r = client.post("/annotations", json={
        "image_id": uploaded["image_id"], "frame": 0, "label": "x",
        "prompts": {"points": [], "boxes": []}})
    assert r.status_code == 400


def test_preview_shows_the_brush_too(client, uploaded):
    """The live preview composites strokes, so what is committed is what was seen.

    Args:
        client: The test client.
        uploaded: An uploaded MR file.
    """
    r = client.post("/segment/preview.png", json={
        "image_id": uploaded["image_id"], "frame": 0, "prompts": PROMPT,
        "strokes": [{"mode": "add", "radius": 6, "points": [[5, 5]]}]})
    assert r.status_code == 200

    plain = client.post("/segment/preview.png", json={
        "image_id": uploaded["image_id"], "frame": 0, "prompts": PROMPT})
    assert float(r.headers["X-Mask-Area"]) > float(plain.headers["X-Mask-Area"])


def test_export_records_the_strokes(client, uploaded):
    """Strokes reach the export, which is what makes an annotation reproducible.

    Args:
        client: The test client.
        uploaded: An uploaded MR file.
    """
    iid = uploaded["image_id"]
    a = _mk(client, iid)
    client.patch(f"/annotations/{a['id']}", json={
        "strokes": [{"mode": "erase", "radius": 3, "points": [[32, 32]]}]})

    doc = client.get(f"/export.json?image_id={iid}").json()
    ann = doc["images"][0]["annotations"][0]
    assert ann["strokes"][0]["mode"] == "erase"
    assert "strokes" in doc["model"]["mask_composition"]
