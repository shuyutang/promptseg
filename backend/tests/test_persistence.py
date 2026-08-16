"""Work survives a restart, and can be picked up again.

The behaviour under test is the one a user notices only when it is missing: the
server stops -- deliberately or not -- and the folder they were half way through
is still there when it comes back, masks, colours and instance numbers included.

Each test that needs a "restart" reloads the application module, which rebuilds
the store, the runner and the database connection exactly as a fresh process
would. That is a real restart in every way that matters here: nothing in memory
carries over.
"""
from __future__ import annotations
import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def restart(tmp_path):
    """Give a test a persistent server it can restart at will.

    Points the settings at a temporary data directory, so nothing touches the
    user's real one, and restores the memory-only default afterwards for the
    rest of the suite.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Yields:
        A callable returning a client bound to a freshly started server. Calling
        it again is a restart.
    """
    import app as app_module
    from config import settings

    previous = (settings.persist, settings.data_dir)
    settings.persist = True
    settings.data_dir = tmp_path / "data"

    def start() -> TestClient:
        """Start the server again, as a new process would.

        Returns:
            A client bound to the rebuilt application.
        """
        if getattr(app_module, "db", None):
            app_module.db.close()
        importlib.reload(app_module)
        return TestClient(app_module.app)

    yield start

    if getattr(app_module, "db", None):
        app_module.db.close()
    settings.persist, settings.data_dir = previous
    importlib.reload(app_module)


def _upload(client, dicom_bytes, name="scan/slice.dcm"):
    """Upload one DICOM and return the workspace and image ids.

    Args:
        client: The test client.
        dicom_bytes: The DICOM fixtures.
        name: Path within the picked folder to upload it under.

    Returns:
        ``(workspace_id, image_id)``.
    """
    r = client.post("/upload", files=[("files", (name, dicom_bytes["ct"], "application/dicom"))])
    assert r.status_code == 200, r.text
    body = r.json()
    return body["workspace_id"], body["images"][0]["image_id"]


def _annotate(client, image_id, label, x=32, y=32):
    """Commit one annotation from a single point prompt.

    Args:
        client: The test client.
        image_id: Image to annotate.
        label: Label to use.
        x: Point x, in image pixels.
        y: Point y, in image pixels.

    Returns:
        The created annotation as JSON.
    """
    r = client.post("/annotations", json={
        "image_id": image_id, "frame": 0, "label": label,
        "prompts": {"points": [{"x": x, "y": y, "label": 1}], "boxes": []},
    })
    assert r.status_code == 200, r.text
    return r.json()


def test_annotations_survive_a_restart(restart, dicom_bytes):
    """The headline promise: stopping the server does not cost unexported work.

    Args:
        restart: Starts the persistent server.
        dicom_bytes: The DICOM fixtures.
    """
    c = restart()
    wid, iid = _upload(c, dicom_bytes)
    before = _annotate(c, iid, "lung")

    c = restart()
    listed = c.get("/sessions").json()
    assert listed["persist"] is True
    assert [s["workspace_id"] for s in listed["sessions"]] == [wid]
    assert listed["sessions"][0]["annotation_count"] == 1
    assert listed["sessions"][0]["labels"] == ["lung"]

    opened = c.post(f"/sessions/{wid}/open")
    assert opened.status_code == 200, opened.text
    assert opened.json()["errors"] == []

    after = c.get(f"/annotations?image_id={iid}").json()
    assert len(after) == 1
    assert after[0] == before                        # mask, prompts, timestamps and all

    # The pixels came back too, not just the mask that was drawn on them.
    assert c.get(f"/frame.png?image_id={iid}&frame=0").status_code == 200
    assert c.get(f"/annotations/{before['id']}/mask.png").status_code == 200


def test_a_reopened_session_is_the_same_session(restart, dicom_bytes):
    """Identifiers, colours and instance numbers continue rather than restarting.

    A reopened session that renumbered ``vertebra #1`` or recoloured ``lung``
    would be a copy of the work, not the work.

    Args:
        restart: Starts the persistent server.
        dicom_bytes: The DICOM fixtures.
    """
    c = restart()
    wid, iid = _upload(c, dicom_bytes)
    first = _annotate(c, iid, "lung")
    _annotate(c, iid, "vertebra", x=20, y=20)
    _annotate(c, iid, "lung", x=44, y=44)
    c.patch(f"/images/{iid}", json={"reviewed": True})

    c = restart()
    ws = c.post(f"/sessions/{wid}/open").json()
    assert ws["workspace_id"] == wid
    assert ws["images"][0]["image_id"] == iid
    assert ws["images"][0]["reviewed"] is True

    anns = c.get(f"/annotations?image_id={iid}").json()
    assert [(a["label"], a["instance"]) for a in anns] == [
        ("lung", 1), ("vertebra", 1), ("lung", 2)]
    assert anns[0]["color"] == first["color"]
    assert anns[1]["color"] != anns[0]["color"]      # the collision stayed resolved

    # Numbering continues from what was saved instead of colliding with it.
    assert _annotate(c, iid, "lung", x=10, y=10)["instance"] == 3


def test_work_done_after_reopening_is_saved_too(restart, dicom_bytes):
    """A reopened session keeps writing; it does not go quietly read-only.

    Args:
        restart: Starts the persistent server.
        dicom_bytes: The DICOM fixtures.
    """
    c = restart()
    wid, iid = _upload(c, dicom_bytes)
    _annotate(c, iid, "lung")

    c = restart()
    c.post(f"/sessions/{wid}/open")
    _annotate(c, iid, "liver", x=20, y=20)
    c.patch(f"/images/{iid}", json={"reviewed": True})

    c = restart()
    ws = c.post(f"/sessions/{wid}/open").json()
    assert ws["annotation_count"] == 2
    assert ws["images"][0]["reviewed"] is True
    assert sorted(l["name"] for l in ws["labels"]) == ["liver", "lung"]


def test_deletions_do_not_come_back(restart, dicom_bytes):
    """Deleting a mask or a file deletes it on disk as well as in memory.

    Args:
        restart: Starts the persistent server.
        dicom_bytes: The DICOM fixtures.
    """
    c = restart()
    r = c.post("/upload", files=[
        ("files", ("scan/a.dcm", dicom_bytes["ct"], "application/dicom")),
        ("files", ("scan/b.dcm", dicom_bytes["mr"], "application/dicom")),
    ]).json()
    wid = r["workspace_id"]
    keep, drop = r["images"][0]["image_id"], r["images"][1]["image_id"]
    doomed = _annotate(c, keep, "lung")
    _annotate(c, keep, "liver", x=20, y=20)
    _annotate(c, drop, "spleen")

    assert c.delete(f"/annotations/{doomed['id']}").status_code == 200
    assert c.delete(f"/images/{drop}").status_code == 200

    c = restart()
    ws = c.post(f"/sessions/{wid}/open").json()
    assert ws["image_count"] == 1
    assert [a["label"] for a in c.get(f"/annotations?image_id={keep}").json()] == ["liver"]


def test_deleting_a_session_reclaims_its_disk(restart, dicom_bytes):
    """Forgetting a session removes its rows and the image bytes behind them.

    Args:
        restart: Starts the persistent server.
        dicom_bytes: The DICOM fixtures.
    """
    import app as app_module

    c = restart()
    wid, iid = _upload(c, dicom_bytes)
    _annotate(c, iid, "lung")
    assert app_module.db.stats()["bytes"] > 0

    assert c.delete(f"/sessions/{wid}").status_code == 200
    assert c.delete(f"/sessions/{wid}").status_code == 404
    assert c.get("/sessions").json()["sessions"] == []
    assert app_module.db.stats() | {"path": None} == {
        "path": None, "sessions": 0, "images": 0, "annotations": 0, "bytes": 0, "error": None}
    assert not any(p.is_file() for p in app_module.db.blob_dir.rglob("*"))


def test_the_same_file_in_two_sessions_is_stored_once(restart, dicom_bytes):
    """Blobs are content-addressed, so re-picking a folder costs no extra disk.

    Args:
        restart: Starts the persistent server.
        dicom_bytes: The DICOM fixtures.
    """
    import app as app_module

    c = restart()
    _upload(c, dicom_bytes, "first/slice.dcm")
    once = app_module.db.stats()["bytes"]
    _upload(c, dicom_bytes, "second/slice.dcm")

    assert len(c.get("/sessions").json()["sessions"]) == 2
    assert app_module.db.stats()["bytes"] == once
    assert app_module.db.stats()["images"] == 2


def test_a_failed_write_does_not_fail_the_request(restart, dicom_bytes):
    """An unwritable database costs persistence, never the annotation itself.

    Args:
        restart: Starts the persistent server.
        dicom_bytes: The DICOM fixtures.
    """
    import app as app_module

    c = restart()
    _, iid = _upload(c, dicom_bytes)
    app_module.db._db.close()                        # as a full or locked disk would

    ann = _annotate(c, iid, "lung")                  # the user's work still lands
    assert ann["area"] > 0
    assert c.get(f"/annotations?image_id={iid}").json()[0]["id"] == ann["id"]
    assert app_module.db.last_error                  # and the failure is visible
    assert c.get("/health").json()["persist"] is True


def test_old_sessions_are_pruned(tmp_path):
    """Past the cap, the least recently touched sessions go.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    from persistence import SessionDB

    db = SessionDB(tmp_path / "data")
    for i in range(5):
        db.save_workspace(f"ws{i}", f"folder {i}", "2026-01-01T00:00:00Z", {})
        sha = db.put_blob(f"file {i}".encode())
        db.save_image(f"im{i}", f"ws{i}", 0, "a.dcm", sha, False, {}, "2026-01-01T00:00:00Z")

    dropped = db.prune(2)
    kept = [s["workspace_id"] for s in db.sessions()]
    assert len(kept) == 2 and len(dropped) == 3
    assert set(kept).isdisjoint(dropped)
    assert db.stats()["images"] == 2
    assert len(list(p for p in db.blob_dir.rglob("*") if p.is_file())) == 2
    db.close()


def test_a_missing_blob_costs_one_file_not_the_session(restart, dicom_bytes):
    """A hand-cleaned data directory degrades to a named error, not a failure.

    Args:
        restart: Starts the persistent server.
        dicom_bytes: The DICOM fixtures.
    """
    import app as app_module

    c = restart()
    r = c.post("/upload", files=[
        ("files", ("scan/a.dcm", dicom_bytes["ct"], "application/dicom")),
        ("files", ("scan/b.dcm", dicom_bytes["mr"], "application/dicom")),
    ]).json()
    wid = r["workspace_id"]
    _annotate(c, r["images"][0]["image_id"], "lung")

    c = restart()
    for p in sorted(Path(app_module.db.blob_dir).rglob("*")):
        if p.is_file():
            p.unlink()                               # lose one file's bytes
            break

    ws = c.post(f"/sessions/{wid}/open").json()
    assert ws["image_count"] == 1
    assert len(ws["errors"]) == 1 and "missing" in ws["errors"][0]


def test_memory_only_server_says_so(client):
    """With ``SAM2_PERSIST=0`` the resume list is honest rather than empty.

    Args:
        client: The test client, which the suite runs memory-only.
    """
    r = client.get("/sessions").json()
    assert r == {"persist": False, "sessions": []}
    assert client.get("/health").json()["persist"] is False
    assert client.post("/sessions/whatever/open").status_code == 503
    assert client.delete("/sessions/whatever").status_code == 503
