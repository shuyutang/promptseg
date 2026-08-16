"""Durable session storage: the work survives a server restart.

The in-memory :mod:`store` is the working copy; this is the copy on disk. Every
mutation writes through, so there is no "save" button to forget and no window
in which a crash loses the last annotation. A saved session can be reopened
later from the file list, which is what makes the export a deliberate act
rather than the only way to keep anything.

Two decisions worth knowing about:

**The original file bytes are saved, not just the annotations.** Reopening a
session has to put the images back on screen, and a mask is meaningless without
the pixels it was drawn on. Blobs are content-addressed by SHA-256 and stored
once, so re-uploading the same folder into a second session costs no extra
disk.

**Persistence never fails a request.** A full disk or a locked database must not
lose the annotation the user just made -- writes are attempted, failures are
recorded in :attr:`SessionDB.last_error` and reported by ``/health``, and the
in-memory copy carries on regardless.

Note that this means images -- DICOM included -- are written to
``SAM2_DATA_DIR`` in the clear. Set ``SAM2_PERSIST=0`` to keep the old
memory-only behaviour on a machine where that is not acceptable.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1
"""Bumped when the tables change. An older database is migrated or discarded."""

_NOW = "strftime('%Y-%m-%dT%H:%M:%SZ', 'now')"
"""SQLite's clock, formatted like every other timestamp the API returns.
SQLite's own ``datetime('now')`` writes ``YYYY-MM-DD HH:MM:SS``, which a browser
parses as *local* time -- the resume list would then be hours out."""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id  TEXT PRIMARY KEY,
    name          TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    label_colors  TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS images (
    image_id      TEXT PRIMARY KEY,
    workspace_id  TEXT NOT NULL,
    idx           INTEGER NOT NULL,
    filename      TEXT NOT NULL,
    blob_sha      TEXT NOT NULL,
    reviewed      INTEGER NOT NULL DEFAULT 0,
    instances     TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS images_by_workspace ON images(workspace_id, idx);
CREATE TABLE IF NOT EXISTS annotations (
    id            TEXT PRIMARY KEY,
    image_id      TEXT NOT NULL,
    frame         INTEGER NOT NULL,
    label         TEXT NOT NULL,
    instance      INTEGER NOT NULL,
    prompts       TEXT NOT NULL,
    window        TEXT,
    threshold     REAL NOT NULL,
    mask_index    INTEGER NOT NULL,
    rle           TEXT NOT NULL,
    area          INTEGER NOT NULL,
    bbox          TEXT,
    score         REAL,
    strokes       TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS annotations_by_image ON annotations(image_id);
CREATE TABLE IF NOT EXISTS blobs (
    sha           TEXT PRIMARY KEY,
    size          INTEGER NOT NULL
);
"""


@dataclass
class SavedImage:
    """One persisted file, as read back for reopening.

    Attributes:
        image_id: Image identifier, reused so annotations reattach.
        filename: Path within the picked folder, as uploaded.
        data: The original file bytes, or ``b""`` if the blob has gone missing.
        blob_sha: Digest of those bytes, carried back so a reopened session
            keeps writing its image rows rather than going quietly read-only.
        reviewed: Whether the user had marked this file done.
        instances: Canonical label to highest instance number handed out, so
            numbering continues rather than restarting at 1.
        created_at: UTC ISO-8601 timestamp of the original upload.
        annotations: Annotation rows, in creation order, with JSON columns
            already decoded.
    """
    image_id: str
    filename: str
    data: bytes
    blob_sha: str
    reviewed: bool
    instances: dict[str, int]
    created_at: str
    annotations: list[dict]


@dataclass
class SavedWorkspace:
    """One persisted session, as read back for reopening.

    Attributes:
        workspace_id: Workspace identifier, reused so a reopened session is the
            same session rather than a copy.
        name: Display name, usually the picked folder.
        created_at: UTC ISO-8601 timestamp.
        label_colors: Canonical label to hex colour, restored so a label keeps
            the colour the user learned.
        images: The files, in file-list order.
    """
    workspace_id: str
    name: str
    created_at: str
    label_colors: dict[str, str]
    images: list[SavedImage]


class SessionDB:
    """SQLite-backed session storage, with image bytes beside it on disk.

    One connection is shared across FastAPI's thread pool under a lock, which is
    ample for a single-user tool and avoids a pool that could see half-written
    state. Every write is its own transaction, so an interrupted server leaves a
    consistent database rather than a partial workspace.

    Attributes:
        path: The SQLite file.
        blob_dir: Where original file bytes live, sharded by the first two
            characters of the hash.
        last_error: The most recent write failure, or None. Surfaced by
            ``/health`` so a silently unwritable data directory is visible.
    """

    def __init__(self, data_dir: Path | str) -> None:
        """Open, creating the data directory and schema if needed.

        Args:
            data_dir: Directory to keep the database and blobs in. Created,
                with parents, if it does not exist.

        Raises:
            OSError: If the directory cannot be created.
            sqlite3.Error: If the database cannot be opened.
        """
        self.dir = Path(data_dir).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "sessions.db"
        self.blob_dir = self.dir / "blobs"
        self.blob_dir.mkdir(exist_ok=True)
        self.last_error: str | None = None

        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")     # a reader never blocks the writer
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_SCHEMA)
        self._db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        self._db.commit()

    def close(self) -> None:
        """Close the connection. Safe to call twice."""
        with self._lock:
            try:
                self._db.close()
            except sqlite3.Error:
                pass

    # ---- blobs --------------------------------------------------------

    def put_blob(self, data: bytes) -> str:
        """Store file bytes, deduplicated by content.

        Args:
            data: The original uploaded file.

        Returns:
            The SHA-256 hex digest, which is the blob's name and the value an
            image row keeps.
        """
        sha = hashlib.sha256(data).hexdigest()
        dest = self._blob_path(sha)
        try:
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Write beside the target and rename, so a crash never leaves a
                # truncated blob under a hash that claims to describe it.
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(dest)
        except OSError as e:
            # No blob means no reopening this file, but the upload itself is
            # fine and the user is told by ``/health`` rather than by a failure.
            self.last_error = f"{type(e).__name__}: {e}"
            return ""
        self._write("INSERT OR REPLACE INTO blobs(sha, size) VALUES (?, ?)", (sha, len(data)))
        return sha

    def get_blob(self, sha: str) -> bytes:
        """Read file bytes back.

        Args:
            sha: The digest an image row recorded.

        Returns:
            The bytes, or ``b""`` if the blob is missing -- a hand-cleaned data
            directory should degrade to one unreadable file, not a failed
            reopen.
        """
        p = self._blob_path(sha)
        try:
            return p.read_bytes()
        except OSError:
            return b""

    def _blob_path(self, sha: str) -> Path:
        """Locate a blob on disk.

        Args:
            sha: The digest.

        Returns:
            ``blobs/ab/abcdef…``; sharded because one flat directory of a few
            thousand slices is slow to list on most filesystems.
        """
        return self.blob_dir / sha[:2] / sha

    # ---- write-through ------------------------------------------------

    def save_workspace(self, workspace_id: str, name: str, created_at: str,
                       label_colors: dict[str, str]) -> None:
        """Create or update a session's own row.

        Also stamps ``updated_at``, which is what the resume list sorts by.

        Args:
            workspace_id: Workspace identifier.
            name: Display name.
            created_at: UTC ISO-8601 timestamp of creation.
            label_colors: Canonical label to hex colour for the whole session.
        """
        self._write(
            f"INSERT INTO workspaces(workspace_id, name, created_at, updated_at, label_colors) "
            f"VALUES (?, ?, ?, {_NOW}, ?) "
            "ON CONFLICT(workspace_id) DO UPDATE SET "
            "  name=excluded.name, label_colors=excluded.label_colors, "
            "  updated_at=excluded.updated_at",
            (workspace_id, name, created_at, json.dumps(label_colors)),
        )

    def save_image(self, image_id: str, workspace_id: str, idx: int, filename: str,
                   blob_sha: str, reviewed: bool, instances: dict[str, int],
                   created_at: str) -> None:
        """Create or update one file's row.

        Args:
            image_id: Image identifier.
            workspace_id: Owning workspace.
            idx: Position in the file list.
            filename: Path within the picked folder.
            blob_sha: Digest of the original bytes, from :meth:`put_blob`.
            reviewed: Whether the user marked this file done.
            instances: Canonical label to highest instance number handed out.
            created_at: UTC ISO-8601 timestamp.
        """
        self._write(
            "INSERT INTO images(image_id, workspace_id, idx, filename, blob_sha, "
            "                   reviewed, instances, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(image_id) DO UPDATE SET "
            "  idx=excluded.idx, reviewed=excluded.reviewed, instances=excluded.instances",
            (image_id, workspace_id, idx, filename, blob_sha,
             int(reviewed), json.dumps(instances), created_at),
            touch=workspace_id,
        )

    def delete_image(self, image_id: str, workspace_id: str) -> None:
        """Forget one file and everything drawn on it.

        Args:
            image_id: Image identifier.
            workspace_id: Owning workspace, so its timestamp moves too.
        """
        self._write("DELETE FROM annotations WHERE image_id = ?", (image_id,))
        self._write("DELETE FROM images WHERE image_id = ?", (image_id,), touch=workspace_id)

    def save_annotation(self, ann, workspace_id: str) -> None:
        """Create or update one annotation.

        Args:
            ann: A :class:`store.Annotation`. Its JSON-shaped fields are encoded
                here; everything that reproduces the mask is kept, matching the
                export.
            workspace_id: Owning workspace, so its timestamp moves too.
        """
        self._write(
            "INSERT INTO annotations(id, image_id, frame, label, instance, prompts, window, "
            "  threshold, mask_index, rle, area, bbox, score, strokes, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "  frame=excluded.frame, label=excluded.label, instance=excluded.instance, "
            "  prompts=excluded.prompts, window=excluded.window, threshold=excluded.threshold, "
            "  mask_index=excluded.mask_index, rle=excluded.rle, area=excluded.area, "
            "  bbox=excluded.bbox, score=excluded.score, strokes=excluded.strokes, "
            "  updated_at=excluded.updated_at",
            (ann.id, ann.image_id, ann.frame, ann.label, ann.instance,
             json.dumps(ann.prompts), json.dumps(ann.window) if ann.window else None,
             ann.threshold, ann.mask_index, json.dumps(ann.rle), ann.area,
             json.dumps(ann.bbox) if ann.bbox else None, ann.score,
             json.dumps(ann.strokes), ann.created_at, ann.updated_at),
            touch=workspace_id,
        )

    def delete_annotation(self, ann_id: str, workspace_id: str) -> None:
        """Forget one annotation.

        Args:
            ann_id: Annotation identifier.
            workspace_id: Owning workspace, so its timestamp moves too.
        """
        self._write("DELETE FROM annotations WHERE id = ?", (ann_id,), touch=workspace_id)

    def _write(self, sql: str, params: tuple, touch: str | None = None) -> None:
        """Run one statement, recording rather than raising any failure.

        A write that fails must not take the user's annotation with it -- the
        in-memory copy is already correct, and ``/health`` reports the problem.

        Args:
            sql: The statement.
            params: Its parameters.
            touch: Workspace id whose ``updated_at`` should be refreshed, so the
                resume list orders by real activity.
        """
        with self._lock:
            try:
                self._db.execute(sql, params)
                if touch:
                    self._db.execute(
                        f"UPDATE workspaces SET updated_at = {_NOW} WHERE workspace_id = ?",
                        (touch,),
                    )
                self._db.commit()
                self.last_error = None
            except (sqlite3.Error, OSError) as e:
                self.last_error = f"{type(e).__name__}: {e}"
                try:
                    self._db.rollback()
                except sqlite3.Error:
                    pass        # a closed connection cannot roll back, and need not

    # ---- reading ------------------------------------------------------

    def sessions(self) -> list[dict]:
        """Summarise every saved session, for the resume list.

        Returns:
            ``[{workspace_id, name, created_at, updated_at, image_count,
            annotation_count, labels}]``, most recently touched first. Labels
            are the distinct names used, so the list says what a session is
            about without opening it. Empty if the database cannot be read; the
            reason is then in :attr:`last_error`.
        """
        with self._lock:
            try:
                rows = self._db.execute("""
                    SELECT w.workspace_id, w.name, w.created_at, w.updated_at,
                           (SELECT COUNT(*) FROM images i WHERE i.workspace_id = w.workspace_id)
                               AS image_count,
                           (SELECT COUNT(*) FROM annotations a
                                  JOIN images i2 ON i2.image_id = a.image_id
                                 WHERE i2.workspace_id = w.workspace_id) AS annotation_count
                      FROM workspaces w
                     ORDER BY w.updated_at DESC
                """).fetchall()
                out = []
                for r in rows:
                    labels = self._db.execute("""
                        SELECT DISTINCT a.label FROM annotations a
                          JOIN images i ON i.image_id = a.image_id
                         WHERE i.workspace_id = ? ORDER BY a.label COLLATE NOCASE
                    """, (r["workspace_id"],)).fetchall()
                    out.append({**dict(r), "labels": [x["label"] for x in labels]})
                return out
            except sqlite3.Error as e:
                self.last_error = f"{type(e).__name__}: {e}"
                return []

    def load_workspace(self, workspace_id: str) -> SavedWorkspace | None:
        """Read a whole session back, blobs included.

        Args:
            workspace_id: Workspace identifier.

        Returns:
            The session, or None if it was never saved, has been deleted, or
            cannot be read -- in which case :attr:`last_error` says why.
        """
        with self._lock:
            try:
                w = self._db.execute("SELECT * FROM workspaces WHERE workspace_id = ?",
                                     (workspace_id,)).fetchone()
                if w is None:
                    return None
                images = self._db.execute(
                    "SELECT * FROM images WHERE workspace_id = ? ORDER BY idx", (workspace_id,)
                ).fetchall()
                saved = []
                for i in images:
                    anns = self._db.execute(
                        "SELECT * FROM annotations WHERE image_id = ? ORDER BY created_at, rowid",
                        (i["image_id"],),
                    ).fetchall()
                    saved.append(SavedImage(
                        image_id=i["image_id"], filename=i["filename"],
                        data=self.get_blob(i["blob_sha"]), blob_sha=i["blob_sha"],
                        reviewed=bool(i["reviewed"]),
                        instances=json.loads(i["instances"]), created_at=i["created_at"],
                        annotations=[_decode_annotation(a) for a in anns],
                    ))
                return SavedWorkspace(
                    workspace_id=w["workspace_id"], name=w["name"], created_at=w["created_at"],
                    label_colors=json.loads(w["label_colors"]), images=saved,
                )
            except sqlite3.Error as e:
                self.last_error = f"{type(e).__name__}: {e}"
                return None

    # ---- housekeeping -------------------------------------------------

    def delete_workspace(self, workspace_id: str) -> bool:
        """Forget a session and collect the blobs it was the last user of.

        Args:
            workspace_id: Workspace identifier.

        Returns:
            True if it existed.
        """
        with self._lock:
            found = self._db.execute("SELECT 1 FROM workspaces WHERE workspace_id = ?",
                                     (workspace_id,)).fetchone() is not None
            if not found:
                return False
            self._db.execute(
                "DELETE FROM annotations WHERE image_id IN "
                "(SELECT image_id FROM images WHERE workspace_id = ?)", (workspace_id,))
            self._db.execute("DELETE FROM images WHERE workspace_id = ?", (workspace_id,))
            self._db.execute("DELETE FROM workspaces WHERE workspace_id = ?", (workspace_id,))
            self._db.commit()
        self.collect_blobs()
        return True

    def collect_blobs(self) -> int:
        """Delete blobs no image row refers to any more.

        Returns:
            How many were removed.
        """
        with self._lock:
            orphans = [r["sha"] for r in self._db.execute(
                "SELECT sha FROM blobs WHERE sha NOT IN (SELECT blob_sha FROM images)"
            ).fetchall()]
            for sha in orphans:
                self._blob_path(sha).unlink(missing_ok=True)
            self._db.executemany("DELETE FROM blobs WHERE sha = ?", [(s,) for s in orphans])
            self._db.commit()
        return len(orphans)

    def prune(self, keep: int) -> list[str]:
        """Drop the least recently touched sessions past a cap.

        Kept deliberately simple: the tool is single-user and a session is only
        worth keeping while someone might come back to it. Without a cap, a
        machine that annotates folders all day fills its disk with sessions
        nobody will reopen.

        Args:
            keep: How many sessions to retain. Zero or less keeps everything.

        Returns:
            The workspace ids that were removed.
        """
        if keep <= 0:
            return []
        with self._lock:
            doomed = [r["workspace_id"] for r in self._db.execute(
                "SELECT workspace_id FROM workspaces ORDER BY updated_at DESC LIMIT -1 OFFSET ?",
                (keep,),
            ).fetchall()]
        for wid in doomed:
            self.delete_workspace(wid)
        return doomed

    def stats(self) -> dict:
        """Report what is on disk, for ``/health``.

        Returns:
            ``{path, sessions, images, annotations, bytes, error}``. The counts
            are None if the database cannot be read, which ``error`` explains.
        """
        counts = {"sessions": None, "images": None, "annotations": None, "bytes": None}
        with self._lock:
            try:
                counts = dict(self._db.execute(
                    "SELECT (SELECT COUNT(*) FROM workspaces)  AS sessions,"
                    "       (SELECT COUNT(*) FROM images)      AS images,"
                    "       (SELECT COUNT(*) FROM annotations) AS annotations,"
                    "       (SELECT COALESCE(SUM(size), 0) FROM blobs) AS bytes"
                ).fetchone())
            except sqlite3.Error as e:
                self.last_error = f"{type(e).__name__}: {e}"
            return {"path": str(self.dir), **counts, "error": self.last_error}


def _decode_annotation(row: sqlite3.Row) -> dict:
    """Turn one annotation row back into the field names the store uses.

    Args:
        row: A row from the ``annotations`` table.

    Returns:
        A dict ready to pass to :meth:`store.Store.restore_workspace`, with the
        JSON columns decoded.
    """
    return {
        "id": row["id"], "frame": row["frame"], "label": row["label"],
        "instance": row["instance"], "prompts": json.loads(row["prompts"]),
        "window": json.loads(row["window"]) if row["window"] else None,
        "threshold": row["threshold"], "mask_index": row["mask_index"],
        "rle": json.loads(row["rle"]), "area": row["area"],
        "bbox": json.loads(row["bbox"]) if row["bbox"] else None,
        "score": row["score"], "strokes": json.loads(row["strokes"]),
        "created_at": row["created_at"], "updated_at": row["updated_at"],
    }
