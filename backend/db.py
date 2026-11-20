import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_path TEXT NOT NULL UNIQUE,
    route_type TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    file_name TEXT NOT NULL,
    media_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    modified_at REAL NOT NULL,
    UNIQUE(dataset_id, kind)
);
CREATE TABLE IF NOT EXISTS motion_records (
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    direction TEXT NOT NULL,
    steps INTEGER NOT NULL,
    PRIMARY KEY(dataset_id, sequence)
);
CREATE INDEX IF NOT EXISTS idx_artifacts_dataset ON artifacts(dataset_id);
"""


@contextmanager
def connect(database_path):
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize(database_path):
    Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    with connect(database_path) as connection:
        connection.executescript(SCHEMA)


def sync_dataset(connection, dataset, name, source_path, route_type):
    source_path = Path(source_path).resolve()
    connection.execute(
        """INSERT INTO datasets(id, name, source_path, route_type)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET name=excluded.name,
             source_path=excluded.source_path, route_type=excluded.route_type,
             updated_at=CURRENT_TIMESTAMP""",
        (dataset, name, str(source_path), route_type),
    )

    artifacts = []
    for kind, filename in {
        "image": "global_map.png",
        "mask": "global_mask_color.png",
        "skeleton": "global_skeleton_all.png",
        "route": "cutting_path_coordinates.json" if route_type == "cutting" else "global_path.json",
        "skeleton_data": "skeleton_coordinates.json",
    }.items():
        path = source_path / filename
        if path.is_file():
            artifacts.append((dataset, kind, filename, _media_type(path), path.stat().st_size, path.stat().st_mtime))

    connection.execute("DELETE FROM artifacts WHERE dataset_id = ?", (dataset,))
    connection.executemany(
        """INSERT INTO artifacts(dataset_id, kind, file_name, media_type, size_bytes, modified_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        artifacts,
    )

    motion_path = source_path / "motion_steps.json"
    records = []
    if motion_path.is_file():
        try:
            payload = json.loads(motion_path.read_text(encoding="utf-8"))
            records = payload.get("records", [])
        except (OSError, json.JSONDecodeError):
            records = []
    connection.execute("DELETE FROM motion_records WHERE dataset_id = ?", (dataset,))
    connection.executemany(
        "INSERT INTO motion_records(dataset_id, sequence, direction, steps) VALUES (?, ?, ?, ?)",
        [(dataset, index, str(record.get("direction", "SCAN")), int(record.get("steps", 0)))
         for index, record in enumerate(records)],
    )


def _media_type(path):
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".json": "application/json",
    }.get(path.suffix.lower(), "application/octet-stream")
