import json
import os
from pathlib import Path

from flask import Flask, abort, jsonify, send_file, send_from_directory

from .db import connect, initialize, sync_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = (
    ("toupview-test4", "ToupView · test4", "ToupView/test4", "cutting"),
    ("map-global", "Map · 全局扫描", "Map", "longest"),
)


def create_app(test_config=None):
    app = Flask(__name__, static_folder=None)
    app.config.from_mapping(
        DATABASE=os.getenv("CORESG_DATABASE", str(PROJECT_ROOT / "instance" / "coresegment.db")),
        SCAN_ROOT=os.getenv("CORESG_SCAN_ROOT", str(PROJECT_ROOT / "scan_output")),
        UI_ROOT=str(PROJECT_ROOT / "ui"),
    )
    if test_config:
        app.config.update(test_config)

    initialize(app.config["DATABASE"])
    _sync_samples(app.config["DATABASE"], Path(app.config["SCAN_ROOT"]))

    @app.get("/")
    def index():
        return send_from_directory(app.config["UI_ROOT"], "index.html")

    @app.get("/api/health")
    def health():
        with connect(app.config["DATABASE"]) as connection:
            count = connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        return jsonify(status="ok", database="sqlite", datasets=count)

    @app.get("/api/datasets")
    def list_datasets():
        with connect(app.config["DATABASE"]) as connection:
            rows = connection.execute(
                """SELECT d.id, d.name, d.route_type, a.kind, a.file_name
                   FROM datasets d LEFT JOIN artifacts a ON a.dataset_id=d.id
                   ORDER BY CASE WHEN d.id = 'toupview-test4' THEN 0 ELSE 1 END, d.id, a.kind"""
            ).fetchall()
        datasets = {}
        for row in rows:
            item = datasets.setdefault(row["id"], {
                "id": row["id"], "name": row["name"], "routeType": row["route_type"], "artifacts": {}
            })
            if row["kind"]:
                item["artifacts"][row["kind"]] = f"/api/datasets/{row['id']}/artifacts/{row['kind']}"
        return jsonify(items=list(datasets.values()))

    @app.get("/api/datasets/<dataset_id>/motion")
    def get_motion(dataset_id):
        with connect(app.config["DATABASE"]) as connection:
            dataset = connection.execute(
                "SELECT id, source_path FROM datasets WHERE id = ?", (dataset_id,)
            ).fetchone()
            if dataset is None:
                abort(404)
            rows = connection.execute(
                "SELECT direction, steps FROM motion_records WHERE dataset_id = ? ORDER BY sequence",
                (dataset_id,),
            ).fetchall()
        skeleton_info = {}
        try:
            payload = json.loads((Path(dataset["source_path"]) / "motion_steps.json").read_text(encoding="utf-8"))
            skeleton_info = payload.get("skeleton_info", {})
        except (OSError, json.JSONDecodeError):
            pass
        return jsonify(records=[dict(row) for row in rows], skeleton_info=skeleton_info)

    @app.get("/api/datasets/<dataset_id>/artifacts/<kind>")
    def get_artifact(dataset_id, kind):
        with connect(app.config["DATABASE"]) as connection:
            row = connection.execute(
                """SELECT d.source_path, a.file_name, a.media_type
                   FROM datasets d JOIN artifacts a ON a.dataset_id=d.id
                   WHERE d.id = ? AND a.kind = ?""",
                (dataset_id, kind),
            ).fetchone()
        if row is None:
            abort(404)
        return send_file(Path(row["source_path"]) / row["file_name"], mimetype=row["media_type"], conditional=True)

    @app.errorhandler(404)
    def not_found(_error):
        if os.getenv("FLASK_ENV") == "development":
            return jsonify(error="not_found"), 404
        return jsonify(error="not_found"), 404

    return app


def _sync_samples(database_path, scan_root):
    with connect(database_path) as connection:
        for dataset_id, name, relative_path, route_type in DEFAULT_DATASETS:
            source = scan_root / relative_path
            if source.is_dir():
                sync_dataset(connection, dataset_id, name, source, route_type)
            else:
                connection.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
