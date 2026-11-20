import json
import tempfile
import unittest
from pathlib import Path

from backend.app import create_app


class BackendApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        scan_root = root / "scan_output"
        sample = scan_root / "ToupView" / "test4"
        sample.mkdir(parents=True)
        (sample / "global_map.png").write_bytes(b"fake image")
        (sample / "cutting_path_coordinates.json").write_text(
            json.dumps({"cutting_points": [[1, 2]]}), encoding="utf-8"
        )
        (sample / "motion_steps.json").write_text(
            json.dumps({"skeleton_info": {"total_pixels": 42}, "records": [
                {"direction": "RIGHT", "steps": 10}
            ]}), encoding="utf-8"
        )
        self.app = create_app({
            "TESTING": True,
            "DATABASE": str(root / "db" / "test.sqlite3"),
            "SCAN_ROOT": str(scan_root),
            "UI_ROOT": str(root),
        })
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_health_and_dataset_catalog(self):
        self.assertEqual(self.client.get("/api/health").json["datasets"], 1)
        item = self.client.get("/api/datasets").json["items"][0]
        self.assertEqual(item["id"], "toupview-test4")
        self.assertIn("route", item["artifacts"])

    def test_motion_and_registered_artifact(self):
        motion = self.client.get("/api/datasets/toupview-test4/motion").json
        self.assertEqual(motion["records"], [{"direction": "RIGHT", "steps": 10}])
        self.assertEqual(motion["skeleton_info"]["total_pixels"], 42)
        artifact = self.client.get("/api/datasets/toupview-test4/artifacts/image")
        self.assertEqual(artifact.data, b"fake image")
        artifact.close()

    def test_unknown_artifact_is_not_served(self):
        response = self.client.get("/api/datasets/toupview-test4/artifacts/../../requirements.txt")
        self.assertEqual(response.status_code, 404)

    def test_repository_samples_are_indexed(self):
        app = create_app({
            "TESTING": True,
            "DATABASE": str(Path(self.temp_dir.name) / "repository.sqlite3"),
            "SCAN_ROOT": str(Path(__file__).resolve().parents[1] / "scan_output"),
        })
        client = app.test_client()
        items = client.get("/api/datasets").json["items"]
        by_id = {item["id"]: item for item in items}
        self.assertEqual(items[0]["id"], "toupview-test4")
        self.assertIn("toupview-test4", by_id)
        self.assertIn("map-global", by_id)
        self.assertEqual(by_id["map-global"]["routeType"], "longest")
        motion = client.get("/api/datasets/map-global/motion").json
        self.assertEqual(len(motion["records"]), 937)


if __name__ == "__main__":
    unittest.main()
