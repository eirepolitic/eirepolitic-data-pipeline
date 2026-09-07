import json
import tempfile
import unittest
from pathlib import Path

from tools.completed_post_archive import prepare, update_index


class CompletedPostArchiveTests(unittest.TestCase):
    def test_prepare_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "slide.png").write_bytes(b"final-slide")
            summary = {
                "schema_version": 1,
                "post_id": "post-1",
                "title": "Example",
                "status": "completed",
                "repository": "Eirepolitic-data-pipeline",
                "created_by": "test-agent",
                "created_at": "2026-09-06T00:00:00Z",
                "platform": "Instagram",
                "assets": ["slide.png"],
                "tools_used": ["approved renderer"],
                "process": ["rendered", "reviewed"],
                "qa": ["dimensions checked"],
                "decisions": ["human approved"],
            }
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            out = root / "bundle"
            prepare(source, summary_path, out, "Eirepolitic-data-pipeline", "post-1")
            self.assertTrue((out / "assets/slide.png").is_file())
            provenance = json.loads((out / "provenance.json").read_text())
            self.assertEqual(provenance["files"][0]["path"], "assets/slide.png")
            self.assertEqual(len(provenance["files"][0]["sha256"]), 64)
            index = root / "index.json"
            update_index(index, out / "agent-summary.json", "s3://bucket/prefix/post-1/")
            data = json.loads(index.read_text())
            self.assertEqual(data["posts"][0]["post_id"], "post-1")
            with self.assertRaises(ValueError):
                update_index(index, out / "agent-summary.json", "s3://bucket/prefix/post-1/")

    def test_rejects_non_completed_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"; source.mkdir(); (source / "a").write_text("x")
            summary = {
                "schema_version": 1, "post_id": "p", "title": "x", "status": "draft",
                "repository": "Eirepolitic-data-pipeline", "created_by": "agent", "created_at": "now",
                "platform": "Instagram", "assets": ["a"], "tools_used": ["x"], "process": ["x"],
                "qa": ["x"], "decisions": ["x"]
            }
            path = root / "summary.json"; path.write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                prepare(source, path, root / "out", "Eirepolitic-data-pipeline", "p")


if __name__ == "__main__":
    unittest.main()
