from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from instagram.factory.constituency_batch import generate_constituency_batch, slugify, stable_run_id


class InstagramFactoryBatchTest(TestCase):
    def test_slug_and_run_id_are_deterministic(self) -> None:
        self.assertEqual(slugify("Dún Laoghaire"), "dun-laoghaire")
        first = stable_run_id("project", 1, "batch-1", "abc123")
        second = stable_run_id("project", 1, "batch-1", "abc123")
        self.assertEqual(first, second)
        self.assertNotEqual(first, stable_run_id("project", 1, "batch-2", "abc123"))

    def test_local_batch_generates_isolated_items_and_manifests(self) -> None:
        with TemporaryDirectory() as temp_dir:
            report = generate_constituency_batch(
                "instagram/projects/constituency_issue_profile_v1/project.yml",
                data_source="local",
                output_root=temp_dir,
                git_sha="test-sha",
                workflow_run_id="test-run",
            )
            root = Path(temp_dir)
            self.assertIn(report["state"], {"succeeded", "partially_failed"})
            self.assertGreater(report["succeeded_item_count"], 0)
            self.assertEqual(
                report["expected_item_count"],
                report["succeeded_item_count"] + report["failed_item_count"],
            )
            self.assertFalse(report["approved"])
            self.assertFalse(report["publishing_allowed"])
            self.assertTrue((root / "run_manifest.json").is_file())
            self.assertTrue((root / "review/review_state.json").is_file())
            self.assertTrue((root / "review/batch_sample_issue_profiles.png").is_file())

            manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
            for slug, summary in manifest["items"].items():
                item_root = root / "generated" / slug
                self.assertTrue((item_root / "item_manifest.json").is_file())
                if summary["status"] == "succeeded":
                    self.assertTrue((item_root / "slides/01_cover.png").is_file())
                    self.assertTrue((item_root / "slides/02_issue_profile.png").is_file())
