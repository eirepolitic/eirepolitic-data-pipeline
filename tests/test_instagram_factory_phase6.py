from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from instagram.factory.constituency_batch import generate_constituency_batch
from instagram.factory.ready import mark_ready_for_posting
from instagram.factory.review import mark_review
from instagram.factory.review_index import build_review_index

PROJECT = "instagram/projects/constituency_issue_profile_v1/project.yml"


class InstagramFactoryPhase6Test(TestCase):
    def test_review_index_lists_all_items_and_never_enables_publishing(self) -> None:
        with TemporaryDirectory() as temp_dir:
            batch = generate_constituency_batch(PROJECT, data_source="local", output_root=temp_dir, git_sha="phase6-index")
            root = Path(batch["output_root"])
            report = build_review_index(root)
            self.assertEqual(report["item_count"], batch["expected_item_count"])
            self.assertFalse(report["publishing_allowed"])
            self.assertTrue((root / "review/review_index.html").is_file())
            self.assertTrue((root / "review/review_index_manifest.json").is_file())

    def test_ready_gate_rejects_unreviewed_run(self) -> None:
        with TemporaryDirectory() as temp_dir:
            batch = generate_constituency_batch(PROJECT, data_source="local", output_root=temp_dir, git_sha="phase6-blocked")
            with self.assertRaisesRegex(ValueError, "Run is not fully approved"):
                mark_ready_for_posting(batch["output_root"], reviewer="tester")

    def test_fully_approved_run_can_be_marked_ready_without_publication(self) -> None:
        with TemporaryDirectory() as temp_dir:
            batch = generate_constituency_batch(PROJECT, data_source="local", output_root=temp_dir, git_sha="phase6-ready")
            root = Path(batch["output_root"])
            for slug in batch["items"]:
                mark_review(root, item_slug=slug, status="approved", reviewer="tester")
            report = mark_ready_for_posting(root, reviewer="tester", note="manual review complete")
            self.assertTrue(report["approved"])
            self.assertTrue(report["ready_for_posting"])
            self.assertFalse(report["publishing_allowed"])
            manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["ready_for_posting"])
            self.assertFalse(manifest["publishing_allowed"])
