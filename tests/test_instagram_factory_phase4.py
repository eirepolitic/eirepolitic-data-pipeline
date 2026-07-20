from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from instagram.factory.constituency_batch import generate_constituency_batch
from instagram.factory.regeneration import regenerate_selected
from instagram.factory.review import mark_review

PROJECT = "instagram/projects/constituency_issue_profile_v1/project.yml"


class InstagramFactoryPhase4Test(TestCase):
    def test_review_updates_item_and_slide_state_without_enabling_publication(self) -> None:
        with TemporaryDirectory() as temp_dir:
            batch = generate_constituency_batch(PROJECT, data_source="local", output_root=temp_dir, git_sha="phase4")
            root = Path(batch["output_root"])
            slug = next(iter(batch["items"]))
            state = mark_review(root, item_slug=slug, slide_id="cover", status="approved", note="cover checked")
            self.assertEqual(state["items"][slug]["slides"]["cover"], "approved")
            self.assertFalse(state["publishing_allowed"])
            self.assertEqual(state["history"][-1]["note"], "cover checked")

    def test_targeted_regeneration_preserves_unaffected_slide_hash(self) -> None:
        with TemporaryDirectory() as temp_dir:
            batch = generate_constituency_batch(PROJECT, data_source="local", output_root=temp_dir, git_sha="phase4")
            source_root = Path(batch["output_root"])
            slug = next(iter(batch["items"]))
            source_item = json.loads((source_root / batch["items"][slug]["manifest"]).read_text(encoding="utf-8"))
            original_hashes = {slide["slide_id"]: slide["sha256"] for slide in source_item["slides"]}
            destination = Path(temp_dir) / "derived"
            report = regenerate_selected(
                PROJECT,
                source_root,
                destination,
                new_run_id="derived-test-run",
                item_slugs=[slug],
                slide_ids=["issue_profile"],
                reason="chart correction",
                data_source="local",
            )
            derived_item = json.loads((destination / batch["items"][slug]["manifest"]).read_text(encoding="utf-8"))
            derived_hashes = {slide["slide_id"]: slide["sha256"] for slide in derived_item["slides"]}
            self.assertEqual(derived_hashes["cover"], original_hashes["cover"])
            self.assertEqual(report["parent_run_id"], batch["run_id"])
            self.assertFalse(report["publishing_allowed"])
