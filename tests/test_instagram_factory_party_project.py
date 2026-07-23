from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image

from instagram.factory.generic_batch import generate_project_batch
from instagram.factory.generic_regeneration import regenerate_project_items
from instagram.factory.generic_tests import render_project_tests
from instagram.factory.validation_scenarios import HORIZONTAL_BAR_REQUIRED_SCENARIOS

PROJECT = "instagram/projects/party_issue_profile_v1/project.yml"


class PartyIssueProfileProjectTest(TestCase):
    def test_party_scenarios_and_batch_use_generic_core(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scenarios = render_project_tests(PROJECT, data_source="local", output_root=root / "scenarios")
            self.assertEqual(scenarios["grain"], "party")
            self.assertEqual(scenarios["adapter_id"], "party_issue_profile_v1")
            self.assertTrue(set(HORIZONTAL_BAR_REQUIRED_SCENARIOS).issubset(scenarios["scenario_manifests"]))
            self.assertIn("minimum", scenarios["scenario_manifests"])
            self.assertIn("maximum", scenarios["scenario_manifests"])

            for manifest in scenarios["scenario_manifests"].values():
                if manifest["status"] == "waived":
                    self.assertEqual(manifest["data_origin"], "waived_no_real_case")
                    self.assertTrue(manifest["waiver_reason"])
                    self.assertFalse(manifest["slides"])
                else:
                    self.assertEqual(manifest["data_origin"], "current_real")
                    self.assertFalse(manifest["synthetic"])
                    self.assertTrue(manifest["selection_reason"])
                    self.assertTrue(manifest["source_item_key"])
                    self.assertTrue(manifest["slides"])
            self.assertGreater(scenarios["rendered_scenario_count"], 0)
            self.assertFalse(scenarios["publishing_allowed"])

            sheet = scenarios["validation_contact_sheet"]
            self.assertEqual(sheet["layout"], "large_readable_tiles")
            self.assertEqual(sheet["scenario_count"], len(scenarios["required_scenarios"]))
            self.assertTrue(sheet["pages"])
            for page in sheet["pages"]:
                page_path = Path(scenarios["output_root"]) / page
                self.assertTrue(page_path.is_file())
                with Image.open(page_path) as image:
                    self.assertEqual(image.width, 2800)
                    self.assertGreater(image.height, 1000)
            self.assertTrue((Path(scenarios["output_root"]) / "validation_contact_sheet_manifest.json").is_file())

            batch = generate_project_batch(PROJECT, data_source="local", output_root=root / "batch", git_sha="party-test")
            self.assertEqual(batch["grain"], "party")
            self.assertEqual(batch["adapter_id"], "party_issue_profile_v1")
            self.assertGreater(batch["expected_item_count"], 0)
            self.assertEqual(batch["failed_item_count"], 0)
            self.assertFalse(batch["publishing_allowed"])

    def test_party_targeted_regeneration_preserves_cover(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            batch = generate_project_batch(PROJECT, data_source="local", output_root=root / "batch", git_sha="party-regen")
            source = Path(batch["output_root"])
            slug = next(iter(batch["items"]))
            manifest_path = source / batch["items"][slug]["manifest"]
            source_item = json.loads(manifest_path.read_text(encoding="utf-8"))
            hashes = {slide["slide_id"]: slide["sha256"] for slide in source_item["slides"]}

            destination = root / "derived"
            report = regenerate_project_items(
                PROJECT,
                source,
                destination,
                new_run_id="party-derived-test",
                item_slugs=[slug],
                slide_ids=["issue_profile"],
                reason="party chart correction",
                data_source="local",
            )
            derived = json.loads((destination / batch["items"][slug]["manifest"]).read_text(encoding="utf-8"))
            derived_hashes = {slide["slide_id"]: slide["sha256"] for slide in derived["slides"]}
            self.assertEqual(derived_hashes["cover"], hashes["cover"])
            self.assertEqual(report["grain"], "party")
            self.assertFalse(report["publishing_allowed"])
