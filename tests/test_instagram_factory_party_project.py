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
            for gate in (
                "layout_utilization",
                "media_slot_fill",
                "visual_plot_utilization",
                "title_text_bounds",
                "chart_text_bounds",
                "dynamic_text_sizing",
                "dense_real_examples",
            ):
                self.assertTrue(scenarios["quality_gates"][gate])

            self.assertTrue(scenarios["density_validation"]["success"])
            self.assertGreaterEqual(
                scenarios["density_validation"]["metrics"]["item_count_max"]["displayed_item_count"],
                6,
            )
            self.assertGreaterEqual(
                scenarios["density_validation"]["metrics"]["real_example"]["displayed_item_count"],
                5,
            )

            for manifest in scenarios["scenario_manifests"].values():
                if manifest["status"] == "waived":
                    self.assertEqual(manifest["data_origin"], "waived_no_real_case")
                    self.assertTrue(manifest["waiver_reason"])
                    continue
                for slide in manifest["slides"]:
                    self.assertTrue(slide["layout_quality"]["success"])
                    if slide["visual_quality"] is not None:
                        self.assertTrue(slide["visual_quality"]["success"])

            sheet = scenarios["validation_contact_sheet"]
            self.assertEqual(
                sheet["layout"],
                "two_column_full_review_plus_deduplicated_summary_plus_complete_audit",
            )
            self.assertEqual(sheet["full"]["columns"], 2)
            self.assertEqual(sheet["full"]["preview_width"], 760)
            self.assertEqual(sheet["full"]["preview_height"], 950)
            self.assertTrue(sheet["full"]["metric_first"])
            self.assertTrue(sheet["full"]["waivers_inline"])
            self.assertTrue(sheet["full"]["cover_metadata_compact"])
            self.assertEqual(sheet["full"]["waiver_card_count"], scenarios["waived_scenario_count"])
            self.assertTrue(sheet["full"]["cover_shown_once"])

            expected_primary = {
                name
                for name in HORIZONTAL_BAR_REQUIRED_SCENARIOS
                if scenarios["scenario_manifests"][name]["status"] == "rendered"
            }
            waived_primary = {
                name
                for name in HORIZONTAL_BAR_REQUIRED_SCENARIOS
                if scenarios["scenario_manifests"][name]["status"] == "waived"
            }
            self.assertEqual(set(sheet["full"]["scenario_rows"]), expected_primary)
            self.assertEqual(set(sheet["full"]["waived_scenarios"]), waived_primary)

            for page in sheet["full"]["pages"] + sheet["summary"]["pages"] + sheet["audit"]["pages"]:
                page_path = Path(scenarios["output_root"]) / page
                self.assertTrue(page_path.is_file())
                with Image.open(page_path) as image:
                    self.assertEqual(image.width, 2400)
                    self.assertGreater(image.height, 1000)

            batch = generate_project_batch(PROJECT, data_source="local", output_root=root / "batch", git_sha="party-test")
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
