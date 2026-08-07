from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image

from instagram.factory.generic_tests import render_project_tests
from instagram.factory.validation_scenarios import HORIZONTAL_BAR_REQUIRED_SCENARIOS

PROJECT = "instagram/projects/constituency_issue_profile_v1/project.yml"


class ConstituencyIssueProfileProjectTest(TestCase):
    def test_constituency_validation_uses_shared_quality_contract(self) -> None:
        with TemporaryDirectory() as temp_dir:
            report = render_project_tests(
                PROJECT,
                data_source="local",
                output_root=Path(temp_dir) / "constituency-tests",
            )

            self.assertEqual(report["grain"], "constituency")
            self.assertEqual(report["adapter_id"], "constituency_issue_profile_v1")
            self.assertTrue(set(HORIZONTAL_BAR_REQUIRED_SCENARIOS).issubset(report["scenario_manifests"]))
            self.assertFalse(report["publishing_allowed"])
            self.assertTrue(report["quality_gates"]["dense_real_examples"])
            self.assertTrue(report["density_validation"]["success"])
            self.assertGreaterEqual(
                report["density_validation"]["metrics"]["item_count_max"]["displayed_item_count"],
                6,
            )
            self.assertGreaterEqual(
                report["density_validation"]["metrics"]["real_example"]["displayed_item_count"],
                5,
            )

            sheet = report["validation_contact_sheet"]
            self.assertEqual(
                sheet["layout"],
                "two_column_full_review_plus_deduplicated_summary_plus_complete_audit",
            )
            self.assertTrue(sheet["full"]["cover_shown_once"])
            self.assertTrue(sheet["summary"]["cover_shown_once"])
            self.assertEqual(sheet["full"]["columns"], 2)
            self.assertEqual(sheet["full"]["preview_width"], 760)
            self.assertEqual(sheet["full"]["preview_height"], 950)
            self.assertTrue(sheet["full"]["metric_first"])
            self.assertTrue(sheet["full"]["waivers_inline"])
            self.assertTrue(sheet["full"]["cover_metadata_compact"])
            self.assertEqual(sheet["full"]["waiver_card_count"], report["waived_scenario_count"])

            expected_primary = {
                name
                for name in HORIZONTAL_BAR_REQUIRED_SCENARIOS
                if report["scenario_manifests"][name]["status"] == "rendered"
            }
            waived_primary = {
                name
                for name in HORIZONTAL_BAR_REQUIRED_SCENARIOS
                if report["scenario_manifests"][name]["status"] == "waived"
            }
            self.assertEqual(set(sheet["full"]["scenario_rows"]), expected_primary)
            self.assertEqual(len(sheet["full"]["scenario_rows"]), len(expected_primary))
            self.assertEqual(set(sheet["full"]["waived_scenarios"]), waived_primary)
            self.assertEqual(len(sheet["full"]["waived_scenarios"]), len(waived_primary))
            self.assertNotIn("minimum", sheet["full"]["scenario_rows"])
            self.assertNotIn("maximum", sheet["full"]["scenario_rows"])
            self.assertEqual(sheet["full"]["pages"], ["validation_contact_sheet.png"])
            self.assertEqual(sheet["summary"]["pages"], ["validation_summary_contact_sheet.png"])

            rendered = [
                scenario
                for scenario in report["scenario_manifests"].values()
                if scenario["status"] == "rendered"
            ]
            self.assertTrue(rendered)
            for scenario in rendered:
                self.assertIn(scenario["data_origin"], {"current_real", "historical_real"})
                for slide in scenario["slides"]:
                    self.assertTrue(slide["layout_quality"]["success"])
                    for text_metric in slide["layout_quality"].get("text", []):
                        self.assertFalse(text_metric["clipped"])
                        self.assertFalse(text_metric["truncated"])
                        self.assertGreaterEqual(text_metric["final_font_size"], 42)
                    for media in slide["layout_quality"]["media"]:
                        if media.get("measurable") and media.get("fit") == "contain":
                            self.assertGreaterEqual(media["vertical_fill_ratio"], 0.96)
                            self.assertGreaterEqual(media["area_fill_ratio"], 0.90)
                    if slide["visual_quality"] is not None:
                        quality = slide["visual_quality"]
                        self.assertTrue(quality["success"])
                        self.assertEqual(quality["metrics"]["category_text_clipped_count"], 0)
                        self.assertEqual(quality["metrics"]["value_text_clipped_count"], 0)
                        self.assertEqual(quality["metrics"]["truncated_label_count"], 0)

            for page in sheet["full"]["pages"] + sheet["summary"]["pages"] + sheet["audit"]["pages"]:
                page_path = Path(report["output_root"]) / page
                self.assertTrue(page_path.is_file())
                with Image.open(page_path) as image:
                    self.assertEqual(image.width, 2400)
                    self.assertGreater(image.height, 1000)

            real_cover = next(
                slide
                for slide in report["scenario_manifests"]["real_example"]["slides"]
                if slide["slide_id"] == "cover"
            )
            with Image.open(Path(report["output_root"]) / real_cover["path"]) as image:
                self.assertEqual(image.size, (1080, 1350))
