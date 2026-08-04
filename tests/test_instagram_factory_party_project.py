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
            for gate in (
                "layout_utilization",
                "media_slot_fill",
                "visual_plot_utilization",
                "title_text_bounds",
                "chart_text_bounds",
                "dynamic_text_sizing",
            ):
                self.assertTrue(scenarios["quality_gates"][gate])

            for manifest in scenarios["scenario_manifests"].values():
                if manifest["status"] == "waived":
                    self.assertEqual(manifest["data_origin"], "waived_no_real_case")
                    self.assertTrue(manifest["waiver_reason"])
                    self.assertFalse(manifest["slides"])
                    continue

                self.assertEqual(manifest["data_origin"], "current_real")
                self.assertFalse(manifest["synthetic"])
                self.assertTrue(manifest["selection_reason"])
                self.assertTrue(manifest["source_item_key"])
                self.assertTrue(manifest["slides"])
                for slide in manifest["slides"]:
                    self.assertTrue(slide["layout_quality"]["success"])
                    self.assertGreaterEqual(slide["layout_quality"]["whitespace"]["occupied_height_ratio"], 0.88)
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
                        self.assertGreaterEqual(quality["metrics"]["plot_vertical_fill_ratio"], 0.88)
                        self.assertGreaterEqual(quality["metrics"]["plot_area_ratio"], 0.55)
                        self.assertGreaterEqual(quality["metrics"]["category_label_font_size"], 14)
                        self.assertGreaterEqual(quality["metrics"]["value_label_font_size"], 14)
                        self.assertGreaterEqual(quality["metrics"]["axis_font_size"], 11)
                        self.assertGreaterEqual(quality["metrics"]["bar_thickness_px"], 70)
                        self.assertLessEqual(quality["metrics"]["max_wrapped_label_lines"], 2)
                        self.assertLessEqual(quality["metrics"]["max_value_label_x_ratio"], 0.98)
                        self.assertEqual(quality["metrics"]["category_text_clipped_count"], 0)
                        self.assertEqual(quality["metrics"]["value_text_clipped_count"], 0)
                        self.assertEqual(quality["metrics"]["truncated_label_count"], 0)
            self.assertGreater(scenarios["rendered_scenario_count"], 0)
            self.assertFalse(scenarios["publishing_allowed"])

            sheet = scenarios["validation_contact_sheet"]
            self.assertEqual(sheet["layout"], "deduplicated_summary_plus_complete_audit")
            self.assertEqual(sheet["scenario_count"], len(scenarios["required_scenarios"]))
            self.assertTrue(sheet["summary"]["cover_shown_once"])
            self.assertGreater(sheet["summary"]["unique_visual_count"], 0)
            self.assertLessEqual(sheet["summary"]["unique_visual_count"], scenarios["rendered_scenario_count"])
            self.assertNotIn("minimum", [scenario for group in sheet["summary"]["render_groups"] for scenario in group["scenarios"]])
            self.assertNotIn("maximum", [scenario for group in sheet["summary"]["render_groups"] for scenario in group["scenarios"]])

            for page in sheet["summary"]["pages"] + sheet["audit"]["pages"]:
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
