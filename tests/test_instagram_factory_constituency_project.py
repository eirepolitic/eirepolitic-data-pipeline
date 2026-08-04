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
            self.assertTrue(report["validation_contact_sheet"]["summary"]["cover_shown_once"])

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
                    for media in slide["layout_quality"]["media"]:
                        if media.get("measurable") and media.get("fit") == "contain":
                            self.assertGreaterEqual(media["vertical_fill_ratio"], 0.96)
                            self.assertGreaterEqual(media["area_fill_ratio"], 0.90)

            real_cover = next(
                slide
                for slide in report["scenario_manifests"]["real_example"]["slides"]
                if slide["slide_id"] == "cover"
            )
            with Image.open(Path(report["output_root"]) / real_cover["path"]) as image:
                self.assertEqual(image.size, (1080, 1350))
