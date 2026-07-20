from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from instagram.factory.catalogues import load_catalogues, validate_catalogues
from instagram.factory.constituency_pilot import build_constituency_records, build_scenarios, render_project_tests
from instagram.factory.project import load_project, validate_project


class InstagramFactoryCatalogueTest(TestCase):
    def test_repository_catalogues_are_valid(self) -> None:
        report = validate_catalogues()
        self.assertTrue(report["success"], msg="\n".join(report["errors"]))
        self.assertEqual(report["counts"]["post_types"], 4)
        self.assertEqual(report["counts"]["visual_types"], 17)
        self.assertEqual(report["counts"]["metrics"], 4)


class InstagramFactoryProjectTest(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cls.repo_root = repo_root
        cls.catalogues = load_catalogues()
        cls.template = load_project(repo_root / "instagram/projects/_template/project.yml")

    def test_project_template_is_valid(self) -> None:
        report = validate_project(project=self.template, catalogues=self.catalogues)
        self.assertTrue(report["success"], msg="\n".join(report["errors"]))
        self.assertEqual(report["slide_count"], 1)

    def test_constituency_pilot_is_valid(self) -> None:
        project = load_project(self.repo_root / "instagram/projects/constituency_issue_profile_v1/project.yml")
        report = validate_project(project=project, catalogues=self.catalogues)
        self.assertTrue(report["success"], msg="\n".join(report["errors"]))
        self.assertEqual(report["slide_count"], 2)

    def test_unknown_visual_type_fails_actionably(self) -> None:
        project = deepcopy(self.template)
        project["slides"][0]["visual"]["visual_type_id"] = "missing_visual"
        report = validate_project(project=project, catalogues=self.catalogues)
        self.assertFalse(report["success"])
        self.assertTrue(any("unknown visual_type_id" in error for error in report["errors"]))

    def test_duplicate_slide_order_fails(self) -> None:
        project = deepcopy(self.template)
        duplicate = deepcopy(project["slides"][0])
        duplicate["slide_id"] = "duplicate"
        project["slides"].append(duplicate)
        report = validate_project(project=project, catalogues=self.catalogues)
        self.assertFalse(report["success"])
        self.assertTrue(any("Duplicate slide order" in error for error in report["errors"]))

    def test_approved_project_cannot_use_draft_components(self) -> None:
        project = deepcopy(self.template)
        project["status"] = "approved"
        report = validate_project(project=project, catalogues=self.catalogues)
        self.assertFalse(report["success"])
        self.assertIn("Approved projects cannot reference draft catalogue entries", report["errors"])


class InstagramFactoryConstituencyPilotTest(TestCase):
    def test_scenarios_include_synthetic_and_real_cases(self) -> None:
        members = [
            {"full_name": "Aoife Byrne", "constituency": "Wicklow-Wexford"},
            {"full_name": "Brendan Walsh", "constituency": "Wicklow-Wexford"},
            {"full_name": "Ciara Doyle", "constituency": "Dublin Bay South"},
        ]
        speeches = [
            {"Speaker Name": "Aoife Byrne", "issue": "Housing"},
            {"Speaker Name": "Aoife Byrne", "issue": "Transport"},
            {"Speaker Name": "Brendan Walsh", "issue": "Education"},
            {"Speaker Name": "Ciara Doyle", "issue": "Health"},
        ]
        records, manifest = build_constituency_records(members, speeches)
        scenarios = build_scenarios(records)
        self.assertEqual(manifest["matched_speeches"], 4)
        self.assertTrue(scenarios["minimum"]["synthetic"])
        self.assertTrue(scenarios["maximum"]["synthetic"])
        self.assertFalse(scenarios["real_example"]["synthetic"])
        self.assertTrue(all(scenario["no_publication"] for scenario in scenarios.values()))
        self.assertLessEqual(len(scenarios["minimum"]["issue_rows"]), 2)
        self.assertLessEqual(len(scenarios["maximum"]["issue_rows"]), 7)
        maximum_labels = [row["label"] for row in scenarios["maximum"]["issue_rows"]]
        self.assertEqual(len(maximum_labels), len(set(maximum_labels)))
        self.assertLessEqual(len(scenarios["real_example"]["issue_rows"]), 7)

    def test_synthetic_name_and_result_extremes_are_selected_independently(self) -> None:
        records = [
            {
                "constituency": "Mayo",
                "constituency_key": "mayo",
                "member_names": ["A"],
                "member_count": 1,
                "issue_rows": [{"label": "Housing", "value": 5}],
                "issue_count": 1,
                "speech_count": 5,
                "max_issue_label_length": 7,
            },
            {
                "constituency": "Dublin Bay North",
                "constituency_key": "dublin-bay-north",
                "member_names": ["B"],
                "member_count": 1,
                "issue_rows": [{"label": "Health", "value": 1}],
                "issue_count": 1,
                "speech_count": 1,
                "max_issue_label_length": 6,
            },
            {
                "constituency": "Limerick City",
                "constituency_key": "limerick-city",
                "member_names": ["C"],
                "member_count": 1,
                "issue_rows": [
                    {"label": "Housing", "value": 20},
                    {"label": "Transport", "value": 10},
                ],
                "issue_count": 2,
                "speech_count": 30,
                "max_issue_label_length": 9,
            },
        ]

        scenarios = build_scenarios(records)

        self.assertEqual(scenarios["minimum"]["display_constituency"], "Mayo")
        self.assertEqual(scenarios["minimum"]["result_constituency"], "Dublin Bay North")
        self.assertEqual(scenarios["minimum"]["issue_rows"], [{"label": "Health", "value": 1}])
        self.assertEqual(scenarios["maximum"]["display_constituency"], "Dublin Bay North")
        self.assertEqual(scenarios["maximum"]["result_constituency"], "Limerick City")
        self.assertEqual(scenarios["maximum"]["result_speech_count"], 30)

    def test_local_complete_slide_render(self) -> None:
        with TemporaryDirectory() as temp_dir:
            manifest = render_project_tests(
                "instagram/projects/constituency_issue_profile_v1/project.yml",
                data_source="local",
                output_root=temp_dir,
            )
            root = Path(temp_dir)
            self.assertTrue(manifest["success"])
            self.assertEqual(manifest["review_state"], "needs_review")
            for scenario in ("minimum", "maximum", "real_example"):
                self.assertTrue((root / scenario / "01_cover.png").is_file())
                self.assertTrue((root / scenario / "02_issue_profile.png").is_file())
                self.assertTrue((root / scenario / "contact_sheet.png").is_file())
                self.assertTrue((root / scenario / "scenario_manifest.json").is_file())
            self.assertTrue((root / "contact_sheets/cover.png").is_file())
            self.assertTrue((root / "contact_sheets/issue_profile.png").is_file())
            self.assertTrue((root / "project_validation_manifest.json").is_file())
