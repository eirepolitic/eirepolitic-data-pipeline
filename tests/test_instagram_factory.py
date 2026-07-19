from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest import TestCase

from instagram.factory.catalogues import load_catalogues, validate_catalogues
from instagram.factory.project import load_project, validate_project


class InstagramFactoryCatalogueTest(TestCase):
    def test_repository_catalogues_are_valid(self) -> None:
        report = validate_catalogues()
        self.assertTrue(report["success"], msg="\n".join(report["errors"]))
        self.assertEqual(report["counts"]["post_types"], 4)
        self.assertEqual(report["counts"]["visual_types"], 17)
        self.assertEqual(report["counts"]["metrics"], 3)


class InstagramFactoryProjectTest(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cls.catalogues = load_catalogues()
        cls.template = load_project(repo_root / "instagram/projects/_template/project.yml")

    def test_project_template_is_valid(self) -> None:
        report = validate_project(project=self.template, catalogues=self.catalogues)
        self.assertTrue(report["success"], msg="\n".join(report["errors"]))
        self.assertEqual(report["slide_count"], 1)

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
