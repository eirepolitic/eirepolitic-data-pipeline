from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import yaml
from PIL import Image

from instagram.factory.adapters import ADAPTERS, FactoryAdapter
from instagram.factory.generic_batch import generate_project_batch
from instagram.factory.generic_tests import render_project_tests


def _load_records(_: str):
    return ([{"scope_id": "ireland", "scope_name": "Ireland"}], {"mode": "local"}, {"row_count": 1})


def _context(record, project):
    return {**record, "display_label": record["scope_name"], "no_publication": True}


def _scenarios(records, project):
    row = records[0]
    return {
        "minimum": {
            **row,
            "scenario": "minimum",
            "synthetic": True,
            "synthetic_reason": "Deterministic non-production fixture used to test the grain-agnostic orchestrator.",
            "data_origin": "synthetic_contract_edge",
        },
        "maximum": {
            **row,
            "scenario": "maximum",
            "synthetic": True,
            "synthetic_reason": "Deterministic non-production fixture used to test the grain-agnostic orchestrator.",
            "data_origin": "synthetic_contract_edge",
        },
        "real_example": {
            **row,
            "scenario": "real_example",
            "synthetic": False,
            "data_origin": "current_real",
            "selection_reason": "Only available deterministic fixture record.",
            "source_item_key": "ireland",
            "source_item_label": "Ireland",
        },
    }


def _assets(item_dir, context, project):
    path = item_dir / "assets" / "card.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1080, 1080), "white").save(path)
    return {"paths": {"cover": path}, "visual_manifest": None}


def _media(slide, assets):
    return assets["cover"]


class GenericFactoryCoreTest(TestCase):
    def setUp(self) -> None:
        ADAPTERS["national_fixture_v1"] = FactoryAdapter(
            adapter_id="national_fixture_v1",
            load_records=_load_records,
            build_context=_context,
            build_scenarios=_scenarios,
            render_assets=_assets,
            media_for_slide=_media,
        )

    def tearDown(self) -> None:
        ADAPTERS.pop("national_fixture_v1", None)

    def _project(self, root: Path) -> Path:
        path = root / "project.yml"
        path.write_text(yaml.safe_dump({
            "project_id": "national_fixture_v1",
            "version": 1,
            "status": "draft",
            "purpose": "Prove the factory core supports a non-constituency grain.",
            "factory": {"adapter": "national_fixture_v1"},
            "granularity": {
                "grain": "national",
                "key_fields": ["scope_id"],
                "label_field": "scope_name",
                "source_metric": "fixture_issue_counts",
                "selector": {"mode": "all"},
                "ordering": {"field": "scope_name", "direction": "ascending"},
            },
            "period": {"mode": "latest_available", "field": None},
            "slides": [{
                "slide_id": "cover",
                "order": 1,
                "post_type_id": "title_text_media_v1",
                "text": {
                    "slide_title": "{{ scope_name }}",
                    "body_text": "National test post.",
                    "footer_text": "Synthetic fixture",
                },
                "media": {"main_media": "cover", "type": "generated_card"},
                "accessibility": {"alt_text_template": "National test cover"},
                "fallback_behavior": {"missing_media": "fail"},
            }],
            "validation": {
                "scenarios": ["minimum", "maximum", "real_example"],
                "real_example_selector": {"mode": "first_complete"},
                "require_explicit_approval": True,
            },
            "output": {
                "local_root": str(root / "tests"),
                "s3_prefix": "processed/instagram_factory/projects/national_fixture_v1",
                "preview_branch": "instagram-preview-output",
            },
            "review": {"require_all_items_reviewed": True, "allow_targeted_regeneration": True},
            "schedule": {"enabled": False, "trigger": "manual"},
        }, sort_keys=False), encoding="utf-8")
        return path

    def test_batch_and_scenarios_support_national_grain(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = self._project(root)
            batch = generate_project_batch(project, data_source="local", output_root=root / "batch", git_sha="generic")
            self.assertEqual(batch["grain"], "national")
            self.assertEqual(batch["adapter_id"], "national_fixture_v1")
            self.assertEqual(batch["expected_item_count"], 1)
            manifest = json.loads((Path(batch["output_root"]) / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["publishing_allowed"])

            tests = render_project_tests(project, data_source="local", output_root=root / "scenarios")
            self.assertEqual(tests["grain"], "national")
            self.assertEqual(set(tests["scenario_manifests"]), {"minimum", "maximum", "real_example"})
            self.assertEqual(tests["rendered_scenario_count"], 3)
            self.assertEqual(tests["waived_scenario_count"], 0)
