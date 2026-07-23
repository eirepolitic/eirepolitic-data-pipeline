from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from instagram.renderer.template_renderer import render_template
from instagram.visuals.renderers.common import write_json

from .adapters import get_adapter
from .catalogues import REPO_ROOT, CatalogueSet, load_catalogues
from .common import replace_tokens
from .project import load_project, validate_project
from .validation_contact_sheet import build_validation_contact_sheet


def _required_scenarios(project: dict[str, Any], catalogues: CatalogueSet) -> list[str]:
    required = list(project["validation"]["scenarios"])
    for slide in sorted(project["slides"], key=lambda row: row["order"]):
        visual = slide.get("visual")
        if not isinstance(visual, dict):
            continue
        visual_type = catalogues.visual_types[visual["visual_type_id"]]
        profile = visual_type.get("validation_profile")
        if not isinstance(profile, dict):
            continue
        for scenario in profile.get("required_scenarios", []):
            if scenario not in required:
                required.append(scenario)
    return required


def render_project_tests(
    project_path: str | Path,
    *,
    data_source: str = "local",
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    project = load_project(project_path)
    catalogues = load_catalogues()
    validation = validate_project(project=project, catalogues=catalogues)
    if not validation["success"]:
        raise ValueError("Invalid project:\n" + "\n".join(validation["errors"]))

    adapter = get_adapter(project)
    records, source_manifest, join_manifest = adapter.load_records(data_source)
    scenarios = adapter.build_scenarios(records, project)
    required_scenarios = _required_scenarios(project, catalogues)
    root = Path(output_root or project.get("output", {}).get("local_root") or f"generated_factory_tests/{project['project_id']}")
    if not root.is_absolute():
        root = REPO_ROOT / root
    root.mkdir(parents=True, exist_ok=True)

    template_cache: dict[str, dict[str, Any]] = {}
    scenario_manifests: dict[str, Any] = {}
    rendered_count = 0
    waived_count = 0

    for scenario_name in required_scenarios:
        if scenario_name not in scenarios:
            raise ValueError(f"Adapter did not provide required scenario: {scenario_name}")
        scenario = scenarios[scenario_name]
        scenario_dir = root / scenario_name

        if scenario.get("waived") is True:
            waiver_reason = str(scenario.get("waiver_reason") or "").strip()
            if not waiver_reason:
                raise ValueError(f"Waived scenario '{scenario_name}' is missing waiver_reason")
            scenario_manifest = {
                "scenario": scenario_name,
                "adapter_id": adapter.adapter_id,
                "grain": project["granularity"]["grain"],
                "status": "waived",
                "data_origin": str(scenario.get("data_origin", "waived_no_real_case")),
                "waiver_reason": waiver_reason,
                "synthetic": False,
                "no_publication": True,
                "slides": [],
                "visual_manifest": None,
            }
            scenario_dir.mkdir(parents=True, exist_ok=True)
            write_json(scenario_dir / "scenario_manifest.json", scenario_manifest)
            scenario_manifests[scenario_name] = scenario_manifest
            waived_count += 1
            continue

        context = adapter.build_context(scenario, project)
        context["scenario"] = scenario_name
        asset_result = adapter.render_assets(scenario_dir, context, project)
        rendered: list[dict[str, Any]] = []

        for slide in sorted(project["slides"], key=lambda row: row["order"]):
            post_type = catalogues.post_types[slide["post_type_id"]]
            layout_path = str(post_type["layout_path"])
            if layout_path not in template_cache:
                template_cache[layout_path] = json.loads((REPO_ROOT / layout_path).read_text(encoding="utf-8"))
            bindings = {key: replace_tokens(value, context) for key, value in slide.get("text", {}).items()}
            bindings["main_media"] = str(adapter.media_for_slide(slide, asset_result["paths"]))
            output_path = scenario_dir / f"{slide['order']:02d}_{slide['slide_id']}.png"
            result = render_template(template_cache[layout_path], bindings, output_path)
            if result.warnings:
                raise ValueError(f"Render warnings for {scenario_name}/{slide['slide_id']}: {result.warnings}")
            rendered.append({
                "slide_id": slide["slide_id"],
                "path": str(output_path.relative_to(root)),
                "warnings": [],
            })

        scenario_manifest = {
            "scenario": scenario_name,
            "adapter_id": adapter.adapter_id,
            "grain": project["granularity"]["grain"],
            "status": "rendered",
            "data_origin": str(context.get("data_origin", "unknown")),
            "selection_reason": context.get("selection_reason"),
            "source_item_key": context.get("source_item_key"),
            "source_item_label": context.get("source_item_label"),
            "scenario_metrics": context.get("scenario_metrics"),
            "synthetic": bool(context.get("synthetic", False)),
            "synthetic_reason": context.get("synthetic_reason"),
            "no_publication": True,
            "slides": rendered,
            "visual_manifest": asset_result.get("visual_manifest"),
        }
        if scenario_manifest["synthetic"] and not scenario_manifest["synthetic_reason"]:
            raise ValueError(f"Synthetic scenario '{scenario_name}' is missing a documented synthetic_reason")
        write_json(scenario_dir / "scenario_manifest.json", scenario_manifest)
        scenario_manifests[scenario_name] = scenario_manifest
        rendered_count += 1

    contact_sheet = build_validation_contact_sheet(
        root=root,
        project_id=str(project["project_id"]),
        scenario_manifests=scenario_manifests,
        scenario_order=required_scenarios,
    )

    report = {
        "success": True,
        "project_id": project["project_id"],
        "project_version": project["version"],
        "adapter_id": adapter.adapter_id,
        "grain": project["granularity"]["grain"],
        "data_source": data_source,
        "source_manifest": source_manifest,
        "join_manifest": join_manifest,
        "required_scenarios": required_scenarios,
        "rendered_scenario_count": rendered_count,
        "waived_scenario_count": waived_count,
        "scenario_manifests": scenario_manifests,
        "validation_contact_sheet": contact_sheet,
        "review_state": "needs_review",
        "approved": False,
        "publishing_allowed": False,
        "warnings": validation["warnings"],
    }
    write_json(root / "project_validation_manifest.json", report)
    return {**report, "output_root": str(root)}
