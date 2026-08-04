from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from instagram.renderer.template_renderer import render_template
from instagram.visuals.renderers.common import load_yaml, write_json

from .adapters import get_adapter
from .catalogues import REPO_ROOT, CatalogueSet, load_catalogues
from .common import replace_tokens
from .layout_quality import validate_slide_layout, validate_visual_manifest
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


def _stage(stage: str, status: str, **details: Any) -> dict[str, Any]:
    return {"stage": stage, "status": status, **details}


def _combine_scenarios(
    *,
    required_scenarios: list[str],
    current: dict[str, dict[str, Any]],
    historical: dict[str, dict[str, Any]],
    historical_manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Prefer current real data, then historical real data, then record a waiver.

    Synthetic contract-edge data is deliberately not generated here. It remains a
    separately approved future fallback for recurring projects only.
    """
    combined: dict[str, dict[str, Any]] = {}
    loaded_batches = int(historical_manifest.get("loaded_batch_count", 0) or 0)
    historical_status = str(historical_manifest.get("status") or "not_configured")

    for scenario_name in required_scenarios:
        current_candidate = current.get(scenario_name)
        historical_candidate = historical.get(scenario_name)
        current_matches = bool(current_candidate and current_candidate.get("waived") is not True)
        historical_matches = bool(historical_candidate and historical_candidate.get("waived") is not True)

        if current_matches:
            selected = deepcopy(current_candidate)
            selected["scenario"] = scenario_name
            selected["data_origin"] = str(selected.get("data_origin") or "current_real")
            selected["search_stages"] = [
                _stage("current_real", "matched"),
                _stage("historical_real", "not_needed", loaded_batch_count=loaded_batches),
                _stage("synthetic_contract_edge", "not_needed"),
                _stage("waived", "not_needed"),
            ]
            combined[scenario_name] = selected
            continue

        if historical_matches:
            selected = deepcopy(historical_candidate)
            selected["scenario"] = scenario_name
            selected["data_origin"] = "historical_real"
            selected["historical_fallback"] = True
            original_reason = str(
                selected.get("selection_reason")
                or "A qualifying historical production record was selected."
            ).replace("Current real record", "Historical real record")
            selected["selection_reason"] = (
                "No qualifying current production record existed. "
                f"Historical real data selected: {original_reason}"
            )
            selected["search_stages"] = [
                _stage("current_real", "no_qualifying_case"),
                _stage(
                    "historical_real",
                    "matched",
                    loaded_batch_count=loaded_batches,
                    source_batch_id=selected.get("source_batch_id"),
                ),
                _stage("synthetic_contract_edge", "not_needed"),
                _stage("waived", "not_needed"),
            ]
            combined[scenario_name] = selected
            continue

        selected = deepcopy(current_candidate or historical_candidate or {})
        selected.update({
            "scenario": scenario_name,
            "waived": True,
            "synthetic": False,
            "no_publication": True,
            "data_origin": "waived_no_real_case",
        })
        current_reason = str(
            (current_candidate or {}).get("waiver_reason")
            or "No qualifying current production record exists."
        )
        historical_reason = str(
            (historical_candidate or {}).get("waiver_reason")
            or "No qualifying historical production record exists."
        )
        selected["waiver_reason"] = (
            f"Current production data and {loaded_batches} loaded batch(es) of historical production data were checked. "
            f"{current_reason} {historical_reason} Synthetic contract-edge data was not generated."
        )
        selected["search_stages"] = [
            _stage("current_real", "no_qualifying_case"),
            _stage(
                "historical_real",
                "no_qualifying_case" if historical_status == "completed" else historical_status,
                loaded_batch_count=loaded_batches,
            ),
            _stage("synthetic_contract_edge", "not_generated"),
            _stage("waived", "selected"),
        ]
        combined[scenario_name] = selected

    return combined


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
    current_scenarios = adapter.build_scenarios(records, project)
    required_scenarios = _required_scenarios(project, catalogues)

    historical_search: dict[str, Any] = {
        "status": "not_configured",
        "record_count": 0,
        "replacement_count": 0,
        "loaded_batch_count": 0,
    }
    historical_scenarios: dict[str, dict[str, Any]] = {}
    if adapter.load_historical_records is not None:
        historical_records, historical_manifest = adapter.load_historical_records(
            data_source,
            project,
            source_manifest,
        )
        historical_search = dict(historical_manifest)
        if historical_records:
            historical_scenarios = adapter.build_scenarios(historical_records, project)

    scenarios = _combine_scenarios(
        required_scenarios=required_scenarios,
        current=current_scenarios,
        historical=historical_scenarios,
        historical_manifest=historical_search,
    )
    historical_replacements = [
        name for name, scenario in scenarios.items()
        if scenario.get("data_origin") == "historical_real"
    ]
    historical_search["replacement_count"] = len(historical_replacements)
    historical_search["replacement_scenarios"] = historical_replacements

    root = Path(output_root or project.get("output", {}).get("local_root") or f"generated_factory_tests/{project['project_id']}")
    if not root.is_absolute():
        root = REPO_ROOT / root
    root.mkdir(parents=True, exist_ok=True)

    layout_cache: dict[str, dict[str, Any]] = {}
    visual_template_cache: dict[str, dict[str, Any]] = {}
    scenario_manifests: dict[str, Any] = {}
    rendered_count = 0
    waived_count = 0

    for scenario_name in required_scenarios:
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
                "search_stages": scenario.get("search_stages", []),
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
            if layout_path not in layout_cache:
                layout_cache[layout_path] = json.loads((REPO_ROOT / layout_path).read_text(encoding="utf-8"))
            layout = layout_cache[layout_path]
            bindings = {key: replace_tokens(value, context) for key, value in slide.get("text", {}).items()}
            bindings["main_media"] = str(adapter.media_for_slide(slide, asset_result["paths"]))
            output_path = scenario_dir / f"{slide['order']:02d}_{slide['slide_id']}.png"
            result = render_template(layout, bindings, output_path)
            if result.warnings:
                raise ValueError(f"Render warnings for {scenario_name}/{slide['slide_id']}: {result.warnings}")

            layout_quality = validate_slide_layout(
                template=layout,
                bindings=bindings,
                output_path=output_path,
                text_metrics=result.text_metrics,
            )
            if not layout_quality["success"]:
                raise ValueError(
                    f"Layout quality failed for {scenario_name}/{slide['slide_id']}: "
                    + "; ".join(layout_quality["errors"])
                )

            visual_quality = None
            visual = slide.get("visual")
            if isinstance(visual, dict):
                visual_type = catalogues.visual_types[visual["visual_type_id"]]
                visual_template_path = str(visual_type["template_path"])
                if visual_template_path not in visual_template_cache:
                    visual_template_cache[visual_template_path] = load_yaml(REPO_ROOT / visual_template_path)
                visual_quality = validate_visual_manifest(
                    visual_manifest=asset_result.get("visual_manifest"),
                    template=visual_template_cache[visual_template_path],
                )
                if not visual_quality["success"]:
                    raise ValueError(
                        f"Visual quality failed for {scenario_name}/{slide['slide_id']}: "
                        + "; ".join(visual_quality["errors"])
                    )

            rendered.append({
                "slide_id": slide["slide_id"],
                "path": str(output_path.relative_to(root)),
                "warnings": [],
                "text_metrics": result.text_metrics,
                "layout_quality": layout_quality,
                "visual_quality": visual_quality,
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
            "source_batch_id": context.get("source_batch_id"),
            "source_member_key": context.get("source_member_key"),
            "source_speech_key": context.get("source_speech_key"),
            "source_last_modified": context.get("source_last_modified"),
            "historical_fallback": bool(context.get("historical_fallback", False)),
            "search_stages": context.get("search_stages", []),
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
        "historical_search": historical_search,
        "required_scenarios": required_scenarios,
        "rendered_scenario_count": rendered_count,
        "waived_scenario_count": waived_count,
        "scenario_manifests": scenario_manifests,
        "validation_contact_sheet": contact_sheet,
        "quality_gates": {
            "layout_utilization": True,
            "media_slot_fill": True,
            "visual_plot_utilization": True,
            "title_text_bounds": True,
            "chart_text_bounds": True,
            "dynamic_text_sizing": True,
            "historical_real_data_fallback": True,
        },
        "review_state": "needs_review",
        "approved": False,
        "publishing_allowed": False,
        "warnings": validation["warnings"],
    }
    write_json(root / "project_validation_manifest.json", report)
    return {**report, "output_root": str(root)}
