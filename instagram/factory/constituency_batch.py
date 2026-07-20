from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from instagram.renderer.template_renderer import render_template
from instagram.visuals.renderers.common import write_json

from .catalogues import REPO_ROOT, load_catalogues
from .constituency_pilot import (
    build_constituency_records,
    build_contact_sheet,
    load_source_rows,
    render_visual,
    utc_now,
    write_cover_asset,
)
from .project import load_project, validate_project


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "item"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_tokens(value: Any, constituency: str) -> Any:
    if isinstance(value, str):
        return value.replace("{{ constituency }}", constituency)
    return value


def _source_batch_id(source_manifest: dict[str, Any]) -> str:
    for source_name in ("members", "speeches"):
        source = source_manifest.get(source_name, {})
        batch_id = source.get("resolution", {}).get("batch_id")
        if batch_id:
            return str(batch_id)
    return "local-fixture"


def _run_id(project: dict[str, Any], batch_id: str, git_sha: str) -> str:
    seed = f"{project['project_id']}|{project['version']}|{batch_id}|{git_sha}"
    suffix = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    return f"v{project['version']}-{_slug(batch_id)}-{suffix}"


def generate_constituency_batch(
    project_path: str | Path,
    *,
    data_source: str = "s3",
    output_root: str | Path | None = None,
    git_sha: str | None = None,
    workflow_run_id: str | None = None,
) -> dict[str, Any]:
    project = load_project(project_path)
    validation = validate_project(project=project, catalogues=load_catalogues())
    if not validation["success"]:
        raise ValueError("Invalid project:\n" + "\n".join(validation["errors"]))
    if project.get("granularity", {}).get("grain") != "constituency":
        raise ValueError("This batch generator currently supports constituency projects only")

    members, speeches, source_manifest = load_source_rows(data_source)
    records, join_manifest = build_constituency_records(members, speeches)
    records = sorted(records, key=lambda row: row["constituency"])
    batch_id = _source_batch_id(source_manifest)
    resolved_git_sha = git_sha or os.getenv("GITHUB_SHA") or "local"
    run_id = _run_id(project, batch_id, resolved_git_sha)

    root = Path(output_root or "generated_factory_batches")
    if not root.is_absolute():
        root = REPO_ROOT / root
    run_root = root / project["project_id"] / "runs" / run_id
    generated_root = run_root / "generated"
    template = json.loads(
        (REPO_ROOT / "instagram/templates/layouts/title_text_media_v1.json").read_text(encoding="utf-8")
    )

    item_manifests: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    preview_paths: list[Path] = []
    preview_labels: list[str] = []

    for record in records:
        constituency = record["constituency"]
        item_slug = _slug(constituency)
        item_dir = generated_root / item_slug
        try:
            scenario = {
                **record,
                "display_constituency": constituency,
                "display_constituency_key": record["constituency_key"],
                "result_constituency": constituency,
                "result_constituency_key": record["constituency_key"],
                "result_issue_count": record["issue_count"],
                "result_speech_count": record["speech_count"],
                "issue_rows": [dict(row) for row in record["issue_rows"][:7]],
                "scenario": "batch_item",
                "synthetic": False,
                "no_publication": True,
            }
            assets_dir = item_dir / "assets"
            cover_asset = assets_dir / "constituency_cover.png"
            visual_asset = assets_dir / "issue_profile.png"
            write_cover_asset(cover_asset, scenario)
            visual_manifest = render_visual(
                visual_asset,
                item_dir / "metadata/issue_profile.visual.json",
                item_dir / "manifests/issue_profile.visual_manifest.json",
                scenario,
            )

            slide_manifests: list[dict[str, Any]] = []
            rendered: list[Path] = []
            for slide in sorted(project["slides"], key=lambda item: item["order"]):
                slide_id = slide["slide_id"]
                media_path = cover_asset if slide_id == "cover" else visual_asset
                bindings = {
                    key: _replace_tokens(value, constituency)
                    for key, value in slide.get("text", {}).items()
                }
                bindings["main_media"] = str(media_path)
                output_path = item_dir / f"slide-{slide['order']:02d}-{slide_id}.png"
                result = render_template(template, bindings, output_path)
                if result.warnings:
                    raise ValueError(f"Render warnings for {constituency}/{slide_id}: {result.warnings}")
                rendered.append(output_path)
                slide_manifests.append(
                    {
                        "slide_id": slide_id,
                        "order": slide["order"],
                        "post_type_id": slide["post_type_id"],
                        "visual_type_id": slide.get("visual", {}).get("visual_type_id"),
                        "resolved_text": bindings,
                        "output": output_path.name,
                        "sha256": _sha256(output_path),
                        "width": 1080,
                        "height": 1350,
                        "warnings": [],
                    }
                )

            item_manifest = {
                "project_id": project["project_id"],
                "project_version": project["version"],
                "run_id": run_id,
                "item_key": record["constituency_key"],
                "item_label": constituency,
                "item_slug": item_slug,
                "source_batch_id": batch_id,
                "source_summary": {
                    "member_names": record["member_names"],
                    "member_count": record["member_count"],
                    "issue_count": record["issue_count"],
                    "speech_count": record["speech_count"],
                },
                "slides": slide_manifests,
                "visual_manifest": visual_manifest,
                "review_state": "unreviewed",
                "approved": False,
                "no_publication": True,
                "generated_at": utc_now(),
            }
            write_json(item_dir / "item_manifest.json", item_manifest)
            item_manifests.append(item_manifest)
            if len(preview_paths) < 12:
                preview_paths.append(rendered[1])
                preview_labels.append(constituency)
        except Exception as exc:
            failures.append({"item_label": constituency, "item_slug": item_slug, "error": str(exc)})

    if preview_paths:
        build_contact_sheet(preview_paths, run_root / "review/contact_sheet.png", preview_labels)

    status = "succeeded"
    if failures and item_manifests:
        status = "partially_failed"
    elif failures:
        status = "failed"

    run_manifest = {
        "status": status,
        "project_id": project["project_id"],
        "project_version": project["version"],
        "run_id": run_id,
        "git_sha": resolved_git_sha,
        "workflow_run_id": workflow_run_id or os.getenv("GITHUB_RUN_ID"),
        "source_batch_id": batch_id,
        "data_source": data_source,
        "source_manifest": source_manifest,
        "join_manifest": join_manifest,
        "item_count_expected": len(records),
        "item_count_succeeded": len(item_manifests),
        "item_count_failed": len(failures),
        "items": [
            {
                "item_key": item["item_key"],
                "item_label": item["item_label"],
                "item_slug": item["item_slug"],
                "manifest": f"generated/{item['item_slug']}/item_manifest.json",
                "review_state": item["review_state"],
            }
            for item in item_manifests
        ],
        "failures": failures,
        "review_state": "unreviewed",
        "approved": False,
        "publishing_allowed": False,
        "created_at": utc_now(),
    }
    write_json(run_root / "run_manifest.json", run_manifest)
    write_json(
        run_root / "review/review_state.json",
        {
            "project_id": project["project_id"],
            "run_id": run_id,
            "items": {
                item["item_slug"]: {
                    "status": "unreviewed",
                    "slides": {slide["slide_id"]: "unreviewed" for slide in item["slides"]},
                }
                for item in item_manifests
            },
        },
    )
    return run_manifest
