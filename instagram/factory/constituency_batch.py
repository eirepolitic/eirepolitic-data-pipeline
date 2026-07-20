from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from instagram.renderer.template_renderer import render_template
from instagram.visuals.renderers.common import write_json

from .catalogues import REPO_ROOT, load_catalogues
from .constituency_pilot import (
    build_constituency_records,
    build_contact_sheet,
    load_source_rows,
    render_visual,
    write_cover_asset,
)
from .project import load_project, validate_project


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    from .constituency_pilot import normalize_text

    return normalize_text(value).replace(" ", "-")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_run_id(project_id: str, project_version: int, batch_id: str, git_sha: str) -> str:
    identity = f"{project_id}|{project_version}|{batch_id}|{git_sha}"
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"{project_id}-v{project_version}-{suffix}"


def _batch_id(source_manifest: dict[str, Any]) -> str:
    for source_name in ("members", "speeches"):
        source = source_manifest.get(source_name, {})
        batch_id = source.get("resolution", {}).get("batch_id")
        if batch_id:
            return str(batch_id)
    return "local-fixture"


def _bindings(slide: dict[str, Any], constituency: str, media_path: Path) -> dict[str, Any]:
    values = {
        key: str(value).replace("{{ constituency }}", constituency)
        for key, value in slide.get("text", {}).items()
    }
    values["main_media"] = str(media_path)
    return values


def _item_context(record: dict[str, Any]) -> dict[str, Any]:
    rows = [dict(row) for row in record["issue_rows"][:7]]
    return {
        **record,
        "display_constituency": record["constituency"],
        "result_constituency": record["constituency"],
        "result_constituency_key": record["constituency_key"],
        "result_issue_count": record["issue_count"],
        "result_speech_count": record["speech_count"],
        "issue_rows": rows,
        "issue_count": len(rows),
        "scenario": "batch_item",
        "synthetic": False,
        "no_publication": True,
    }


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

    members, speeches, source_manifest = load_source_rows(data_source)
    records, join_manifest = build_constituency_records(members, speeches)
    records = sorted(records, key=lambda row: row["constituency"])
    batch_id = _batch_id(source_manifest)
    resolved_git_sha = git_sha or os.getenv("GITHUB_SHA") or "local"
    run_id = stable_run_id(project["project_id"], int(project["version"]), batch_id, resolved_git_sha)

    root = Path(output_root or f"generated_factory_batches/{project['project_id']}/{run_id}")
    if not root.is_absolute():
        root = REPO_ROOT / root
    generated_root = root / "generated"
    generated_root.mkdir(parents=True, exist_ok=True)

    template_path = REPO_ROOT / "instagram/templates/layouts/title_text_media_v1.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    item_manifests: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    preview_paths: list[Path] = []
    preview_labels: list[str] = []

    for record in records:
        constituency = record["constituency"]
        item_slug = slugify(constituency)
        item_dir = generated_root / item_slug
        assets_dir = item_dir / "assets"
        slides_dir = item_dir / "slides"
        assets_dir.mkdir(parents=True, exist_ok=True)
        slides_dir.mkdir(parents=True, exist_ok=True)
        context = _item_context(record)
        slide_results: list[dict[str, Any]] = []
        item_errors: list[str] = []

        try:
            cover_asset = assets_dir / "constituency_cover.png"
            visual_asset = assets_dir / "issue_profile.png"
            write_cover_asset(cover_asset, context)
            visual_manifest = render_visual(
                visual_asset,
                item_dir / "metadata/issue_profile.visual.json",
                item_dir / "manifests/issue_profile.visual_manifest.json",
                context,
            )

            for slide in sorted(project["slides"], key=lambda item: item["order"]):
                media_path = cover_asset if slide["slide_id"] == "cover" else visual_asset
                output_path = slides_dir / f"{slide['order']:02d}_{slide['slide_id']}.png"
                result = render_template(template, _bindings(slide, constituency, media_path), output_path)
                if result.warnings:
                    raise ValueError(f"Render warnings for {constituency}/{slide['slide_id']}: {result.warnings}")
                slide_results.append(
                    {
                        "slide_id": slide["slide_id"],
                        "order": slide["order"],
                        "layout_id": slide["post_type_id"],
                        "path": str(output_path.relative_to(root)),
                        "sha256": file_sha256(output_path),
                        "width": Image.open(output_path).width,
                        "height": Image.open(output_path).height,
                        "warnings": [],
                    }
                )

            status = "succeeded"
            if len(preview_paths) < 12:
                preview_paths.append(slides_dir / "02_issue_profile.png")
                preview_labels.append(constituency)
        except Exception as exc:
            status = "failed"
            item_errors.append(str(exc))
            failures.append({"item_slug": item_slug, "constituency": constituency, "error": str(exc)})

        item_manifest = {
            "project_id": project["project_id"],
            "project_version": project["version"],
            "run_id": run_id,
            "item_key": constituency,
            "item_label": constituency,
            "item_slug": item_slug,
            "status": status,
            "review_state": "unreviewed",
            "no_publication": True,
            "source_batch_id": batch_id,
            "member_names": record.get("member_names", []),
            "member_count": record.get("member_count", 0),
            "issue_rows": context["issue_rows"],
            "result_speech_count": record["speech_count"],
            "slides": slide_results,
            "errors": item_errors,
            "generated_at": utc_now(),
        }
        if status == "succeeded":
            item_manifest["visual_manifest"] = visual_manifest
        write_json(item_dir / "item_manifest.json", item_manifest)
        item_manifests[item_slug] = item_manifest

    if preview_paths:
        build_contact_sheet(preview_paths, root / "review/batch_sample_issue_profiles.png", preview_labels)

    succeeded = sum(1 for item in item_manifests.values() if item["status"] == "succeeded")
    failed = len(item_manifests) - succeeded
    state = "succeeded" if failed == 0 else ("partially_failed" if succeeded else "failed")
    run_manifest = {
        "project_id": project["project_id"],
        "project_version": project["version"],
        "run_id": run_id,
        "state": state,
        "draft": True,
        "approved": False,
        "publishing_allowed": False,
        "review_state": "unreviewed",
        "created_at": utc_now(),
        "git_sha": resolved_git_sha,
        "workflow_run_id": workflow_run_id or os.getenv("GITHUB_RUN_ID"),
        "data_source": data_source,
        "source_batch_id": batch_id,
        "source_manifest": source_manifest,
        "join_manifest": join_manifest,
        "expected_item_count": len(records),
        "succeeded_item_count": succeeded,
        "failed_item_count": failed,
        "item_order": [slugify(record["constituency"]) for record in records],
        "items": {
            slug: {
                "status": manifest["status"],
                "label": manifest["item_label"],
                "manifest": f"generated/{slug}/item_manifest.json",
            }
            for slug, manifest in item_manifests.items()
        },
        "failures": failures,
        "review_sample": "review/batch_sample_issue_profiles.png" if preview_paths else None,
        "warnings": validation["warnings"],
    }
    write_json(root / "run_manifest.json", run_manifest)
    write_json(
        root / "review/review_state.json",
        {
            "project_id": project["project_id"],
            "run_id": run_id,
            "overall_status": "unreviewed",
            "publishing_allowed": False,
            "items": {
                slug: {
                    "status": "unreviewed" if item["status"] == "succeeded" else "generation_failed",
                    "slides": {slide["slide_id"]: "unreviewed" for slide in item["slides"]},
                }
                for slug, item in item_manifests.items()
            },
        },
    )
    return {**run_manifest, "output_root": str(root)}


def upload_batch_to_s3(output_root: str | Path, bucket: str, prefix: str) -> dict[str, Any]:
    import boto3

    root = Path(output_root)
    if not root.is_dir():
        raise ValueError(f"Batch output root does not exist: {root}")
    client = boto3.client("s3")
    uploaded: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        key = f"{prefix.rstrip('/')}/{path.relative_to(root).as_posix()}"
        client.upload_file(str(path), bucket, key)
        uploaded.append(key)
    return {"success": True, "bucket": bucket, "prefix": prefix, "uploaded_file_count": len(uploaded)}
