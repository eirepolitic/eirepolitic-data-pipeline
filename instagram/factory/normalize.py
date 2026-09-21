"""Per-project `normalize_result()` shims (EirePolitic Director, Phase 3, §3.2).

Each shim translates one adapter's raw `generate()` return value into the
shared `contract.RenderResult` shape. Neither adapter is touched: this module
reads the raw dict (and, for `party_issue_monthly_profile_v2`, the
`run_manifest.json` file that adapter already writes to disk — the raw
top-level return value doesn't carry the per-slide path list, only the
manifest file does) and produces a `RenderResult` on the way out.

Deliberately NOT imported by, or imported from, `instagram/factory/recurring.py`
or either adapter — those files are frozen and byte-identity-checked against
commit 386b933 by `.github/workflows/director_factory_v1_identity_ci.yml`
(plan §1.8). This module is called by `instagram/factory/render_pipeline.py`
*after* `recurring.run_project()` has already produced and validated the raw
result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from instagram.factory.contract import ContactSheetRef, QASummary, RenderResult, SlideRef


def _normalize_party_issue_monthly_profile_v2(
    project: dict[str, Any], raw: dict[str, Any], *, output_root: Path
) -> RenderResult:
    render_cfg = project.get("render") or {}
    width = int(render_cfg.get("width", 1080))
    height = int(render_cfg.get("height", 1350))

    period_root = Path(raw["output_root"])
    run_manifest_path = period_root / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))

    slides: list[SlideRef] = []
    for party_manifest in run_manifest.get("parties", []):
        party_key = str(party_manifest["party_key"])
        for relative_slide_path in party_manifest.get("slides", []):
            slide_id = f"{party_key}/{Path(str(relative_slide_path)).stem}"
            slides.append(
                SlideRef(
                    id=slide_id,
                    path=str(period_root / str(relative_slide_path)),
                    width=width,
                    height=height,
                )
            )

    contact_sheets = tuple(
        ContactSheetRef(id=str(key), path=str(path))
        for key, path in (raw.get("contact_sheets") or {}).items()
    )

    declared_qa = project.get("qa") or {}
    expected_slide_count = int(declared_qa.get("expected_slide_count", raw.get("slide_count", len(slides))))
    package = None
    if raw.get("zip_path"):
        package = {"zip_path": raw["zip_path"], "sha256": raw.get("zip_sha256")}

    return RenderResult(
        project_id=str(raw["project_id"]),
        period_key=str(raw["period"]),
        output_root=str(period_root),
        slides=tuple(slides),
        contact_sheets=contact_sheets,
        # This project has no single caption.txt today (unlike the polling
        # project) — nothing to point at, so this is honestly None rather
        # than a guessed path.
        caption_path=None,
        manifest_path=str(run_manifest_path),
        package=package,
        qa=QASummary(
            expected_slide_count=expected_slide_count,
            actual_slide_count=int(raw.get("slide_count", len(slides))),
            dimensions=(width, height),
        ),
        review_state=str(raw["review_state"]),
        publication_enabled=raw["publication_enabled"],
        source_batch_id=raw.get("source_batch_id"),
        raw={**raw, "run_manifest": run_manifest},
    )


def _normalize_ipi_polling_factory_v1(
    project: dict[str, Any], raw: dict[str, Any], *, output_root: Path
) -> RenderResult:
    qa_block = raw.get("qa") or {}
    dims = qa_block.get("dimensions")
    width = int(dims[0]) if dims else None
    height = int(dims[1]) if dims else None

    slide_paths = [str(path) for path in (raw.get("slides") or [])]
    definition_ids = [str(item["id"]) for item in (project.get("slides") or {}).get("definitions", [])]
    slides: list[SlideRef] = []
    for index, path in enumerate(slide_paths):
        slide_id = definition_ids[index] if index < len(definition_ids) else Path(path).stem
        slides.append(SlideRef(id=slide_id, path=path, width=width, height=height))

    contact_sheet_path = raw.get("contact_sheet")
    contact_sheets = (
        (ContactSheetRef(id="four_slide_overview", path=str(contact_sheet_path)),)
        if contact_sheet_path
        else ()
    )

    # metadata/manifest.json and the period root are siblings of the slides/
    # directory this adapter writes every rendered slide into; there's no
    # single field in the raw return carrying the period root directly
    # (unlike party_issue_monthly_profile_v2's "output_root"), so it's
    # derived from a rendered slide's own path.
    if slide_paths:
        period_root = Path(slide_paths[0]).parent.parent
        candidate_manifest = period_root / "metadata" / "manifest.json"
        manifest_path = str(candidate_manifest) if candidate_manifest.exists() else None
        output_root_str = str(period_root)
    else:
        manifest_path = None
        output_root_str = str(output_root)

    expected_slide_count = int(qa_block.get("expected_slide_count", len(slide_paths)))

    return RenderResult(
        project_id=str(raw.get("project_id", project.get("project_id"))),
        period_key=str((raw.get("latest_poll") or {}).get("publication_date") or "unknown"),
        output_root=output_root_str,
        slides=tuple(slides),
        contact_sheets=contact_sheets,
        caption_path=raw.get("caption"),
        manifest_path=manifest_path,
        package=raw.get("package"),
        qa=QASummary(
            expected_slide_count=expected_slide_count,
            actual_slide_count=int(qa_block.get("actual_slide_count", len(slide_paths))),
            dimensions=(width, height) if width and height else None,
        ),
        review_state=str(raw["review_state"]),
        publication_enabled=raw["publication_enabled"],
        # This project has no immutable-batch concept (plan §1.4/§1.7) —
        # its lineage is source_uri/source_id in raw, not a batch id.
        source_batch_id=None,
        raw=raw,
    )


def _normalize_pq_monthly_overview_v1(
    project: dict[str, Any], raw: dict[str, Any], *, output_root: Path
) -> RenderResult:
    """pq_monthly_overview_v1 (prototype, single-slide build — director
    session 2026-09-21-monthly-questions-overview). The adapter returns a
    flat `slides` list of PNG paths directly (unlike the party project's
    per-party manifest structure), and its own `run_manifest.json` (written
    to `output_root`) carries the full readability/QA/dedupe detail for the
    session log, but nothing here needs to re-open it — the raw dict already
    has everything the contract needs.
    """
    render_cfg = project.get("render") or {}
    width = int(render_cfg.get("width", 1080))
    height = int(render_cfg.get("height", 1350))

    period_root = Path(raw["output_root"])
    slides = [
        SlideRef(id=Path(str(slide_path)).stem, path=str(slide_path), width=width, height=height)
        for slide_path in (raw.get("slides") or [])
    ]

    declared_qa = project.get("qa") or {}
    expected_slide_count = int(declared_qa.get("expected_slide_count", raw.get("slide_count", len(slides))))

    return RenderResult(
        project_id=str(raw["project_id"]),
        period_key=str(raw["period"]),
        output_root=str(period_root),
        slides=tuple(slides),
        contact_sheets=(),
        caption_path=None,
        manifest_path=raw.get("manifest_path"),
        package=None,
        qa=QASummary(
            expected_slide_count=expected_slide_count,
            actual_slide_count=int(raw.get("slide_count", len(slides))),
            dimensions=(width, height),
        ),
        review_state=str(raw["review_state"]),
        publication_enabled=raw["publication_enabled"],
        source_batch_id=raw.get("source_batch_id"),
        raw=raw,
    )


NORMALIZERS: dict[str, Callable[..., RenderResult]] = {
    "party_issue_monthly_profile_v2": _normalize_party_issue_monthly_profile_v2,
    "ipi_polling_factory_v1": _normalize_ipi_polling_factory_v1,
    "pq_monthly_overview_v1": _normalize_pq_monthly_overview_v1,
}


def normalize_result(project: dict[str, Any], raw: dict[str, Any], *, output_root: Path) -> RenderResult:
    """Dispatch to the registered shim for `project["project_id"]`.

    Raises RuntimeError (not a silent pass-through) for an unregistered
    project, so a new project added to instagram/projects/ without a shim
    fails loudly here rather than producing a malformed RenderResult later.
    """
    project_id = str(project.get("project_id") or raw.get("project_id") or "")
    shim = NORMALIZERS.get(project_id)
    if shim is None:
        raise RuntimeError(
            f"No normalize_result() shim registered for project_id={project_id!r} in "
            "instagram/factory/normalize.py. Add one and register it in NORMALIZERS "
            "before this project can use the generic render pipeline/workflow."
        )
    return shim(project, raw, output_root=output_root)
