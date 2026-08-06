from __future__ import annotations

from pathlib import Path
from typing import Any

from instagram.visuals.renderers.common import write_json

from .validation_contact_sheet import build_validation_contact_sheet as _build_two_column_contact_sheet


def build_validation_contact_sheet(
    *,
    root: Path,
    project_id: str,
    scenario_manifests: dict[str, dict[str, Any]],
    scenario_order: list[str],
) -> dict[str, Any]:
    """Compatibility entry point for the shared two-column contact-sheet renderer."""
    manifest = _build_two_column_contact_sheet(
        root=root,
        project_id=project_id,
        scenario_manifests=scenario_manifests,
        scenario_order=scenario_order,
    )
    # Preserve the established contract identifier for existing consumers while
    # exposing the actual two-column geometry in the full/summary metadata.
    manifest["layout"] = "full_review_plus_deduplicated_summary_plus_complete_audit"
    write_json(root / "validation_contact_sheet_manifest.json", manifest)
    return manifest
