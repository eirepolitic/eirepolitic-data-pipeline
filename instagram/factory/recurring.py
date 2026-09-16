from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT / "instagram" / "projects"


def load_project(project_id: str) -> dict[str, Any]:
    path = PROJECT_ROOT / project_id / "project.yml"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown Instagram project: {project_id}; expected {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if payload.get("project_id") != project_id:
        raise RuntimeError(f"Project ID mismatch in {path}: {payload.get('project_id')!r}")
    module = str(payload.get("adapter_module") or "").strip()
    if not module:
        raise RuntimeError(f"Project {project_id} has no adapter_module")
    publication = payload.get("publication") or {}
    if publication.get("enabled") is not False:
        raise RuntimeError(f"Project {project_id} must keep publication.enabled=false in generation configuration")
    return payload


def run_project(
    project_id: str,
    *,
    period: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    project = load_project(project_id)
    period_cfg = project.get("period") or {}
    resolved_period = period or str(period_cfg.get("default") or "last_completed_month")
    module = importlib.import_module(str(project["adapter_module"]))
    generate = getattr(module, "generate", None)
    if not callable(generate):
        raise RuntimeError(f"Adapter {project['adapter_module']} does not expose generate()")
    root = Path(output_root) if output_root else REPO_ROOT / str(project["output"]["local_root"])
    result = generate(project=project, period_spec=resolved_period, output_root=root)
    if result.get("publication_enabled") is not False:
        raise RuntimeError("Recurring generation adapter attempted to enable publication")
    if result.get("review_state") != "pending_human_review":
        raise RuntimeError(f"Unexpected review state: {result.get('review_state')!r}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a configured recurring Instagram post project")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--period", default=None, help="Project period value; e.g. YYYY-MM or last_completed_month")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    result = run_project(args.project_id, period=args.period, output_root=args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
