"""Generic render pipeline entrypoint (EirePolitic Director, Phase 3, §3.4)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from instagram.factory import recurring
from instagram.factory.contract import RenderResult
from instagram.factory.normalize import normalize_result
from instagram.factory.publication_handoff import write_publication_handoff
from instagram.factory.qa import qa_with_declarative_checks


def run_and_normalize(
    project_id: str,
    *,
    period: str | None = None,
    output_root: str | Path | None = None,
) -> RenderResult:
    project = recurring.load_project(project_id)
    root = Path(output_root) if output_root else recurring.REPO_ROOT / str(project["output"]["local_root"])
    raw = recurring.run_project(project_id, period=period, output_root=output_root)
    result = normalize_result(project, raw, output_root=root)
    result = qa_with_declarative_checks(project, result)
    if result.qa.all_passed:
        write_publication_handoff(result, artifact_root=root)
    return result


def _summarize(result: RenderResult) -> dict[str, Any]:
    return {
        "project_id": result.project_id,
        "period_key": result.period_key,
        "output_root": result.output_root,
        "slide_count": len(result.slides),
        "contact_sheet_count": len(result.contact_sheets),
        "review_state": result.review_state,
        "publication_enabled": result.publication_enabled,
        "source_batch_id": result.source_batch_id,
        "qa": {
            "expected_slide_count": result.qa.expected_slide_count,
            "actual_slide_count": result.qa.actual_slide_count,
            "all_passed": result.qa.all_passed,
            "checks": [
                {"name": check.name, "passed": check.passed, "detail": check.detail}
                for check in result.qa.checks
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a project through the generic render pipeline")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--period", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    result = run_and_normalize(args.project_id, period=args.period, output_root=args.output_root)
    print(json.dumps(_summarize(result), indent=2, sort_keys=True))
    if not result.qa.all_passed:
        failed_names = [check.name for check in result.qa.failed_checks()]
        raise SystemExit(f"Declarative QA failed for {result.project_id}: {failed_names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
