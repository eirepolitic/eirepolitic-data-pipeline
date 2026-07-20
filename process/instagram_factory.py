from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from instagram.factory.catalogues import CatalogueValidationError, list_options, load_catalogues, validate_catalogues
from instagram.factory.project import validate_project


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def _validate_catalogues(_: argparse.Namespace) -> int:
    report = validate_catalogues()
    _print(report)
    return 0 if report["success"] else 1


def _list_options(args: argparse.Namespace) -> int:
    try:
        catalogues = load_catalogues()
        names = None if args.catalogue == "all" else [args.catalogue]
        _print(list_options(catalogues, names))
        return 0
    except CatalogueValidationError as exc:
        _print({"success": False, "errors": str(exc).splitlines()})
        return 1


def _validate_project(args: argparse.Namespace) -> int:
    report = validate_project(project_path=args.project)
    _print(report)
    return 0 if report["success"] else 1


def _render_tests(args: argparse.Namespace) -> int:
    try:
        from instagram.factory.constituency_pilot import render_project_tests

        report = render_project_tests(args.project, data_source=args.data_source, output_root=args.output_root)
        _print(report)
        return 0
    except Exception as exc:
        _print({"success": False, "errors": [str(exc)]})
        return 1


def _generate_batch(args: argparse.Namespace) -> int:
    try:
        from instagram.factory.constituency_batch import generate_constituency_batch, upload_batch_to_s3

        report = generate_constituency_batch(
            args.project,
            data_source=args.data_source,
            output_root=args.output_root,
            git_sha=args.git_sha,
            workflow_run_id=args.workflow_run_id,
        )
        if args.s3_bucket:
            s3_prefix = f"{args.s3_prefix.rstrip('/')}/runs/{report['run_id']}"
            report["s3_upload"] = upload_batch_to_s3(report["output_root"], args.s3_bucket, s3_prefix)
            report["s3_output_prefix"] = f"s3://{args.s3_bucket}/{s3_prefix}"
        _print(report)
        return 0 if report["state"] in {"succeeded", "succeeded_with_warnings"} else 1
    except Exception as exc:
        _print({"success": False, "errors": [str(exc)]})
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and run Instagram Content Factory projects.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_catalogues_parser = subparsers.add_parser("validate-catalogues", help="Validate catalogues.")
    validate_catalogues_parser.set_defaults(handler=_validate_catalogues)

    list_parser = subparsers.add_parser("list-options", help="List catalogue menu options.")
    list_parser.add_argument("--catalogue", choices=["all", "post_types", "visual_types", "metrics"], default="all")
    list_parser.set_defaults(handler=_list_options)

    project_parser = subparsers.add_parser("validate-project", help="Validate one project.yml specification.")
    project_parser.add_argument("--project", required=True)
    project_parser.set_defaults(handler=_validate_project)

    render_parser = subparsers.add_parser("render-tests", help="Render complete-slide validation scenarios.")
    render_parser.add_argument("--project", required=True)
    render_parser.add_argument("--data-source", choices=["local", "s3"], default="local")
    render_parser.add_argument("--output-root")
    render_parser.set_defaults(handler=_render_tests)

    batch_parser = subparsers.add_parser("generate-batch", help="Generate a draft complete-slide batch by project grain.")
    batch_parser.add_argument("--project", required=True)
    batch_parser.add_argument("--data-source", choices=["local", "s3"], default="s3")
    batch_parser.add_argument("--output-root")
    batch_parser.add_argument("--git-sha")
    batch_parser.add_argument("--workflow-run-id")
    batch_parser.add_argument("--s3-bucket")
    batch_parser.add_argument("--s3-prefix", default="processed/instagram_factory/projects/constituency_issue_profile_v1")
    batch_parser.set_defaults(handler=_generate_batch)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
