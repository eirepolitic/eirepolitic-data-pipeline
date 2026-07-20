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


def _guard(handler):
    def wrapped(args: argparse.Namespace) -> int:
        try:
            return handler(args)
        except Exception as exc:
            _print({"success": False, "errors": [str(exc)]})
            return 1
    return wrapped


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


@_guard
def _render_tests(args: argparse.Namespace) -> int:
    from instagram.factory.constituency_pilot import render_project_tests
    report = render_project_tests(args.project, data_source=args.data_source, output_root=args.output_root)
    _print(report)
    return 0


@_guard
def _generate_batch(args: argparse.Namespace) -> int:
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


@_guard
def _mark_review(args: argparse.Namespace) -> int:
    from instagram.factory.review import mark_review
    report = mark_review(
        args.run_root,
        item_slug=args.item,
        status=args.status,
        slide_id=args.slide,
        note=args.note,
        reviewer=args.reviewer,
    )
    _print(report)
    return 0


@_guard
def _regenerate(args: argparse.Namespace) -> int:
    from instagram.factory.regeneration import regenerate_selected
    report = regenerate_selected(
        args.project,
        args.source_run_root,
        args.destination_run_root,
        new_run_id=args.new_run_id,
        item_slugs=args.items,
        slide_ids=args.slides,
        reason=args.reason,
        data_source=args.data_source,
    )
    _print(report)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and run Instagram Content Factory projects.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("validate-catalogues")
    p.set_defaults(handler=_validate_catalogues)

    p = subparsers.add_parser("list-options")
    p.add_argument("--catalogue", choices=["all", "post_types", "visual_types", "metrics"], default="all")
    p.set_defaults(handler=_list_options)

    p = subparsers.add_parser("validate-project")
    p.add_argument("--project", required=True)
    p.set_defaults(handler=_validate_project)

    p = subparsers.add_parser("render-tests")
    p.add_argument("--project", required=True)
    p.add_argument("--data-source", choices=["local", "s3"], default="local")
    p.add_argument("--output-root")
    p.set_defaults(handler=_render_tests)

    p = subparsers.add_parser("generate-batch")
    p.add_argument("--project", required=True)
    p.add_argument("--data-source", choices=["local", "s3"], default="s3")
    p.add_argument("--output-root")
    p.add_argument("--git-sha")
    p.add_argument("--workflow-run-id")
    p.add_argument("--s3-bucket")
    p.add_argument("--s3-prefix", default="processed/instagram_factory/projects/constituency_issue_profile_v1")
    p.set_defaults(handler=_generate_batch)

    p = subparsers.add_parser("mark-review")
    p.add_argument("--run-root", required=True)
    p.add_argument("--item", required=True)
    p.add_argument("--slide")
    p.add_argument("--status", required=True, choices=["unreviewed", "approved", "changes_requested", "rejected"])
    p.add_argument("--note")
    p.add_argument("--reviewer", default="human")
    p.set_defaults(handler=_mark_review)

    p = subparsers.add_parser("regenerate")
    p.add_argument("--project", required=True)
    p.add_argument("--source-run-root", required=True)
    p.add_argument("--destination-run-root", required=True)
    p.add_argument("--new-run-id", required=True)
    p.add_argument("--items", nargs="+", required=True)
    p.add_argument("--slides", nargs="+")
    p.add_argument("--reason", required=True)
    p.add_argument("--data-source", choices=["local", "s3"], default="s3")
    p.set_defaults(handler=_regenerate)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
