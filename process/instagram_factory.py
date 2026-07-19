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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and inspect Instagram Content Factory configuration.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_catalogues_parser = subparsers.add_parser(
        "validate-catalogues",
        help="Validate post, visual, and metric catalogues.",
    )
    validate_catalogues_parser.set_defaults(handler=_validate_catalogues)

    list_parser = subparsers.add_parser("list-options", help="List catalogue menu options.")
    list_parser.add_argument(
        "--catalogue",
        choices=["all", "post_types", "visual_types", "metrics"],
        default="all",
    )
    list_parser.set_defaults(handler=_list_options)

    project_parser = subparsers.add_parser("validate-project", help="Validate one project.yml specification.")
    project_parser.add_argument("--project", required=True, help="Repository-relative or absolute project.yml path.")
    project_parser.set_defaults(handler=_validate_project)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
