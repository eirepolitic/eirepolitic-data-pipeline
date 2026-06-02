"""CLI entry point for unified Oireachtas table builds.

F01 implementation intentionally provides the command surface and registry
inspection only. API/S3/table execution is added in later bounded packets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .schemas import DEFAULT_TABLES_CONFIG, get_table_schema, load_table_registry


VALID_MODES = ("discover", "test", "incremental", "full", "backfill")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m extract.oireachtas.build_table",
        description="Build or inspect unified Oireachtas tables.",
    )
    parser.add_argument("--table", help="Table registry key, e.g. silver_members.")
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="test",
        help="Build mode. F01 supports command validation only.",
    )
    parser.add_argument("--chamber", default="dail", help="Oireachtas chamber, e.g. dail or seanad.")
    parser.add_argument("--house-no", default="34", help="House number where applicable.")
    parser.add_argument("--date-start", help="Start date YYYY-MM-DD for fact tables.")
    parser.add_argument("--date-end", help="End date YYYY-MM-DD for fact tables.")
    parser.add_argument("--limit", type=int, default=25, help="API/test row limit.")
    parser.add_argument("--sample-rows", type=int, default=10, help="Rows to publish in review sample.")
    parser.add_argument(
        "--write-review-sample",
        action="store_true",
        help="Later packets write review samples when this is set.",
    )
    parser.add_argument(
        "--list-tables",
        action="store_true",
        help="List configured tables and exit.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_TABLES_CONFIG),
        help="Path to table registry YAML.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable output for list/validation commands.",
    )
    return parser


def list_tables(config_path: str, *, as_json: bool = False) -> int:
    registry = load_table_registry(Path(config_path))
    if as_json:
        payload = {
            name: {
                "layer": schema.layer,
                "status": schema.status,
                "cadence": schema.cadence,
                "primary_key": schema.primary_key,
                "endpoint": schema.endpoint,
            }
            for name, schema in sorted(registry.items())
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    for name, schema in sorted(registry.items()):
        endpoint = f" endpoint={schema.endpoint}" if schema.endpoint else ""
        print(
            f"{name} layer={schema.layer} status={schema.status} "
            f"cadence={schema.cadence} pk={schema.primary_key_display}{endpoint}"
        )
    return 0


def validate_command(args: argparse.Namespace) -> int:
    if args.list_tables:
        return list_tables(args.config, as_json=args.json)

    if not args.table:
        print("ERROR: --table is required unless --list-tables is used.", file=sys.stderr)
        return 2

    schema = get_table_schema(args.table, Path(args.config))
    payload = {
        "status": "validated",
        "message": "F01 skeleton only: table execution is implemented in later packets.",
        "table": schema.name,
        "mode": args.mode,
        "layer": schema.layer,
        "cadence": schema.cadence,
        "primary_key": schema.primary_key,
        "columns": schema.columns,
        "endpoint": schema.endpoint,
        "params": {
            "chamber": args.chamber,
            "house_no": args.house_no,
            "date_start": args.date_start,
            "date_end": args.date_end,
            "limit": args.limit,
            "sample_rows": args.sample_rows,
            "write_review_sample": bool(args.write_review_sample),
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(payload["message"])
        print(f"table={schema.name}")
        print(f"mode={args.mode}")
        print(f"primary_key={schema.primary_key_display}")
        print(f"columns={len(schema.columns)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return validate_command(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
