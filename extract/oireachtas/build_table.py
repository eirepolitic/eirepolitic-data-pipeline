"""CLI entry point for unified Oireachtas table builds.

F03 supports `_discovery` mode to probe Oireachtas API endpoint payload shapes
before real table builders are implemented.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .discovery import DISCOVERY_TABLE, discovery_dq, discovery_schema, run_endpoint_discovery
from .io_s3 import DEFAULT_BUCKET, DEFAULT_REGION, get_json, make_s3_client, put_json
from .normalize import utc_now_iso
from .review import REVIEW_ROOT, raw_review_url, write_review_bundle
from .schemas import DEFAULT_TABLES_CONFIG, get_table_schema, load_table_registry


VALID_MODES = ("discover", "test", "incremental", "full", "backfill")
SMOKE_TABLE = "_smoke"
REVIEW_BRANCH = "oireachtas-review-output"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m extract.oireachtas.build_table",
        description="Build or inspect unified Oireachtas tables.",
    )
    parser.add_argument("--table", help="Table registry key, _smoke, or _discovery.")
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="test",
        help="Build mode.",
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
        help="Write review sample files when supported.",
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
    parser.add_argument(
        "--s3-bucket",
        default=os.getenv("S3_BUCKET", DEFAULT_BUCKET),
        help="S3 bucket for smoke/review outputs.",
    )
    parser.add_argument(
        "--aws-region",
        default=os.getenv("AWS_REGION", DEFAULT_REGION),
        help="AWS region for S3 client.",
    )
    parser.add_argument(
        "--review-root",
        default=str(REVIEW_ROOT),
        help="Local review output root for workflow publishing.",
    )
    parser.add_argument(
        "--github-repository",
        default=os.getenv("GITHUB_REPOSITORY", "eirepolitic/eirepolitic-data-pipeline"),
        help="owner/repo for raw review URL generation inside GitHub Actions.",
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


def run_smoke(args: argparse.Namespace) -> int:
    """Run F02 S3/review smoke test."""
    if args.mode != "test":
        print("ERROR: _smoke only supports --mode test.", file=sys.stderr)
        return 2

    now = utc_now_iso()
    manifest_key = "processed/oireachtas_unified/review/_smoke/latest/manifest.json"
    sample_rows = [
        {
            "check_name": "s3_put_get",
            "status": "pending",
            "bucket": args.s3_bucket,
            "key": manifest_key,
            "created_at_utc": now,
        }
    ]

    manifest: dict[str, Any] = {
        "table": SMOKE_TABLE,
        "mode": args.mode,
        "status": "started",
        "created_at_utc": now,
        "bucket": args.s3_bucket,
        "manifest_key": manifest_key,
        "aws_region": args.aws_region,
        "review_branch": REVIEW_BRANCH,
    }

    s3 = make_s3_client(region_name=args.aws_region)
    put_json(s3, bucket=args.s3_bucket, key=manifest_key, payload=manifest)
    readback = get_json(s3, bucket=args.s3_bucket, key=manifest_key)
    if readback.get("table") != SMOKE_TABLE:
        raise RuntimeError("S3 readback verification failed: table mismatch")

    manifest["status"] = "success"
    manifest["verified_at_utc"] = utc_now_iso()
    put_json(s3, bucket=args.s3_bucket, key=manifest_key, payload=manifest)

    sample_rows[0]["status"] = "success"
    schema = {
        "table": SMOKE_TABLE,
        "primary_key": ["check_name"],
        "columns": list(sample_rows[0].keys()),
    }
    dq = {
        "table": SMOKE_TABLE,
        "dq_status": "pass",
        "checks": [
            {"check_name": "s3_put_get", "status": "pass", "message": "S3 PutObject/GetObject succeeded."},
            {"check_name": "review_bundle", "status": "pass", "message": "Local review bundle written."},
        ],
    }

    review_dir = write_review_bundle(
        table=SMOKE_TABLE,
        manifest=manifest,
        schema=schema,
        dq=dq,
        sample_rows=sample_rows,
        root=Path(args.review_root),
    )
    review_url = raw_review_url(
        repo=args.github_repository,
        branch=REVIEW_BRANCH,
        table=SMOKE_TABLE,
        filename="manifest.json",
    )

    print(f"TABLE={SMOKE_TABLE}")
    print(f"MODE={args.mode}")
    print("ROWS=1")
    print(f"COLUMNS={len(schema['columns'])}")
    print("PRIMARY_KEY=check_name")
    print("PRIMARY_KEY_UNIQUE=true")
    print(f"MANIFEST_KEY=s3://{args.s3_bucket}/{manifest_key}")
    print(f"REVIEW_LOCAL_DIR={review_dir}")
    print(f"REVIEW_SAMPLE_RAW_URL={review_url}")
    print("DQ_STATUS=pass")

    return 0


def run_discovery(args: argparse.Namespace) -> int:
    """Run F03 endpoint discovery and write review bundle."""
    if args.mode != "discover":
        print("ERROR: _discovery only supports --mode discover.", file=sys.stderr)
        return 2

    rows, manifest = run_endpoint_discovery(limit=max(1, min(args.limit, 10)))
    schema = discovery_schema(rows)
    dq = discovery_dq(rows)
    review_dir = write_review_bundle(
        table=DISCOVERY_TABLE,
        manifest=manifest,
        schema=schema,
        dq=dq,
        sample_rows=rows,
        root=Path(args.review_root),
    )
    review_url = raw_review_url(
        repo=args.github_repository,
        branch=REVIEW_BRANCH,
        table=DISCOVERY_TABLE,
        filename="manifest.json",
    )

    print(f"TABLE={DISCOVERY_TABLE}")
    print(f"MODE={args.mode}")
    print(f"ROWS={len(rows)}")
    print(f"COLUMNS={len(schema.get('columns', []))}")
    print("PRIMARY_KEY=endpoint_name")
    print("PRIMARY_KEY_UNIQUE=true")
    print(f"REVIEW_LOCAL_DIR={review_dir}")
    print(f"REVIEW_SAMPLE_RAW_URL={review_url}")
    print(f"DQ_STATUS={dq.get('dq_status')}")
    print(f"ENDPOINT_OK_COUNT={manifest.get('ok_count')}")
    print(f"ENDPOINT_FAILED_COUNT={manifest.get('failed_count')}")
    return 0


def validate_command(args: argparse.Namespace) -> int:
    if args.list_tables:
        return list_tables(args.config, as_json=args.json)

    if not args.table:
        print("ERROR: --table is required unless --list-tables is used.", file=sys.stderr)
        return 2

    if args.table == SMOKE_TABLE:
        return run_smoke(args)

    if args.table == DISCOVERY_TABLE:
        return run_discovery(args)

    schema = get_table_schema(args.table, Path(args.config))
    payload = {
        "status": "validated",
        "message": "F03 skeleton: real table execution is implemented in later packets.",
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
