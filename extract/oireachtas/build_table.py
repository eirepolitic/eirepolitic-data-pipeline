"""CLI entry point for unified Oireachtas table builds."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .client import OireachtasClient
from .discovery import DISCOVERY_TABLE, discovery_dq, discovery_schema, run_endpoint_discovery
from .io_s3 import DEFAULT_BUCKET, DEFAULT_REGION, get_json, make_s3_client, put_json
from .normalize import utc_now_iso
from .review import REVIEW_ROOT, raw_review_url, write_review_bundle
from .schemas import DEFAULT_TABLES_CONFIG, get_table_schema, load_table_registry
from .table_bill_debates import TABLE_NAME as BILL_DEBATES_TABLE, build_silver_bill_debates
from .table_bill_events import TABLE_NAME as BILL_EVENTS_TABLE, build_silver_bill_events
from .table_bill_related_docs import TABLE_NAME as BILL_RELATED_DOCS_TABLE, build_silver_bill_related_docs
from .table_bill_sponsors import TABLE_NAME as BILL_SPONSORS_TABLE, build_silver_bill_sponsors
from .table_bill_stages import TABLE_NAME as BILL_STAGES_TABLE, build_silver_bill_stages
from .table_bill_versions import TABLE_NAME as BILL_VERSIONS_TABLE, build_silver_bill_versions
from .table_bills import TABLE_NAME as BILLS_TABLE, build_silver_bills
from .table_constituencies import TABLE_NAME as CONSTITUENCIES_TABLE, build_silver_constituencies
from .table_control_data_quality_results import TABLE_NAME as CONTROL_DATA_QUALITY_RESULTS_TABLE, build_control_data_quality_results
from .table_control_pipeline_runs import TABLE_NAME as CONTROL_PIPELINE_RUNS_TABLE, build_control_pipeline_runs
from .table_control_table_manifests import TABLE_NAME as CONTROL_TABLE_MANIFESTS_TABLE, build_control_table_manifests
from .table_debate_records import TABLE_NAME as DEBATE_RECORDS_TABLE, build_silver_debate_records
from .table_debate_sections import TABLE_NAME as DEBATE_SECTIONS_TABLE, build_silver_debate_sections
from .table_division_tallies import TABLE_NAME as DIVISION_TALLIES_TABLE, build_silver_division_tallies
from .table_divisions import TABLE_NAME as DIVISIONS_TABLE, build_silver_divisions
from .table_gold_constituency_activity_yearly import TABLE_NAME as GOLD_CONSTITUENCY_ACTIVITY_YEARLY_TABLE, build_gold_constituency_activity_yearly
from .table_gold_content_fact_pool import TABLE_NAME as GOLD_CONTENT_FACT_POOL_TABLE, build_gold_content_fact_pool
from .table_gold_current_members import TABLE_NAME as GOLD_CURRENT_MEMBERS_TABLE, build_gold_current_members
from .table_gold_member_activity_monthly import TABLE_NAME as GOLD_MEMBER_ACTIVITY_MONTHLY_TABLE, build_gold_member_activity_monthly
from .table_gold_member_activity_yearly import TABLE_NAME as GOLD_MEMBER_ACTIVITY_YEARLY_TABLE, build_gold_member_activity_yearly
from .table_houses import TABLE_NAME as HOUSES_TABLE, build_silver_houses
from .table_member_constituencies import TABLE_NAME as MEMBER_CONSTITUENCIES_TABLE, build_silver_member_constituencies
from .table_member_memberships import TABLE_NAME as MEMBER_MEMBERSHIPS_TABLE, build_silver_member_memberships
from .table_member_offices import TABLE_NAME as MEMBER_OFFICES_TABLE, build_silver_member_offices
from .table_member_parties import TABLE_NAME as MEMBER_PARTIES_TABLE, build_silver_member_parties
from .table_member_votes import TABLE_NAME as MEMBER_VOTES_TABLE, build_silver_member_votes
from .table_members import TABLE_NAME as MEMBERS_TABLE, build_silver_members
from .table_parties import TABLE_NAME as PARTIES_TABLE, build_silver_parties
from .table_questions import TABLE_NAME as QUESTIONS_TABLE, build_silver_questions
from .table_source_files import TABLE_NAME as SOURCE_FILES_TABLE, build_silver_source_files
from .table_speeches import TABLE_NAME as SPEECHES_TABLE, build_silver_speeches

VALID_MODES = ("discover", "test", "incremental", "full", "backfill")
SMOKE_TABLE = "_smoke"
REVIEW_BRANCH = "oireachtas-review-output"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m extract.oireachtas.build_table", description="Build or inspect unified Oireachtas tables.")
    parser.add_argument("--table", help="Table registry key, _smoke, or _discovery.")
    parser.add_argument("--mode", choices=VALID_MODES, default="test", help="Build mode.")
    parser.add_argument("--chamber", default="dail", help="Oireachtas chamber.")
    parser.add_argument("--house-no", default="34", help="House number where applicable.")
    parser.add_argument("--date-start", help="Start date YYYY-MM-DD for fact tables.")
    parser.add_argument("--date-end", help="End date YYYY-MM-DD for fact tables.")
    parser.add_argument("--limit", type=int, default=25, help="API/test row limit.")
    parser.add_argument("--sample-rows", type=int, default=10, help="Rows to publish in review sample.")
    parser.add_argument("--write-review-sample", action="store_true", help="Write review sample files when supported.")
    parser.add_argument("--publish-latest", choices=("auto", "true", "false"), default="auto", help="Control writes to processed/oireachtas_unified/latest/*. auto disables latest for mode=test and enables it otherwise.")
    parser.add_argument("--list-tables", action="store_true", help="List configured tables and exit.")
    parser.add_argument("--config", default=str(DEFAULT_TABLES_CONFIG), help="Path to table registry YAML.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    parser.add_argument("--s3-bucket", default=os.getenv("S3_BUCKET", DEFAULT_BUCKET), help="S3 bucket for outputs.")
    parser.add_argument("--aws-region", default=os.getenv("AWS_REGION", DEFAULT_REGION), help="AWS region for S3 client.")
    parser.add_argument("--review-root", default=str(REVIEW_ROOT), help="Local review output root.")
    parser.add_argument("--github-repository", default=os.getenv("GITHUB_REPOSITORY", "eirepolitic/eirepolitic-data-pipeline"), help="owner/repo for raw review URL generation.")
    return parser


def _publish_latest_enabled(args: argparse.Namespace) -> bool:
    if args.publish_latest == "true":
        return True
    if args.publish_latest == "false":
        return False
    return args.mode != "test"


def _set_latest_env(args: argparse.Namespace) -> bool:
    enabled = _publish_latest_enabled(args)
    os.environ["OIREACHTAS_PUBLISH_LATEST"] = "true" if enabled else "false"
    return enabled


def list_tables(config_path: str, *, as_json: bool = False) -> int:
    registry = load_table_registry(Path(config_path))
    if as_json:
        payload = {name: {"layer": s.layer, "status": s.status, "cadence": s.cadence, "primary_key": s.primary_key, "endpoint": s.endpoint} for name, s in sorted(registry.items())}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    for name, schema in sorted(registry.items()):
        endpoint = f" endpoint={schema.endpoint}" if schema.endpoint else ""
        print(f"{name} layer={schema.layer} status={schema.status} cadence={schema.cadence} pk={schema.primary_key_display}{endpoint}")
    return 0


def run_smoke(args: argparse.Namespace) -> int:
    if args.mode != "test":
        print("ERROR: _smoke only supports --mode test.", file=sys.stderr)
        return 2
    now = utc_now_iso()
    manifest_key = "processed/oireachtas_unified/review/_smoke/latest/manifest.json"
    manifest: dict[str, Any] = {"table": SMOKE_TABLE, "mode": args.mode, "status": "started", "created_at_utc": now, "bucket": args.s3_bucket, "manifest_key": manifest_key, "aws_region": args.aws_region, "review_branch": REVIEW_BRANCH}
    s3 = make_s3_client(region_name=args.aws_region)
    put_json(s3, bucket=args.s3_bucket, key=manifest_key, payload=manifest)
    readback = get_json(s3, bucket=args.s3_bucket, key=manifest_key)
    if readback.get("table") != SMOKE_TABLE:
        raise RuntimeError("S3 readback verification failed: table mismatch")
    manifest["status"] = "success"
    manifest["verified_at_utc"] = utc_now_iso()
    put_json(s3, bucket=args.s3_bucket, key=manifest_key, payload=manifest)
    rows = [{"check_name": "s3_put_get", "status": "success", "bucket": args.s3_bucket, "key": manifest_key, "created_at_utc": now}]
    schema = {"table": SMOKE_TABLE, "primary_key": ["check_name"], "columns": list(rows[0].keys())}
    dq = {"table": SMOKE_TABLE, "dq_status": "pass", "checks": [{"check_name": "s3_put_get", "status": "pass"}, {"check_name": "review_bundle", "status": "pass"}]}
    review_dir = write_review_bundle(table=SMOKE_TABLE, manifest=manifest, schema=schema, dq=dq, sample_rows=rows, root=Path(args.review_root))
    print(f"TABLE={SMOKE_TABLE}\nMODE={args.mode}\nROWS=1\nCOLUMNS={len(schema['columns'])}\nPRIMARY_KEY=check_name\nPRIMARY_KEY_UNIQUE=true")
    print(f"MANIFEST_KEY=s3://{args.s3_bucket}/{manifest_key}\nREVIEW_LOCAL_DIR={review_dir}\nREVIEW_SAMPLE_RAW_URL={raw_review_url(repo=args.github_repository, branch=REVIEW_BRANCH, table=SMOKE_TABLE, filename='manifest.json')}\nDQ_STATUS=pass")
    return 0


def run_discovery(args: argparse.Namespace) -> int:
    if args.mode != "discover":
        print("ERROR: _discovery only supports --mode discover.", file=sys.stderr)
        return 2
    rows, manifest = run_endpoint_discovery(limit=max(1, min(args.limit, 10)))
    schema = discovery_schema(rows)
    dq = discovery_dq(rows)
    review_dir = write_review_bundle(table=DISCOVERY_TABLE, manifest=manifest, schema=schema, dq=dq, sample_rows=rows, root=Path(args.review_root))
    print(f"TABLE={DISCOVERY_TABLE}\nMODE={args.mode}\nROWS={len(rows)}\nCOLUMNS={len(schema.get('columns', []))}\nPRIMARY_KEY=endpoint_name\nPRIMARY_KEY_UNIQUE=true")
    print(f"REVIEW_LOCAL_DIR={review_dir}\nREVIEW_SAMPLE_RAW_URL={raw_review_url(repo=args.github_repository, branch=REVIEW_BRANCH, table=DISCOVERY_TABLE, filename='manifest.json')}\nDQ_STATUS={dq.get('dq_status')}\nENDPOINT_OK_COUNT={manifest.get('ok_count')}\nENDPOINT_FAILED_COUNT={manifest.get('failed_count')}")
    return 0


def run_real_table(args: argparse.Namespace) -> int:
    if args.mode not in {"test", "full", "incremental", "backfill"}:
        print(f"ERROR: {args.table} does not support --mode {args.mode}.", file=sys.stderr)
        return 2
    publish_latest = _set_latest_env(args)
    schema = get_table_schema(args.table, Path(args.config))
    s3 = make_s3_client(region_name=args.aws_region)
    client = OireachtasClient(timeout_seconds=30, retries=5, backoff_seconds=2.0, sleep_seconds=0.2)
    common = {"client": client, "s3": s3, "bucket": args.s3_bucket, "schema": schema, "limit": args.limit, "mode": args.mode}
    filtered = {**common, "chamber": args.chamber, "house_no": args.house_no}

    if args.table == HOUSES_TABLE:
        result = build_silver_houses(**common)
    elif args.table == CONSTITUENCIES_TABLE:
        result = build_silver_constituencies(**filtered)
    elif args.table == PARTIES_TABLE:
        result = build_silver_parties(**filtered)
    elif args.table == MEMBERS_TABLE:
        result = build_silver_members(**filtered)
    elif args.table == MEMBER_MEMBERSHIPS_TABLE:
        result = build_silver_member_memberships(**filtered)
    elif args.table == MEMBER_PARTIES_TABLE:
        result = build_silver_member_parties(**filtered)
    elif args.table == MEMBER_CONSTITUENCIES_TABLE:
        result = build_silver_member_constituencies(**filtered)
    elif args.table == MEMBER_OFFICES_TABLE:
        result = build_silver_member_offices(**filtered)
    elif args.table == SOURCE_FILES_TABLE:
        result = build_silver_source_files(**filtered, date_start=args.date_start, date_end=args.date_end)
    elif args.table == DEBATE_RECORDS_TABLE:
        result = build_silver_debate_records(**filtered, date_start=args.date_start, date_end=args.date_end)
    elif args.table == DEBATE_SECTIONS_TABLE:
        result = build_silver_debate_sections(**filtered, date_start=args.date_start, date_end=args.date_end)
    elif args.table == SPEECHES_TABLE:
        result = build_silver_speeches(**filtered, date_start=args.date_start, date_end=args.date_end)
    elif args.table == DIVISIONS_TABLE:
        result = build_silver_divisions(**filtered, date_start=args.date_start, date_end=args.date_end)
    elif args.table == DIVISION_TALLIES_TABLE:
        result = build_silver_division_tallies(**filtered, date_start=args.date_start, date_end=args.date_end)
    elif args.table == MEMBER_VOTES_TABLE:
        result = build_silver_member_votes(**filtered, date_start=args.date_start, date_end=args.date_end)
    elif args.table == QUESTIONS_TABLE:
        result = build_silver_questions(**filtered, date_start=args.date_start, date_end=args.date_end)
    elif args.table == BILLS_TABLE:
        result = build_silver_bills(**filtered, date_start=args.date_start, date_end=args.date_end)
    elif args.table == BILL_VERSIONS_TABLE:
        result = build_silver_bill_versions(**filtered, date_start=args.date_start, date_end=args.date_end)
    elif args.table == BILL_STAGES_TABLE:
        result = build_silver_bill_stages(**filtered, date_start=args.date_start, date_end=args.date_end)
    elif args.table == BILL_RELATED_DOCS_TABLE:
        result = build_silver_bill_related_docs(**filtered, date_start=args.date_start, date_end=args.date_end)
    elif args.table == BILL_SPONSORS_TABLE:
        result = build_silver_bill_sponsors(**filtered, date_start=args.date_start, date_end=args.date_end)
    elif args.table == BILL_DEBATES_TABLE:
        result = build_silver_bill_debates(**filtered, date_start=args.date_start, date_end=args.date_end)
    elif args.table == BILL_EVENTS_TABLE:
        result = build_silver_bill_events(**filtered, date_start=args.date_start, date_end=args.date_end)
    elif args.table == GOLD_CURRENT_MEMBERS_TABLE:
        result = build_gold_current_members(**filtered)
    elif args.table == GOLD_MEMBER_ACTIVITY_YEARLY_TABLE:
        result = build_gold_member_activity_yearly(**filtered)
    elif args.table == GOLD_MEMBER_ACTIVITY_MONTHLY_TABLE:
        result = build_gold_member_activity_monthly(**filtered)
    elif args.table == GOLD_CONSTITUENCY_ACTIVITY_YEARLY_TABLE:
        result = build_gold_constituency_activity_yearly(**filtered)
    elif args.table == GOLD_CONTENT_FACT_POOL_TABLE:
        result = build_gold_content_fact_pool(**filtered)
    elif args.table == CONTROL_PIPELINE_RUNS_TABLE:
        result = build_control_pipeline_runs(**filtered)
    elif args.table == CONTROL_TABLE_MANIFESTS_TABLE:
        result = build_control_table_manifests(**filtered)
    elif args.table == CONTROL_DATA_QUALITY_RESULTS_TABLE:
        result = build_control_data_quality_results(**filtered)
    else:
        payload = {"status": "validated", "message": "Real table execution is not implemented for this table yet.", "table": schema.name, "mode": args.mode, "layer": schema.layer, "cadence": schema.cadence, "primary_key": schema.primary_key, "columns": schema.columns, "endpoint": schema.endpoint}
        print(json.dumps(payload, indent=2, sort_keys=True) if args.json else payload["message"])
        return 0

    result.manifest["publish_latest"] = publish_latest
    result.manifest["latest_write_policy"] = "enabled" if publish_latest else "suppressed"
    review_dir = write_review_bundle(table=result.table, manifest=result.manifest, schema=result.schema, dq=result.dq, sample_rows=result.rows, root=Path(args.review_root))
    print(f"TABLE={result.table}\nMODE={args.mode}\nROWS={result.manifest.get('output_rows')}\nCOLUMNS={len(result.schema.get('columns', []))}\nPRIMARY_KEY={','.join(result.schema.get('primary_key', []))}\nPRIMARY_KEY_UNIQUE={str(result.manifest.get('primary_key_unique')).lower()}\nPUBLISH_LATEST={str(publish_latest).lower()}")
    print(f"CSV_KEY=s3://{args.s3_bucket}/{result.s3_keys.get('csv')}\nPARQUET_KEY=s3://{args.s3_bucket}/{result.s3_keys.get('parquet')}\nMANIFEST_KEY=s3://{args.s3_bucket}/{result.s3_keys.get('manifest')}")
    print(f"REVIEW_LOCAL_DIR={review_dir}\nREVIEW_SAMPLE_RAW_URL={raw_review_url(repo=args.github_repository, branch=REVIEW_BRANCH, table=result.table, filename='manifest.json')}\nDQ_STATUS={result.dq.get('dq_status')}")
    return 0 if result.dq.get("dq_status") != "fail" else 1


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
    return run_real_table(args)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return validate_command(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
