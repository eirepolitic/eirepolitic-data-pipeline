from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

# Support direct execution as `python process/oireachtas_batch_control.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extract.oireachtas.batch import (
    PRODUCTION_POINTER_KEY,
    PREVIOUS_POINTER_KEY,
    assemble_batch_manifest,
    promote_batch,
    read_json_if_exists,
    rollback_batch,
    validate_batch_id,
)
from extract.oireachtas.io_s3 import DEFAULT_BUCKET, DEFAULT_REGION, make_s3_client, production_publishing_enabled


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Validate, promote, inspect, or roll back Oireachtas immutable batches.")
    root.add_argument("--bucket", default=os.getenv("S3_BUCKET", DEFAULT_BUCKET))
    root.add_argument("--region", default=os.getenv("AWS_REGION", DEFAULT_REGION))
    sub = root.add_subparsers(dest="command", required=True)

    assemble = sub.add_parser("assemble")
    assemble.add_argument("--batch-id", required=True)
    assemble.add_argument("--required-table", action="append", default=[])

    promote = sub.add_parser("promote")
    promote.add_argument("--batch-id", required=True)

    rollback = sub.add_parser("rollback")
    rollback.add_argument("--batch-id", required=True)

    sub.add_parser("status")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    s3 = make_s3_client(region_name=args.region)
    actor = os.getenv("GITHUB_ACTOR", "")
    workflow_run_id = os.getenv("GITHUB_RUN_ID", "")

    if args.command == "assemble":
        payload = assemble_batch_manifest(
            s3,
            bucket=args.bucket,
            batch_id=validate_batch_id(args.batch_id),
            required_tables=args.required_table,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload.get("status") == "validated" else 1

    if args.command in {"promote", "rollback"}:
        if not production_publishing_enabled():
            raise RuntimeError(
                "Batch pointer updates require both OIREACHTAS_PUBLISH_ENABLED=true "
                "and OIREACHTAS_PUBLISH_LATEST=true"
            )
        if args.command == "promote":
            payload = promote_batch(
                s3,
                bucket=args.bucket,
                batch_id=validate_batch_id(args.batch_id),
                actor=actor,
                workflow_run_id=workflow_run_id,
            )
        else:
            payload = rollback_batch(
                s3,
                bucket=args.bucket,
                target_batch_id=validate_batch_id(args.batch_id),
                actor=actor,
                workflow_run_id=workflow_run_id,
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    payload = {
        "production": read_json_if_exists(s3, bucket=args.bucket, key=PRODUCTION_POINTER_KEY),
        "previous": read_json_if_exists(s3, bucket=args.bucket, key=PREVIOUS_POINTER_KEY),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
