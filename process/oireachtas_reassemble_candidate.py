from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extract.oireachtas.batch import assemble_batch_manifest, batch_manifest_key, read_json_required, validate_batch_id
from extract.oireachtas.io_s3 import DEFAULT_BUCKET, DEFAULT_REGION, make_s3_client


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reassemble a candidate while preserving its original required tables.")
    p.add_argument("--batch-id", default=os.getenv("OIREACHTAS_BATCH_ID", ""))
    p.add_argument("--bucket", default=os.getenv("S3_BUCKET", DEFAULT_BUCKET))
    p.add_argument("--region", default=os.getenv("AWS_REGION", DEFAULT_REGION))
    p.add_argument("--require", action="append", default=[])
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    batch_id = validate_batch_id(args.batch_id)
    s3 = make_s3_client(region_name=args.region)
    existing = read_json_required(s3, bucket=args.bucket, key=batch_manifest_key(batch_id))
    required = sorted(set(existing.get("required_tables") or []) | set(args.require))
    result = assemble_batch_manifest(
        s3,
        bucket=args.bucket,
        batch_id=batch_id,
        required_tables=required,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
