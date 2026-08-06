from __future__ import annotations

import argparse
import io
import json
import os
from typing import Sequence

import pandas as pd

from extract.oireachtas.enrichment_contracts import (
    assert_valid_enrichment_manifest,
    assert_valid_publish_contract,
)
from extract.oireachtas.io_s3 import DEFAULT_BUCKET, DEFAULT_REGION, make_s3_client
from process.oireachtas_speech_issue_classifier import TABLE_NAME, run_manifest_key


def validate_run_from_s3(
    *,
    s3: object,
    bucket: str,
    run_id: str,
    publish_contract: bool,
) -> dict[str, object]:
    manifest_key = run_manifest_key(run_id)
    manifest_response = s3.get_object(Bucket=bucket, Key=manifest_key)
    manifest = json.loads(manifest_response["Body"].read().decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest is not a JSON object: s3://{bucket}/{manifest_key}")

    if publish_contract:
        table_key = str(manifest.get("table_parquet_key") or "").strip()
        if not table_key:
            raise ValueError("Manifest does not contain table_parquet_key")
        table_response = s3.get_object(Bucket=bucket, Key=table_key)
        frame = pd.read_parquet(io.BytesIO(table_response["Body"].read()))
        assert_valid_publish_contract(TABLE_NAME, manifest, frame)
        mode = "publish"
        row_count = int(len(frame))
    else:
        assert_valid_enrichment_manifest(
            TABLE_NAME,
            manifest,
            require_candidate_artifacts=False,
        )
        mode = "manifest"
        row_count = None

    return {
        "table": TABLE_NAME,
        "run_id": run_id,
        "validation_mode": mode,
        "status": "pass",
        "manifest_key": manifest_key,
        "row_count": row_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate immutable Oireachtas speech-enrichment contracts"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--publish-contract", action="store_true")
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", DEFAULT_REGION))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_run_from_s3(
        s3=make_s3_client(region_name=args.region),
        bucket=args.bucket,
        run_id=args.run_id,
        publish_contract=args.publish_contract,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
