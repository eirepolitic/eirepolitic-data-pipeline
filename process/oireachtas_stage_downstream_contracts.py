from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extract.oireachtas.batch import batch_key_for_production_key, current_batch_id, resolve_production_key
from extract.oireachtas.contracts import load_contract_config
from extract.oireachtas.io_s3 import DEFAULT_BUCKET, DEFAULT_REGION, make_s3_client
from extract.oireachtas.normalize import stable_json_dumps


AUXILIARY_CONTRACTS = [
    "member_photo_urls",
    "member_summaries",
    "constituency_images",
    "debate_issue_labels",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage validated downstream enrichment inputs into an immutable candidate batch.")
    p.add_argument("--bucket", default=os.getenv("S3_BUCKET", DEFAULT_BUCKET))
    p.add_argument("--region", default=os.getenv("AWS_REGION", DEFAULT_REGION))
    p.add_argument("--batch-id", default=os.getenv("OIREACHTAS_BATCH_ID", ""), required=False)
    p.add_argument("--contract", action="append", default=[])
    return p


def _source_key(s3, *, bucket: str, logical_key: str) -> str:
    try:
        return resolve_production_key(s3, bucket=bucket, production_key=logical_key)
    except Exception:
        return logical_key


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    batch_id = args.batch_id or current_batch_id()
    if not batch_id:
        raise ValueError("batch_id is required")
    contracts, _ = load_contract_config()
    selected = args.contract or AUXILIARY_CONTRACTS
    unknown = sorted(set(selected) - set(contracts))
    if unknown:
        raise KeyError(f"Unknown contracts: {unknown}")

    s3 = make_s3_client(region_name=args.region)
    now = datetime.now(timezone.utc)
    results: list[dict[str, object]] = []
    for name in selected:
        contract = contracts[name]
        source_key = _source_key(s3, bucket=args.bucket, logical_key=contract.logical_key)
        source = s3.get_object(Bucket=args.bucket, Key=source_key)
        body = source["Body"].read()
        head = s3.head_object(Bucket=args.bucket, Key=source_key)
        modified = head.get("LastModified")
        if modified is None:
            raise RuntimeError(f"Missing LastModified for {source_key}")
        age_days = (now.date() - modified.astimezone(timezone.utc).date()).days
        if age_days > contract.maximum_age_days:
            raise RuntimeError(
                f"Refusing stale enrichment {name}: age {age_days} days exceeds {contract.maximum_age_days}"
            )
        target_key = batch_key_for_production_key(contract.logical_key, batch_id)
        s3.put_object(
            Bucket=args.bucket,
            Key=target_key,
            Body=body,
            ContentType=str(source.get("ContentType") or "text/csv"),
            Metadata={
                "source-key": source_key[:2000],
                "source-etag": str(head.get("ETag", "")).strip('"')[:1024],
                "source-last-modified": modified.astimezone(timezone.utc).isoformat()[:1024],
                "contract": name,
            },
        )
        results.append(
            {
                "contract": name,
                "logical_key": contract.logical_key,
                "source_key": source_key,
                "target_key": target_key,
                "source_age_days": age_days,
                "bytes": len(body),
            }
        )

    provenance_key = f"processed/oireachtas_unified/batches/{batch_id}/provenance/downstream_enrichments.json"
    payload = {
        "batch_id": batch_id,
        "staged_at_utc": now.isoformat(),
        "status": "success",
        "datasets": results,
    }
    s3.put_object(
        Bucket=args.bucket,
        Key=provenance_key,
        Body=(stable_json_dumps(payload) + "\n").encode("utf-8"),
        ContentType="application/json",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
