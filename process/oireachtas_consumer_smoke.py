from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extract.oireachtas.batch import batch_key_for_production_key, validate_batch_id
from extract.oireachtas.io_s3 import DEFAULT_BUCKET, DEFAULT_REGION, make_s3_client
from process.instagram_render_post import S3CSVLoader, build_post_context, render_slides


LOGICAL_KEYS = {
    "members": "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv",
    "member_summaries": "processed/oireachtas_unified/compat/text/members_summaries_compat.csv",
    "member_photos": "processed/oireachtas_unified/compat/media/members_photo_urls_compat.csv",
    "debate_issues": "processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv",
    "constituency_images": "processed/oireachtas_unified/compat/media/constituency_images_compat.csv",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Smoke-test downstream consumers against one immutable Oireachtas batch.")
    p.add_argument("--batch-id", default=os.getenv("OIREACHTAS_BATCH_ID", ""), required=False)
    p.add_argument("--target-year", default=os.getenv("TARGET_YEAR", ""), required=False)
    p.add_argument("--bucket", default=os.getenv("S3_BUCKET", DEFAULT_BUCKET))
    p.add_argument("--region", default=os.getenv("AWS_REGION", DEFAULT_REGION))
    p.add_argument("--spec", default="instagram/specs/constituency_test_post.yml")
    p.add_argument("--output", default="consumer_smoke")
    return p


def _read_csv(s3, *, bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str, keep_default_na=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    batch_id = validate_batch_id(args.batch_id)
    target_year = int(args.target_year)
    s3 = make_s3_client(region_name=args.region)
    physical = {name: batch_key_for_production_key(key, batch_id) for name, key in LOGICAL_KEYS.items()}

    members = _read_csv(s3, bucket=args.bucket, key=physical["members"])
    if len(members) < 150:
        raise RuntimeError(f"Members consumer input is incomplete: {len(members)} rows")
    if members["member_code"].duplicated().any():
        raise RuntimeError("Members consumer input contains duplicate member_code values")

    metrics_key = (
        f"processed/oireachtas_unified/batches/{batch_id}/consumers/member_profile_metrics/"
        f"member_profile_metrics_{target_year}.csv"
    )
    metrics = _read_csv(s3, bucket=args.bucket, key=metrics_key)
    if len(metrics) != len(members):
        raise RuntimeError(f"Metrics/member row mismatch: metrics={len(metrics)} members={len(members)}")
    if set(metrics["member_code"]) != set(members["member_code"]):
        raise RuntimeError("Metrics member_code set does not match candidate members")

    for label, env_name in {
        "members": "INSTAGRAM_MEMBERS_DATASET_KEYS",
        "member_summaries": "INSTAGRAM_MEMBER_SUMMARIES_DATASET_KEYS",
        "member_photos": "INSTAGRAM_MEMBER_PHOTOS_DATASET_KEYS",
        "debate_issues": "INSTAGRAM_DEBATE_ISSUES_DATASET_KEYS",
        "constituency_images": "INSTAGRAM_CONSTITUENCY_IMAGES_DATASET_KEYS",
    }.items():
        os.environ[env_name] = physical[label]

    constituency_counts = members["constituency"].fillna("").astype(str).str.strip()
    constituency = constituency_counts[constituency_counts != ""].value_counts().index[0]
    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    spec.setdefault("data", {})["constituency"] = constituency
    spec.setdefault("post", {})["output_root"] = args.output

    loader = S3CSVLoader(bucket=args.bucket, region=args.region)
    context = build_post_context(spec, loader)
    output_dir = Path(args.output) / f"batch-{batch_id}"
    html_paths = render_slides(spec, context, output_dir)
    if not html_paths:
        raise RuntimeError("Instagram consumer smoke produced no HTML slides")

    report = {
        "status": "pass",
        "batch_id": batch_id,
        "target_year": target_year,
        "member_rows": int(len(members)),
        "metrics_rows": int(len(metrics)),
        "constituency": constituency,
        "html_slide_count": len(html_paths),
        "datasets": physical,
        "metrics_key": metrics_key,
        "output_dir": str(output_dir),
    }
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.output, "consumer_smoke.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
