from __future__ import annotations

import boto3

from instagram.factory import party_monthly_profile as profile
from instagram.factory.oireachtas_production import resolve_production_key


def main() -> None:
    s3 = boto3.client("s3", region_name="ca-central-1")
    resolved_key, pointer = resolve_production_key(
        s3,
        bucket=profile.S3_BUCKET,
        production_key=profile.CLASSIFIED_KEY,
    )
    print(f"Resolved unified production batch: {pointer.get('batch_id') or pointer.get('mode')}")
    print(f"Resolved classified source: {resolved_key}")
    profile.CLASSIFIED_KEY = resolved_key
    profile.main()


if __name__ == "__main__":
    main()
