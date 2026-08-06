from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

from extract.oireachtas.io_s3 import DEFAULT_BUCKET, DEFAULT_REGION, make_s3_client
from extract.oireachtas.speech_issue_compat import resolve_speech_issue_compatibility


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve the active speech issue compatibility dataset"
    )
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", DEFAULT_REGION))
    parser.add_argument("--cutover-enabled", action="store_true")
    parser.add_argument("--disable-legacy-fallback", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolution = resolve_speech_issue_compatibility(
        make_s3_client(region_name=args.region),
        bucket=args.bucket,
        cutover_enabled=args.cutover_enabled,
        allow_legacy_fallback=not args.disable_legacy_fallback,
    )
    print(json.dumps(resolution.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
