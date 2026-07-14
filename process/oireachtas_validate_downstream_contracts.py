from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extract.oireachtas.contracts import validate_contract_set
from extract.oireachtas.io_s3 import DEFAULT_BUCKET, DEFAULT_REGION, make_s3_client


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate Oireachtas downstream schema and freshness contracts.")
    p.add_argument("--bucket", default=os.getenv("S3_BUCKET", DEFAULT_BUCKET))
    p.add_argument("--region", default=os.getenv("AWS_REGION", DEFAULT_REGION))
    p.add_argument("--contract", action="append", default=[])
    p.add_argument("--as-of-date", default="")
    p.add_argument("--output", default="")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    as_of = datetime.strptime(args.as_of_date, "%Y-%m-%d").date() if args.as_of_date else None
    result = validate_contract_set(
        make_s3_client(region_name=args.region),
        bucket=args.bucket,
        names=args.contract or None,
        as_of=as_of,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
