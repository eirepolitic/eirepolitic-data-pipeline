from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import pandas as pd

DEFAULT_REGION = "ca-central-1"
DEFAULT_SCORE_COLUMNS = [
    "full_name",
    "party",
    "constituency",
    "top_issue_2025",
    "vote_participation_pct_2025",
    "speech_count_2025",
    "speech_rank_2025",
]


def read_csv(path: str, region: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        _, _, rest = path.partition("s3://")
        bucket, _, key = rest.partition("/")
        if not bucket or not key:
            raise RuntimeError(f"Invalid S3 URI: {path}")
        s3 = boto3.client("s3", region_name=region)
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(io.StringIO(obj["Body"].read().decode("utf-8-sig", errors="replace")))
    return pd.read_csv(path)


def text_len(value: Any) -> int:
    if pd.isna(value):
        return 0
    return len(str(value).strip())


def build_max_length_fixture(df: pd.DataFrame, score_columns: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    available = [col for col in score_columns if col in df.columns]
    if not available:
        raise RuntimeError(f"None of the score columns exist in the input table: {score_columns}")

    scored = df.copy()
    scored["_max_length_score"] = scored[available].apply(lambda row: sum(text_len(v) for v in row), axis=1)
    scored["_max_length_name"] = scored.get("full_name", pd.Series([""] * len(scored))).fillna("").astype(str)
    scored = scored.sort_values(by=["_max_length_score", "_max_length_name"], ascending=[False, True])
    selected = scored.head(1).drop(columns=["_max_length_score", "_max_length_name"])

    row = scored.iloc[0]
    metadata = {
        "success": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_mode": "single_real_record_max_text_length",
        "score_columns_requested": score_columns,
        "score_columns_used": available,
        "selected_full_name": str(row.get("full_name", "")),
        "selected_party": str(row.get("party", "")),
        "selected_constituency": str(row.get("constituency", "")),
        "max_length_score": int(row.get("_max_length_score", 0)),
        "input_rows": int(len(df)),
    }
    return selected, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a one-row max-length real-record fixture for Instagram template stress tests.")
    parser.add_argument("--source-table", required=True, help="CSV path or S3 URI for source member metrics.")
    parser.add_argument("--output", required=True, help="Output CSV path for one selected row.")
    parser.add_argument("--metadata-output", help="Optional JSON metadata path.")
    parser.add_argument("--aws-region", default=DEFAULT_REGION)
    parser.add_argument("--score-columns", nargs="*", default=DEFAULT_SCORE_COLUMNS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_csv(args.source_table, args.aws_region)
    selected, metadata = build_max_length_fixture(df, list(args.score_columns))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False, encoding="utf-8-sig")

    metadata_output = Path(args.metadata_output) if args.metadata_output else output.with_suffix(".metadata.json")
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({**metadata, "output": str(output), "metadata_output": str(metadata_output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
