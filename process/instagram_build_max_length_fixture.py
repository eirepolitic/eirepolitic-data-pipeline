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
DEFAULT_STRESS_COLUMNS = [
    "full_name",
    "party",
    "constituency",
    "top_issue_2025",
    "vote_participation_pct_2025",
    "speech_count_2025",
    "speech_rank_2025",
]
PHOTO_COLUMNS = ["photo_url"]


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


def clean_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def text_len(value: Any) -> int:
    return len(clean_value(value))


def longest_value(df: pd.DataFrame, column: str) -> tuple[Any, int | None, int]:
    """Return the longest non-empty value for a column, plus source row index and text length."""
    if column not in df.columns:
        return "", None, 0

    best_index: int | None = None
    best_value: Any = ""
    best_length = -1
    best_tiebreak = ""

    for idx, raw in df[column].items():
        value = clean_value(raw)
        if not value:
            continue
        length = len(value)
        # Deterministic tie-break: alphabetically earliest value.
        tiebreak = value.lower()
        if length > best_length or (length == best_length and (not best_tiebreak or tiebreak < best_tiebreak)):
            best_index = int(idx)
            best_value = raw
            best_length = length
            best_tiebreak = tiebreak

    if best_index is None:
        return "", None, 0
    return best_value, best_index, best_length


def longest_valid_photo(df: pd.DataFrame) -> tuple[str, int | None, int]:
    for column in PHOTO_COLUMNS:
        if column not in df.columns:
            continue
        candidates = df[column].fillna("").astype(str).str.strip()
        candidates = candidates[candidates.str.startswith(("http://", "https://"))]
        if not candidates.empty:
            # Prefer the longest URL only as a stress value; any valid photo works for layout testing.
            ordered = candidates.sort_values(key=lambda s: s.str.len(), ascending=False)
            idx = int(ordered.index[0])
            value = str(ordered.iloc[0]).strip()
            return value, idx, len(value)
    return "", None, 0


def build_max_length_fixture(df: pd.DataFrame, stress_columns: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    available = [col for col in stress_columns if col in df.columns]
    if not available:
        raise RuntimeError(f"None of the stress columns exist in the input table: {stress_columns}")

    # Start from the longest-name row to preserve any extra columns expected downstream,
    # then overwrite display fields with the longest real value found independently per field.
    base_col = "full_name" if "full_name" in df.columns else available[0]
    _, base_index, _ = longest_value(df, base_col)
    if base_index is None:
        base_index = int(df.index[0])

    synthetic = df.loc[[base_index]].copy()
    field_sources: dict[str, dict[str, Any]] = {}

    for column in available:
        value, source_index, length = longest_value(df, column)
        if source_index is not None:
            synthetic.at[synthetic.index[0], column] = value
        field_sources[column] = {
            "source_row_index": source_index,
            "source_full_name": clean_value(df.at[source_index, "full_name"]) if source_index is not None and "full_name" in df.columns else "",
            "value": clean_value(value),
            "length": int(length),
        }

    photo_value, photo_source_index, photo_length = longest_valid_photo(df)
    if "photo_url" in df.columns and photo_value:
        synthetic.at[synthetic.index[0], "photo_url"] = photo_value
        field_sources["photo_url"] = {
            "source_row_index": photo_source_index,
            "source_full_name": clean_value(df.at[photo_source_index, "full_name"]) if photo_source_index is not None and "full_name" in df.columns else "",
            "value": photo_value,
            "length": int(photo_length),
        }

    # Make the synthetic nature explicit in stable ID/code-style columns if present.
    if "member_code" in synthetic.columns:
        synthetic.at[synthetic.index[0], "member_code"] = "synthetic-max-length"

    row = synthetic.iloc[0]
    metadata = {
        "success": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_mode": "synthetic_longest_value_per_field",
        "stress_columns_requested": stress_columns,
        "stress_columns_used": available,
        "selected_full_name": clean_value(row.get("full_name", "")),
        "selected_party": clean_value(row.get("party", "")),
        "selected_constituency": clean_value(row.get("constituency", "")),
        "synthetic_row": True,
        "input_rows": int(len(df)),
        "field_sources": field_sources,
    }
    return synthetic, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a one-row synthetic max-length fixture for Instagram template stress tests.")
    parser.add_argument("--source-table", required=True, help="CSV path or S3 URI for source member metrics.")
    parser.add_argument("--output", required=True, help="Output CSV path for one synthetic row.")
    parser.add_argument("--metadata-output", help="Optional JSON metadata path.")
    parser.add_argument("--aws-region", default=DEFAULT_REGION)
    parser.add_argument("--score-columns", nargs="*", default=DEFAULT_STRESS_COLUMNS, help="Display fields to stress with their longest available values.")
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
