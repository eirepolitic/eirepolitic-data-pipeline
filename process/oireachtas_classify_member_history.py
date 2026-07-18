from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import boto3
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extract.oireachtas.io_s3 import get_bytes

TABLE_CONFIG = {
    "silver_member_parties": {
        "value_columns": ["party_uri", "party_name"],
        "start_column": "party_start",
        "end_column": "party_end",
        "current_column": "is_current",
        "allow_multiple_distinct_current": False,
    },
    "silver_member_constituencies": {
        "value_columns": ["constituency_uri", "constituency_name"],
        "start_column": "represent_start",
        "end_column": "represent_end",
        "current_column": "is_current",
        "allow_multiple_distinct_current": False,
    },
    "silver_member_offices": {
        "value_columns": ["office_uri", "office_name"],
        "start_column": "office_start",
        "end_column": "office_end",
        "current_column": "is_current",
        "allow_multiple_distinct_current": True,
    },
}


def truthy(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def read_live_csv(s3: Any, *, bucket: str, table: str) -> pd.DataFrame:
    key = f"processed/oireachtas_unified/latest/csv/{table}.csv"
    return pd.read_csv(io.BytesIO(get_bytes(s3, bucket=bucket, key=key)), dtype=str, keep_default_na=False)


def normalized_tuple(row: pd.Series, columns: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "") or "").strip() for column in columns)


def classify_table(table: str, df: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    current = df[truthy(df[config["current_column"]])].copy()
    details: list[dict[str, Any]] = []
    repeated_members = 0
    repeated_same_value = 0
    conflicting_values = 0
    exact_business_duplicates = 0
    multi_role_current = 0

    business_columns = ["member_code", *config["value_columns"], config["start_column"], config["end_column"]]
    duplicate_mask = current.duplicated(subset=business_columns, keep=False)
    exact_business_duplicates = int(duplicate_mask.sum())

    for member_code, group in current.groupby("member_code", sort=True):
        if len(group) <= 1:
            continue
        repeated_members += 1
        values = {
            normalized_tuple(row, config["value_columns"])
            for _, row in group.iterrows()
        }
        starts = sorted({str(value).strip() for value in group[config["start_column"]].tolist() if str(value).strip()})
        ends = sorted({str(value).strip() for value in group[config["end_column"]].tolist() if str(value).strip()})

        if len(values) == 1:
            classification = "repeated_same_current_value"
            repeated_same_value += 1
        elif config["allow_multiple_distinct_current"]:
            classification = "valid_multiple_current_roles"
            multi_role_current += 1
        else:
            classification = "conflicting_current_values"
            conflicting_values += 1

        details.append(
            {
                "table": table,
                "member_code": str(member_code),
                "classification": classification,
                "current_row_count": int(len(group)),
                "distinct_value_count": int(len(values)),
                "values": [list(value) for value in sorted(values)],
                "start_dates": starts,
                "nonblank_end_dates": ends,
                "rows": group.to_dict(orient="records"),
            }
        )

    summary = {
        "table": table,
        "total_rows": int(len(df)),
        "current_rows": int(len(current)),
        "members_with_multiple_current_rows": repeated_members,
        "members_with_repeated_same_current_value": repeated_same_value,
        "members_with_conflicting_current_values": conflicting_values,
        "members_with_valid_multiple_current_roles": multi_role_current,
        "exact_business_duplicate_rows": exact_business_duplicates,
    }
    return summary, details


def write_report(summaries: list[dict[str, Any]], details: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "member_history_classification_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "member_history_classification_details.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(summaries).to_csv(output_dir / "member_history_classification_summary.csv", index=False)

    lines = [
        "# Member history current-row classification",
        "",
        "| Table | Rows | Current rows | Members with repeated current rows | Same value repeated | Conflicting values | Valid multiple roles | Exact business duplicate rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['table']} | {item['total_rows']} | {item['current_rows']} | "
            f"{item['members_with_multiple_current_rows']} | {item['members_with_repeated_same_current_value']} | "
            f"{item['members_with_conflicting_current_values']} | {item['members_with_valid_multiple_current_roles']} | "
            f"{item['exact_business_duplicate_rows']} |"
        )

    conflicts = [item for item in details if item["classification"] == "conflicting_current_values"]
    repeats = [item for item in details if item["classification"] == "repeated_same_current_value"]
    roles = [item for item in details if item["classification"] == "valid_multiple_current_roles"]
    lines.extend(["", "## Conflicting current values", ""])
    if conflicts:
        for item in conflicts:
            lines.append(f"- `{item['table']}` / `{item['member_code']}`: {json.dumps(item['values'], ensure_ascii=False)}")
    else:
        lines.append("None.")

    lines.extend(["", "## Repeated same current value", ""])
    if repeats:
        for item in repeats[:30]:
            lines.append(
                f"- `{item['table']}` / `{item['member_code']}`: {item['current_row_count']} rows, "
                f"starts={json.dumps(item['start_dates'])}"
            )
        if len(repeats) > 30:
            lines.append(f"- …and {len(repeats) - 30} more. See the JSON details file.")
    else:
        lines.append("None.")

    lines.extend(["", "## Valid multiple current offices", ""])
    if roles:
        for item in roles[:30]:
            lines.append(f"- `{item['member_code']}`: {json.dumps(item['values'], ensure_ascii=False)}")
        if len(roles) > 30:
            lines.append(f"- …and {len(roles) - 30} more. See the JSON details file.")
    else:
        lines.append("None.")

    (output_dir / "member_history_classification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Classify repeated current rows in Oireachtas member-history tables.")
    p.add_argument("--bucket", default="eirepolitic-data")
    p.add_argument("--region", default="ca-central-1")
    p.add_argument("--output-dir", type=Path, default=Path("oireachtas_validation_output/checkpoint2_classification"))
    return p


def main() -> int:
    args = parser().parse_args()
    s3 = boto3.client("s3", region_name=args.region)
    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for table, config in TABLE_CONFIG.items():
        summary, table_details = classify_table(table, read_live_csv(s3, bucket=args.bucket, table=table), config)
        summaries.append(summary)
        details.extend(table_details)
    write_report(summaries, details, args.output_dir)
    print(json.dumps({"summaries": summaries, "detail_records": len(details)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
