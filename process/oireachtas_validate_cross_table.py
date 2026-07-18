from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import boto3
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extract.oireachtas.io_s3 import get_bytes
from process.oireachtas_validate_table import DEFAULT_EXPECTATIONS, ValidationResult, load_expectations


def validate_relationship(
    *,
    name: str,
    child: pd.DataFrame,
    parent: pd.DataFrame,
    child_columns: list[str],
    parent_columns: list[str],
    allow_blank: bool = False,
) -> ValidationResult:
    missing_child = [column for column in child_columns if column not in child.columns]
    missing_parent = [column for column in parent_columns if column not in parent.columns]
    if missing_child or missing_parent:
        return ValidationResult(
            table="cross_table",
            test_name=name,
            expected_result="all relationship columns exist and every child key resolves",
            actual_result=f"missing child columns={missing_child}; missing parent columns={missing_parent}",
            status="fail",
        )

    child_keys = child[child_columns].fillna("").astype(str).apply(tuple, axis=1)
    parent_keys = set(parent[parent_columns].fillna("").astype(str).apply(tuple, axis=1))
    if allow_blank:
        candidate_mask = child[child_columns].fillna("").astype(str).apply(lambda row: any(value.strip() for value in row), axis=1)
    else:
        candidate_mask = pd.Series(True, index=child.index)
    orphans = child_keys[candidate_mask & ~child_keys.isin(parent_keys)]
    samples = [list(value) for value in orphans.head(5)]
    return ValidationResult(
        table="cross_table",
        test_name=name,
        expected_result="0 orphan child keys",
        actual_result=str(len(orphans)),
        status="pass" if orphans.empty else "fail",
        details=json.dumps({"sample_orphans": samples}, ensure_ascii=False),
    )


def _read_csv(*, s3: Any, bucket: str, prefix: str, table: str) -> pd.DataFrame:
    key = f"{prefix}/{table}.csv"
    return pd.read_csv(io.BytesIO(get_bytes(s3, bucket=bucket, key=key)), dtype=str, keep_default_na=False)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate configured Oireachtas cross-table relationships.")
    p.add_argument("--expectations", type=Path, default=DEFAULT_EXPECTATIONS)
    p.add_argument("--bucket", default="eirepolitic-data")
    p.add_argument("--region", default="ca-central-1")
    p.add_argument("--output-dir", type=Path, default=Path("oireachtas_validation_output"))
    return p


def main() -> int:
    args = parser().parse_args()
    expectations = load_expectations(args.expectations)
    relationships = expectations.get("relationships") or []
    prefix = str(expectations.get("logical_prefix") or "processed/oireachtas_unified/latest/csv")
    s3 = boto3.client("s3", region_name=args.region)
    cache: dict[str, pd.DataFrame] = {}

    def frame(table: str) -> pd.DataFrame:
        if table not in cache:
            cache[table] = _read_csv(s3=s3, bucket=args.bucket, prefix=prefix, table=table)
        return cache[table]

    results: list[ValidationResult] = []
    for relationship in relationships:
        results.append(
            validate_relationship(
                name=str(relationship["name"]),
                child=frame(str(relationship["child_table"])),
                parent=frame(str(relationship["parent_table"])),
                child_columns=list(relationship["child_columns"]),
                parent_columns=list(relationship["parent_columns"]),
                allow_blank=bool(relationship.get("allow_blank", False)),
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    (args.output_dir / "cross_table_validation.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.output_dir / "cross_table_validation.csv", index=False)
    failed = [result for result in results if result.status == "fail"]
    print(json.dumps({"relationships": len(results), "failed": len(failed)}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
