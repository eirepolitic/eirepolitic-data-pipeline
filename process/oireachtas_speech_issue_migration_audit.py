from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from process.oireachtas_speech_issue_classifier import (
    LEGACY_CLASSIFIED_KEY,
    SILVER_SPEECHES_KEY,
    canonicalize_label,
    make_s3_client,
    normalize_date,
    normalize_name,
    normalize_order,
    normalize_text,
    read_s3_csv,
    text_hash,
)


def _col(df: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype="object")


def unique_label_lookup(frame: pd.DataFrame, key_column: str) -> tuple[dict[object, str], int]:
    lookup: dict[object, str] = {}
    ambiguous = 0
    for key, group in frame.groupby(key_column, sort=False):
        labels = sorted(set(group["issue_label"].tolist()))
        if len(group) == 1 and len(labels) == 1:
            lookup[key] = labels[0]
        else:
            ambiguous += 1
    return lookup, ambiguous


def audit(silver: pd.DataFrame, legacy: pd.DataFrame) -> dict[str, object]:
    legacy_work = pd.DataFrame(index=legacy.index)
    legacy_text = _col(legacy, "Speech Text", "speech_text")
    legacy_work["debate_date"] = _col(legacy, "Debate Date", "debate_date", "date").map(normalize_date)
    legacy_work["speech_order"] = _col(legacy, "Speech Order", "speech_order").map(normalize_order)
    legacy_work["speaker_name"] = _col(legacy, "Speaker Name", "speaker_name", "member_name").map(normalize_name)
    legacy_work["issue_label"] = _col(legacy, "PoliticalIssues", "political_issues", "issue_label").map(canonicalize_label)
    legacy_work["raw_text_hash"] = legacy_text.map(text_hash)
    legacy_work["normalized_text_hash"] = legacy_text.map(lambda value: text_hash(normalize_text(value)))
    legacy_work = legacy_work[legacy_work["issue_label"] != ""].copy()
    legacy_work["raw_date_hash"] = list(zip(legacy_work["debate_date"], legacy_work["raw_text_hash"]))
    legacy_work["norm_date_hash"] = list(zip(legacy_work["debate_date"], legacy_work["normalized_text_hash"]))
    legacy_work["norm_exact"] = list(
        zip(
            legacy_work["debate_date"],
            legacy_work["speech_order"],
            legacy_work["speaker_name"],
            legacy_work["normalized_text_hash"],
        )
    )

    raw_lookup, raw_ambiguous = unique_label_lookup(legacy_work, "raw_date_hash")
    norm_lookup, norm_ambiguous = unique_label_lookup(legacy_work, "norm_date_hash")
    norm_exact_lookup: dict[object, str] = {}
    norm_exact_ambiguous = 0
    for key, group in legacy_work.groupby("norm_exact", sort=False):
        labels = sorted(set(group["issue_label"].tolist()))
        if len(labels) == 1:
            norm_exact_lookup[key] = labels[0]
        else:
            norm_exact_ambiguous += 1

    silver_work = pd.DataFrame(index=silver.index)
    silver_work["speech_id"] = _col(silver, "speech_id")
    silver_work["debate_date"] = _col(silver, "debate_date").map(normalize_date)
    silver_work["speech_order"] = _col(silver, "speech_order").map(normalize_order)
    silver_work["speaker_name"] = _col(silver, "speaker_name").map(normalize_name)
    silver_work["source_hash"] = _col(silver, "speech_text_hash")
    silver_text = _col(silver, "speech_text")
    silver_work["normalized_text_hash"] = silver_text.map(lambda value: text_hash(normalize_text(value)))
    silver_work["raw_date_hash"] = list(zip(silver_work["debate_date"], silver_work["source_hash"]))
    silver_work["norm_date_hash"] = list(zip(silver_work["debate_date"], silver_work["normalized_text_hash"]))
    silver_work["norm_exact"] = list(
        zip(
            silver_work["debate_date"],
            silver_work["speech_order"],
            silver_work["speaker_name"],
            silver_work["normalized_text_hash"],
        )
    )

    raw_match = silver_work["raw_date_hash"].map(raw_lookup).fillna("")
    norm_exact_match = silver_work["norm_exact"].map(norm_exact_lookup).fillna("")
    norm_date_match = silver_work["norm_date_hash"].map(norm_lookup).fillna("")

    baseline_mask = raw_match != ""
    recovered_exact_mask = (~baseline_mask) & (norm_exact_match != "")
    recovered_date_mask = (~baseline_mask) & (~recovered_exact_mask) & (norm_date_match != "")
    recovered_mask = baseline_mask | recovered_exact_mask | recovered_date_mask

    legacy_max = pd.to_datetime(legacy_work["debate_date"], errors="coerce").max()
    silver_dates = pd.to_datetime(silver_work["debate_date"], errors="coerce")
    historical_silver_mask = silver_dates <= legacy_max

    legacy_norm_keys = set(legacy_work["norm_date_hash"].tolist())
    silver_norm_keys = set(silver_work["norm_date_hash"].tolist())
    legacy_keys_absent_from_silver = legacy_norm_keys - silver_norm_keys
    legacy_rows_absent = int(legacy_work["norm_date_hash"].isin(legacy_keys_absent_from_silver).sum())

    unmatched_historical = historical_silver_mask & (~recovered_mask)
    sample = []
    for idx in silver_work.index[unmatched_historical][:25]:
        sample.append(
            {
                "speech_id": silver_work.at[idx, "speech_id"],
                "debate_date": silver_work.at[idx, "debate_date"],
                "speech_order": silver_work.at[idx, "speech_order"],
                "speaker_name": silver_work.at[idx, "speaker_name"],
                "speech_excerpt": normalize_text(silver.at[idx, "speech_text"])[:300],
            }
        )

    legacy_valid = int(len(legacy_work))
    migrated_with_normalization = int(recovered_mask.sum())
    return {
        "silver_rows": int(len(silver_work)),
        "legacy_valid_rows": legacy_valid,
        "legacy_max_date": legacy_max.date().isoformat() if not pd.isna(legacy_max) else None,
        "baseline_raw_date_hash_matches": int(baseline_mask.sum()),
        "additional_normalized_exact_matches": int(recovered_exact_mask.sum()),
        "additional_normalized_date_hash_matches": int(recovered_date_mask.sum()),
        "total_silver_rows_reusing_legacy_after_normalization": migrated_with_normalization,
        "migration_pct_of_legacy_rows_after_normalization": round(migrated_with_normalization / legacy_valid * 100, 2) if legacy_valid else 0.0,
        "legacy_raw_date_hash_ambiguous_keys": int(raw_ambiguous),
        "legacy_normalized_exact_ambiguous_keys": int(norm_exact_ambiguous),
        "legacy_normalized_date_hash_ambiguous_keys": int(norm_ambiguous),
        "legacy_rows_with_normalized_date_text_absent_from_silver": legacy_rows_absent,
        "historical_silver_rows_through_legacy_max_date": int(historical_silver_mask.sum()),
        "historical_silver_rows_still_without_legacy_match": int(unmatched_historical.sum()),
        "unmatched_historical_sample": sample,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit legacy-to-unified speech issue migration coverage")
    parser.add_argument("--bucket", default="eirepolitic-data")
    parser.add_argument("--region", default="ca-central-1")
    parser.add_argument("--report-path", default="speech_issue_migration_audit.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    s3 = make_s3_client(region_name=args.region)
    silver = read_s3_csv(s3, bucket=args.bucket, key=SILVER_SPEECHES_KEY)
    legacy = read_s3_csv(s3, bucket=args.bucket, key=LEGACY_CLASSIFIED_KEY)
    report = audit(silver, legacy)
    Path(args.report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
