from __future__ import annotations

import pandas as pd

APPROVED_ISSUES = {
    "Macroeconomics",
    "Civil Rights, Minority Issues and Civil Liberties",
    "Health",
    "Agriculture",
    "Labor, Employment and Immigration",
    "Education",
    "Environment",
    "Energy",
    "Transportation",
    "Law/Crime and Family Issues",
    "Social Welfare",
    "Housing and Community Development",
    "Banking/Finance and Domestic Commerce",
    "Defense",
    "Space, Science, and Technology",
    "Foreign Trade",
    "International Affairs and Foreign Aid",
    "Government Operations",
    "Public Lands and Water Management",
    "State and Local Government Administration",
    "Culture and Arts",
    "Sports and Recreation",
    "Other/Miscellaneous",
    "Domestic Terrorism",
    "NONE",
}

FINAL_STATUSES = {"classified", "none", "skipped_short_text"}


def audit_issue_classification(
    speeches: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    period_start: str | None = None,
    period_end: str | None = None,
) -> dict:
    """Audit whether issue labels are safe to use for metric calculations.

    Public issue metrics require a one-to-one classification row for every speech
    in scope, a matching source-text hash, an approved label, and a final status.
    """
    required_speech = {"speech_id", "speech_text_hash", "debate_date"}
    required_labels = {
        "speech_id", "source_speech_text_hash", "issue_label", "issue_label_source",
        "model_name", "classification_status", "classified_at_utc",
    }
    missing_speech = sorted(required_speech - set(speeches.columns))
    missing_labels = sorted(required_labels - set(labels.columns))
    if missing_speech or missing_labels:
        return {
            "ready": False,
            "missing_speech_columns": missing_speech,
            "missing_label_columns": missing_labels,
        }

    source = speeches.copy()
    enrichment = labels.copy()
    source["speech_id"] = source["speech_id"].fillna("").astype(str)
    enrichment["speech_id"] = enrichment["speech_id"].fillna("").astype(str)

    if period_start is not None and period_end is not None:
        dates = pd.to_datetime(source["debate_date"], errors="coerce").dt.normalize()
        start = pd.Timestamp(period_start)
        end = pd.Timestamp(period_end)
        source = source.loc[dates.between(start, end, inclusive="both")].copy()
        enrichment = enrichment[enrichment["speech_id"].isin(set(source["speech_id"]))].copy()

    source_ids = set(source["speech_id"])
    label_ids = set(enrichment["speech_id"])
    missing_label_ids = sorted(source_ids - label_ids)
    orphan_label_ids = sorted(label_ids - source_ids)

    duplicate_source = int(source["speech_id"].duplicated().sum())
    duplicate_labels = int(enrichment["speech_id"].duplicated().sum())

    joined = source[["speech_id", "speech_text_hash"]].merge(
        enrichment[[
            "speech_id", "source_speech_text_hash", "issue_label", "issue_label_source",
            "model_name", "classification_status", "classified_at_utc",
        ]],
        on="speech_id",
        how="left",
        validate="one_to_one" if duplicate_source == 0 and duplicate_labels == 0 else "many_to_many",
    )

    hash_match = joined["speech_text_hash"].fillna("").astype(str) == joined["source_speech_text_hash"].fillna("").astype(str)
    invalid_label = ~joined["issue_label"].fillna("").astype(str).isin(APPROVED_ISSUES)
    final_status = joined["classification_status"].fillna("").astype(str).isin(FINAL_STATUSES)
    blank_source = joined["issue_label_source"].fillna("").astype(str).str.strip().eq("")
    model_missing = (
        joined["issue_label_source"].fillna("").astype(str).eq("openai_model")
        & joined["model_name"].fillna("").astype(str).str.strip().eq("")
    )

    source_counts = enrichment["issue_label_source"].fillna("").replace("", "<blank>").value_counts(dropna=False).to_dict()
    model_counts = enrichment["model_name"].fillna("").replace("", "<blank>").value_counts(dropna=False).to_dict()
    status_counts = enrichment["classification_status"].fillna("").replace("", "<blank>").value_counts(dropna=False).to_dict()
    label_counts = enrichment["issue_label"].fillna("").replace("", "<blank>").value_counts(dropna=False).to_dict()

    total = int(len(source))
    policy_count = int(joined["issue_label"].fillna("").astype(str).ne("NONE").sum()) if total else 0
    none_count = int(joined["issue_label"].fillna("").astype(str).eq("NONE").sum()) if total else 0

    checks = {
        "source_speech_id_unique": duplicate_source == 0,
        "label_speech_id_unique": duplicate_labels == 0,
        "every_speech_has_label_row": len(missing_label_ids) == 0,
        "no_orphan_label_rows": len(orphan_label_ids) == 0,
        "source_hash_matches": bool(hash_match.all()) if total else True,
        "all_labels_approved": bool((~invalid_label).all()) if total else True,
        "all_classifications_final": bool(final_status.all()) if total else True,
        "all_label_sources_populated": bool((~blank_source).all()) if total else True,
        "model_rows_have_model_name": bool((~model_missing).all()) if total else True,
    }

    return {
        "ready": all(checks.values()),
        "scope_rows": total,
        "policy_labelled_rows": policy_count,
        "none_rows": none_count,
        "policy_label_rate": (policy_count / total) if total else 1.0,
        "missing_label_rows": len(missing_label_ids),
        "orphan_label_rows": len(orphan_label_ids),
        "hash_mismatch_rows": int((~hash_match).sum()),
        "invalid_label_rows": int(invalid_label.sum()),
        "non_final_status_rows": int((~final_status).sum()),
        "blank_label_source_rows": int(blank_source.sum()),
        "model_missing_name_rows": int(model_missing.sum()),
        "checks": checks,
        "classification_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "issue_label_source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "model_name_counts": {str(k): int(v) for k, v in model_counts.items()},
        "issue_label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "missing_label_examples": missing_label_ids[:10],
        "orphan_label_examples": orphan_label_ids[:10],
    }
