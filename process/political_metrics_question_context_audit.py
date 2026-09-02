#!/usr/bin/env python3
"""Audit deterministic parliamentary-question / speech relationships against production.

No writes to S3 are performed. This verifies that oral question records, oral-question
sections, observed exchange participants, and transcript interventions reconcile without
double counting or inferred question-taker attribution.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import boto3
import pandas as pd

from extract.oireachtas.batch import PRODUCTION_POINTER_KEY, read_json_required, resolve_production_key
from political_metrics.materialize import get_dataset_contract, load_materialization_contract, validate_materialized_frame
from political_metrics.question_context import (
    build_oral_question_exchange_participants,
    build_oral_question_sections,
    build_speech_question_context,
)

BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")
REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ca-central-1"
OUT_DIR = Path(os.getenv("POLITICAL_METRICS_AUDIT_DIR", "artifacts/political-metrics-audit"))
CONTRACT_PATH = REPO_ROOT / "configs/political_metrics/materialization.yml"

KEYS = {
    "questions": "processed/oireachtas_unified/latest/csv/silver_questions.csv",
    "speeches": "processed/oireachtas_unified/latest/csv/silver_speeches.csv",
    "sections": "processed/oireachtas_unified/latest/csv/silver_debate_sections.csv",
    "offices": "processed/oireachtas_unified/latest/csv/silver_member_offices.csv",
}

ALLOWED_PARTICIPANT_ROLES = {"ministerial", "chair", "ordinary_member", "collective_or_unidentified"}


def _read_csv(s3, logical_key: str) -> tuple[pd.DataFrame, str]:
    resolved = resolve_production_key(s3, bucket=BUCKET, production_key=logical_key)
    obj = s3.get_object(Bucket=BUCKET, Key=resolved)
    frame = pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str, keep_default_na=False, na_values=[""])
    return frame, resolved


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name=REGION)
    pointer = read_json_required(s3, bucket=BUCKET, key=PRODUCTION_POINTER_KEY)
    batch_id = str(pointer.get("batch_id") or pointer.get("mode") or "unknown")
    contract = load_materialization_contract(CONTRACT_PATH)
    contract_version = int(contract["contract_version"])

    frames = {}
    resolved = {}
    for name, key in KEYS.items():
        frames[name], resolved[name] = _read_csv(s3, key)

    oral_sections = build_oral_question_sections(
        questions=frames["questions"],
        speeches=frames["speeches"],
        debate_sections=frames["sections"],
        member_offices=frames["offices"],
        source_batch_id=batch_id,
        contract_version=contract_version,
    )
    exchange_participants = build_oral_question_exchange_participants(
        questions=frames["questions"],
        speeches=frames["speeches"],
        member_offices=frames["offices"],
        source_batch_id=batch_id,
        contract_version=contract_version,
    )
    speech_context = build_speech_question_context(
        speeches=frames["speeches"],
        questions=frames["questions"],
        source_batch_id=batch_id,
        contract_version=contract_version,
    )

    section_errors = validate_materialized_frame(
        oral_sections,
        get_dataset_contract(contract, "oral_question_sections"),
        expected_source_batch_id=batch_id,
    )
    participant_errors = validate_materialized_frame(
        exchange_participants,
        get_dataset_contract(contract, "oral_question_exchange_participants"),
        expected_source_batch_id=batch_id,
    )
    context_errors = validate_materialized_frame(
        speech_context,
        get_dataset_contract(contract, "speech_question_context"),
        expected_source_batch_id=batch_id,
    )

    questions = frames["questions"].copy()
    qtype = questions["question_type"].fillna("").astype(str).str.strip().str.lower()
    oral = questions[qtype.eq("oral")].copy()
    written = questions[qtype.eq("written")].copy()

    expected_sections = set(oral["debate_section_id"].dropna().astype(str))
    actual_sections = set(oral_sections["debate_section_id"].astype(str))
    if actual_sections != expected_sections:
        section_errors.append(
            f"oral section identity mismatch: expected {len(expected_sections)}, found {len(actual_sections)}"
        )

    speeches = frames["speeches"].copy()
    speeches["speech_id"] = speeches["speech_id"].astype(str)
    speeches["debate_section_id"] = speeches["debate_section_id"].fillna("").astype(str)
    expected_related_frame = speeches[speeches["debate_section_id"].isin(expected_sections)].copy()
    expected_related = set(expected_related_frame["speech_id"])
    related_mask = _truthy(speech_context["is_oral_question_related"])
    actual_related = set(speech_context.loc[related_mask, "speech_id"].astype(str))
    if actual_related != expected_related:
        context_errors.append(
            f"related speech identity mismatch: expected {len(expected_related)}, found {len(actual_related)}"
        )

    written_sections = set(written["debate_section_id"].dropna().astype(str))
    written_only = written_sections - expected_sections
    overlap = actual_sections & written_only
    if overlap:
        section_errors.append(f"{len(overlap)} written-only sections incorrectly materialized as oral sections")

    source_words = pd.to_numeric(expected_related_frame["word_count"], errors="coerce")
    if source_words.isna().any():
        participant_errors.append(f"{int(source_words.isna().sum())} oral-exchange source speeches have invalid word_count")
    else:
        expected_word_total = int(source_words.sum())
        participant_interventions = int(pd.to_numeric(exchange_participants["intervention_count"], errors="coerce").fillna(0).sum())
        participant_words = int(pd.to_numeric(exchange_participants["word_count"], errors="coerce").fillna(0).sum())
        section_interventions = int(pd.to_numeric(oral_sections["related_speech_count"], errors="coerce").fillna(0).sum())
        section_words = int(pd.to_numeric(oral_sections["related_speech_word_count"], errors="coerce").fillna(0).sum())
        expected_interventions = len(expected_related)
        if participant_interventions != expected_interventions:
            participant_errors.append(
                f"participant intervention reconciliation mismatch: expected {expected_interventions}, found {participant_interventions}"
            )
        if participant_words != expected_word_total:
            participant_errors.append(
                f"participant word reconciliation mismatch: expected {expected_word_total}, found {participant_words}"
            )
        if section_interventions != expected_interventions:
            section_errors.append(
                f"section intervention reconciliation mismatch: expected {expected_interventions}, found {section_interventions}"
            )
        if section_words != expected_word_total:
            section_errors.append(f"section word reconciliation mismatch: expected {expected_word_total}, found {section_words}")

    roles = set(exchange_participants["participant_role"].dropna().astype(str))
    unexpected_roles = sorted(roles - ALLOWED_PARTICIPANT_ROLES)
    if unexpected_roles:
        participant_errors.append(f"unexpected participant roles: {unexpected_roles}")

    collective = exchange_participants[exchange_participants["participant_role"].eq("collective_or_unidentified")]
    bad_collective = collective["member_code"].fillna("").astype(str).str.strip().ne("")
    if bad_collective.any():
        participant_errors.append(
            f"{int(bad_collective.sum())} collective_or_unidentified participant rows incorrectly contain member_code"
        )

    role_intervention_cols = [
        "ministerial_intervention_count",
        "chair_intervention_count",
        "ordinary_member_intervention_count",
        "collective_or_unidentified_intervention_count",
    ]
    role_word_cols = [
        "ministerial_word_count",
        "chair_word_count",
        "ordinary_member_word_count",
        "collective_or_unidentified_word_count",
    ]
    role_interventions = sum(
        int(pd.to_numeric(oral_sections[col], errors="coerce").fillna(0).sum()) for col in role_intervention_cols
    )
    role_words = sum(int(pd.to_numeric(oral_sections[col], errors="coerce").fillna(0).sum()) for col in role_word_cols)
    if role_interventions != len(expected_related):
        section_errors.append(
            f"section role intervention partition mismatch: expected {len(expected_related)}, found {role_interventions}"
        )
    if not source_words.isna().any() and role_words != int(source_words.sum()):
        section_errors.append(f"section role word partition mismatch: expected {int(source_words.sum())}, found {role_words}")

    all_errors = section_errors + participant_errors + context_errors
    report = {
        "status": "pass" if not all_errors else "fail",
        "production_batch_id": batch_id,
        "resolved_source_keys": resolved,
        "question_classifier_calls": 0,
        "question_taker_attribution_materialized": False,
        "source_counts": {
            "question_records": int(len(questions)),
            "oral_question_records": int(len(oral)),
            "written_question_records": int(len(written)),
            "speech_interventions": int(speeches["speech_id"].nunique()),
        },
        "derived_counts": {
            "oral_question_sections": int(len(oral_sections)),
            "oral_question_exchange_participant_rows": int(len(exchange_participants)),
            "oral_question_related_speeches": int(len(actual_related)),
            "oral_question_related_words": int(source_words.sum()) if not source_words.isna().any() else None,
            "other_speeches": int((~related_mask).sum()),
            "collective_or_unidentified_participant_rows": int(len(collective)),
        },
        "validation": {
            "oral_question_sections": section_errors,
            "oral_question_exchange_participants": participant_errors,
            "speech_question_context": context_errors,
        },
    }
    path = OUT_DIR / "question_context_audit.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_path = OUT_DIR / "question_context_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Parliamentary question relationship audit",
                "",
                f"Status: **{report['status']}**",
                f"Production batch: `{batch_id}`",
                f"Oral question records: **{len(oral):,}**",
                f"Oral question sections: **{len(oral_sections):,}**",
                f"Exchange participant-role rows: **{len(exchange_participants):,}**",
                f"Speech interventions linked to oral-question sections: **{len(actual_related):,}**",
                f"Question-classifier calls: **0**",
                "",
                "Question counts represent submitted Oireachtas question records. Oral-question sections, observed participant-role rows, and transcript interventions are audited separately. Participant rows do not assert who formally took a question.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
