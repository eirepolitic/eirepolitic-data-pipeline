#!/usr/bin/env python3
"""Audit deterministic parliamentary-question / speech relationships against production.

No writes to S3 are performed. This verifies that oral question records, oral-question
sections, and transcript speech interventions can be related without double counting.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import boto3
import pandas as pd

from extract.oireachtas.batch import PRODUCTION_POINTER_KEY, read_json_required, resolve_production_key
from political_metrics.materialize import get_dataset_contract, load_materialization_contract, validate_materialized_frame
from political_metrics.question_context import build_oral_question_sections, build_speech_question_context

BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")
REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ca-central-1"
OUT_DIR = Path(os.getenv("POLITICAL_METRICS_AUDIT_DIR", "artifacts/political-metrics-audit"))
CONTRACT_PATH = Path(__file__).resolve().parents[1] / "configs/political_metrics/materialization.yml"

KEYS = {
    "questions": "processed/oireachtas_unified/latest/csv/silver_questions.csv",
    "speeches": "processed/oireachtas_unified/latest/csv/silver_speeches.csv",
    "sections": "processed/oireachtas_unified/latest/csv/silver_debate_sections.csv",
}


def _read_csv(s3, logical_key: str) -> tuple[pd.DataFrame, str]:
    resolved = resolve_production_key(s3, bucket=BUCKET, production_key=logical_key)
    obj = s3.get_object(Bucket=BUCKET, Key=resolved)
    frame = pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str, keep_default_na=False, na_values=[""])
    return frame, resolved


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
    expected_related = set(
        speeches.loc[speeches["debate_section_id"].astype(str).isin(expected_sections), "speech_id"].astype(str)
    )
    actual_related = set(
        speech_context.loc[
            speech_context["is_oral_question_related"].astype(str).str.lower().eq("true"), "speech_id"
        ].astype(str)
    )
    if actual_related != expected_related:
        context_errors.append(
            f"related speech identity mismatch: expected {len(expected_related)}, found {len(actual_related)}"
        )

    written_sections = set(written["debate_section_id"].dropna().astype(str))
    written_only = written_sections - expected_sections
    overlap = actual_sections & written_only
    if overlap:
        section_errors.append(f"{len(overlap)} written-only sections incorrectly materialized as oral sections")

    report = {
        "status": "pass" if not section_errors and not context_errors else "fail",
        "production_batch_id": batch_id,
        "resolved_source_keys": resolved,
        "question_classifier_calls": 0,
        "source_counts": {
            "question_records": int(len(questions)),
            "oral_question_records": int(len(oral)),
            "written_question_records": int(len(written)),
            "speech_interventions": int(speeches["speech_id"].nunique()),
        },
        "derived_counts": {
            "oral_question_sections": int(len(oral_sections)),
            "oral_question_related_speeches": int(len(actual_related)),
            "other_speeches": int((~speech_context["is_oral_question_related"].astype(bool)).sum()),
        },
        "validation": {
            "oral_question_sections": section_errors,
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
                f"Speech interventions linked to oral-question sections: **{len(actual_related):,}**",
                f"Question-classifier calls: **0**",
                "",
                "Question counts represent submitted Oireachtas question records. Oral-question sections and their transcript speech interventions are audited separately.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
