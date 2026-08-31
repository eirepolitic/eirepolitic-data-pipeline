from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI

from process.oireachtas_speech_issue_classifier import (
    SILVER_SPEECHES_KEY,
    build_classifier_prompt,
    canonicalize_label,
    classification_schema,
    make_s3_client,
    read_s3_csv,
)


def classify_medium(client: OpenAI, text: str, model: str) -> str:
    response = client.responses.create(
        model=model,
        input=build_classifier_prompt(text),
        reasoning={"effort": "medium"},
        text={
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "speech_issue_classification",
                "strict": True,
                "schema": classification_schema(),
            },
        },
        max_output_tokens=128,
        store=False,
    )
    payload = json.loads(str(response.output_text or "").strip())
    label = canonicalize_label(payload.get("issue_label"))
    if not label:
        raise ValueError(f"Invalid Luna label: {payload!r}")
    return label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-test Luna disagreement cases with medium reasoning")
    parser.add_argument("--baseline-report", default="diagnostics/speech_classifier/latest_luna_benchmark.json")
    parser.add_argument("--control-report", default="diagnostics/speech_classifier/latest_model_control.json")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", "eirepolitic-data"))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "ca-central-1"))
    parser.add_argument("--report-path", default="speech_issue_luna_medium_benchmark.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")

    baseline = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
    control = json.loads(Path(args.control_report).read_text(encoding="utf-8"))
    control_by_id = {row["speech_id"]: row["control_label"] for row in control.get("results") or []}
    disagreements = [row for row in baseline.get("results") or [] if not row.get("exact_match")]
    speech_ids = [row["speech_id"] for row in disagreements]

    s3 = make_s3_client(region_name=args.region)
    silver = read_s3_csv(s3, bucket=args.bucket, key=SILVER_SPEECHES_KEY)
    silver = silver[silver["speech_id"].isin(speech_ids)].set_index("speech_id", drop=False)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    results = []
    failures = []
    medium_matches_legacy = 0
    medium_matches_control = 0
    medium_matches_baseline = 0
    improved_vs_baseline = 0

    for prior in disagreements:
        speech_id = prior["speech_id"]
        if speech_id not in silver.index:
            failures.append({"speech_id": speech_id, "error": "speech missing from silver"})
            continue
        row = silver.loc[speech_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        try:
            medium = classify_medium(client, str(row["speech_text"]), args.model)
        except Exception as exc:
            failures.append({"speech_id": speech_id, "error_type": type(exc).__name__, "error": str(exc)[:1000]})
            continue
        legacy = prior["expected_legacy_label"]
        baseline_label = prior["luna_label"]
        control_label = control_by_id.get(speech_id, "")
        medium_matches_legacy += int(medium == legacy)
        medium_matches_control += int(bool(control_label) and medium == control_label)
        medium_matches_baseline += int(medium == baseline_label)
        improved_vs_baseline += int(medium == legacy and baseline_label != legacy)
        results.append({
            "speech_id": speech_id,
            "debate_date": str(row.get("debate_date", "")),
            "legacy_label": legacy,
            "baseline_luna_low": baseline_label,
            "luna_medium": medium,
            "control_gpt_4_1_mini": control_label,
            "medium_matches_legacy": medium == legacy,
            "medium_matches_control": bool(control_label) and medium == control_label,
            "speech_excerpt": str(row.get("speech_text", ""))[:500],
        })

    succeeded = len(results)
    report = {
        "mode": "luna_medium_disagreement_benchmark",
        "model": args.model,
        "reasoning_effort": "medium",
        "sample_requested": len(disagreements),
        "sample_succeeded": succeeded,
        "sample_failed": len(failures),
        "medium_matches_legacy": medium_matches_legacy,
        "medium_legacy_agreement_pct_on_disagreements": round(medium_matches_legacy / succeeded * 100, 1) if succeeded else 0.0,
        "medium_matches_control": medium_matches_control,
        "medium_control_agreement_pct_on_disagreements": round(medium_matches_control / succeeded * 100, 1) if succeeded else 0.0,
        "medium_matches_baseline_low": medium_matches_baseline,
        "medium_changed_from_low": succeeded - medium_matches_baseline,
        "legacy_agreement_improvements_vs_low": improved_vs_baseline,
        "writes_performed": False,
        "results": results,
        "failures": failures,
    }
    Path(args.report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
