from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI

from process.oireachtas_speech_issue_classifier import (
    ISSUE_CATEGORIES,
    SILVER_SPEECHES_KEY,
    build_classifier_prompt,
    canonicalize_label,
    classification_schema,
    make_s3_client,
    read_s3_csv,
)


def classify_control(client: OpenAI, text: str, model: str) -> str:
    response = client.responses.create(
        model=model,
        input=build_classifier_prompt(text),
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
        raise ValueError(f"Invalid control label: {payload!r}")
    return label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GPT-4.1 mini against Luna historical benchmark sample")
    parser.add_argument("--luna-report", default="diagnostics/speech_classifier/latest_luna_benchmark.json")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", "eirepolitic-data"))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "ca-central-1"))
    parser.add_argument("--report-path", default="speech_issue_model_control.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")

    luna = json.loads(Path(args.luna_report).read_text(encoding="utf-8"))
    benchmark_rows = luna.get("results") or []
    speech_ids = [row["speech_id"] for row in benchmark_rows]
    expected_by_id = {row["speech_id"]: row["expected_legacy_label"] for row in benchmark_rows}
    luna_by_id = {row["speech_id"]: row["luna_label"] for row in benchmark_rows}

    s3 = make_s3_client(region_name=args.region)
    silver = read_s3_csv(s3, bucket=args.bucket, key=SILVER_SPEECHES_KEY)
    silver = silver[silver["speech_id"].isin(speech_ids)].copy()
    silver = silver.set_index("speech_id", drop=False)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    results = []
    failures = []
    expected_match = 0
    luna_match = 0
    all_three_match = 0
    luna_disagreements = 0
    control_sides_with_legacy = 0
    control_sides_with_luna = 0
    control_differs_from_both = 0

    for speech_id in speech_ids:
        if speech_id not in silver.index:
            failures.append({"speech_id": speech_id, "error": "speech_id not found in silver_speeches"})
            continue
        row = silver.loc[speech_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        expected = expected_by_id[speech_id]
        luna_label = luna_by_id[speech_id]
        try:
            control_label = classify_control(client, str(row["speech_text"]), args.model)
        except Exception as exc:
            failures.append({"speech_id": speech_id, "error_type": type(exc).__name__, "error": str(exc)[:1000]})
            continue

        if control_label == expected:
            expected_match += 1
        if control_label == luna_label:
            luna_match += 1
        if expected == luna_label == control_label:
            all_three_match += 1
        if luna_label != expected:
            luna_disagreements += 1
            if control_label == expected:
                control_sides_with_legacy += 1
            elif control_label == luna_label:
                control_sides_with_luna += 1
            else:
                control_differs_from_both += 1

        results.append(
            {
                "speech_id": speech_id,
                "debate_date": str(row.get("debate_date", "")),
                "speaker_name": str(row.get("speaker_name", "")),
                "expected_legacy_label": expected,
                "luna_label": luna_label,
                "control_label": control_label,
                "control_matches_legacy": control_label == expected,
                "control_matches_luna": control_label == luna_label,
                "speech_excerpt": str(row.get("speech_text", ""))[:500],
            }
        )

    succeeded = len(results)
    report = {
        "mode": "model_control_benchmark",
        "control_model": args.model,
        "luna_model": luna.get("model"),
        "sample_requested": len(speech_ids),
        "sample_succeeded": succeeded,
        "sample_failed": len(failures),
        "legacy_exact_agreement_pct": round(expected_match / succeeded * 100, 1) if succeeded else 0.0,
        "control_luna_agreement_pct": round(luna_match / succeeded * 100, 1) if succeeded else 0.0,
        "all_three_exact_pct": round(all_three_match / succeeded * 100, 1) if succeeded else 0.0,
        "luna_disagreement_rows": luna_disagreements,
        "on_luna_disagreements_control_matches_legacy": control_sides_with_legacy,
        "on_luna_disagreements_control_matches_luna": control_sides_with_luna,
        "on_luna_disagreements_control_differs_from_both": control_differs_from_both,
        "writes_performed": False,
        "results": results,
        "failures": failures,
    }
    Path(args.report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
