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
    canonicalize_label,
    classification_schema,
    make_s3_client,
    read_s3_csv,
)

GUIDANCE = """
Apply these tie-breakers consistently:
- Energy: electricity, fuels, power grids, energy prices, energy supply, data-centre electricity demand, home heating energy, carbon taxes when the speech is mainly about energy/fuel cost or use.
- Environment: climate policy, emissions, pollution, waste/recycling, biodiversity, environmental regulation, conservation. Prefer Energy when the concrete subject is power/fuel infrastructure or energy supply.
- Labor, Employment and Immigration: jobs, unemployment, employment schemes, labour conditions, workforce shortages, apprenticeships primarily discussed as workforce supply, immigration/asylum administration.
- Education: schools, teachers, pupils, universities, student support, educational institutions and training primarily discussed as education provision. Prefer Labor when apprenticeships/training are mainly about labour supply or employment.
- Social Welfare: welfare payments, disability income supports, pensions, social protection schemes and income support. Prefer Health when the core issue is medical care or health services; prefer Culture and Arts when a payment is mainly an arts-sector policy.
- Domestic Terrorism: terrorism law, terrorist offences, proscription of terrorist organisations, terrorist attacks, political violence where terrorism is the core subject. Do not choose it merely because far-right politics, protest, conflict or civil-rights questions are mentioned.
- International Affairs and Foreign Aid: foreign conflicts, diplomacy, international humanitarian issues, overseas aid and relations between states. Prefer Domestic Terrorism only when terrorism/proscription/terrorist offences are the core subject.
- Macroeconomics: national fiscal/economic policy, budgets, inflation, broad economic growth, taxation at economy-wide level and overall economic performance.
- Banking/Finance and Domestic Commerce: banks, financial products, investment funds, competition/consumer markets, firm-level commercial conditions and domestic business regulation. Prefer Macroeconomics for economy-wide fiscal or growth questions.
- Government Operations: operation of the Oireachtas, government administration, public-sector governance, procedural/state machinery and general government accountability where no more specific policy topic dominates.
- Law/Crime and Family Issues: policing, prisons, criminal justice, courts, family law and victim/witness justice processes. Prefer Civil Rights when equality, discrimination or fundamental rights is the dominant issue.
- NONE: use for procedural fragments, greetings, generic political rhetoric, very short context-dependent remarks, or passages with no sufficiently clear single policy topic.

Classify the subject actually expressed in the supplied speech text. Do not assume the wider debate topic when the excerpt itself does not support it.
""".strip()


def prompt(text: str) -> list[dict[str, str]]:
    categories = "\n".join(f"- {category}" for category in ISSUE_CATEGORIES)
    return [
        {
            "role": "system",
            "content": (
                "Classify Irish parliamentary speeches by their single core policy topic. "
                "Choose exactly one allowed issue label. Do not infer party position, sentiment, or importance.\n\n"
                + GUIDANCE
            ),
        },
        {"role": "user", "content": f"Allowed issue labels:\n{categories}\n\nSpeech:\n{text}"},
    ]


def classify(client: OpenAI, text: str, model: str) -> str:
    response = client.responses.create(
        model=model,
        input=prompt(text),
        reasoning={"effort": "low"},
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
    parser = argparse.ArgumentParser(description="Benchmark improved taxonomy guidance for Luna")
    parser.add_argument("--baseline-report", default="diagnostics/speech_classifier/latest_luna_benchmark.json")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", "eirepolitic-data"))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "ca-central-1"))
    parser.add_argument("--report-path", default="speech_issue_luna_prompt_benchmark.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")

    baseline = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
    rows = baseline.get("results") or []
    speech_ids = [row["speech_id"] for row in rows]
    expected = {row["speech_id"]: row["expected_legacy_label"] for row in rows}
    old_luna = {row["speech_id"]: row["luna_label"] for row in rows}

    s3 = make_s3_client(region_name=args.region)
    silver = read_s3_csv(s3, bucket=args.bucket, key=SILVER_SPEECHES_KEY)
    silver = silver[silver["speech_id"].isin(speech_ids)].set_index("speech_id", drop=False)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    results = []
    failures = []
    exact = 0
    old_exact = 0
    changed = 0
    improved = 0
    regressed = 0
    for speech_id in speech_ids:
        if speech_id not in silver.index:
            failures.append({"speech_id": speech_id, "error": "speech missing from silver"})
            continue
        row = silver.loc[speech_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        try:
            label = classify(client, str(row["speech_text"]), args.model)
        except Exception as exc:
            failures.append({"speech_id": speech_id, "error_type": type(exc).__name__, "error": str(exc)[:1000]})
            continue
        exp = expected[speech_id]
        old = old_luna[speech_id]
        is_exact = label == exp
        was_exact = old == exp
        exact += int(is_exact)
        old_exact += int(was_exact)
        changed += int(label != old)
        improved += int((not was_exact) and is_exact)
        regressed += int(was_exact and (not is_exact))
        results.append({
            "speech_id": speech_id,
            "debate_date": str(row.get("debate_date", "")),
            "expected_legacy_label": exp,
            "baseline_luna_label": old,
            "guided_luna_label": label,
            "baseline_exact": was_exact,
            "guided_exact": is_exact,
            "speech_excerpt": str(row.get("speech_text", ""))[:500],
        })

    succeeded = len(results)
    report = {
        "mode": "luna_taxonomy_guidance_benchmark",
        "model": args.model,
        "sample_requested": len(speech_ids),
        "sample_succeeded": succeeded,
        "sample_failed": len(failures),
        "baseline_exact_agreement_pct": round(old_exact / succeeded * 100, 1) if succeeded else 0.0,
        "guided_exact_agreement_pct": round(exact / succeeded * 100, 1) if succeeded else 0.0,
        "labels_changed": changed,
        "legacy_agreement_improvements": improved,
        "legacy_agreement_regressions": regressed,
        "writes_performed": False,
        "results": results,
        "failures": failures,
    }
    Path(args.report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
