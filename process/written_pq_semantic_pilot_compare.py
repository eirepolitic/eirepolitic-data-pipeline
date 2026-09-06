#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FULL = Path("analysis/written_pq_semantic_pilot")
CHEAP = Path("analysis/written_pq_semantic_pilot_cheap")


def _parse(path: Path) -> dict[str, dict]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {str(r.debate_section_id): json.loads(r.classification_json) for r in df.itertuples(index=False)}


def _jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main() -> int:
    full = _parse(FULL / "classifications.csv")
    cheap = _parse(CHEAP / "classifications.csv")
    shared = sorted(set(full) & set(cheap))
    rows = []
    for sid in shared:
        a, b = full[sid], cheap[sid]
        aq = {str(x["question_id"]): x for x in a.get("questions", [])}
        bq = {str(x["question_id"]): x for x in b.get("questions", [])}
        qids = sorted(set(aq) & set(bq))
        q_topic = [_jaccard(aq[q].get("topic_tags", []), bq[q].get("topic_tags", [])) for q in qids]
        q_intent = [_jaccard(aq[q].get("question_intents", []), bq[q].get("question_intents", [])) for q in qids]
        rows.append({
            "debate_section_id": sid,
            "question_topic_jaccard": sum(q_topic)/len(q_topic) if q_topic else None,
            "question_intent_jaccard": sum(q_intent)/len(q_intent) if q_intent else None,
            "answer_topic_jaccard": _jaccard(a.get("answer", {}).get("topic_tags", []), b.get("answer", {}).get("topic_tags", [])),
            "answer_characteristic_jaccard": _jaccard(a.get("answer", {}).get("answer_characteristics", []), b.get("answer", {}).get("answer_characteristics", [])),
            "combined_topic_jaccard": _jaccard(a.get("combined_exchange", {}).get("topic_tags", []), b.get("combined_exchange", {}).get("topic_tags", [])),
        })
    df = pd.DataFrame(rows)
    df.to_csv(CHEAP / "comparison_to_v2.csv", index=False)
    full_summary = json.loads((FULL / "summary.json").read_text())
    cheap_summary = json.loads((CHEAP / "summary.json").read_text())
    metrics = ["question_topic_jaccard","question_intent_jaccard","answer_topic_jaccard","answer_characteristic_jaccard","combined_topic_jaccard"]
    summary = {
        "shared_sections": len(shared),
        "agreement": {m: round(float(df[m].mean()),4) if not df.empty else None for m in metrics},
        "full_v2_total_tokens": int(full_summary["usage"]["total_tokens"]),
        "cheap_total_tokens": int(cheap_summary["usage"]["total_tokens"]),
        "full_v2_avg_tokens_per_section": full_summary.get("average_tokens_per_successful_section"),
        "cheap_avg_tokens_per_section": cheap_summary.get("average_tokens_per_successful_section"),
        "token_reduction_pct": round(100*(1-int(cheap_summary["usage"]["total_tokens"])/int(full_summary["usage"]["total_tokens"])),2),
        "cheap_failed_sections": int(cheap_summary["failed_sections"]),
        "cheap_validation_error_sections": int(cheap_summary["validation_error_sections"]),
        "note": "Agreement is model-vs-model on a 25-section research sample, not accuracy against human labels.",
    }
    (CHEAP / "comparison_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,indent=2,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
