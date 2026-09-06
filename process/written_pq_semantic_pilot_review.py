#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path("analysis/written_pq_semantic_pilot")


def _tag_sets(result: dict, scope: str) -> set[str]:
    if scope == "question":
        tags: set[str] = set()
        for q in result.get("questions", []):
            tags.update(q.get("topic_tags", []))
        return tags
    if scope == "answer":
        return set(result.get("answer", {}).get("topic_tags", []))
    return set(result.get("combined_exchange", {}).get("topic_tags", []))


def main() -> int:
    frame = pd.read_csv(ROOT / "classifications.csv", dtype=str, keep_default_na=False)
    parsed = [json.loads(value) for value in frame["classification_json"]]
    validation = frame[frame["validation_errors"].astype(str).str.strip().ne("")].copy()

    comparison_rows = []
    for row, result in zip(frame.to_dict("records"), parsed):
        question = _tag_sets(result, "question")
        answer = _tag_sets(result, "answer")
        combined = _tag_sets(result, "combined")
        union = question | answer
        comparison_rows.append(
            {
                "debate_section_id": row["debate_section_id"],
                "question_tags": sorted(question),
                "answer_tags": sorted(answer),
                "combined_tags": sorted(combined),
                "question_answer_same": question == answer,
                "combined_equals_union": combined == union,
                "combined_missing_from_union": sorted(combined - union),
                "union_missing_from_combined": sorted(union - combined),
            }
        )

    counts: dict[str, int] = {}
    for value in validation["validation_errors"].tolist():
        for item in str(value).split(";"):
            if item:
                counts[item] = counts.get(item, 0) + 1

    comparison = pd.DataFrame(comparison_rows)
    review = {
        "sections": int(len(frame)),
        "validation_error_sections": int(len(validation)),
        "validation_error_counts": dict(sorted(counts.items(), key=lambda x: (-x[1], x[0]))),
        "validation_error_rows": validation[
            ["debate_section_id", "question_date", "recipient", "answer_chars", "validation_errors"]
        ].to_dict("records"),
        "question_answer_same_tag_set_sections": int(comparison["question_answer_same"].sum()),
        "question_answer_different_tag_set_sections": int((~comparison["question_answer_same"]).sum()),
        "combined_equals_question_answer_union_sections": int(comparison["combined_equals_union"].sum()),
        "combined_differs_from_union_sections": int((~comparison["combined_equals_union"]).sum()),
        "examples_question_answer_differ": comparison.loc[
            ~comparison["question_answer_same"],
            ["debate_section_id", "question_tags", "answer_tags", "combined_tags"],
        ].head(10).to_dict("records"),
    }
    (ROOT / "review_summary.json").write_text(
        json.dumps(review, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(review, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
