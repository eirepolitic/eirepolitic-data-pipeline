from __future__ import annotations

import pandas as pd

from process.oireachtas_speech_issue_classifier import (
    ClassificationResult,
    canonicalize_label,
    merge_results,
    select_delta,
)


def test_canonicalize_label_is_case_insensitive() -> None:
    assert canonicalize_label("health") == "Health"
    assert canonicalize_label("not-a-category") is None


def test_select_delta_returns_new_and_changed_speeches() -> None:
    speeches = pd.DataFrame(
        [
            {"speech_id": "s1", "speech_text_hash": "h1", "speech_text": "one"},
            {"speech_id": "s2", "speech_text_hash": "h2-new", "speech_text": "two"},
            {"speech_id": "s3", "speech_text_hash": "h3", "speech_text": "three"},
        ]
    )
    existing = pd.DataFrame(
        [
            {"speech_id": "s1", "speech_text_hash": "h1", "classification_status": "classified"},
            {"speech_id": "s2", "speech_text_hash": "h2-old", "classification_status": "classified"},
        ]
    )

    delta = select_delta(speeches, existing)

    assert delta["speech_id"].tolist() == ["s2", "s3"]


def test_select_delta_retries_failed_rows() -> None:
    speeches = pd.DataFrame(
        [{"speech_id": "s1", "speech_text_hash": "h1", "speech_text": "one"}]
    )
    existing = pd.DataFrame(
        [{"speech_id": "s1", "speech_text_hash": "h1", "classification_status": "failed"}]
    )

    assert select_delta(speeches, existing)["speech_id"].tolist() == ["s1"]


def test_merge_results_replaces_old_version_by_speech_id() -> None:
    existing = pd.DataFrame(
        [
            {
                "speech_id": "s1",
                "speech_text_hash": "old",
                "issue_label": "Health",
                "classification_status": "classified",
            }
        ]
    )
    new = pd.DataFrame(
        [
            {
                "speech_id": "s1",
                "speech_text_hash": "new",
                "issue_label": "Education",
                "classification_status": "classified",
            }
        ]
    )

    merged = merge_results(existing, new)

    assert len(merged) == 1
    assert merged.iloc[0]["speech_text_hash"] == "new"
    assert merged.iloc[0]["issue_label"] == "Education"


def test_classification_result_defaults_are_safe() -> None:
    result = ClassificationResult(label="NONE")
    assert result.response_id == ""
    assert result.input_tokens is None
    assert result.output_tokens is None
