from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from process.oireachtas_speech_issue_classifier import (
    ISSUE_CATEGORIES,
    OUTPUT_COLUMNS,
    ParsedBatchResult,
    build_batch_requests,
    build_compatibility_output,
    candidate_staleness_reasons,
    combine_batch_results,
    evaluate_predictions,
    materialize_batch_rows,
    merge_results,
    parse_batch_output_jsonl,
    reconcile_results_to_source,
    select_delta,
    structured_response_body,
    validate_candidate,
    validate_label,
)


def speech_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "speech_id": "s1",
                "speech_text_hash": "h1",
                "speech_text": "A substantive speech about hospital capacity and waiting lists.",
                "debate_date": "2026-01-01",
                "speaker_name": "Member One",
                "speech_order": 1,
            },
            {
                "speech_id": "s2",
                "speech_text_hash": "h2",
                "speech_text": "A substantive speech about schools and teacher recruitment.",
                "debate_date": "2026-01-01",
                "speaker_name": "Member Two",
                "speech_order": 2,
            },
        ]
    )


def label_row(
    speech_id: str,
    speech_hash: str,
    label: str = "Health",
    status: str = "classified",
) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in OUTPUT_COLUMNS}
    row.update(
        {
            "speech_id": speech_id,
            "speech_text_hash": speech_hash,
            "issue_label": label,
            "classification_status": status,
            "model_name": "test-model",
            "prompt_version": "test-prompt",
            "taxonomy_version": "test-taxonomy",
            "classified_at_utc": "2026-01-01T00:00:00Z",
            "source_batch_id": "batch-1",
            "source_batch_speech_key": "batch/silver_speeches.parquet",
            "classification_run_id": "run-1",
            "review_status": "unreviewed",
            "attempt_count": 1,
            "retry_eligible_after_utc": "",
        }
    )
    return row


def test_taxonomy_has_exact_expected_size_and_none() -> None:
    assert len(ISSUE_CATEGORIES) == 25
    assert len(set(ISSUE_CATEGORIES)) == 25
    assert ISSUE_CATEGORIES[-1] == "NONE"


def test_validate_label_accepts_only_exact_taxonomy_values() -> None:
    assert validate_label("Health") == "Health"
    with pytest.raises(ValueError, match="Invalid issue label"):
        validate_label("health")
    with pytest.raises(ValueError, match="Invalid issue label"):
        validate_label("not-a-category")


def test_select_delta_returns_new_changed_failed_and_forced_rows() -> None:
    speeches = pd.concat(
        [
            speech_rows(),
            pd.DataFrame(
                [
                    {"speech_id": "s3", "speech_text_hash": "h3", "speech_text": "new"},
                    {"speech_id": "s4", "speech_text_hash": "h4", "speech_text": "failed"},
                ]
            ),
        ],
        ignore_index=True,
    )
    failed = label_row("s4", "h4", "", "failed")
    failed["retry_eligible_after_utc"] = "2026-01-01T00:00:00Z"
    existing = pd.DataFrame(
        [
            label_row("s1", "h1"),
            label_row("s2", "h2-old", "Education"),
            failed,
        ]
    )

    delta = select_delta(
        speeches,
        existing,
        force_speech_ids=["s1"],
        now_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    assert delta["speech_id"].tolist() == ["s1", "s2", "s3", "s4"]
    assert delta["selection_reason"].tolist() == [
        "forced",
        "changed_hash",
        "new",
        "retry_failed",
    ]


def test_select_delta_empty_when_all_hashes_are_current() -> None:
    speeches = speech_rows()
    existing = pd.DataFrame([label_row("s1", "h1"), label_row("s2", "h2", "Education")])
    assert select_delta(speeches, existing).empty


def test_select_delta_rejects_duplicate_source_ids() -> None:
    speeches = pd.concat([speech_rows(), speech_rows().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate speech_id"):
        select_delta(speeches, pd.DataFrame())


def test_failed_retry_waits_until_eligible_time() -> None:
    failed = label_row("s1", "h1", "", "failed")
    failed["attempt_count"] = 1
    failed["retry_eligible_after_utc"] = "2026-02-01T00:00:00Z"

    before = select_delta(
        speech_rows().iloc[[0]],
        pd.DataFrame([failed]),
        now_utc=datetime(2026, 1, 31, tzinfo=timezone.utc),
    )
    after = select_delta(
        speech_rows().iloc[[0]],
        pd.DataFrame([failed]),
        now_utc=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )

    assert before.empty
    assert after["selection_reason"].tolist() == ["retry_failed"]
    assert after["prior_attempt_count"].tolist() == [1]


def test_failed_retry_stops_at_retry_limit_but_force_overrides() -> None:
    failed = label_row("s1", "h1", "", "failed")
    failed["attempt_count"] = 3
    failed["retry_eligible_after_utc"] = "2026-01-01T00:00:00Z"
    existing = pd.DataFrame([failed])
    now = datetime(2026, 1, 2, tzinfo=timezone.utc)

    assert select_delta(
        speech_rows().iloc[[0]],
        existing,
        max_retries=3,
        now_utc=now,
    ).empty
    forced = select_delta(
        speech_rows().iloc[[0]],
        existing,
        max_retries=3,
        force_speech_ids=["s1"],
        now_utc=now,
    )
    assert forced["selection_reason"].tolist() == ["forced"]


def test_merge_results_is_idempotent_and_replaces_changed_hash() -> None:
    existing = pd.DataFrame([label_row("s1", "old")])
    new = pd.DataFrame([label_row("s1", "new", "Education")])

    once = merge_results(existing, new)
    twice = merge_results(once, new)

    assert len(twice) == 1
    assert twice.iloc[0]["speech_text_hash"] == "new"
    assert twice.iloc[0]["issue_label"] == "Education"


def test_merge_results_rejects_duplicate_incoming_ids() -> None:
    duplicate = pd.DataFrame([label_row("s1", "h1"), label_row("s1", "h1")])
    with pytest.raises(ValueError, match="duplicate speech_id"):
        merge_results(pd.DataFrame(), duplicate)


def test_reconcile_results_prunes_removed_and_old_hash_rows() -> None:
    existing = pd.DataFrame(
        [
            label_row("s1", "h1"),
            label_row("s2", "old-hash", "Education"),
            label_row("removed", "h3"),
        ]
    )
    new = pd.DataFrame([label_row("s2", "h2", "Education")])

    reconciled = reconcile_results_to_source(existing, new, speech_rows())

    assert reconciled["speech_id"].tolist() == ["s1", "s2"]
    assert reconciled.set_index("speech_id").loc["s2", "speech_text_hash"] == "h2"


def test_batch_request_uses_responses_structured_output_schema() -> None:
    selection, requests = build_batch_requests(speech_rows().iloc[[0]], model="test-model")

    assert len(selection) == 1
    assert requests[0]["url"] == "/v1/responses"
    assert requests[0]["body"] == structured_response_body(
        model="test-model",
        speech_text=speech_rows().iloc[0]["speech_text"],
    )
    schema = requests[0]["body"]["text"]["format"]
    assert schema["strict"] is True
    assert schema["schema"]["properties"]["issue_label"]["enum"] == ISSUE_CATEGORIES


def test_parse_batch_output_accepts_valid_structured_response() -> None:
    payload = {
        "custom_id": "speech-1",
        "response": {
            "status_code": 200,
            "body": {
                "id": "resp_1",
                "output_text": json.dumps({"issue_label": "Health"}),
                "usage": {"input_tokens": 100, "output_tokens": 5},
            },
        },
        "error": None,
    }

    result = parse_batch_output_jsonl((json.dumps(payload) + "\n").encode())[0]

    assert result.status == "classified"
    assert result.label == "Health"
    assert result.response_id == "resp_1"
    assert result.input_tokens == 100


def test_parse_batch_output_marks_invalid_label_failed() -> None:
    payload = {
        "custom_id": "speech-1",
        "response": {
            "status_code": 200,
            "body": {"id": "resp_1", "output_text": json.dumps({"issue_label": "Invalid"})},
        },
    }
    result = parse_batch_output_jsonl((json.dumps(payload) + "\n").encode())[0]
    assert result.status == "failed"
    assert "Invalid issue label" in result.error


def test_parse_batch_output_rejects_duplicate_custom_ids() -> None:
    line = json.dumps({"custom_id": "same", "error": {"message": "failed"}})
    with pytest.raises(ValueError, match="duplicate custom_id"):
        parse_batch_output_jsonl(f"{line}\n{line}\n".encode())


def test_output_and_error_files_cannot_overlap_custom_ids() -> None:
    output = [ParsedBatchResult("same", "classified", "Health")]
    errors = [ParsedBatchResult("same", "failed", error="error")]
    with pytest.raises(ValueError, match="overlapping custom_id"):
        combine_batch_results(output, errors)


def test_materialize_batch_rows_marks_missing_result_failed_and_schedules_retry() -> None:
    selection, _ = build_batch_requests(speech_rows(), model="test-model")
    selection["prior_attempt_count"] = [0, 1]
    results = [ParsedBatchResult(selection.iloc[0]["custom_id"], "classified", "Health")]

    before = datetime.now(timezone.utc)
    rows = materialize_batch_rows(
        selection,
        results,
        model="test-model",
        source_batch_id="batch-1",
        source_key="batch/silver_speeches.parquet",
        run_id="run-1",
        openai_batch_id="batch_openai_1",
        retry_delay_hours=24,
    )

    assert rows["classification_status"].tolist() == ["classified", "failed"]
    assert rows["attempt_count"].tolist() == [1, 2]
    assert rows.iloc[0]["retry_eligible_after_utc"] == ""
    assert "Missing result" in rows.iloc[1]["classification_error"]
    retry_after = datetime.fromisoformat(
        rows.iloc[1]["retry_eligible_after_utc"].replace("Z", "+00:00")
    )
    assert retry_after >= before + timedelta(hours=23, minutes=59)


def test_candidate_validation_rejects_invalid_label_hash_mismatch_and_failures() -> None:
    speeches = speech_rows()
    candidate = pd.DataFrame(
        [
            label_row("s1", "wrong-hash", "Invalid"),
            label_row("s2", "h2", "", "failed"),
        ]
    )

    report = validate_candidate(candidate, speeches=speeches, new_rows=candidate, max_failure_rate=0.0)

    assert report["dq_status"] == "fail"
    failed_checks = {check["check_name"] for check in report["checks"] if check["status"] == "fail"}
    assert {"classified_labels_valid", "source_hash_matches", "failure_rate_acceptable"} <= failed_checks


def test_candidate_validation_marks_incomplete_source_as_partial() -> None:
    candidate = pd.DataFrame([label_row("s1", "h1")])

    report = validate_candidate(
        candidate,
        speeches=speech_rows(),
        new_rows=candidate,
        max_failure_rate=0.0,
    )

    assert report["dq_status"] == "fail"
    assert report["candidate_status"] == "validated_partial"
    coverage = next(
        check for check in report["checks"] if check["check_name"] == "source_coverage_complete"
    )
    assert coverage["metric_value"]["missing_count"] == 1


def test_candidate_validation_passes_empty_delta_against_valid_current_table() -> None:
    speeches = speech_rows()
    candidate = pd.DataFrame([label_row("s1", "h1"), label_row("s2", "h2", "Education")])
    report = validate_candidate(
        candidate,
        speeches=speeches,
        new_rows=pd.DataFrame(columns=OUTPUT_COLUMNS),
        max_failure_rate=0.0,
    )
    assert report["dq_status"] == "pass"
    assert report["candidate_status"] == "validated"
    assert report["failure_rate"] == 0.0


def test_staleness_reasons_detect_source_and_classification_pointer_changes() -> None:
    manifest = {
        "source_batch_id": "batch-1",
        "source_batch_speech_key": "batch-1/silver.parquet",
        "base_classification_run_id": "labels-1",
        "base_classification_table_key": "labels-1/table.parquet",
    }

    assert candidate_staleness_reasons(
        manifest=manifest,
        active_source_batch_id="batch-1",
        active_source_key="batch-1/silver.parquet",
        current_classification_pointer={
            "run_id": "labels-1",
            "table_parquet_key": "labels-1/table.parquet",
        },
    ) == []

    assert candidate_staleness_reasons(
        manifest=manifest,
        active_source_batch_id="batch-2",
        active_source_key="batch-2/silver.parquet",
        current_classification_pointer={
            "run_id": "labels-2",
            "table_parquet_key": "labels-2/table.parquet",
        },
    ) == [
        "active_source_batch_changed",
        "active_source_key_changed",
        "classification_pointer_changed",
    ]


def test_compatibility_output_uses_new_enrichment_labels() -> None:
    compat = build_compatibility_output(
        speech_rows(),
        pd.DataFrame([label_row("s1", "h1", "Health"), label_row("s2", "h2", "Education")]),
    )
    assert compat["PoliticalIssues"].tolist() == ["Health", "Education"]
    assert compat["classification_status"].tolist() == ["classified", "classified"]


def test_evaluation_metrics_include_none_and_invalid_output() -> None:
    metrics = evaluate_predictions(
        pd.DataFrame(
            [
                {"expected_issue_label": "NONE", "predicted_issue_label": "NONE"},
                {"expected_issue_label": "Health", "predicted_issue_label": "Health"},
                {"expected_issue_label": "Education", "predicted_issue_label": "Invalid"},
            ]
        )
    )
    assert metrics["overall_accuracy"] == pytest.approx(2 / 3)
    assert metrics["none_precision"] == 1.0
    assert metrics["none_recall"] == 1.0
    assert metrics["invalid_output_rate"] == pytest.approx(1 / 3)
