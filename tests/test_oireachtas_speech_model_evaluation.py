from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from process.oireachtas_speech_model_evaluation import (
    build_evaluation_requests,
    compare_results,
    evaluate_model_result,
    fixture_report,
    load_fixture,
    pairwise_agreement,
    prepare_evaluation_files,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/oireachtas_speech_issue_evaluation_v1.csv"


def approved_fixture(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "speech_text": "Hospital waiting lists and staffing are the dominant issue.",
                "expected_issue_label": "Health",
                "review_status": "approved",
                "reviewer": "reviewer-1",
                "reviewed_at_utc": "2026-08-05T00:00:00Z",
                "review_notes": "Clear health example.",
            },
            {
                "sample_id": "s2",
                "speech_text": "Question put and agreed to.",
                "expected_issue_label": "NONE",
                "review_status": "approved",
                "reviewer": "reviewer-1",
                "reviewed_at_utc": "2026-08-05T00:00:00Z",
                "review_notes": "Procedural example.",
            },
        ]
    )
    path = tmp_path / "fixture.csv"
    frame.to_csv(path, index=False)
    return path


def model_results(tmp_path: Path, model: str, predictions: list[str]) -> Path:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "model_name": model,
                "predicted_issue_label": predictions[0],
                "classification_status": "classified",
                "input_tokens": 20,
                "output_tokens": 4,
                "latency_seconds": 10,
                "batch_id": f"batch-{model}",
                "batch_status": "completed",
                "error": "",
            },
            {
                "sample_id": "s2",
                "model_name": model,
                "predicted_issue_label": predictions[1],
                "classification_status": "classified",
                "input_tokens": 10,
                "output_tokens": 3,
                "latency_seconds": 10,
                "batch_id": f"batch-{model}",
                "batch_status": "completed",
                "error": "",
            },
        ]
    )
    path = tmp_path / f"{model}.csv"
    frame.to_csv(path, index=False)
    return path


def test_seed_fixture_is_valid_but_not_approved() -> None:
    report = fixture_report(FIXTURE)
    assert report["rows"] >= 25
    assert report["fully_reviewed"] is False
    assert report["missing_categories"] == []
    with pytest.raises(ValueError, match="requires every fixture row to be approved"):
        load_fixture(FIXTURE, require_reviewed=True)


def test_approved_fixture_passes_paid_run_gate(tmp_path: Path) -> None:
    fixture = approved_fixture(tmp_path)
    frame = load_fixture(fixture, require_reviewed=True)
    assert len(frame) == 2


def test_prepare_requests_uses_same_fixture_and_prompt_for_two_models(tmp_path: Path) -> None:
    fixture = approved_fixture(tmp_path)
    report = prepare_evaluation_files(
        fixture_path=fixture,
        models=["model-a", "model-b"],
        output_dir=tmp_path / "out",
        require_reviewed=True,
    )
    assert report["rows"] == 2
    assert len(report["models"]) == 2
    assert all(item["request_rows"] == 2 for item in report["models"])
    assert Path(report["manifest_path"]).exists()

    fixture_frame = load_fixture(fixture, require_reviewed=True)
    _, left = build_evaluation_requests(fixture_frame, model="model-a")
    _, right = build_evaluation_requests(fixture_frame, model="model-b")
    for left_request, right_request in zip(left, right):
        left_body = dict(left_request["body"])
        right_body = dict(right_request["body"])
        left_body.pop("model")
        right_body.pop("model")
        assert left_body == right_body


def test_model_metrics_include_accuracy_none_tokens_latency_and_cost(tmp_path: Path) -> None:
    fixture = load_fixture(approved_fixture(tmp_path), require_reviewed=True)
    result = pd.read_csv(model_results(tmp_path, "model-a", ["Health", "NONE"]))
    metrics = evaluate_model_result(
        fixture,
        result,
        input_price_per_million=1.0,
        output_price_per_million=2.0,
        backfill_rows=1000,
        recurring_rows=100,
    )
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["none_precision"] == 1.0
    assert metrics["none_recall"] == 1.0
    assert metrics["batch_completion_success"] is True
    assert metrics["input_tokens_total"] == 30
    assert metrics["output_tokens_total"] == 7
    assert metrics["latency_seconds"]["p95"] == 10.0
    assert metrics["cost"]["estimated_backfill_cost"] is not None


def test_pairwise_agreement_and_comparison_never_auto_select(tmp_path: Path) -> None:
    fixture = approved_fixture(tmp_path)
    left_path = model_results(tmp_path, "model-a", ["Health", "NONE"])
    right_path = model_results(tmp_path, "model-b", ["Health", "Health"])
    left = pd.read_csv(left_path)
    right = pd.read_csv(right_path)

    agreement = pairwise_agreement([left, right])
    assert agreement[0]["agreement_rate"] == 0.5

    report = compare_results(
        fixture_path=fixture,
        result_paths=[left_path, right_path],
        pricing_path=None,
        backfill_rows=0,
        recurring_rows=0,
        require_reviewed=True,
    )
    assert report["selection_status"] == "requires_human_approval"
    assert report["selected_model"] is None
    assert len(report["models"]) == 2


def test_invalid_output_and_failed_batch_are_visible(tmp_path: Path) -> None:
    fixture = load_fixture(approved_fixture(tmp_path), require_reviewed=True)
    result = pd.read_csv(model_results(tmp_path, "model-a", ["Invalid", "NONE"]))
    result.loc[0, "classification_status"] = "failed"
    result.loc[0, "batch_status"] = "failed"

    metrics = evaluate_model_result(fixture, result)

    assert metrics["invalid_output_rate"] == 0.5
    assert metrics["classification_failure_rate"] == 0.5
    assert metrics["batch_completion_success"] is False


def test_prepared_manifest_records_checksums(tmp_path: Path) -> None:
    report = prepare_evaluation_files(
        fixture_path=approved_fixture(tmp_path),
        models=["model-a", "model-b"],
        output_dir=tmp_path / "out",
        require_reviewed=True,
    )
    manifest = json.loads(Path(report["manifest_path"]).read_text())
    assert manifest["fixture_sha256"]
    assert all(item["mapping_sha256"] for item in manifest["models"])
    assert all(item["requests_sha256"] for item in manifest["models"])
