from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

import process.oireachtas_speech_classifier_trigger_guard as guard_module
from process.oireachtas_speech_classifier_trigger_guard import inspect_automatic_trigger
from process.oireachtas_speech_issue_classifier import OUTPUT_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_WORKFLOW = ROOT / ".github/workflows/oireachtas_speech_issue_classifier_v2.yml"
ORCHESTRATOR_WORKFLOW = ROOT / ".github/workflows/oireachtas_refresh_validation_orchestrator.yml"


def source_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "speech_id": "s1",
                "speech_text_hash": "h1",
                "speech_text": "A substantive health policy speech with enough detail for classification.",
            },
            {
                "speech_id": "s2",
                "speech_text_hash": "h2",
                "speech_text": "A substantive education policy speech with enough detail for classification.",
            },
        ]
    )


def label_rows(*, source_batch_id: str = "core-1") -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for speech_id, speech_hash, label in (
        ("s1", "h1", "Health"),
        ("s2", "h2", "Education"),
    ):
        row: dict[str, Any] = {column: "" for column in OUTPUT_COLUMNS}
        row.update(
            {
                "speech_id": speech_id,
                "speech_text_hash": speech_hash,
                "issue_label": label,
                "classification_status": "classified",
                "model_name": "test-model",
                "prompt_version": "test-prompt",
                "taxonomy_version": "legacy-25-v1",
                "classified_at_utc": "2026-08-05T00:00:00Z",
                "source_batch_id": source_batch_id,
                "source_batch_speech_key": "batch/source.parquet",
                "classification_run_id": "labels-1",
                "review_status": "unreviewed",
                "attempt_count": 1,
            }
        )
        output.append(row)
    return pd.DataFrame(output, columns=OUTPUT_COLUMNS)


def patch_guard_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    active_batch_id: str,
    speeches: pd.DataFrame,
    pointer: dict[str, Any] | None,
    existing: pd.DataFrame,
) -> None:
    monkeypatch.setattr(
        guard_module,
        "read_source_context",
        lambda _s3, *, bucket: (active_batch_id, "batch/source.parquet", speeches),
    )
    monkeypatch.setattr(
        guard_module,
        "read_current_enrichment",
        lambda _s3, *, bucket: (pointer, existing),
    )


def test_trigger_guard_rejects_promoted_batch_that_is_no_longer_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_guard_sources(
        monkeypatch,
        active_batch_id="core-2",
        speeches=source_rows(),
        pointer=None,
        existing=pd.DataFrame(columns=OUTPUT_COLUMNS),
    )

    with pytest.raises(RuntimeError, match="no longer active"):
        inspect_automatic_trigger(
            s3=object(),
            bucket="test",
            expected_source_batch_id="core-1",
        )


def test_trigger_guard_skips_when_active_batch_is_fully_classified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_guard_sources(
        monkeypatch,
        active_batch_id="core-1",
        speeches=source_rows(),
        pointer={
            "run_id": "labels-1",
            "table_parquet_key": "labels/table.parquet",
            "source_batch_id": "core-1",
        },
        existing=label_rows(),
    )

    result = inspect_automatic_trigger(
        s3=object(),
        bucket="test",
        expected_source_batch_id="core-1",
    )

    assert result["should_prepare"] is False
    assert result["reason"] == "no_classification_work"
    assert result["delta_rows"] == 0


def test_trigger_guard_prepares_new_or_changed_speech_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = source_rows()
    changed.loc[changed["speech_id"] == "s2", "speech_text_hash"] = "h2-new"
    patch_guard_sources(
        monkeypatch,
        active_batch_id="core-1",
        speeches=changed,
        pointer={
            "run_id": "labels-1",
            "table_parquet_key": "labels/table.parquet",
            "source_batch_id": "core-1",
        },
        existing=label_rows(),
    )

    result = inspect_automatic_trigger(
        s3=object(),
        bucket="test",
        expected_source_batch_id="core-1",
    )

    assert result["should_prepare"] is True
    assert result["reason"] == "speech_delta_present"
    assert result["delta_rows"] == 1


def test_trigger_guard_prepares_source_batch_maintenance_even_without_paid_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_guard_sources(
        monkeypatch,
        active_batch_id="core-2",
        speeches=source_rows(),
        pointer={
            "run_id": "labels-1",
            "table_parquet_key": "labels/table.parquet",
            "source_batch_id": "core-1",
        },
        existing=label_rows(source_batch_id="core-1"),
    )

    result = inspect_automatic_trigger(
        s3=object(),
        bucket="test",
        expected_source_batch_id="core-2",
    )

    assert result["should_prepare"] is True
    assert result["reason"] == "source_batch_changed"
    assert result["delta_rows"] == 0


def load_workflow(path: Path) -> dict[str, Any]:
    parsed = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(parsed, dict)
    assert isinstance(parsed.get("jobs"), dict)
    return parsed


def test_workflow_yaml_files_parse() -> None:
    classifier = load_workflow(CLASSIFIER_WORKFLOW)
    orchestrator = load_workflow(ORCHESTRATOR_WORKFLOW)
    assert classifier["name"] == "Oireachtas Speech Issue Classifier v2"
    assert orchestrator["name"] == "Oireachtas Refresh Validation Orchestrator"


def test_classifier_has_no_broad_workflow_run_trigger() -> None:
    data = load_workflow(CLASSIFIER_WORKFLOW)
    triggers = data["on"]
    assert "workflow_dispatch" in triggers
    assert "workflow_run" not in triggers


def test_classifier_requires_exact_source_batch_guard_for_automatic_preparation() -> None:
    text = CLASSIFIER_WORKFLOW.read_text(encoding="utf-8")
    assert "expected_source_batch_id" in text
    assert "oireachtas_speech_classifier_trigger_guard.py" in text
    assert "--expected-source-batch-id" in text
    assert "steps.trigger_guard.outputs.should_prepare == 'true'" in text
    assert "Automatic post-refresh backfill is prohibited" in text
    assert "Automatic post-refresh forced reclassification is prohibited" in text


def test_classifier_switches_are_independent() -> None:
    text = CLASSIFIER_WORKFLOW.read_text(encoding="utf-8")
    required_switches = {
        "OIREACHTAS_SPEECH_CLASSIFIER_ENABLED",
        "OIREACHTAS_SPEECH_CLASSIFIER_CANDIDATE_ENABLED",
        "OIREACHTAS_SPEECH_CLASSIFIER_AUTOMATIC_SUBMIT_ENABLED",
        "OIREACHTAS_SPEECH_CLASSIFIER_PUBLISH_ENABLED",
        "OIREACHTAS_SPEECH_CLASSIFIER_BACKFILL_ENABLED",
    }
    assert all(name in text for name in required_switches)
    assert "publish_latest confirmation is required" in text
    assert "Backfill switch is disabled" in text


def test_orchestrator_dispatches_only_after_pointer_verification() -> None:
    data = load_workflow(ORCHESTRATOR_WORKFLOW)
    steps = data["jobs"]["promote"]["steps"]
    names = [step.get("name", "") for step in steps]
    verify_index = names.index("Verify production pointer")
    dispatch_index = names.index("Dispatch independent speech-classification candidate")
    assert dispatch_index > verify_index

    dispatch = steps[dispatch_index]
    assert dispatch["continue-on-error"] == "true"
    assert "CLASSIFIER_ENABLED" in dispatch["if"]
    assert "CLASSIFIER_CANDIDATE_ENABLED" in dispatch["if"]
    assert "expected_source_batch_id=\"${BATCH_ID}\"" in dispatch["run"]
    assert "operation=prepare" in dispatch["run"]
    assert "publish_latest=false" in dispatch["run"]


def test_orchestrator_dispatch_failure_cannot_trigger_core_rollback() -> None:
    data = load_workflow(ORCHESTRATOR_WORKFLOW)
    steps = data["jobs"]["promote"]["steps"]
    rollback = next(
        step
        for step in steps
        if step.get("name") == "Roll back after failed pointer verification"
    )
    dispatch = next(
        step
        for step in steps
        if step.get("name") == "Dispatch independent speech-classification candidate"
    )
    assert dispatch["continue-on-error"] == "true"
    assert rollback["if"] == "failure() && steps.promote.outcome == 'success'"


def test_orchestrator_grants_only_required_dispatch_permission() -> None:
    data = load_workflow(ORCHESTRATOR_WORKFLOW)
    assert data["permissions"] == {"contents": "read", "actions": "write"}
