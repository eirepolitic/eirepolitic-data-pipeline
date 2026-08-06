from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from extract.oireachtas.enrichment_contracts import (
    EnrichmentContractError,
    assert_valid_enrichment_manifest,
    assert_valid_publish_contract,
    get_enrichment_contract,
    load_enrichment_registry,
    load_enrichment_write_policies,
    validate_enrichment_manifest,
    validate_enrichment_table,
    validate_publish_contract,
)
from process.oireachtas_speech_issue_classifier import OUTPUT_COLUMNS, TABLE_NAME

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_WORKFLOW = ROOT / ".github/workflows/oireachtas_speech_issue_classifier_v2.yml"
CORE_TABLE_REGISTRY = ROOT / "configs/oireachtas/tables.yml"


def valid_row(speech_id: str = "s1", label: str = "Health") -> dict[str, Any]:
    row: dict[str, Any] = {column: "" for column in OUTPUT_COLUMNS}
    row.update(
        {
            "speech_id": speech_id,
            "speech_text_hash": f"hash-{speech_id}",
            "issue_label": label,
            "classification_status": "classified",
            "model_name": "test-model",
            "prompt_version": "speech-issue-v2.1",
            "taxonomy_version": "legacy-25-v1",
            "classified_at_utc": "2026-08-05T00:00:00Z",
            "input_tokens": 100,
            "output_tokens": 5,
            "source_batch_id": "core-1",
            "source_batch_speech_key": "batches/core-1/silver_speeches.parquet",
            "classification_run_id": "run-1",
            "openai_response_id": "resp-1",
            "openai_batch_id": "batch-1",
            "review_status": "unreviewed",
            "classification_error": "",
            "attempt_count": 1,
            "retry_eligible_after_utc": "",
        }
    )
    return row


def valid_manifest() -> dict[str, Any]:
    return {
        "table": TABLE_NAME,
        "run_id": "run-1",
        "status": "validated",
        "created_at_utc": "2026-08-05T00:00:00Z",
        "updated_at_utc": "2026-08-05T01:00:00Z",
        "source_batch_id": "core-1",
        "source_batch_speech_key": "batches/core-1/silver_speeches.parquet",
        "base_classification_run_id": "",
        "base_classification_table_key": "",
        "model_name": "test-model",
        "prompt_version": "speech-issue-v2.1",
        "taxonomy_version": "legacy-25-v1",
        "historical_backfill": False,
        "source_rows": 1,
        "existing_rows": 0,
        "full_delta_rows": 1,
        "delta_rows_selected": 1,
        "delta_truncated": False,
        "maintenance_needed": False,
        "deterministic_none_rows": 0,
        "batch_request_rows": 1,
        "max_rows": 25,
        "max_retries": 3,
        "short_speech_word_limit": 20,
        "selection_csv_key": "runs/run-1/selection.csv",
        "selection_parquet_key": "runs/run-1/selection.parquet",
        "requests_jsonl_key": "runs/run-1/requests.jsonl",
        "deterministic_results_key": "runs/run-1/deterministic.csv",
        "manifest_key": "runs/run-1/manifest.json",
        "artifact_checksums": {"candidate_parquet": {"sha256": "abc"}},
        "batch_submission_attempts": 1,
        "published": False,
        "candidate_validation_status": "validated",
        "dq_status": "pass",
        "stale_reasons": [],
        "table_csv_key": "runs/run-1/table.csv",
        "table_parquet_key": "runs/run-1/table.parquet",
        "compat_csv_key": "runs/run-1/compat.csv",
        "compat_parquet_key": "runs/run-1/compat.parquet",
        "dq_key": "runs/run-1/dq.json",
    }


def test_enrichment_registry_is_separate_from_core_registry() -> None:
    enrichments = load_enrichment_registry()
    core = yaml.safe_load(CORE_TABLE_REGISTRY.read_text(encoding="utf-8"))

    assert TABLE_NAME in enrichments
    assert TABLE_NAME not in core["tables"]
    assert enrichments[TABLE_NAME]["core_refresh_required"] is False
    assert enrichments[TABLE_NAME]["source_table"] == "silver_speeches"


def test_write_policy_requires_independent_atomic_publication() -> None:
    policies = load_enrichment_write_policies()
    policy = policies[TABLE_NAME]

    assert policy["storage_mode"] == "immutable_run"
    assert policy["candidate_generation_required"] is True
    assert policy["validation_required"] is True
    assert policy["atomic_pointer_publish_required"] is True
    assert policy["previous_pointer_required"] is True
    assert policy["allow_core_transaction_coupling"] is False
    assert policy["allow_partial_publication"] is False
    assert policy["allow_stale_publication"] is False


def test_registry_columns_match_classifier_output_columns() -> None:
    contract, _ = get_enrichment_contract(TABLE_NAME)
    assert contract["columns"] == OUTPUT_COLUMNS
    assert contract["required_columns"] == OUTPUT_COLUMNS


def test_valid_manifest_and_table_pass_contracts() -> None:
    manifest = valid_manifest()
    frame = pd.DataFrame([valid_row()])

    assert validate_enrichment_manifest(
        TABLE_NAME,
        manifest,
        require_candidate_artifacts=True,
    ) == []
    assert validate_enrichment_table(TABLE_NAME, frame) == []
    assert validate_publish_contract(TABLE_NAME, manifest, frame) == []
    assert_valid_publish_contract(TABLE_NAME, manifest, frame)


def test_manifest_rejects_unknown_status_and_missing_fields() -> None:
    manifest = valid_manifest()
    manifest["status"] = "made-up"
    del manifest["source_batch_id"]

    errors = validate_enrichment_manifest(TABLE_NAME, manifest)

    assert "manifest missing field: source_batch_id" in errors
    assert "manifest status is not allowed: 'made-up'" in errors
    with pytest.raises(EnrichmentContractError):
        assert_valid_enrichment_manifest(TABLE_NAME, manifest)


def test_table_rejects_duplicate_ids_invalid_label_and_blank_required_values() -> None:
    first = valid_row("s1", "Invalid")
    second = valid_row("s1", "Health")
    second["model_name"] = ""
    frame = pd.DataFrame([first, second])

    errors = validate_enrichment_table(TABLE_NAME, frame)

    assert any("duplicate primary key rows" in error for error in errors)
    assert any("invalid conditional values in issue_label" in error for error in errors)
    assert any("blank required values in model_name" in error for error in errors)


def test_failed_rows_may_have_blank_issue_label() -> None:
    row = valid_row()
    row["classification_status"] = "failed"
    row["issue_label"] = ""
    row["classification_error"] = "temporary failure"

    assert validate_enrichment_table(TABLE_NAME, pd.DataFrame([row])) == []


def test_publish_contract_rejects_partial_stale_or_failed_dq() -> None:
    frame = pd.DataFrame([valid_row()])

    partial = deepcopy(valid_manifest())
    partial["status"] = "validated_partial"
    partial["candidate_validation_status"] = "validated_partial"
    partial_errors = validate_publish_contract(TABLE_NAME, partial, frame)
    assert any("not publishable" in error for error in partial_errors)
    assert "write policy rejects partial publication" in partial_errors

    stale = deepcopy(valid_manifest())
    stale["stale_reasons"] = ["active_source_batch_changed"]
    assert "write policy rejects stale publication" in validate_publish_contract(
        TABLE_NAME,
        stale,
        frame,
    )

    failed = deepcopy(valid_manifest())
    failed["dq_status"] = "fail"
    assert "write policy requires dq_status=pass" in validate_publish_contract(
        TABLE_NAME,
        failed,
        frame,
    )


def test_unknown_enrichment_contract_is_rejected() -> None:
    with pytest.raises(EnrichmentContractError, match="Unknown enrichment table"):
        get_enrichment_contract("not_a_table")


def test_workflow_enforces_contract_after_prepare_collect_and_before_publish() -> None:
    text = CLASSIFIER_WORKFLOW.read_text(encoding="utf-8")

    assert text.count("oireachtas_validate_speech_enrichment.py") == 3
    assert "Validate prepared manifest contract" in text
    assert "Collect and validate candidate" in text
    assert "Atomically publish validated candidate" in text
    publish_section = text.split("- name: Atomically publish validated candidate", 1)[1]
    assert publish_section.index("oireachtas_validate_speech_enrichment.py") < publish_section.index(
        "oireachtas_speech_issue_classifier.py publish"
    )
