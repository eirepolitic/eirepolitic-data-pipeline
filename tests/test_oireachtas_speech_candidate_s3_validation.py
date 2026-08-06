from __future__ import annotations

import io
import json
from typing import Any

import pytest

from process.oireachtas_speech_candidate_s3_validation import (
    POINTER_KEYS,
    snapshot_pointers,
    verify_candidate_only_run,
)
from process.oireachtas_speech_issue_classifier import (
    PREVIOUS_CLASSIFICATION_POINTER_KEY,
    PRODUCTION_CLASSIFICATION_POINTER_KEY,
    RUN_ROOT,
    run_manifest_key,
    sha256_bytes,
)

BUCKET = "test-bucket"
RUN_ID = "candidate-test"


class FakeBody(io.BytesIO):
    pass


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_json(self, key: str, payload: dict[str, Any]) -> None:
        self.objects[key] = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")

    def put_bytes(self, key: str, payload: bytes) -> None:
        self.objects[key] = payload

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if Key not in self.objects:
            from botocore.exceptions import ClientError

            raise ClientError(
                {
                    "Error": {"Code": "NoSuchKey", "Message": "missing"},
                    "ResponseMetadata": {"HTTPStatusCode": 404},
                },
                "GetObject",
            )
        return {"Body": FakeBody(self.objects[Key])}


def seed_candidate(s3: FakeS3, *, status: str = "prepared") -> dict[str, Any]:
    prefix = f"{RUN_ROOT}/run_id={RUN_ID}"
    artifacts = {
        "selection_csv": (f"{prefix}/selection.csv", b"speech_id\ns1\n"),
        "selection_parquet": (f"{prefix}/selection.parquet", b"parquet"),
        "openai_requests_jsonl": (f"{prefix}/openai_requests.jsonl", b'{"custom_id":"speech-1"}\n'),
        "deterministic_results_csv": (f"{prefix}/deterministic_results.csv", b"speech_id\n"),
    }
    checksums: dict[str, Any] = {}
    for name, (key, payload) in artifacts.items():
        s3.put_bytes(key, payload)
        checksums[name] = {
            "key": key,
            "sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
        }
    manifest = {
        "table": "enrichment_speech_issue_labels",
        "run_id": RUN_ID,
        "status": status,
        "manifest_key": run_manifest_key(RUN_ID),
        "source_batch_id": "core-1",
        "source_batch_speech_key": "batches/core-1/silver_speeches.parquet",
        "model_name": "candidate-model",
        "source_rows": 100,
        "existing_rows": 90,
        "full_delta_rows": 10,
        "delta_rows_selected": 10,
        "deterministic_none_rows": 2,
        "batch_request_rows": 8,
        "delta_truncated": False,
        "maintenance_needed": False,
        "selection_csv_key": artifacts["selection_csv"][0],
        "selection_parquet_key": artifacts["selection_parquet"][0],
        "requests_jsonl_key": artifacts["openai_requests_jsonl"][0],
        "deterministic_results_key": artifacts["deterministic_results_csv"][0],
        "artifact_checksums": checksums,
        "published": False,
        "openai_batch_id": "",
        "batch_submission_attempts": 0,
    }
    s3.put_json(run_manifest_key(RUN_ID), manifest)
    return manifest


def seed_pointers(s3: FakeS3) -> None:
    s3.put_json(
        PRODUCTION_CLASSIFICATION_POINTER_KEY,
        {"run_id": "production-1", "table_parquet_key": "production/table.parquet"},
    )
    s3.put_json(
        PREVIOUS_CLASSIFICATION_POINTER_KEY,
        {"run_id": "production-0", "table_parquet_key": "previous/table.parquet"},
    )


def test_snapshot_records_both_pointer_values_and_hashes() -> None:
    s3 = FakeS3()
    seed_pointers(s3)

    snapshot = snapshot_pointers(s3=s3, bucket=BUCKET)

    assert snapshot["pointer_keys"] == POINTER_KEYS
    assert snapshot["pointers"]["production"]["run_id"] == "production-1"
    assert snapshot["pointers"]["previous"]["run_id"] == "production-0"
    assert len(snapshot["pointer_sha256"]["production"]) == 64
    assert len(snapshot["pointer_sha256"]["previous"]) == 64


def test_verify_candidate_only_run_passes_without_pointer_mutation() -> None:
    s3 = FakeS3()
    seed_pointers(s3)
    before = snapshot_pointers(s3=s3, bucket=BUCKET)
    seed_candidate(s3)

    report = verify_candidate_only_run(
        s3=s3,
        bucket=BUCKET,
        run_id=RUN_ID,
        before_snapshot=before,
    )

    assert report["status"] == "pass"
    assert report["production_pointer_unchanged"] is True
    assert report["previous_pointer_unchanged"] is True
    assert report["openai_batch_submitted"] is False
    assert len(report["artifact_checks"]) == 4
    assert all(item["status"] == "pass" for item in report["artifact_checks"])


def test_verify_rejects_changed_production_pointer() -> None:
    s3 = FakeS3()
    seed_pointers(s3)
    before = snapshot_pointers(s3=s3, bucket=BUCKET)
    seed_candidate(s3)
    s3.put_json(
        PRODUCTION_CLASSIFICATION_POINTER_KEY,
        {"run_id": "unexpected", "table_parquet_key": "unexpected/table.parquet"},
    )

    with pytest.raises(RuntimeError, match="changed a classification pointer"):
        verify_candidate_only_run(
            s3=s3,
            bucket=BUCKET,
            run_id=RUN_ID,
            before_snapshot=before,
        )


def test_verify_rejects_openai_submission_metadata() -> None:
    s3 = FakeS3()
    seed_pointers(s3)
    before = snapshot_pointers(s3=s3, bucket=BUCKET)
    manifest = seed_candidate(s3)
    manifest["openai_batch_id"] = "batch-paid"
    manifest["batch_submission_attempts"] = 1
    s3.put_json(run_manifest_key(RUN_ID), manifest)

    with pytest.raises(RuntimeError, match="submitted an OpenAI Batch"):
        verify_candidate_only_run(
            s3=s3,
            bucket=BUCKET,
            run_id=RUN_ID,
            before_snapshot=before,
        )


def test_verify_rejects_checksum_corruption() -> None:
    s3 = FakeS3()
    seed_pointers(s3)
    before = snapshot_pointers(s3=s3, bucket=BUCKET)
    manifest = seed_candidate(s3)
    request_key = manifest["requests_jsonl_key"]
    s3.put_bytes(request_key, b"corrupt")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_candidate_only_run(
            s3=s3,
            bucket=BUCKET,
            run_id=RUN_ID,
            before_snapshot=before,
        )


def test_verify_rejects_collected_or_published_artifacts() -> None:
    s3 = FakeS3()
    seed_pointers(s3)
    before = snapshot_pointers(s3=s3, bucket=BUCKET)
    manifest = seed_candidate(s3)
    manifest["table_parquet_key"] = f"{RUN_ROOT}/run_id={RUN_ID}/table.parquet"
    s3.put_json(run_manifest_key(RUN_ID), manifest)

    with pytest.raises(RuntimeError, match="unexpectedly contains collected artifact"):
        verify_candidate_only_run(
            s3=s3,
            bucket=BUCKET,
            run_id=RUN_ID,
            before_snapshot=before,
        )


def test_verify_rejects_artifact_outside_immutable_run_prefix() -> None:
    s3 = FakeS3()
    seed_pointers(s3)
    before = snapshot_pointers(s3=s3, bucket=BUCKET)
    manifest = seed_candidate(s3)
    manifest["selection_csv_key"] = "processed/latest/selection.csv"
    s3.put_json(run_manifest_key(RUN_ID), manifest)

    with pytest.raises(RuntimeError, match="escaped immutable run prefix"):
        verify_candidate_only_run(
            s3=s3,
            bucket=BUCKET,
            run_id=RUN_ID,
            before_snapshot=before,
        )
