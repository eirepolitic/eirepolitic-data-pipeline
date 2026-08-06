from __future__ import annotations

import hashlib
import io
import json
from typing import Any

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from extract.oireachtas.batch import (
    PRODUCTION_POINTER_KEY,
    batch_key_for_production_key,
)
from process.oireachtas_speech_issue_classifier import (
    OUTPUT_COLUMNS,
    PREVIOUS_CLASSIFICATION_POINTER_KEY,
    PRODUCTION_CLASSIFICATION_POINTER_KEY,
    SOURCE_LOGICAL_KEY,
    collect_run,
    dataframe_artifacts,
    prepare_run,
    publish_run,
    put_json_direct,
    read_json_optional,
    run_manifest_key,
    submit_run,
)

BUCKET = "test-bucket"


class FakeBody(io.BytesIO):
    pass


class FakeS3:
    """Small in-memory S3 double with ETag and conditional-write behavior."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}
        self.denied_get_keys: set[str] = set()
        self.conflict_on_put_keys: set[str] = set()

    @staticmethod
    def _etag(payload: bytes) -> str:
        digest = hashlib.md5(payload, usedforsecurity=False).hexdigest()
        return f'"{digest}"'

    @staticmethod
    def _client_error(code: str, status: int, operation: str) -> ClientError:
        return ClientError(
            {
                "Error": {"Code": code, "Message": code},
                "ResponseMetadata": {"HTTPStatusCode": status},
            },
            operation,
        )

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if Key in self.denied_get_keys:
            raise self._client_error("AccessDenied", 403, "GetObject")
        stored = self.objects.get((Bucket, Key))
        if stored is None:
            raise self._client_error("NoSuchKey", 404, "GetObject")
        return {
            "Body": FakeBody(stored["Body"]),
            "ContentType": stored.get("ContentType"),
            "ETag": stored["ETag"],
        }

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes | bytearray | str,
        ContentType: str | None = None,
        IfMatch: str | None = None,
        IfNoneMatch: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if Key in self.conflict_on_put_keys and (IfMatch or IfNoneMatch):
            raise self._client_error("PreconditionFailed", 412, "PutObject")
        current = self.objects.get((Bucket, Key))
        if IfNoneMatch == "*" and current is not None:
            raise self._client_error("PreconditionFailed", 412, "PutObject")
        if IfMatch is not None and (current is None or current["ETag"] != IfMatch):
            raise self._client_error("PreconditionFailed", 412, "PutObject")
        payload = Body.encode("utf-8") if isinstance(Body, str) else bytes(Body)
        etag = self._etag(payload)
        self.objects[(Bucket, Key)] = {
            "Body": payload,
            "ContentType": ContentType,
            "ETag": etag,
        }
        return {"ETag": etag}

    def payload(self, key: str) -> bytes:
        return self.objects[(BUCKET, key)]["Body"]

    def json(self, key: str) -> dict[str, Any]:
        value = json.loads(self.payload(key).decode("utf-8"))
        assert isinstance(value, dict)
        return value

    def exists(self, key: str) -> bool:
        return (BUCKET, key) in self.objects


class FakeModels:
    def retrieve(self, model: str) -> object:
        return type("Model", (), {"id": model})()


class FakeFiles:
    def __init__(self, content_by_id: dict[str, bytes] | None = None) -> None:
        self.content_by_id = content_by_id or {}
        self.create_called = False

    def content(self, file_id: str) -> FakeBody:
        return FakeBody(self.content_by_id[file_id])

    def create(self, **_: Any) -> object:
        self.create_called = True
        return type("File", (), {"id": "file-input"})()


class FakeBatches:
    def create(self, **_: Any) -> object:
        return type("Batch", (), {"id": "batch-created", "status": "validating"})()


class FakeOpenAI:
    def __init__(self, content_by_id: dict[str, bytes] | None = None) -> None:
        self.models = FakeModels()
        self.files = FakeFiles(content_by_id)
        self.batches = FakeBatches()


def source_rows(*, second_hash: str = "h2") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "speech_id": "s1",
                "speech_text_hash": "h1",
                "speech_text": (
                    "The health service needs additional hospital capacity, staffing, primary care "
                    "investment, waiting-list reductions, and improved access throughout the country."
                ),
                "debate_date": "2026-01-01",
                "speaker_name": "Member One",
                "speech_order": 1,
            },
            {
                "speech_id": "s2",
                "speech_text_hash": second_hash,
                "speech_text": (
                    "Schools require more teachers, modern buildings, special education resources, "
                    "curriculum support, and equitable opportunities for every child in the State."
                ),
                "debate_date": "2026-01-01",
                "speaker_name": "Member Two",
                "speech_order": 2,
            },
        ]
    )


def classification_rows(*, source_batch_id: str = "core-1") -> pd.DataFrame:
    now = "2026-08-05T00:00:00Z"
    rows: list[dict[str, Any]] = []
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
                "model_name": "old-model",
                "prompt_version": "old-prompt",
                "taxonomy_version": "legacy-25-v1",
                "classified_at_utc": now,
                "input_tokens": 10,
                "output_tokens": 2,
                "source_batch_id": source_batch_id,
                "source_batch_speech_key": "old/source.parquet",
                "classification_run_id": "labels-old",
                "review_status": "unreviewed",
                "attempt_count": 1,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def put_dataframe(s3: FakeS3, *, key: str, frame: pd.DataFrame) -> None:
    _, parquet = dataframe_artifacts(frame)
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=parquet,
        ContentType="application/x-parquet",
    )


def seed_core_source(
    s3: FakeS3,
    *,
    batch_id: str = "core-1",
    frame: pd.DataFrame | None = None,
) -> str:
    put_json_direct(
        s3,
        bucket=BUCKET,
        key=PRODUCTION_POINTER_KEY,
        payload={"mode": "batch", "batch_id": batch_id},
    )
    source_key = batch_key_for_production_key(SOURCE_LOGICAL_KEY, batch_id)
    put_dataframe(s3, key=source_key, frame=frame if frame is not None else source_rows())
    return source_key


def seed_classification_pointer(
    s3: FakeS3,
    *,
    source_batch_id: str,
    run_id: str = "labels-old",
) -> dict[str, Any]:
    table_key = f"classification/{run_id}/table.parquet"
    put_dataframe(
        s3,
        key=table_key,
        frame=classification_rows(source_batch_id=source_batch_id),
    )
    pointer = {
        "table": "enrichment_speech_issue_labels",
        "run_id": run_id,
        "table_parquet_key": table_key,
        "table_csv_key": f"classification/{run_id}/table.csv",
        "source_batch_id": source_batch_id,
        "model_name": "old-model",
        "prompt_version": "old-prompt",
        "taxonomy_version": "legacy-25-v1",
    }
    put_json_direct(
        s3,
        bucket=BUCKET,
        key=PRODUCTION_CLASSIFICATION_POINTER_KEY,
        payload=pointer,
    )
    return pointer


def mark_batch_ready(
    s3: FakeS3,
    *,
    run_id: str,
    output_file_id: str = "file-output",
    error_file_id: str = "",
) -> dict[str, Any]:
    manifest = s3.json(run_manifest_key(run_id))
    manifest.update(
        {
            "status": "ready_to_collect",
            "openai_batch_id": "batch-openai-1",
            "openai_output_file_id": output_file_id,
            "openai_error_file_id": error_file_id,
        }
    )
    put_json_direct(
        s3,
        bucket=BUCKET,
        key=run_manifest_key(run_id),
        payload=manifest,
    )
    return manifest


def batch_output_for(frame: pd.DataFrame) -> bytes:
    labels = {"s1": "Health", "s2": "Education"}
    lines = []
    for row in frame.to_dict(orient="records"):
        lines.append(
            json.dumps(
                {
                    "custom_id": row["custom_id"],
                    "response": {
                        "status_code": 200,
                        "body": {
                            "id": f"resp-{row['speech_id']}",
                            "output_text": json.dumps(
                                {"issue_label": labels[str(row["speech_id"])]}
                            ),
                            "usage": {"input_tokens": 30, "output_tokens": 4},
                        },
                    },
                    "error": None,
                }
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def test_prepare_and_collect_are_candidate_only() -> None:
    s3 = FakeS3()
    seed_core_source(s3)

    prepared = prepare_run(
        s3=s3,
        bucket=BUCKET,
        model="test-model",
        max_rows=10,
        historical_backfill=False,
        short_speech_word_limit=0,
        run_id="candidate-only",
    )

    assert prepared["status"] == "prepared"
    assert prepared["delta_rows_selected"] == 2
    assert not s3.exists(PRODUCTION_CLASSIFICATION_POINTER_KEY)
    assert "openai_requests_jsonl" in prepared["artifact_checksums"]

    selection = pd.read_parquet(io.BytesIO(s3.payload(prepared["selection_parquet_key"])))
    mark_batch_ready(s3, run_id="candidate-only")
    collected = collect_run(
        s3=s3,
        bucket=BUCKET,
        client=FakeOpenAI({"file-output": batch_output_for(selection)}),
        run_id="candidate-only",
        max_failure_rate=0.0,
    )

    assert collected["status"] == "validated"
    assert collected["dq_status"] == "pass"
    assert collected["output_rows"] == 2
    assert s3.exists(collected["table_parquet_key"])
    assert s3.exists(collected["compat_parquet_key"])
    assert not s3.exists(PRODUCTION_CLASSIFICATION_POINTER_KEY)
    assert not s3.exists(PREVIOUS_CLASSIFICATION_POINTER_KEY)


def test_zero_delta_is_clean_no_op_and_keeps_pointer_unchanged() -> None:
    s3 = FakeS3()
    seed_core_source(s3)
    original = seed_classification_pointer(s3, source_batch_id="core-1")

    prepared = prepare_run(
        s3=s3,
        bucket=BUCKET,
        model="test-model",
        max_rows=10,
        historical_backfill=False,
        run_id="no-op",
    )

    assert prepared["status"] == "no_op"
    assert prepared["delta_rows_selected"] == 0
    assert s3.json(PRODUCTION_CLASSIFICATION_POINTER_KEY) == original
    assert not s3.exists(PREVIOUS_CLASSIFICATION_POINTER_KEY)


def test_maintenance_run_publishes_atomically_and_records_previous_pointer() -> None:
    s3 = FakeS3()
    seed_core_source(s3, batch_id="core-2")
    old_pointer = seed_classification_pointer(s3, source_batch_id="core-1")

    prepared = prepare_run(
        s3=s3,
        bucket=BUCKET,
        model="test-model",
        max_rows=10,
        historical_backfill=False,
        run_id="maintenance",
    )
    assert prepared["status"] == "ready_to_collect"
    assert prepared["maintenance_needed"] is True
    assert prepared["batch_request_rows"] == 0

    collected = collect_run(
        s3=s3,
        bucket=BUCKET,
        client=None,
        run_id="maintenance",
        max_failure_rate=0.0,
    )
    assert collected["status"] == "validated"

    published = publish_run(
        s3=s3,
        bucket=BUCKET,
        run_id="maintenance",
        publish_enabled=True,
    )

    assert published["status"] == "published"
    current = s3.json(PRODUCTION_CLASSIFICATION_POINTER_KEY)
    previous = s3.json(PREVIOUS_CLASSIFICATION_POINTER_KEY)
    assert current["run_id"] == "maintenance"
    assert current["source_batch_id"] == "core-2"
    assert previous["run_id"] == old_pointer["run_id"]
    assert previous["table_parquet_key"] == old_pointer["table_parquet_key"]
    assert previous["superseded_by_run_id"] == "maintenance"


def test_collect_preserves_stale_candidate_without_publishing() -> None:
    s3 = FakeS3()
    seed_core_source(s3, batch_id="core-1")
    prepared = prepare_run(
        s3=s3,
        bucket=BUCKET,
        model="test-model",
        max_rows=10,
        historical_backfill=False,
        short_speech_word_limit=0,
        run_id="stale-run",
    )
    selection = pd.read_parquet(io.BytesIO(s3.payload(prepared["selection_parquet_key"])))
    mark_batch_ready(s3, run_id="stale-run")

    seed_core_source(s3, batch_id="core-2", frame=source_rows(second_hash="h2-new"))
    collected = collect_run(
        s3=s3,
        bucket=BUCKET,
        client=FakeOpenAI({"file-output": batch_output_for(selection)}),
        run_id="stale-run",
        max_failure_rate=0.0,
    )

    assert collected["status"] == "stale_candidate"
    assert "active_source_batch_changed" in collected["stale_reasons"]
    assert "active_source_key_changed" in collected["stale_reasons"]
    assert s3.exists(collected["table_parquet_key"])
    assert s3.exists(collected["dq_key"])
    assert not s3.exists(PRODUCTION_CLASSIFICATION_POINTER_KEY)
    with pytest.raises(RuntimeError, match="complete validated"):
        publish_run(
            s3=s3,
            bucket=BUCKET,
            run_id="stale-run",
            publish_enabled=True,
        )


def test_corrupted_prepared_requests_are_rejected_before_openai_upload() -> None:
    s3 = FakeS3()
    seed_core_source(s3)
    prepared = prepare_run(
        s3=s3,
        bucket=BUCKET,
        model="test-model",
        max_rows=10,
        historical_backfill=False,
        short_speech_word_limit=0,
        run_id="corrupt-run",
    )
    s3.put_object(
        Bucket=BUCKET,
        Key=prepared["requests_jsonl_key"],
        Body=b"corrupt\n",
        ContentType="application/jsonl",
    )
    client = FakeOpenAI()

    with pytest.raises(ValueError, match="Checksum mismatch"):
        submit_run(
            s3=s3,
            bucket=BUCKET,
            client=client,
            run_id="corrupt-run",
        )

    assert client.files.create_called is False
    assert s3.json(run_manifest_key("corrupt-run"))["batch_submission_attempts"] == 0


def test_permission_failure_is_not_treated_as_missing_pointer() -> None:
    s3 = FakeS3()
    seed_core_source(s3)
    s3.denied_get_keys.add(PRODUCTION_CLASSIFICATION_POINTER_KEY)

    with pytest.raises(ClientError) as exc_info:
        prepare_run(
            s3=s3,
            bucket=BUCKET,
            model="test-model",
            max_rows=10,
            historical_backfill=False,
            run_id="permission-run",
        )

    assert exc_info.value.response["Error"]["Code"] == "AccessDenied"


def test_conditional_publication_conflict_keeps_existing_pointer() -> None:
    s3 = FakeS3()
    seed_core_source(s3, batch_id="core-2")
    old_pointer = seed_classification_pointer(s3, source_batch_id="core-1")
    prepare_run(
        s3=s3,
        bucket=BUCKET,
        model="test-model",
        max_rows=10,
        historical_backfill=False,
        run_id="conflict-run",
    )
    collect_run(
        s3=s3,
        bucket=BUCKET,
        client=None,
        run_id="conflict-run",
        max_failure_rate=0.0,
    )
    s3.conflict_on_put_keys.add(PRODUCTION_CLASSIFICATION_POINTER_KEY)

    with pytest.raises(RuntimeError, match="changed concurrently"):
        publish_run(
            s3=s3,
            bucket=BUCKET,
            run_id="conflict-run",
            publish_enabled=True,
        )

    assert s3.json(PRODUCTION_CLASSIFICATION_POINTER_KEY) == old_pointer
    assert not s3.exists(PREVIOUS_CLASSIFICATION_POINTER_KEY)
    assert s3.json(run_manifest_key("conflict-run"))["published"] is False


def test_optional_json_reader_only_swallows_missing_objects() -> None:
    s3 = FakeS3()
    assert read_json_optional(s3, bucket=BUCKET, key="missing.json") is None

    s3.denied_get_keys.add("denied.json")
    with pytest.raises(ClientError):
        read_json_optional(s3, bucket=BUCKET, key="denied.json")
