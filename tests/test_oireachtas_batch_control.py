from __future__ import annotations

import io
import json
import os
import unittest
from unittest.mock import patch

from extract.oireachtas.batch import (
    PRODUCTION_POINTER_KEY,
    PREVIOUS_POINTER_KEY,
    assemble_batch_manifest,
    batch_entry_key,
    batch_key_for_production_key,
    batch_manifest_key,
    promote_batch,
    rollback_batch,
)
from extract.oireachtas.io_s3 import put_bytes


class FakePaginator:
    def __init__(self, s3: "FakeS3") -> None:
        self.s3 = s3

    def paginate(self, *, Bucket: str, Prefix: str):
        keys = sorted(key for key in self.s3.objects if key.startswith(Prefix))
        yield {"Contents": [{"Key": key} for key in keys]}


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str = "application/octet-stream") -> None:
        self.objects[Key] = bytes(Body)

    def get_object(self, *, Bucket: str, Key: str):
        if Key not in self.objects:
            raise FileNotFoundError(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def head_object(self, *, Bucket: str, Key: str):
        if Key not in self.objects:
            raise FileNotFoundError(Key)
        return {"ContentLength": len(self.objects[Key]), "ETag": '"etag"', "VersionId": "v1"}

    def get_paginator(self, name: str) -> FakePaginator:
        self.assert_name(name)
        return FakePaginator(self)

    @staticmethod
    def assert_name(name: str) -> None:
        if name != "list_objects_v2":
            raise AssertionError(name)


def put_json(s3: FakeS3, key: str, payload: dict) -> None:
    s3.put_object(Bucket="bucket", Key=key, Body=json.dumps(payload).encode("utf-8"), ContentType="application/json")


class BatchKeyTests(unittest.TestCase):
    def test_mutable_latest_maps_to_immutable_batch_key(self) -> None:
        key = batch_key_for_production_key(
            "processed/oireachtas_unified/latest/csv/silver_members.csv",
            "batch-123",
        )
        self.assertEqual(
            key,
            "processed/oireachtas_unified/batches/batch-123/tables/silver_members/csv/silver_members.csv",
        )

    def test_enabled_write_requires_batch_id(self) -> None:
        s3 = FakeS3()
        with patch.dict(
            os.environ,
            {"OIREACHTAS_PUBLISH_ENABLED": "true", "OIREACHTAS_PUBLISH_LATEST": "true"},
            clear=True,
        ):
            with self.assertRaises(RuntimeError):
                put_bytes(
                    s3,
                    bucket="bucket",
                    key="processed/oireachtas_unified/latest/csv/silver_members.csv",
                    body=b"data",
                )

    def test_enabled_write_is_redirected_not_written_to_logical_latest(self) -> None:
        s3 = FakeS3()
        logical = "processed/oireachtas_unified/latest/csv/silver_members.csv"
        with patch.dict(
            os.environ,
            {
                "OIREACHTAS_PUBLISH_ENABLED": "true",
                "OIREACHTAS_PUBLISH_LATEST": "true",
                "OIREACHTAS_BATCH_ID": "batch-123",
            },
            clear=True,
        ):
            put_bytes(s3, bucket="bucket", key=logical, body=b"data")
        self.assertNotIn(logical, s3.objects)
        self.assertIn(batch_key_for_production_key(logical, "batch-123"), s3.objects)


class AtomicPromotionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.s3 = FakeS3()

    def add_entry(self, batch_id: str, table: str, *, status: str = "validated", object_exists: bool = True) -> None:
        logical = f"processed/oireachtas_unified/latest/csv/{table}.csv"
        batch_key = batch_key_for_production_key(logical, batch_id)
        if object_exists:
            self.s3.put_object(Bucket="bucket", Key=batch_key, Body=b"x", ContentType="text/csv")
        put_json(
            self.s3,
            batch_entry_key(batch_id, table),
            {
                "batch_id": batch_id,
                "table": table,
                "status": status,
                "dq_status": "pass" if status == "validated" else "fail",
                "objects": [
                    {
                        "logical_key": logical,
                        "batch_key": batch_key,
                        "exists": object_exists,
                    }
                ],
            },
        )

    def test_failed_batch_cannot_move_production_pointer(self) -> None:
        self.add_entry("bad-batch", "silver_members", status="failed")
        manifest = assemble_batch_manifest(
            self.s3,
            bucket="bucket",
            batch_id="bad-batch",
            required_tables=["silver_members"],
        )
        self.assertEqual(manifest["status"], "failed")
        with self.assertRaises(ValueError):
            promote_batch(self.s3, bucket="bucket", batch_id="bad-batch")
        self.assertNotIn(PRODUCTION_POINTER_KEY, self.s3.objects)

    def test_validated_batch_promotes_with_one_pointer_write(self) -> None:
        self.add_entry("good-batch", "silver_members")
        manifest = assemble_batch_manifest(
            self.s3,
            bucket="bucket",
            batch_id="good-batch",
            required_tables=["silver_members"],
        )
        self.assertEqual(manifest["status"], "validated")
        pointer = promote_batch(self.s3, bucket="bucket", batch_id="good-batch", actor="tester", workflow_run_id="1")
        self.assertEqual(pointer["batch_id"], "good-batch")
        stored = json.loads(self.s3.objects[PRODUCTION_POINTER_KEY])
        self.assertEqual(stored["batch_id"], "good-batch")

    def test_rollback_restores_validated_prior_batch(self) -> None:
        for batch_id in ("batch-a", "batch-b"):
            self.add_entry(batch_id, "silver_members")
            assemble_batch_manifest(
                self.s3,
                bucket="bucket",
                batch_id=batch_id,
                required_tables=["silver_members"],
            )
        promote_batch(self.s3, bucket="bucket", batch_id="batch-a")
        promote_batch(self.s3, bucket="bucket", batch_id="batch-b")
        pointer = rollback_batch(self.s3, bucket="bucket", target_batch_id="batch-a")
        self.assertEqual(pointer["batch_id"], "batch-a")
        self.assertEqual(pointer["operation"], "rollback")
        self.assertIn(PREVIOUS_POINTER_KEY, self.s3.objects)

    def test_manifest_fails_when_required_table_is_missing(self) -> None:
        self.add_entry("partial", "silver_members")
        manifest = assemble_batch_manifest(
            self.s3,
            bucket="bucket",
            batch_id="partial",
            required_tables=["silver_members", "silver_member_votes"],
        )
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["validation"]["missing_tables"], ["silver_member_votes"])


if __name__ == "__main__":
    unittest.main()
