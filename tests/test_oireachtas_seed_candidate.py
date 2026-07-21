from __future__ import annotations

import io
import json
import unittest

from extract.oireachtas.batch import (
    PRODUCTION_POINTER_KEY,
    batch_entry_key,
    batch_key_for_production_key,
    batch_manifest_key,
)
from process.oireachtas_seed_candidate import seed_candidate


class _Paginator:
    def __init__(self, s3: "FakeS3") -> None:
        self.s3 = s3

    def paginate(self, *, Bucket: str, Prefix: str):
        keys = sorted(key for key in self.s3.objects if key.startswith(Prefix))
        yield {"Contents": [{"Key": key} for key in keys]}


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str | None = None):
        self.objects[Key] = bytes(Body)
        return {"ETag": '"put"'}

    def get_object(self, *, Bucket: str, Key: str):
        return {"Body": io.BytesIO(self.objects[Key])}

    def copy_object(self, *, Bucket: str, Key: str, CopySource: dict[str, str], MetadataDirective: str):
        self.objects[Key] = self.objects[CopySource["Key"]]
        return {"CopyObjectResult": {"ETag": '"copy"'}}

    def head_object(self, *, Bucket: str, Key: str):
        body = self.objects[Key]
        return {"ContentLength": len(body), "ETag": '"head"'}

    def get_paginator(self, name: str):
        if name != "list_objects_v2":
            raise ValueError(name)
        return _Paginator(self)


def _put_json(s3: FakeS3, key: str, payload: dict) -> None:
    s3.objects[key] = (json.dumps(payload) + "\n").encode("utf-8")


class SeedCandidateTests(unittest.TestCase):
    def test_clones_complete_validated_production_batch(self) -> None:
        s3 = FakeS3()
        source_batch = "production-1"
        destination_batch = "scheduled-weekly-2"
        table = "silver_members"
        logical_csv = f"processed/oireachtas_unified/latest/csv/{table}.csv"
        logical_parquet = f"processed/oireachtas_unified/latest/parquet/{table}.parquet"
        source_csv = batch_key_for_production_key(logical_csv, source_batch)
        source_parquet = batch_key_for_production_key(logical_parquet, source_batch)
        s3.objects[source_csv] = b"member_code\nA\n"
        s3.objects[source_parquet] = b"parquet"

        entry = {
            "batch_id": source_batch,
            "table": table,
            "status": "validated",
            "dq_status": "pass",
            "objects": [
                {"logical_key": logical_csv, "batch_key": source_csv, "exists": True},
                {"logical_key": logical_parquet, "batch_key": source_parquet, "exists": True},
            ],
        }
        _put_json(s3, batch_entry_key(source_batch, table), entry)
        _put_json(
            s3,
            batch_manifest_key(source_batch),
            {
                "batch_id": source_batch,
                "status": "validated",
                "required_tables": [table],
                "table_count": 1,
                "tables": [entry],
                "validation": {
                    "missing_tables": [],
                    "failed_tables": [],
                    "missing_objects": [],
                    "duplicate_tables": [],
                },
            },
        )
        _put_json(s3, PRODUCTION_POINTER_KEY, {"mode": "batch", "batch_id": source_batch})

        result = seed_candidate(s3, bucket="test", batch_id=destination_batch)

        self.assertEqual(result["status"], "seeded")
        self.assertEqual(result["source_batch_id"], source_batch)
        self.assertEqual(result["copied_entries"], 1)
        self.assertEqual(result["copied_objects"], 2)
        destination_csv = batch_key_for_production_key(logical_csv, destination_batch)
        destination_parquet = batch_key_for_production_key(logical_parquet, destination_batch)
        self.assertEqual(s3.objects[destination_csv], s3.objects[source_csv])
        self.assertEqual(s3.objects[destination_parquet], s3.objects[source_parquet])
        seeded_manifest = json.loads(s3.objects[batch_manifest_key(destination_batch)])
        self.assertEqual(seeded_manifest["status"], "validated")
        self.assertEqual(seeded_manifest["table_count"], 1)


if __name__ == "__main__":
    unittest.main()
