from __future__ import annotations

import importlib
import io
import os
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from extract.oireachtas.batch import batch_key_for_production_key
from extract.oireachtas.contracts import DatasetContract, validate_dataset_contract
from extract.oireachtas.io_s3 import get_bytes, put_bytes


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.modified: dict[str, datetime] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str = "application/octet-stream", **kwargs) -> None:
        self.objects[Key] = bytes(Body)
        self.modified[Key] = datetime(2026, 7, 14, tzinfo=timezone.utc)

    def get_object(self, *, Bucket: str, Key: str):
        if Key not in self.objects:
            raise FileNotFoundError(Key)
        return {"Body": io.BytesIO(self.objects[Key]), "ContentType": "text/csv"}

    def head_object(self, *, Bucket: str, Key: str):
        if Key not in self.objects:
            raise FileNotFoundError(Key)
        return {
            "ContentLength": len(self.objects[Key]),
            "ETag": '"etag"',
            "LastModified": self.modified[Key],
        }


class BatchAwareReadWriteTests(unittest.TestCase):
    def test_candidate_write_does_not_require_production_promotion_switch(self) -> None:
        s3 = FakeS3()
        logical = "processed/oireachtas_unified/compat/members/test.csv"
        with patch.dict(
            os.environ,
            {
                "OIREACHTAS_PUBLISH_ENABLED": "false",
                "OIREACHTAS_PUBLISH_LATEST": "true",
                "OIREACHTAS_BATCH_ID": "candidate-1",
            },
            clear=True,
        ):
            put_bytes(s3, bucket="bucket", key=logical, body=b"value\n1\n")
        self.assertNotIn(logical, s3.objects)
        self.assertIn(batch_key_for_production_key(logical, "candidate-1"), s3.objects)

    def test_candidate_read_never_falls_back_to_logical_production_object(self) -> None:
        s3 = FakeS3()
        logical = "processed/oireachtas_unified/compat/members/test.csv"
        s3.put_object(Bucket="bucket", Key=logical, Body=b"production", ContentType="text/plain")
        with patch.dict(os.environ, {"OIREACHTAS_BATCH_ID": "candidate-1"}, clear=True):
            with self.assertRaises(FileNotFoundError):
                get_bytes(s3, bucket="bucket", key=logical)

    def test_candidate_read_uses_candidate_physical_key(self) -> None:
        s3 = FakeS3()
        logical = "processed/oireachtas_unified/compat/members/test.csv"
        physical = batch_key_for_production_key(logical, "candidate-1")
        s3.put_object(Bucket="bucket", Key=physical, Body=b"candidate", ContentType="text/plain")
        with patch.dict(os.environ, {"OIREACHTAS_BATCH_ID": "candidate-1"}, clear=True):
            self.assertEqual(get_bytes(s3, bucket="bucket", key=logical), b"candidate")


class ContractValidationTests(unittest.TestCase):
    def contract(self) -> DatasetContract:
        return DatasetContract(
            name="members",
            logical_key="processed/oireachtas_unified/compat/members/test.csv",
            required_columns=("member_code", "full_name"),
            primary_key=("member_code",),
            minimum_rows=2,
            maximum_age_days=5,
        )

    def test_contract_passes_complete_fresh_unique_dataset(self) -> None:
        s3 = FakeS3()
        logical = self.contract().logical_key
        physical = batch_key_for_production_key(logical, "candidate-1")
        s3.put_object(
            Bucket="bucket",
            Key=physical,
            Body=b"member_code,full_name\nm1,One\nm2,Two\n",
            ContentType="text/csv",
        )
        with patch.dict(os.environ, {"OIREACHTAS_BATCH_ID": "candidate-1"}, clear=True):
            result = validate_dataset_contract(
                s3,
                bucket="bucket",
                contract=self.contract(),
                as_of=datetime(2026, 7, 14).date(),
            )
        self.assertEqual(result["status"], "pass")

    def test_contract_fails_missing_columns_duplicates_and_staleness(self) -> None:
        s3 = FakeS3()
        logical = self.contract().logical_key
        physical = batch_key_for_production_key(logical, "candidate-1")
        s3.put_object(
            Bucket="bucket",
            Key=physical,
            Body=b"member_code\nm1\nm1\n",
            ContentType="text/csv",
        )
        s3.modified[physical] = datetime(2026, 6, 1, tzinfo=timezone.utc)
        with patch.dict(os.environ, {"OIREACHTAS_BATCH_ID": "candidate-1"}, clear=True):
            result = validate_dataset_contract(
                s3,
                bucket="bucket",
                contract=self.contract(),
                as_of=datetime(2026, 7, 14).date(),
            )
        self.assertEqual(result["status"], "fail")
        self.assertIn("full_name", result["missing_columns"])
        self.assertGreater(result["duplicate_primary_key_rows"], 0)
        self.assertGreater(result["age_days"], result["maximum_age_days"])


class GenericMetricsTests(unittest.TestCase):
    def test_metric_columns_follow_target_year(self) -> None:
        with patch.dict(os.environ, {"TARGET_YEAR": "2024"}, clear=False):
            import process.build_member_profile_metrics as metrics
            metrics = importlib.reload(metrics)
            members = pd.DataFrame(
                [{"member_code": "m1", "full_name": "One", "constituency": "A", "party": "P"}]
            )
            votes = pd.DataFrame(
                [{"memberCode": "m1", "unique_vote_id": "v1", "date": "2024-03-01"}]
            )
            photos = pd.DataFrame([{"member_code": "m1", "photo_url": "https://example.test/a.jpg"}])
            debates = pd.DataFrame(
                [{"member_code": "m1", "PoliticalIssues": "Health", "Debate Date": "2024-04-01"}]
            )
            output = metrics.build_metrics(members, votes, photos, debates)
        self.assertIn("speech_count_2024", output.columns)
        self.assertIn("vote_participation_pct_2024", output.columns)
        self.assertNotIn("speech_count_2025", output.columns)


if __name__ == "__main__":
    unittest.main()
