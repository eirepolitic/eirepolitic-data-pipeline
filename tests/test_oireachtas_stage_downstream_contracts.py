from __future__ import annotations

import unittest
from unittest.mock import patch

from process.oireachtas_stage_downstream_contracts import _source_key


class S3Error(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    def __init__(self, *, missing: bool = False, error_code: str | None = None) -> None:
        self.missing = missing
        self.error_code = error_code
        self.head_calls: list[str] = []

    def head_object(self, *, Bucket: str, Key: str):
        self.head_calls.append(Key)
        if self.error_code:
            raise S3Error(self.error_code)
        if self.missing:
            raise S3Error("NoSuchKey")
        return {"ContentLength": 1}


class StageDownstreamContractSourceTests(unittest.TestCase):
    def test_uses_promoted_batch_object_when_present(self) -> None:
        s3 = FakeS3()
        logical = "processed/oireachtas_unified/compat/media/test.csv"
        resolved = "processed/oireachtas_unified/batches/prod/compat/media/test.csv"
        with patch(
            "process.oireachtas_stage_downstream_contracts.resolve_production_key",
            return_value=resolved,
        ):
            self.assertEqual(_source_key(s3, bucket="bucket", logical_key=logical), resolved)
        self.assertEqual(s3.head_calls, [resolved])

    def test_falls_back_to_logical_key_when_promoted_batch_object_is_missing(self) -> None:
        s3 = FakeS3(missing=True)
        logical = "processed/oireachtas_unified/compat/media/test.csv"
        resolved = "processed/oireachtas_unified/batches/prod/compat/media/test.csv"
        with patch(
            "process.oireachtas_stage_downstream_contracts.resolve_production_key",
            return_value=resolved,
        ):
            self.assertEqual(_source_key(s3, bucket="bucket", logical_key=logical), logical)

    def test_does_not_hide_non_missing_s3_errors(self) -> None:
        s3 = FakeS3(error_code="AccessDenied")
        logical = "processed/oireachtas_unified/compat/media/test.csv"
        resolved = "processed/oireachtas_unified/batches/prod/compat/media/test.csv"
        with patch(
            "process.oireachtas_stage_downstream_contracts.resolve_production_key",
            return_value=resolved,
        ):
            with self.assertRaises(S3Error):
                _source_key(s3, bucket="bucket", logical_key=logical)

    def test_legacy_or_unresolvable_pointer_uses_logical_key(self) -> None:
        s3 = FakeS3()
        logical = "processed/oireachtas_unified/compat/media/test.csv"
        with patch(
            "process.oireachtas_stage_downstream_contracts.resolve_production_key",
            side_effect=FileNotFoundError("no pointer"),
        ):
            self.assertEqual(_source_key(s3, bucket="bucket", logical_key=logical), logical)


if __name__ == "__main__":
    unittest.main()
