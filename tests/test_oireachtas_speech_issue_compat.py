from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from extract.oireachtas.speech_issue_compat import (
    LEGACY_COMPAT_KEY,
    PRODUCTION_POINTER_KEY,
    SpeechIssueCompatibilityError,
    resolve_speech_issue_compatibility,
    validate_compatibility_frame,
    validate_published_compatibility,
)

ROOT = Path(__file__).resolve().parents[1]
STAGING_SCRIPT = ROOT / "process/oireachtas_stage_downstream_contracts.py"
MEMBER_WORKFLOW = ROOT / ".github/workflows/build_member_profile_metrics_2025.yml"
INSTAGRAM_WORKFLOW = ROOT / ".github/workflows/oireachtas_instagram_consumer_smoke.yml"


class FakeBody(io.BytesIO):
    pass


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_json(self, key: str, payload: dict[str, Any]) -> None:
        self.objects[key] = json.dumps(payload).encode("utf-8")

    def put_csv(self, key: str, frame: pd.DataFrame) -> None:
        self.objects[key] = frame.to_csv(index=False).encode("utf-8")

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": FakeBody(self.objects[Key]), "ContentType": "text/csv"}


def compat_frame(*, status: str = "classified", label: str = "Health") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "speech_id": "s1",
                "Debate Date": "2026-01-01",
                "Speaker Name": "Member One",
                "Speech Order": "1",
                "Speech Text": "A speech about health policy.",
                "PoliticalIssues": label,
                "classification_status": status,
                "speech_text_hash": "h1",
            }
        ]
    )


def seed_published_v2(s3: FakeS3) -> tuple[str, str]:
    run_id = "run-1"
    manifest_key = "runs/run-1/manifest.json"
    compat_key = "runs/run-1/compat.csv"
    source_batch_id = "core-1"
    s3.put_csv(compat_key, compat_frame())
    s3.put_json(
        manifest_key,
        {
            "table": "enrichment_speech_issue_labels",
            "run_id": run_id,
            "status": "published",
            "published": True,
            "dq_status": "pass",
            "stale_reasons": [],
            "source_batch_id": source_batch_id,
            "compat_csv_key": compat_key,
            "failed_rows": 0,
            "output_rows": 1,
            "source_rows": 1,
        },
    )
    s3.put_json(
        PRODUCTION_POINTER_KEY,
        {
            "run_id": run_id,
            "manifest_key": manifest_key,
            "compat_csv_key": compat_key,
            "source_batch_id": source_batch_id,
        },
    )
    return compat_key, run_id


def test_valid_published_v2_compatibility_resolves() -> None:
    s3 = FakeS3()
    compat_key, run_id = seed_published_v2(s3)

    resolution, report = validate_published_compatibility(s3, bucket="test")

    assert resolution.key == compat_key
    assert resolution.mode == "v2_published"
    assert resolution.run_id == run_id
    assert report["frame"]["rows"] == 1


def test_cutover_disabled_always_uses_legacy_key() -> None:
    resolution = resolve_speech_issue_compatibility(
        FakeS3(),
        bucket="test",
        cutover_enabled=False,
        allow_legacy_fallback=False,
    )
    assert resolution.key == LEGACY_COMPAT_KEY
    assert resolution.mode == "legacy_pre_cutover"


def test_invalid_v2_falls_back_only_when_enabled() -> None:
    s3 = FakeS3()
    resolution = resolve_speech_issue_compatibility(
        s3,
        bucket="test",
        cutover_enabled=True,
        allow_legacy_fallback=True,
    )
    assert resolution.key == LEGACY_COMPAT_KEY
    assert resolution.mode == "legacy_fallback"
    assert resolution.fallback_reason

    with pytest.raises(KeyError):
        resolve_speech_issue_compatibility(
            s3,
            bucket="test",
            cutover_enabled=True,
            allow_legacy_fallback=False,
        )


def test_partial_or_invalid_compatibility_is_rejected() -> None:
    with pytest.raises(SpeechIssueCompatibilityError, match="fully classified"):
        validate_compatibility_frame(compat_frame(status="failed", label=""))
    with pytest.raises(SpeechIssueCompatibilityError, match="invalid labels"):
        validate_compatibility_frame(compat_frame(label="Invalid"))


def test_duplicate_speech_ids_are_rejected() -> None:
    frame = pd.concat([compat_frame(), compat_frame()], ignore_index=True)
    with pytest.raises(SpeechIssueCompatibilityError, match="duplicate speech_id"):
        validate_compatibility_frame(frame)


def test_staging_resolves_debate_issue_contract_through_guard() -> None:
    text = STAGING_SCRIPT.read_text(encoding="utf-8")
    assert "resolve_speech_issue_compatibility" in text
    assert 'name != "debate_issue_labels"' in text
    assert '"resolution-mode"' in text
    assert '"classification-run-id"' in text


def test_member_metrics_workflow_uses_guarded_resolution_for_production() -> None:
    text = MEMBER_WORKFLOW.read_text(encoding="utf-8")
    assert "oireachtas_resolve_speech_issue_compat.py" in text
    assert "DEBATE_ISSUES_INPUT_KEY=$(jq -r '.key'" in text
    assert "if: inputs.batch_id == ''" in text
    assert "OIREACHTAS_SPEECH_CLASSIFIER_COMPAT_CUTOVER_ENABLED" in text
    assert "OIREACHTAS_SPEECH_CLASSIFIER_COMPAT_LEGACY_FALLBACK_ENABLED" in text


def test_instagram_smoke_uses_and_verifies_resolved_key() -> None:
    text = INSTAGRAM_WORKFLOW.read_text(encoding="utf-8")
    assert "oireachtas_resolve_speech_issue_compat.py" in text
    assert "DEBATE_ISSUES_KEY=$(jq -r '.key'" in text
    assert "Resolved debate issue key not used" in text
    assert "speech-issue-resolution.json" in text
