from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .enrichment_contracts import load_enrichment_registry

TABLE_NAME = "enrichment_speech_issue_labels"
PRODUCTION_POINTER_KEY = (
    "processed/oireachtas_unified/enrichment/"
    "enrichment_speech_issue_labels/pointers/production.json"
)
LEGACY_COMPAT_KEY = (
    "processed/oireachtas_unified/compat/debates/"
    "debate_speeches_classified_compat.csv"
)
CUTOVER_ENV = "OIREACHTAS_SPEECH_CLASSIFIER_COMPAT_CUTOVER_ENABLED"
LEGACY_FALLBACK_ENV = "OIREACHTAS_SPEECH_CLASSIFIER_COMPAT_LEGACY_FALLBACK_ENABLED"


class SpeechIssueCompatibilityError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompatibilityResolution:
    key: str
    mode: str
    run_id: str = ""
    source_batch_id: str = ""
    fallback_reason: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "key": self.key,
            "mode": self.mode,
            "run_id": self.run_id,
            "source_batch_id": self.source_batch_id,
            "fallback_reason": self.fallback_reason,
        }


def _enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() == "true"


def _read_json(s3: Any, *, bucket: str, key: str) -> dict[str, Any]:
    response = s3.get_object(Bucket=bucket, Key=key)
    payload = json.loads(response["Body"].read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise SpeechIssueCompatibilityError(
            f"Expected JSON object at s3://{bucket}/{key}"
        )
    return payload


def _read_csv(s3: Any, *, bucket: str, key: str) -> pd.DataFrame:
    response = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(
        io.BytesIO(response["Body"].read()),
        dtype=str,
        keep_default_na=False,
    )


def _taxonomy() -> set[str]:
    registry = load_enrichment_registry()
    contract = registry[TABLE_NAME]
    conditional = contract.get("conditional_enums") or []
    if not conditional:
        raise SpeechIssueCompatibilityError("Issue-label taxonomy is missing")
    return set(conditional[0].get("values") or [])


def validate_compatibility_frame(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "speech_id",
        "Speaker Name",
        "PoliticalIssues",
        "classification_status",
        "speech_text_hash",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SpeechIssueCompatibilityError(
            f"V2 compatibility output is missing columns: {missing}"
        )
    if frame.empty:
        raise SpeechIssueCompatibilityError("V2 compatibility output is empty")
    if frame["speech_id"].fillna("").astype(str).str.strip().eq("").any():
        raise SpeechIssueCompatibilityError("Compatibility output has blank speech_id values")
    if frame["speech_id"].duplicated().any():
        raise SpeechIssueCompatibilityError("Compatibility output has duplicate speech_id values")
    if frame["speech_text_hash"].fillna("").astype(str).str.strip().eq("").any():
        raise SpeechIssueCompatibilityError(
            "Compatibility output has blank speech_text_hash values"
        )
    statuses = set(frame["classification_status"].astype(str))
    if statuses != {"classified"}:
        raise SpeechIssueCompatibilityError(
            f"Compatibility output is not fully classified: {sorted(statuses)}"
        )
    invalid_labels = sorted(set(frame["PoliticalIssues"].astype(str)) - _taxonomy())
    if invalid_labels:
        raise SpeechIssueCompatibilityError(
            f"Compatibility output has invalid labels: {invalid_labels}"
        )
    return {"rows": int(len(frame)), "status": "pass"}


def validate_published_compatibility(
    s3: Any,
    *,
    bucket: str,
) -> tuple[CompatibilityResolution, dict[str, Any]]:
    pointer = _read_json(s3, bucket=bucket, key=PRODUCTION_POINTER_KEY)
    required_pointer = {
        "run_id",
        "manifest_key",
        "compat_csv_key",
        "source_batch_id",
    }
    missing_pointer = sorted(required_pointer - set(pointer))
    if missing_pointer:
        raise SpeechIssueCompatibilityError(
            f"Classification pointer is missing fields: {missing_pointer}"
        )

    manifest = _read_json(
        s3,
        bucket=bucket,
        key=str(pointer["manifest_key"]),
    )
    if manifest.get("status") != "published" or manifest.get("published") is not True:
        raise SpeechIssueCompatibilityError("Classification run is not published")
    if manifest.get("dq_status") != "pass":
        raise SpeechIssueCompatibilityError("Classification run did not pass DQ")
    if manifest.get("stale_reasons"):
        raise SpeechIssueCompatibilityError("Classification run is stale")
    if str(manifest.get("run_id") or "") != str(pointer["run_id"]):
        raise SpeechIssueCompatibilityError("Pointer and manifest run IDs differ")
    if str(manifest.get("source_batch_id") or "") != str(pointer["source_batch_id"]):
        raise SpeechIssueCompatibilityError("Pointer and manifest source batches differ")
    if str(manifest.get("compat_csv_key") or "") != str(pointer["compat_csv_key"]):
        raise SpeechIssueCompatibilityError("Pointer and manifest compatibility keys differ")
    if int(manifest.get("failed_rows") or 0) != 0:
        raise SpeechIssueCompatibilityError("Published classification contains failed rows")
    if int(manifest.get("output_rows") or 0) != int(manifest.get("source_rows") or -1):
        raise SpeechIssueCompatibilityError("Published classification is not full coverage")

    frame = _read_csv(s3, bucket=bucket, key=str(pointer["compat_csv_key"]))
    frame_report = validate_compatibility_frame(frame)
    if frame_report["rows"] != int(manifest["source_rows"]):
        raise SpeechIssueCompatibilityError(
            "Compatibility row count does not match the published source row count"
        )

    resolution = CompatibilityResolution(
        key=str(pointer["compat_csv_key"]),
        mode="v2_published",
        run_id=str(pointer["run_id"]),
        source_batch_id=str(pointer["source_batch_id"]),
    )
    return resolution, {
        "pointer": pointer,
        "manifest": manifest,
        "frame": frame_report,
    }


def resolve_speech_issue_compatibility(
    s3: Any,
    *,
    bucket: str,
    cutover_enabled: bool | None = None,
    allow_legacy_fallback: bool | None = None,
) -> CompatibilityResolution:
    cutover = _enabled(CUTOVER_ENV) if cutover_enabled is None else cutover_enabled
    fallback = (
        _enabled(LEGACY_FALLBACK_ENV, default=True)
        if allow_legacy_fallback is None
        else allow_legacy_fallback
    )
    if not cutover:
        return CompatibilityResolution(key=LEGACY_COMPAT_KEY, mode="legacy_pre_cutover")
    try:
        resolution, _ = validate_published_compatibility(s3, bucket=bucket)
        return resolution
    except Exception as exc:
        if not fallback:
            raise
        return CompatibilityResolution(
            key=LEGACY_COMPAT_KEY,
            mode="legacy_fallback",
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )
