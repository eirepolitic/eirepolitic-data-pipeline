from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from extract.oireachtas.io_s3 import DEFAULT_BUCKET, DEFAULT_REGION, make_s3_client
from process.oireachtas_speech_issue_classifier import (
    PREVIOUS_CLASSIFICATION_POINTER_KEY,
    PRODUCTION_CLASSIFICATION_POINTER_KEY,
    RUN_ROOT,
    get_bytes_required,
    read_json_optional,
    read_json_required,
    run_manifest_key,
    sha256_bytes,
)

POINTER_KEYS = {
    "production": PRODUCTION_CLASSIFICATION_POINTER_KEY,
    "previous": PREVIOUS_CLASSIFICATION_POINTER_KEY,
}
ALLOWED_PREPARE_STATUSES = {"prepared", "ready_to_collect", "no_op"}
INITIAL_ARTIFACT_NAMES = {
    "selection_csv",
    "selection_parquet",
    "openai_requests_jsonl",
    "deterministic_results_csv",
}


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def snapshot_pointers(*, s3: Any, bucket: str) -> dict[str, Any]:
    pointers = {
        name: read_json_optional(s3, bucket=bucket, key=key)
        for name, key in POINTER_KEYS.items()
    }
    return {
        "bucket": bucket,
        "pointer_keys": POINTER_KEYS,
        "pointers": pointers,
        "pointer_sha256": {
            name: _canonical_sha256(pointer)
            for name, pointer in pointers.items()
        },
    }


def _artifact_checks(
    *,
    s3: Any,
    bucket: str,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    prefix = f"{RUN_ROOT}/run_id={manifest['run_id']}/"
    checksums = manifest.get("artifact_checksums")
    if not isinstance(checksums, Mapping):
        raise ValueError("Manifest artifact_checksums is not a mapping")
    missing = sorted(INITIAL_ARTIFACT_NAMES - set(checksums))
    if missing:
        raise ValueError(f"Manifest is missing initial artifact checksums: {missing}")

    checks: list[dict[str, Any]] = []
    for name in sorted(INITIAL_ARTIFACT_NAMES):
        metadata = checksums[name]
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Artifact metadata is not a mapping: {name}")
        key = str(metadata.get("key") or "").strip()
        expected_sha256 = str(metadata.get("sha256") or "").strip()
        expected_size = int(metadata.get("size_bytes") or -1)
        if not key.startswith(prefix):
            raise ValueError(f"Artifact escaped immutable run prefix: {key}")
        payload = get_bytes_required(s3, bucket=bucket, key=key)
        actual_sha256 = sha256_bytes(payload)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"Checksum mismatch for {name}: {key}")
        if len(payload) != expected_size:
            raise ValueError(f"Size mismatch for {name}: {key}")
        checks.append(
            {
                "artifact": name,
                "key": key,
                "sha256": actual_sha256,
                "size_bytes": len(payload),
                "status": "pass",
            }
        )
    return checks


def verify_candidate_only_run(
    *,
    s3: Any,
    bucket: str,
    run_id: str,
    before_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    after = snapshot_pointers(s3=s3, bucket=bucket)
    before_pointers = before_snapshot.get("pointers")
    if not isinstance(before_pointers, Mapping):
        raise ValueError("Before snapshot does not contain pointers")
    if dict(before_pointers) != after["pointers"]:
        raise RuntimeError("Candidate preparation changed a classification pointer")

    manifest = read_json_required(s3, bucket=bucket, key=run_manifest_key(run_id))
    status = str(manifest.get("status") or "")
    if status not in ALLOWED_PREPARE_STATUSES:
        raise RuntimeError(f"Unexpected candidate-only run status: {status!r}")
    if manifest.get("published") is not False:
        raise RuntimeError("Candidate-only run is marked published")
    if str(manifest.get("openai_batch_id") or "").strip():
        raise RuntimeError("Candidate-only run submitted an OpenAI Batch")
    if int(manifest.get("batch_submission_attempts") or 0) != 0:
        raise RuntimeError("Candidate-only run attempted OpenAI Batch submission")

    prefix = f"{RUN_ROOT}/run_id={run_id}/"
    if str(manifest.get("manifest_key") or "") != run_manifest_key(run_id):
        raise RuntimeError("Manifest key does not match run ID")
    for field in (
        "selection_csv_key",
        "selection_parquet_key",
        "requests_jsonl_key",
        "deterministic_results_key",
    ):
        if not str(manifest.get(field) or "").startswith(prefix):
            raise RuntimeError(f"Manifest field escaped immutable run prefix: {field}")
    for field in ("table_csv_key", "table_parquet_key", "compat_csv_key", "compat_parquet_key"):
        if str(manifest.get(field) or "").strip():
            raise RuntimeError(f"Prepare-only run unexpectedly contains collected artifact: {field}")

    artifact_checks = _artifact_checks(s3=s3, bucket=bucket, manifest=manifest)
    return {
        "status": "pass",
        "validation_mode": "candidate_only_s3",
        "run_id": run_id,
        "run_status": status,
        "source_batch_id": str(manifest.get("source_batch_id") or ""),
        "source_batch_speech_key": str(manifest.get("source_batch_speech_key") or ""),
        "model_name": str(manifest.get("model_name") or ""),
        "source_rows": int(manifest.get("source_rows") or 0),
        "existing_rows": int(manifest.get("existing_rows") or 0),
        "full_delta_rows": int(manifest.get("full_delta_rows") or 0),
        "delta_rows_selected": int(manifest.get("delta_rows_selected") or 0),
        "deterministic_none_rows": int(manifest.get("deterministic_none_rows") or 0),
        "batch_request_rows": int(manifest.get("batch_request_rows") or 0),
        "delta_truncated": bool(manifest.get("delta_truncated")),
        "maintenance_needed": bool(manifest.get("maintenance_needed")),
        "published": bool(manifest.get("published")),
        "openai_batch_submitted": False,
        "production_pointer_unchanged": True,
        "previous_pointer_unchanged": True,
        "pointer_sha256_before": dict(before_snapshot.get("pointer_sha256") or {}),
        "pointer_sha256_after": dict(after.get("pointer_sha256") or {}),
        "artifact_checks": artifact_checks,
    }


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify candidate-only Oireachtas speech-classifier S3 behavior")
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", DEFAULT_REGION))
    commands = parser.add_subparsers(dest="command", required=True)

    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--output", required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--before", required=True)
    verify.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    s3 = make_s3_client(region_name=args.region)
    if args.command == "snapshot":
        result = snapshot_pointers(s3=s3, bucket=args.bucket)
        _write_json(args.output, result)
    else:
        before = json.loads(Path(args.before).read_text(encoding="utf-8"))
        result = verify_candidate_only_run(
            s3=s3,
            bucket=args.bucket,
            run_id=args.run_id,
            before_snapshot=before,
        )
        _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
