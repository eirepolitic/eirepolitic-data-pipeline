from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import yaml
from botocore.exceptions import ClientError

from instagram.renderer.constants import DEFAULT_BUCKET, DEFAULT_REGION

REPO_ROOT = Path(__file__).resolve().parents[3]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_palette(template: dict[str, Any]) -> dict[str, str]:
    palette = template.get("palette", {}) or {}
    return {
        "background": str(palette.get("background", "#0f2f24")),
        "panel": str(palette.get("panel", "#173d30")),
        "panel_alt": str(palette.get("panel_alt", "#214a3b")),
        "text": str(palette.get("text", "#f4ead7")),
        "muted": str(palette.get("muted", "#cbbf9f")),
        "accent": str(palette.get("accent", "#d8b45f")),
        "accent_2": str(palette.get("accent_2", "#9ec5a2")),
        "grid": str(palette.get("grid", "#cbbf9f")),
        "warning": str(palette.get("warning", "#b55b5b")),
    }


def _read_csv_text(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(text))
    rows.extend(dict(row) for row in reader)
    return rows


def _s3_client(region: str):
    return boto3.client("s3", region_name=region)


def _read_s3_csv(input_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bucket = str(input_cfg.get("bucket") or os.getenv("INSTAGRAM_VISUAL_S3_BUCKET") or DEFAULT_BUCKET)
    region = str(input_cfg.get("region") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or DEFAULT_REGION)
    key = str(input_cfg.get("key", "")).strip()
    if not key:
        raise ValueError("S3 visual input requires input.key")

    required = bool(input_cfg.get("required", True))
    source_uri = f"s3://{bucket}/{key}"
    client = _s3_client(region)
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if not required and code in {"404", "NoSuchKey", "NotFound"}:
            return [], {
                "input_mode": "s3_csv",
                "source": source_uri,
                "bucket": bucket,
                "region": region,
                "key": key,
                "missing": True,
            }
        raise

    text = obj["Body"].read().decode("utf-8-sig", errors="replace")
    rows = _read_csv_text(text)
    return rows, {
        "input_mode": "s3_csv",
        "source": source_uri,
        "bucket": bucket,
        "region": region,
        "key": key,
        "row_count": len(rows),
        "missing": False,
    }


def _read_s3_csv_first_available(input_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    keys = [str(key).strip() for key in input_cfg.get("keys", []) if str(key).strip()]
    if not keys:
        raise ValueError("S3 visual input mode s3_csv_first_available requires input.keys")
    bucket = str(input_cfg.get("bucket") or os.getenv("INSTAGRAM_VISUAL_S3_BUCKET") or DEFAULT_BUCKET)
    region = str(input_cfg.get("region") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or DEFAULT_REGION)
    required = bool(input_cfg.get("required", True))
    client = _s3_client(region)

    checked: list[str] = []
    for key in keys:
        checked.append(f"s3://{bucket}/{key}")
        try:
            obj = client.get_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                continue
            raise
        text = obj["Body"].read().decode("utf-8-sig", errors="replace")
        rows = _read_csv_text(text)
        return rows, {
            "input_mode": "s3_csv_first_available",
            "source": f"s3://{bucket}/{key}",
            "bucket": bucket,
            "region": region,
            "key": key,
            "checked": checked,
            "row_count": len(rows),
            "missing": False,
        }

    if required:
        raise FileNotFoundError(f"No S3 CSV found. Checked: {checked}")
    return [], {
        "input_mode": "s3_csv_first_available",
        "source": "missing",
        "bucket": bucket,
        "region": region,
        "checked": checked,
        "row_count": 0,
        "missing": True,
    }


def rows_from_sample(sample: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_cfg = sample.get("input", {}) or {}
    mode = str(input_cfg.get("mode", "inline"))
    if mode == "inline":
        return list(input_cfg.get("rows", []) or []), {
            "input_mode": "inline",
            "source": "inline",
        }
    if mode == "local_csv":
        csv_path = resolve_repo_path(str(input_cfg["path"]))
        rows: list[dict[str, Any]] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(dict(row) for row in reader)
        return rows, {
            "input_mode": "local_csv",
            "source": str(input_cfg["path"]),
            "resolved_source": str(csv_path),
            "row_count": len(rows),
        }
    if mode == "s3_csv":
        return _read_s3_csv(input_cfg)
    if mode == "s3_csv_first_available":
        return _read_s3_csv_first_available(input_cfg)
    raise ValueError(f"Unsupported visual input mode: {mode}")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
