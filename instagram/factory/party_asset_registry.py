from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image

REGISTRY_PATH = Path("configs/reference/party_assets_v1.csv")


@dataclass(frozen=True)
class PartyAsset:
    party_key: str
    party_name: str
    logo_s3_uri: str
    asset_status: str
    source_type: str
    fallback_type: str


def _norm(value: str) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _aliases(row: dict[str, str]) -> set[str]:
    values = {row.get("party_name", "")}
    values.update(part.strip() for part in str(row.get("party_aliases") or "").split(";") if part.strip())
    if _norm(row.get("party_name", "")) == "independent":
        values.add("Independents")
    return {_norm(v) for v in values if v}


def load_registry(path: Path = REGISTRY_PATH) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def resolve_party_asset(party_name: str, path: Path = REGISTRY_PATH) -> PartyAsset:
    target = _norm(party_name)
    matches = [row for row in load_registry(path) if target in _aliases(row)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one approved party asset for {party_name!r}; found {len(matches)}")
    row = matches[0]
    if _norm(row.get("asset_status", "")) != "approved":
        raise RuntimeError(f"Party asset for {party_name!r} is not approved")
    uri = str(row.get("logo_s3_uri") or "").strip()
    if not uri.startswith("s3://"):
        raise RuntimeError(f"Party asset for {party_name!r} has invalid logo_s3_uri")
    return PartyAsset(
        party_key=str(row.get("party_key") or "").strip(),
        party_name=str(row.get("party_name") or "").strip(),
        logo_s3_uri=uri,
        asset_status=str(row.get("asset_status") or "").strip(),
        source_type=str(row.get("source_type") or "").strip(),
        fallback_type=str(row.get("fallback_type") or "").strip(),
    )


def fetch_logo(s3: Any, asset: PartyAsset) -> tuple[Image.Image, dict[str, Any]]:
    parsed = urlparse(asset.logo_s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    obj = s3.get_object(Bucket=bucket, Key=key)
    raw = obj["Body"].read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    if image.size != (1600, 1600):
        raise RuntimeError(f"Logo for {asset.party_name} is {image.size}, expected 1600x1600")
    if str(image.format or "PNG").upper() not in {"PNG", ""}:
        raise RuntimeError(f"Logo for {asset.party_name} is not a PNG")
    return image, {
        "party_key": asset.party_key,
        "registry_party_name": asset.party_name,
        "logo_s3_uri": asset.logo_s3_uri,
        "bucket": bucket,
        "key": key,
        "etag": str(obj.get("ETag") or "").strip('"'),
        "version_id": obj.get("VersionId"),
        "last_modified": obj.get("LastModified").isoformat() if obj.get("LastModified") else None,
        "dimensions": [1600, 1600],
        "asset_status": asset.asset_status,
        "source_type": asset.source_type,
        "fallback_type": asset.fallback_type,
    }
