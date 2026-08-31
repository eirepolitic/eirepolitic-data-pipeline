#!/usr/bin/env python3
"""Fetch reviewed party source assets into a local staging tree.

Safety/quality rules:
- HTTPS only
- only registry rows whose source_type identifies a direct asset are fetched
- generic webpages are reported as unresolved, never scraped/guessed
- supported image/SVG extensions only
- bounded download size and timeout
- existing staging files are not overwritten unless --replace is supplied
- no S3 writes and no publishing
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

import requests

from process.party_assets import DEFAULT_REGISTRY, PartyAsset, load_registry

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".svg"}
DIRECT_SOURCE_TYPES = {
    "official_party_logo_asset",
    "official_party_logo_asset_svg",
    "electoral_commission_registered_emblem_asset",
}
MAX_BYTES = 12 * 1024 * 1024
TIMEOUT_SECONDS = 30


def source_extension(row: PartyAsset) -> str | None:
    if row.source_type not in DIRECT_SOURCE_TYPES:
        return None
    parsed = urlparse(row.source_url)
    if parsed.scheme != "https" or not parsed.netloc:
        return None
    suffix = Path(parsed.path).suffix.lower()
    return suffix if suffix in SUPPORTED_EXTENSIONS else None


def _content_type_ok(content_type: str, suffix: str) -> bool:
    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    if suffix == ".svg":
        return media_type in {"image/svg+xml", "text/xml", "application/xml", "application/octet-stream"}
    return media_type.startswith("image/") or media_type == "application/octet-stream"


def fetch_row(row: PartyAsset, staging_root: Path, replace: bool = False, session=None) -> dict:
    if row.asset_status == "approved_fallback":
        return {"party_key": row.party_key, "status": "fallback", "fallback_type": row.fallback_type}

    suffix = source_extension(row)
    if suffix is None:
        return {
            "party_key": row.party_key,
            "status": "unresolved_source",
            "source_url": row.source_url,
            "source_type": row.source_type,
        }

    destination_dir = staging_root / row.party_key
    destination = destination_dir / f"source{suffix}"
    if destination.exists() and not replace:
        return {
            "party_key": row.party_key,
            "status": "exists",
            "path": str(destination),
            "source_url": row.source_url,
        }

    http = session or requests.Session()
    response = http.get(row.source_url, timeout=TIMEOUT_SECONDS, stream=True, allow_redirects=True)
    response.raise_for_status()

    final_url = urlparse(response.url)
    if final_url.scheme != "https":
        raise ValueError(f"{row.party_key}: source redirected to non-HTTPS URL")

    content_type = response.headers.get("Content-Type", "")
    if not _content_type_ok(content_type, suffix):
        raise ValueError(f"{row.party_key}: unexpected Content-Type {content_type!r}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".part")
    total = 0
    try:
        with temp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_BYTES:
                    raise ValueError(f"{row.party_key}: source exceeds {MAX_BYTES} bytes")
                handle.write(chunk)
        if total == 0:
            raise ValueError(f"{row.party_key}: empty source response")
        temp.replace(destination)
    finally:
        if temp.exists():
            temp.unlink()

    return {
        "party_key": row.party_key,
        "status": "fetched",
        "path": str(destination),
        "bytes": total,
        "content_type": content_type,
        "source_url": row.source_url,
        "final_url": response.url,
        "guessed_mime": mimetypes.guess_type(destination.name)[0],
    }


def fetch_registry(
    staging_root: Path,
    registry_path: Path = DEFAULT_REGISTRY,
    replace: bool = False,
    allow_unresolved: bool = False,
) -> dict:
    rows = load_registry(registry_path)
    results = []
    errors = []

    for row in rows:
        try:
            result = fetch_row(row, staging_root, replace=replace)
        except Exception as exc:
            result = {"party_key": row.party_key, "status": "error", "error": str(exc)}
            errors.append(f"{row.party_key}: {exc}")
        results.append(result)

    unresolved = [item["party_key"] for item in results if item["status"] == "unresolved_source"]
    success = not errors and (allow_unresolved or not unresolved)
    report = {
        "registry": str(registry_path),
        "staging_root": str(staging_root),
        "success": success,
        "fetched_count": sum(item["status"] == "fetched" for item in results),
        "fallback_count": sum(item["status"] == "fallback" for item in results),
        "unresolved_count": len(unresolved),
        "unresolved_parties": unresolved,
        "errors": errors,
        "results": results,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch direct authoritative party logo sources into staging")
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--allow-unresolved", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = fetch_registry(
        Path(args.staging_root),
        Path(args.registry),
        replace=args.replace,
        allow_unresolved=args.allow_unresolved,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
