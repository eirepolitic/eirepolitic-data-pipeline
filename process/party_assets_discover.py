#!/usr/bin/env python3
"""Discover possible logo asset URLs from unresolved official party webpages.

This is deliberately advisory only:
- reads only HTTPS official-party pages from the registry
- parses image/source metadata and linked SVG/PNG/JPEG/WebP assets
- scores likely logo candidates
- never downloads candidate binaries
- never updates the registry
- never writes to S3
"""

from __future__ import annotations

import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

from process.party_assets import DEFAULT_REGISTRY, PartyAsset, canonical_party_key, load_registry

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".svg"}
TIMEOUT_SECONDS = 30
MAX_PAGE_BYTES = 4 * 1024 * 1024


class AssetHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.candidates: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs):
        attrs_dict = {str(key).lower(): str(value or "") for key, value in attrs}
        tag = tag.lower()

        if tag in {"img", "source"}:
            for attr in ("src", "data-src", "data-lazy-src"):
                value = attrs_dict.get(attr, "").strip()
                if value:
                    self.candidates.append({
                        "url": value,
                        "tag": tag,
                        "attr": attr,
                        "alt": attrs_dict.get("alt", ""),
                        "title": attrs_dict.get("title", ""),
                        "class": attrs_dict.get("class", ""),
                    })
            for attr in ("srcset", "data-srcset"):
                value = attrs_dict.get(attr, "").strip()
                for item in value.split(","):
                    candidate = item.strip().split(" ", 1)[0].strip()
                    if candidate:
                        self.candidates.append({
                            "url": candidate,
                            "tag": tag,
                            "attr": attr,
                            "alt": attrs_dict.get("alt", ""),
                            "title": attrs_dict.get("title", ""),
                            "class": attrs_dict.get("class", ""),
                        })

        if tag == "link":
            href = attrs_dict.get("href", "").strip()
            rel = attrs_dict.get("rel", "").lower()
            if href and any(token in rel for token in ("icon", "logo")):
                self.candidates.append({
                    "url": href,
                    "tag": tag,
                    "attr": "href",
                    "alt": "",
                    "title": attrs_dict.get("title", ""),
                    "class": attrs_dict.get("class", ""),
                })


def _is_supported_asset(url: str) -> bool:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix in SUPPORTED_EXTENSIONS


def _same_site(page_url: str, asset_url: str) -> bool:
    page_host = urlparse(page_url).hostname or ""
    asset_host = urlparse(asset_url).hostname or ""
    page_host = page_host.lower().removeprefix("www.")
    asset_host = asset_host.lower().removeprefix("www.")
    return asset_host == page_host or asset_host.endswith("." + page_host)


def _party_tokens(row: PartyAsset) -> set[str]:
    raw = {row.party_name, row.party_key, *row.party_aliases}
    tokens: set[str] = set()
    for value in raw:
        normalized = canonical_party_key(value)
        tokens.update(part for part in normalized.split("-") if len(part) >= 3)
    return tokens


def score_candidate(row: PartyAsset, candidate: dict[str, str], absolute_url: str) -> tuple[int, list[str]]:
    evidence = " ".join([
        absolute_url,
        candidate.get("alt", ""),
        candidate.get("title", ""),
        candidate.get("class", ""),
    ]).lower()
    score = 0
    reasons: list[str] = []

    if "logo" in evidence:
        score += 8
        reasons.append("contains_logo")
    if "brand" in evidence:
        score += 3
        reasons.append("contains_brand")
    if "header" in evidence or "navbar" in evidence or "site-logo" in evidence:
        score += 2
        reasons.append("header_or_site_logo_context")
    if Path(urlparse(absolute_url).path).suffix.lower() == ".svg":
        score += 2
        reasons.append("svg_source")

    normalized_evidence = canonical_party_key(evidence)
    for token in sorted(_party_tokens(row)):
        if token in normalized_evidence:
            score += 1
            reasons.append(f"party_token:{token}")

    negative_terms = ("favicon", "avatar", "author", "footer-social", "facebook", "instagram", "twitter", "x-logo")
    for term in negative_terms:
        if term in evidence:
            score -= 5
            reasons.append(f"negative:{term}")

    return score, reasons


def discover_row(row: PartyAsset, session=None, limit: int = 20) -> dict:
    if row.source_type != "official_party_site":
        return {"party_key": row.party_key, "status": "not_applicable", "candidates": []}

    parsed = urlparse(row.source_url)
    if parsed.scheme != "https" or not parsed.netloc:
        return {"party_key": row.party_key, "status": "invalid_source_url", "candidates": []}

    http = session or requests.Session()
    response = http.get(row.source_url, timeout=TIMEOUT_SECONDS, allow_redirects=True)
    response.raise_for_status()
    content = response.content[: MAX_PAGE_BYTES + 1]
    if len(content) > MAX_PAGE_BYTES:
        raise ValueError(f"{row.party_key}: page exceeds {MAX_PAGE_BYTES} bytes")

    parser = AssetHTMLParser()
    parser.feed(content.decode(response.encoding or "utf-8", errors="replace"))

    dedup: dict[str, dict] = {}
    for candidate in parser.candidates:
        absolute = urljoin(response.url, candidate["url"])
        parsed_asset = urlparse(absolute)
        if parsed_asset.scheme != "https" or not parsed_asset.netloc:
            continue
        if not _is_supported_asset(absolute):
            continue
        if not _same_site(response.url, absolute):
            continue
        score, reasons = score_candidate(row, candidate, absolute)
        existing = dedup.get(absolute)
        record = {
            "url": absolute,
            "score": score,
            "reasons": reasons,
            "tag": candidate.get("tag", ""),
            "attr": candidate.get("attr", ""),
            "alt": candidate.get("alt", ""),
            "title": candidate.get("title", ""),
            "class": candidate.get("class", ""),
        }
        if existing is None or score > existing["score"]:
            dedup[absolute] = record

    ranked = sorted(dedup.values(), key=lambda item: (-item["score"], item["url"]))[:limit]
    return {
        "party_key": row.party_key,
        "party_name": row.party_name,
        "status": "candidates_found" if ranked else "no_candidates",
        "page_url": row.source_url,
        "final_page_url": response.url,
        "candidate_count": len(ranked),
        "candidates": ranked,
    }


def discover_registry(registry_path: Path = DEFAULT_REGISTRY, limit: int = 20) -> dict:
    rows = load_registry(registry_path)
    results = []
    errors = []
    for row in rows:
        if row.source_type != "official_party_site":
            continue
        try:
            result = discover_row(row, limit=limit)
        except Exception as exc:
            result = {"party_key": row.party_key, "party_name": row.party_name, "status": "error", "error": str(exc), "candidates": []}
            errors.append(f"{row.party_key}: {exc}")
        results.append(result)

    return {
        "registry": str(registry_path),
        "success": not errors,
        "errors": errors,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover possible direct logo assets on unresolved official party sites")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output")
    args = parser.parse_args()

    report = discover_registry(Path(args.registry), limit=max(1, args.limit))
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
