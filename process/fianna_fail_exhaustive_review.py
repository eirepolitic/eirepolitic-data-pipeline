#!/usr/bin/env python3
"""Build an exhaustive review-only Fianna Fáil logo contact sheet.

Sources include explicit candidate files plus every plausible same-site image candidate
from the current official homepage. Nothing produced here is canonical and nothing is
written to S3.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, ImageDraw, ImageFont

from process.party_assets import PartyAsset
from process.party_assets_discover import discover_row

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/reference/fianna_fail_search_candidates_v1.csv"
OUT = REPO_ROOT / "review/party_assets_v1/fianna_fail"
MAX_BYTES = 15 * 1024 * 1024
TIMEOUT = 35
COLUMNS = 3
CELL_W = 500
CELL_H = 500


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def load_explicit() -> list[dict]:
    with CONFIG.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def official_discovery() -> list[dict]:
    row = PartyAsset(
        party_key="fianna-fail",
        party_name="Fianna Fáil",
        party_aliases=("Fianna Fail",),
        logo_s3_uri="",
        source_url="https://www.fiannafail.ie/",
        source_type="official_party_site",
        retrieval_date="2026-09-01",
        licence_usage_note="review only",
        asset_status="source_identified_pending_ingest",
        fallback_type="",
    )
    result = discover_row(row, limit=40)
    candidates = []
    for index, item in enumerate(result.get("candidates", []), start=1):
        candidates.append({
            "candidate_id": f"official_site_{index:02d}",
            "label": f"Official-site candidate {index}",
            "source_url": item["url"],
            "source_kind": "official_site_discovery",
            "notes": "; ".join(item.get("reasons", [])),
            "score": item.get("score", 0),
        })
    return candidates


def download(url: str) -> tuple[Image.Image, str]:
    response = requests.get(
        url,
        timeout=TIMEOUT,
        allow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 EirePolitic-FiannaFail-Review/1.0"},
        stream=True,
    )
    response.raise_for_status()
    chunks = []
    total = 0
    for chunk in response.iter_content(128 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_BYTES:
            raise ValueError("candidate exceeds size limit")
        chunks.append(chunk)
    data = b"".join(chunks)
    suffix = Path(urlparse(response.url).path).suffix.lower()
    if suffix == ".svg" or "svg" in (response.headers.get("Content-Type") or "").lower():
        import cairosvg
        data = cairosvg.svg2png(bytestring=data)
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return image.convert("RGBA"), response.url


def dedupe(candidates: list[dict]) -> list[dict]:
    seen = set()
    output = []
    for candidate in candidates:
        key = candidate["source_url"].split("?", 1)[0]
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return output


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = dedupe(load_explicit() + official_discovery())
    entries = []
    errors = []

    for candidate in candidates:
        entry = dict(candidate)
        try:
            image, final_url = download(candidate["source_url"])
            bbox = image.getbbox()
            if bbox is None:
                raise ValueError("empty/transparent image")
            image = image.crop(bbox)
            image.thumbnail((410, 320), Image.Resampling.LANCZOS)
            path = OUT / "previews" / f"{candidate['candidate_id']}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, "PNG")
            entry.update({
                "status": "reviewable",
                "preview": str(path.relative_to(REPO_ROOT)),
                "final_url": final_url,
                "width": image.width,
                "height": image.height,
            })
        except Exception as exc:
            entry.update({"status": "error", "error": str(exc)})
            errors.append(f"{candidate['candidate_id']}: {exc}")
        entries.append(entry)

    reviewable = [item for item in entries if item.get("status") == "reviewable"]
    rows = max(1, (len(reviewable) + COLUMNS - 1) // COLUMNS)
    sheet = Image.new("RGB", (COLUMNS * CELL_W, rows * CELL_H), "white")
    draw = ImageDraw.Draw(sheet)
    title = font(22, True)
    text = font(15)

    for idx, entry in enumerate(reviewable):
        col = idx % COLUMNS
        row = idx // COLUMNS
        left = col * CELL_W
        top = row * CELL_H
        draw.rectangle((left, top, left + CELL_W - 1, top + CELL_H - 1), outline="black", width=1)
        box = (left + 20, top + 20, left + CELL_W - 20, top + 340)
        mid = (box[0] + box[2]) // 2
        draw.rectangle((box[0], box[1], mid, box[3]), fill="#f4f4f4")
        draw.rectangle((mid + 1, box[1], box[2], box[3]), fill="#263238")
        draw.rectangle(box, outline="#999999", width=1)

        with Image.open(REPO_ROOT / entry["preview"]) as image:
            image = image.convert("RGBA")
            x = box[0] + ((box[2] - box[0]) - image.width) // 2
            y = box[1] + ((box[3] - box[1]) - image.height) // 2
            sheet.paste(image, (x, y), image)

        label = entry["label"]
        if len(label) > 34:
            label = label[:31] + "..."
        draw.text((left + 20, top + 360), f"{idx + 1}. {label}", font=title, fill="black")
        draw.text((left + 20, top + 398), entry.get("source_kind", ""), font=text, fill="black")
        note = entry.get("notes", "") or ""
        if len(note) > 58:
            note = note[:55] + "..."
        draw.text((left + 20, top + 425), note, font=text, fill="black")
        draw.text((left + 20, top + 462), "REVIEW ONLY", font=text, fill="#b71c1c")

    sheet_path = OUT / "contact_sheet.png"
    sheet.save(sheet_path, "PNG")
    report = {
        "party_key": "fianna-fail",
        "candidate_count": len(entries),
        "reviewable_count": len(reviewable),
        "errors": errors,
        "entries": entries,
        "contact_sheet": str(sheet_path.relative_to(REPO_ROOT)),
    }
    (OUT / "review.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, ensure_ascii=False))
