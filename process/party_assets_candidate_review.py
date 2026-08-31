#!/usr/bin/env python3
"""Build a temporary visual review sheet for unresolved party logo candidates.

This tool consumes the advisory discovery report. It does not alter the canonical
registry and does not write to S3. Only high-confidence same-site candidates already
identified by party_assets_discover.py are fetched for human review.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from urllib.parse import urlparse

import cairosvg
import requests
from PIL import Image, ImageDraw, ImageFont

MIN_SCORE = 8
MAX_CANDIDATES_PER_PARTY = 4
MAX_BYTES = 12 * 1024 * 1024
TIMEOUT_SECONDS = 30
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".svg"}
CELL_W = 520
CELL_H = 500
COLUMNS = 2


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _fetch_bytes(url: str, session=None) -> tuple[bytes, str]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or Path(parsed.path).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported candidate URL: {url}")
    http = session or requests.Session()
    response = http.get(url, timeout=TIMEOUT_SECONDS, stream=True, allow_redirects=True)
    response.raise_for_status()
    total = 0
    chunks: list[bytes] = []
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_BYTES:
            raise ValueError(f"Candidate exceeds {MAX_BYTES} bytes: {url}")
        chunks.append(chunk)
    if not chunks:
        raise ValueError(f"Empty candidate response: {url}")
    return b"".join(chunks), response.headers.get("Content-Type", "")


def _to_rgba(data: bytes, suffix: str) -> Image.Image:
    if suffix == ".svg":
        data = cairosvg.svg2png(bytestring=data)
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        rgba = image.convert("RGBA")
    if rgba.getbbox() is None:
        raise ValueError("Candidate is fully transparent/empty")
    return rgba


def select_candidates(discovery: dict) -> list[dict]:
    selected: list[dict] = []
    for result in discovery.get("results", []):
        party_key = result.get("party_key")
        candidates = [
            candidate for candidate in result.get("candidates", [])
            if int(candidate.get("score", 0)) >= MIN_SCORE
        ][:MAX_CANDIDATES_PER_PARTY]
        for candidate in candidates:
            selected.append({"party_key": party_key, **candidate})
    return selected


def build_candidate_review(discovery_path: Path, output_root: Path, session=None) -> dict:
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    selected = select_candidates(discovery)
    entries: list[dict] = []
    errors: list[str] = []

    for index, candidate in enumerate(selected, start=1):
        url = candidate["url"]
        suffix = Path(urlparse(url).path).suffix.lower()
        entry = dict(candidate)
        try:
            data, content_type = _fetch_bytes(url, session=session)
            image = _to_rgba(data, suffix)
            bbox = image.getbbox()
            assert bbox is not None
            cropped = image.crop(bbox)
            preview = cropped.copy()
            preview.thumbnail((430, 300), Image.Resampling.LANCZOS)
            preview_path = output_root / "previews" / candidate["party_key"] / f"candidate_{index:02d}.png"
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            preview.save(preview_path, "PNG")
            entry.update({
                "status": "reviewable",
                "content_type": content_type,
                "width": image.width,
                "height": image.height,
                "preview_path": str(preview_path),
            })
        except Exception as exc:
            entry.update({"status": "error", "error": str(exc)})
            errors.append(f"{candidate.get('party_key')}: {exc}")
        entries.append(entry)

    rows = max(1, (len(entries) + COLUMNS - 1) // COLUMNS)
    sheet = Image.new("RGB", (COLUMNS * CELL_W, rows * CELL_H), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = _font(22, True)
    text_font = _font(16)

    for idx, entry in enumerate(entries):
        col = idx % COLUMNS
        row = idx // COLUMNS
        left = col * CELL_W
        top = row * CELL_H
        draw.rectangle((left, top, left + CELL_W - 1, top + CELL_H - 1), outline="black", width=1)
        box = (left + 20, top + 20, left + CELL_W - 20, top + 325)
        mid = (box[0] + box[2]) // 2
        draw.rectangle((box[0], box[1], mid, box[3]), fill="#f2f2f2")
        draw.rectangle((mid + 1, box[1], box[2], box[3]), fill="#263238")
        draw.rectangle(box, outline="#999999", width=1)

        preview_path = entry.get("preview_path")
        if preview_path and Path(preview_path).is_file():
            with Image.open(preview_path) as preview:
                preview = preview.convert("RGBA")
                x = box[0] + ((box[2] - box[0]) - preview.width) // 2
                y = box[1] + ((box[3] - box[1]) - preview.height) // 2
                sheet.paste(preview, (x, y), preview)
        else:
            draw.text((left + CELL_W // 2, top + 170), "CANDIDATE ERROR", font=title_font, anchor="mm", fill="#b71c1c")

        draw.text((left + 20, top + 345), str(entry.get("party_key", "")), font=title_font, fill="black")
        draw.text((left + 20, top + 380), f"score={entry.get('score', 0)}  status={entry.get('status', '')}", font=text_font, fill="black")
        filename = Path(urlparse(str(entry.get("url", ""))).path).name
        if len(filename) > 55:
            filename = filename[:52] + "..."
        draw.text((left + 20, top + 410), filename, font=text_font, fill="black")
        reasons = ", ".join(entry.get("reasons", []))
        if len(reasons) > 65:
            reasons = reasons[:62] + "..."
        draw.text((left + 20, top + 440), reasons, font=text_font, fill="black")

    output_root.mkdir(parents=True, exist_ok=True)
    sheet_path = output_root / "candidate_contact_sheet.png"
    sheet.save(sheet_path, "PNG")
    report = {
        "success": not errors,
        "minimum_score": MIN_SCORE,
        "selected_count": len(selected),
        "reviewable_count": sum(entry.get("status") == "reviewable" for entry in entries),
        "errors": errors,
        "entries": entries,
        "contact_sheet": str(sheet_path),
    }
    (output_root / "candidate_review.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a temporary contact sheet for unresolved logo candidates")
    parser.add_argument("--discovery", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    report = build_candidate_review(Path(args.discovery), Path(args.output_root))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
