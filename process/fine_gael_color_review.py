#!/usr/bin/env python3
"""Build a review-only contact sheet of current colored Fine Gael logo treatments.

Uses official Fine Gael publication PDFs only. Nothing is promoted automatically and
nothing is written to S3.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import fitz
import requests
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "review/party_assets_v1/fine_gael_color"
TIMEOUT = 45
MAX_BYTES = 30 * 1024 * 1024

CANDIDATES = [
    {
        "id": "fg_yellow_eu_2024",
        "label": "Yellow block — EU manifesto 2024",
        "url": "https://www.finegael.ie/app/uploads/2024/05/Fine-Gael-European-Election-Manifesto-2024.pdf",
        "page": 0,
        "clip": (0.66, 0.00, 1.00, 0.30),
    },
    {
        "id": "fg_blue_dublin_2025",
        "label": "Blue block — Dublin report 2025",
        "url": "https://www.finegael.ie/app/uploads/2025/06/Building-A-Better-Dublin-June2025.pdf",
        "page": -1,
        "clip": (0.28, 0.42, 0.72, 0.78),
    },
    {
        "id": "fg_orange_business_2025",
        "label": "Orange block — business survey 2025",
        "url": "https://www.finegael.ie/Fine-Gael-Backing-Business-Survey.pdf",
        "page": -1,
        "clip": (0.28, 0.42, 0.72, 0.78),
    },
    {
        "id": "fg_yellow_meals_2026",
        "label": "Yellow block — meals report 2026",
        "url": "https://www.finegael.ie/app/uploads/2026/03/Hot-School-Meals-Report.pdf",
        "page": -1,
        "clip": (0.28, 0.42, 0.72, 0.78),
    },
]


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _download(url: str) -> bytes:
    response = requests.get(
        url,
        timeout=TIMEOUT,
        stream=True,
        allow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 EirePolitic-FineGael-ColorReview/1.0"},
    )
    response.raise_for_status()
    chunks = []
    total = 0
    for chunk in response.iter_content(128 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_BYTES:
            raise ValueError("PDF exceeds size limit")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data.startswith(b"%PDF"):
        raise ValueError("source is not a PDF")
    return data


def _render_candidate(item: dict) -> Image.Image:
    data = _download(item["url"])
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        page_index = item["page"] if item["page"] >= 0 else doc.page_count + item["page"]
        page = doc.load_page(page_index)
        rect = page.rect
        x0, y0, x1, y1 = item["clip"]
        clip = fitz.Rect(
            rect.x0 + rect.width * x0,
            rect.y0 + rect.height * y0,
            rect.x0 + rect.width * x1,
            rect.y0 + rect.height * y1,
        )
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=clip, alpha=False)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA")
    finally:
        doc.close()


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    errors = []

    for item in CANDIDATES:
        entry = dict(item)
        try:
            image = _render_candidate(item)
            image.thumbnail((520, 360), Image.Resampling.LANCZOS)
            preview = OUT / "previews" / f"{item['id']}.png"
            preview.parent.mkdir(parents=True, exist_ok=True)
            image.save(preview, "PNG")
            entry.update({"status": "reviewable", "preview": str(preview.relative_to(REPO_ROOT))})
        except Exception as exc:
            entry.update({"status": "error", "error": str(exc)})
            errors.append(f"{item['id']}: {exc}")
        entries.append(entry)

    reviewable = [e for e in entries if e.get("status") == "reviewable"]
    cols = 2
    cell_w, cell_h = 620, 500
    rows = max(1, (len(reviewable) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    title = _font(24, True)
    small = _font(16)

    for idx, entry in enumerate(reviewable):
        col = idx % cols
        row = idx // cols
        left = col * cell_w
        top = row * cell_h
        with Image.open(REPO_ROOT / entry["preview"]) as image:
            image = image.convert("RGBA")
            x = left + (cell_w - image.width) // 2
            y = top + 30 + (340 - image.height) // 2
            sheet.paste(image, (x, y), image)
        draw.text((left + 30, top + 390), f"{idx + 1}. {entry['label']}", font=title, fill="black")
        draw.text((left + 30, top + 435), "Official Fine Gael publication · review only", font=small, fill="#555555")

    sheet_path = OUT / "contact_sheet.png"
    sheet.save(sheet_path, "PNG")
    report = {
        "party_key": "fine-gael",
        "candidate_count": len(entries),
        "reviewable_count": len(reviewable),
        "errors": errors,
        "entries": entries,
        "contact_sheet": str(sheet_path.relative_to(REPO_ROOT)),
    }
    (OUT / "review.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    build()
