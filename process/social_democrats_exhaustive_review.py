#!/usr/bin/env python3
"""Build a review-only Social Democrats identity contact sheet.

Includes the clean square logo plus distinct current treatments from official party
manifestos. Nothing produced here is canonical and nothing is written to S3.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from urllib.parse import urlparse

import fitz
import requests
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/reference/social_democrats_search_candidates_v1.csv"
OUT = REPO_ROOT / "review/party_assets_v1/social_democrats"
TIMEOUT = 45
MAX_BYTES = 25 * 1024 * 1024
COLUMNS = 3
CELL_W = 500
CELL_H = 500

PUBLICATION_CANDIDATES = [
    {
        "candidate_id": "sd_manifesto_cover_left",
        "label": "2024 manifesto white-on-purple",
        "url": "https://www.socialdemocrats.ie/wp-content/uploads/2024/11/GEManifesto24-1.pdf",
        "page": 0,
        "clip": (0.00, 0.00, 0.42, 0.28),
        "notes": "Official 2024 manifesto cover treatment; white logo on purple.",
    },
    {
        "candidate_id": "sd_manifesto_cover_right",
        "label": "2024 campaign white-box treatment",
        "url": "https://www.socialdemocrats.ie/wp-content/uploads/2024/11/GE24Manifesto.pdf",
        "page": 0,
        "clip": (0.48, 0.00, 1.00, 0.28),
        "notes": "Official 2024 campaign cover; boxed white Social Democrats logo treatment.",
    },
    {
        "candidate_id": "sd_manifesto_full_header",
        "label": "2024 manifesto header treatment",
        "url": "https://www.socialdemocrats.ie/wp-content/uploads/2024/11/GEManifesto2024.pdf",
        "page": 0,
        "clip": (0.00, 0.00, 0.62, 0.24),
        "notes": "Official 2024 manifesto header treatment.",
    },
]


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _download(url: str) -> tuple[bytes, str]:
    response = requests.get(
        url,
        timeout=TIMEOUT,
        stream=True,
        allow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 EirePolitic-SocialDemocrats-Review/1.0"},
    )
    response.raise_for_status()
    chunks = []
    total = 0
    for chunk in response.iter_content(128 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_BYTES:
            raise ValueError("source exceeds size limit")
        chunks.append(chunk)
    return b"".join(chunks), response.url


def _render_direct(url: str) -> tuple[Image.Image, str]:
    data, final_url = _download(url)
    suffix = Path(urlparse(final_url).path).suffix.lower()
    content = data
    if suffix == ".svg" or data.lstrip().startswith(b"<svg") or b"<svg" in data[:500]:
        import cairosvg
        content = cairosvg.svg2png(bytestring=data)
    with Image.open(io.BytesIO(content)) as image:
        image.load()
        rgba = image.convert("RGBA")
    bbox = rgba.getbbox()
    if bbox is None:
        raise ValueError("empty direct image")
    return rgba.crop(bbox), final_url


def _render_pdf_region(item: dict) -> Image.Image:
    data, _ = _download(item["url"])
    if not data.startswith(b"%PDF"):
        raise ValueError("source is not a PDF")
    document = fitz.open(stream=data, filetype="pdf")
    try:
        page = document.load_page(min(item["page"], document.page_count - 1))
        rect = page.rect
        x0, y0, x1, y1 = item["clip"]
        clip = fitz.Rect(
            rect.x0 + rect.width * x0,
            rect.y0 + rect.height * y0,
            rect.x0 + rect.width * x1,
            rect.y0 + rect.height * y1,
        )
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), clip=clip, alpha=False)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA")
    finally:
        document.close()


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    errors = []

    with CONFIG.open("r", encoding="utf-8-sig", newline="") as handle:
        direct = list(csv.DictReader(handle))

    for row in direct:
        entry = dict(row)
        try:
            image, final_url = _render_direct(row["source_url"])
            image.thumbnail((410, 320), Image.Resampling.LANCZOS)
            preview = OUT / "previews" / f"{row['candidate_id']}.png"
            preview.parent.mkdir(parents=True, exist_ok=True)
            image.save(preview, "PNG")
            entry.update({"status": "reviewable", "preview": str(preview.relative_to(REPO_ROOT)), "final_url": final_url})
        except Exception as exc:
            entry.update({"status": "error", "error": str(exc)})
            errors.append(f"{row['candidate_id']}: {exc}")
        entries.append(entry)

    for item in PUBLICATION_CANDIDATES:
        entry = {
            "candidate_id": item["candidate_id"],
            "label": item["label"],
            "source_kind": "official_party_publication",
            "source_url": item["url"],
            "notes": item["notes"],
        }
        try:
            image = _render_pdf_region(item)
            image.thumbnail((410, 320), Image.Resampling.LANCZOS)
            preview = OUT / "previews" / f"{item['candidate_id']}.png"
            preview.parent.mkdir(parents=True, exist_ok=True)
            image.save(preview, "PNG")
            entry.update({"status": "reviewable", "preview": str(preview.relative_to(REPO_ROOT))})
        except Exception as exc:
            entry.update({"status": "error", "error": str(exc)})
            errors.append(f"{item['candidate_id']}: {exc}")
        entries.append(entry)

    reviewable = [item for item in entries if item.get("status") == "reviewable"]
    rows = max(1, (len(reviewable) + COLUMNS - 1) // COLUMNS)
    sheet = Image.new("RGB", (COLUMNS * CELL_W, rows * CELL_H), "white")
    draw = ImageDraw.Draw(sheet)
    title = font(22, True)
    text = font(15)

    for idx, entry in enumerate(reviewable):
        col = idx % COLUMNS
        row_idx = idx // COLUMNS
        left = col * CELL_W
        top = row_idx * CELL_H
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
        draw.text((left + 20, top + 398), entry["source_kind"], font=text, fill="black")
        notes = entry.get("notes", "")
        if len(notes) > 58:
            notes = notes[:55] + "..."
        draw.text((left + 20, top + 425), notes, font=text, fill="black")
        draw.text((left + 20, top + 462), "REVIEW ONLY", font=text, fill="#b71c1c")

    sheet_path = OUT / "contact_sheet.png"
    sheet.save(sheet_path, "PNG")
    report = {
        "party_key": "social-democrats",
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
