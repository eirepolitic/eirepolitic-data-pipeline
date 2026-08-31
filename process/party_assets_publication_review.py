#!/usr/bin/env python3
"""Build review-only logo candidates from official party publication PDFs.

This module never updates the canonical party registry and never writes to S3. It:
- downloads only configured HTTPS official-party PDFs;
- extracts embedded raster images from the first configured pages;
- renders top-of-page branding regions as additional review candidates;
- produces a labeled contact sheet and JSON report for human review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlparse

import fitz  # PyMuPDF
import requests
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/reference/party_asset_publication_candidates_v1.csv"
MAX_BYTES = 25 * 1024 * 1024
TIMEOUT_SECONDS = 45
CELL_W = 520
CELL_H = 520
COLUMNS = 2


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def load_config(path: Path = DEFAULT_CONFIG) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _download_pdf(url: str, session=None) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"PDF source must be HTTPS: {url}")
    http = session or requests.Session()
    response = http.get(
        url,
        timeout=TIMEOUT_SECONDS,
        stream=True,
        allow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 EirePoliticPartyAssetReview/1.0"},
    )
    response.raise_for_status()
    final = urlparse(response.url)
    if final.scheme != "https":
        raise ValueError("PDF source redirected to non-HTTPS URL")
    content_type = (response.headers.get("Content-Type") or "").lower()
    if "pdf" not in content_type and not response.url.lower().split("?", 1)[0].endswith(".pdf"):
        raise ValueError(f"Unexpected PDF Content-Type: {content_type!r}")
    chunks = []
    total = 0
    for chunk in response.iter_content(chunk_size=128 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_BYTES:
            raise ValueError(f"PDF exceeds {MAX_BYTES} bytes")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data.startswith(b"%PDF"):
        raise ValueError("Downloaded source is not a PDF")
    return data


def _save_preview(image: Image.Image, path: Path, max_size=(430, 330)) -> None:
    rgba = image.convert("RGBA")
    if rgba.getbbox() is None:
        raise ValueError("candidate image is empty")
    rgba.thumbnail(max_size, Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(path, "PNG")


def _render_region(page, clip: fitz.Rect) -> Image.Image:
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), clip=clip, alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    if image.getbbox() is None:
        raise ValueError("rendered page region is empty")
    return image


def extract_candidates(row: dict[str, str], output_root: Path, session=None) -> list[dict]:
    party_key = row["party_key"].strip()
    pdf_url = row["publication_url"].strip()
    page_limit = max(1, int(row.get("page_limit") or 1))
    data = _download_pdf(pdf_url, session=session)
    document = fitz.open(stream=data, filetype="pdf")
    entries: list[dict] = []
    seen_hashes: set[str] = set()
    candidate_index = 0

    try:
        for page_index in range(min(page_limit, document.page_count)):
            page = document.load_page(page_index)

            for image_index, image_info in enumerate(page.get_images(full=True), start=1):
                xref = image_info[0]
                extracted = document.extract_image(xref)
                raw = extracted.get("image", b"")
                digest = hashlib.sha256(raw).hexdigest()
                if not raw or digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                try:
                    with Image.open(io.BytesIO(raw)) as image:
                        image.load()
                        width, height = image.size
                        if width < 100 or height < 60:
                            continue
                        candidate_index += 1
                        preview_path = output_root / "previews" / party_key / f"embedded_{candidate_index:02d}.png"
                        _save_preview(image, preview_path)
                except Exception:
                    continue
                entries.append({
                    "party_key": party_key,
                    "party_name": row["party_name"],
                    "kind": "embedded_image",
                    "page": page_index + 1,
                    "image_index": image_index,
                    "width": width,
                    "height": height,
                    "preview_path": str(preview_path),
                    "publication_url": pdf_url,
                })

            rect = page.rect
            clip = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + rect.height * 0.35)
            image = _render_region(page, clip)
            candidate_index += 1
            preview_path = output_root / "previews" / party_key / f"top_region_p{page_index + 1}.png"
            _save_preview(image, preview_path, max_size=(460, 330))
            entries.append({
                "party_key": party_key,
                "party_name": row["party_name"],
                "kind": "top_page_region",
                "page": page_index + 1,
                "width": image.width,
                "height": image.height,
                "preview_path": str(preview_path),
                "publication_url": pdf_url,
            })
    finally:
        document.close()

    return entries


def build_contact_sheet(entries: list[dict], output: Path) -> None:
    rows = max(1, (len(entries) + COLUMNS - 1) // COLUMNS)
    sheet = Image.new("RGB", (COLUMNS * CELL_W, rows * CELL_H), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = _font(22, True)
    text_font = _font(16)

    for idx, entry in enumerate(entries):
        col = idx % COLUMNS
        row_idx = idx // COLUMNS
        left = col * CELL_W
        top = row_idx * CELL_H
        draw.rectangle((left, top, left + CELL_W - 1, top + CELL_H - 1), outline="black", width=1)
        box = (left + 20, top + 20, left + CELL_W - 20, top + 355)
        draw.rectangle(box, fill="#f4f4f4", outline="#999999", width=1)

        preview = Path(entry["preview_path"])
        if preview.is_file():
            with Image.open(preview) as image:
                image = image.convert("RGBA")
                x = box[0] + ((box[2] - box[0]) - image.width) // 2
                y = box[1] + ((box[3] - box[1]) - image.height) // 2
                sheet.paste(image, (x, y), image)

        draw.text((left + 20, top + 375), entry["party_name"], font=title_font, fill="black")
        draw.text((left + 20, top + 412), f"{entry['kind']} · page {entry['page']}", font=text_font, fill="black")
        if entry["kind"] == "embedded_image":
            draw.text((left + 20, top + 442), f"{entry['width']}×{entry['height']}", font=text_font, fill="black")
        draw.text((left + 20, top + 475), "REVIEW ONLY — not canonical", font=text_font, fill="#b71c1c")

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "PNG")


def build_review(config_path: Path, output_root: Path) -> dict:
    rows = load_config(config_path)
    entries: list[dict] = []
    errors: list[str] = []
    for row in rows:
        try:
            entries.extend(extract_candidates(row, output_root))
        except Exception as exc:
            errors.append(f"{row.get('party_key')}: {exc}")

    sheet_path = output_root / "publication_candidate_contact_sheet.png"
    build_contact_sheet(entries, sheet_path)
    report = {
        "success": not errors,
        "candidate_count": len(entries),
        "errors": errors,
        "entries": entries,
        "contact_sheet": str(sheet_path),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "publication_review.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build review candidates from official party publication PDFs")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    report = build_review(Path(args.config), Path(args.output_root))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
