"""Generic browser review page builder (EirePolitic Director, Phase 3, §3.4).

Replaces the inline f-string HTML heredocs that today live hand-written
inside each bespoke render workflow (e.g.
`.github/workflows/instagram_polling_factory_render.yml`'s "Build browser
review page" step). Works from a normalized `contract.RenderResult` so one
implementation serves every project, regardless of how many slides it has or
whether they're grouped (party_issue_monthly_profile_v2: 55 slides across 11
parties) or flat (ipi_polling_factory_v1: 4 slides).

Two-step design:
  1. `stage_review_assets()` copies the rendered files (slides, contact
     sheets, caption, a slide-download zip) into a flat review directory
     with deterministic relative names, and returns a new RenderResult whose
     paths point at those relative names.
  2. `build_review_html()` is then a pure string-templating function over
     that already-staged RenderResult — no file IO, easy to unit test with a
     synthetic RenderResult.
"""

from __future__ import annotations

import html
import shutil
from dataclasses import replace
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from instagram.factory.contract import ContactSheetRef, RenderResult, SlideRef


def _slugify(value: str) -> str:
    out = []
    for ch in value.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-") or "item"


def stage_review_assets(result: RenderResult, review_dir: Path) -> RenderResult:
    """Copy every asset a RenderResult points at into `review_dir` with
    deterministic, web-safe relative filenames, and return a new
    RenderResult whose paths are those relative filenames.

    Idempotent: safe to call again against the same review_dir (e.g. a
    re-run), since it clears the directory first.
    """
    if review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)

    staged_slides: list[SlideRef] = []
    for index, slide in enumerate(result.slides, start=1):
        ext = Path(slide.path).suffix or ".png"
        relative_name = f"slide-{index:03d}-{_slugify(slide.id)}{ext}"
        shutil.copy2(slide.path, review_dir / relative_name)
        staged_slides.append(replace(slide, path=relative_name))

    staged_contact_sheets: list[ContactSheetRef] = []
    for sheet in result.contact_sheets:
        ext = Path(sheet.path).suffix or ".jpg"
        relative_name = f"contact-{_slugify(sheet.id)}{ext}"
        shutil.copy2(sheet.path, review_dir / relative_name)
        staged_contact_sheets.append(ContactSheetRef(id=sheet.id, path=relative_name))

    staged_caption_path: str | None = None
    staged_caption_text: str | None = None
    if result.caption_path and Path(result.caption_path).is_file():
        staged_caption_text = Path(result.caption_path).read_text(encoding="utf-8")
        (review_dir / "caption.txt").write_text(staged_caption_text, encoding="utf-8")
        staged_caption_path = "caption.txt"

    zip_name = "slides.zip"
    with ZipFile(review_dir / zip_name, "w", ZIP_DEFLATED) as archive:
        for slide in staged_slides:
            archive.write(review_dir / slide.path, arcname=slide.path)

    staged_raw = dict(result.raw)
    if staged_caption_text is not None:
        # build_review_html() reads the caption's *content* from here (it does
        # no file IO itself, see its docstring) — without this, a staged
        # caption.txt would sit on disk unreferenced by the page that's
        # supposed to display it.
        staged_raw["_staged_caption_text"] = staged_caption_text

    return replace(
        result,
        slides=tuple(staged_slides),
        contact_sheets=tuple(staged_contact_sheets),
        caption_path=staged_caption_path,
        package={**(result.package or {}), "review_zip": zip_name},
        raw=staged_raw,
    )


def _group_slides(slides: tuple[SlideRef, ...]) -> dict[str, list[SlideRef]]:
    """Groups slides by the part of their id before a '/', if present
    (party_issue_monthly_profile_v2's "party-key/slide-name" convention).
    Slides with no '/' in their id (ipi_polling_factory_v1) all land in one
    unnamed group, which the page renders as a flat grid."""
    groups: dict[str, list[SlideRef]] = {}
    for slide in slides:
        group = slide.id.split("/", 1)[0] if "/" in slide.id else ""
        groups.setdefault(group, []).append(slide)
    return groups


def build_review_html(result: RenderResult, *, project_label: str | None = None) -> str:
    """Pure function: build the review page HTML from an already-staged
    RenderResult (see `stage_review_assets`). No file IO."""
    label = project_label or result.project_id.replace("_", " ").title()
    qa = result.qa
    checks_rows = "".join(
        f'<tr class="{"pass" if check.passed else "fail"}">'
        f"<td>{html.escape(check.name)}</td>"
        f'<td>{"PASS" if check.passed else "FAIL"}</td>'
        f"<td>{html.escape(check.detail)}</td></tr>"
        for check in qa.checks
    )
    overall_status = "PASS" if qa.all_passed else "FAIL"

    meta_lines = [
        f"Project: {html.escape(result.project_id)}",
        f"Period: {html.escape(result.period_key)}",
        f"Slides: {qa.actual_slide_count} / {qa.expected_slide_count} expected",
        f"Review state: {html.escape(result.review_state)}",
        f"Publication enabled: {result.publication_enabled}",
    ]
    if result.source_batch_id:
        meta_lines.append(f"Source batch: {html.escape(result.source_batch_id)}")
    meta_html = "<br>".join(meta_lines)

    contact_sheets_html = "".join(
        f'<figure><img class="sheet" src="{html.escape(sheet.path)}" alt="{html.escape(sheet.id)} contact sheet">'
        f"<figcaption>{html.escape(sheet.id)}</figcaption></figure>"
        for sheet in result.contact_sheets
    )

    groups = _group_slides(result.slides)
    groups_html_parts = []
    for group_name, slides in groups.items():
        heading = f"<h3>{html.escape(group_name)}</h3>" if group_name else ""
        images = "".join(
            f'<img src="{html.escape(slide.path)}" alt="{html.escape(slide.id)}" loading="lazy">' for slide in slides
        )
        groups_html_parts.append(f'<section class="slide-group">{heading}<div class="slides">{images}</div></section>')
    groups_html = "".join(groups_html_parts)

    caption_html = ""
    # Caller is expected to have staged caption.txt's *content*, not just
    # its path, since this function does no file IO. build_review_html
    # accepts the content via the `raw` dict on RenderResult if present.
    caption_text = result.raw.get("_staged_caption_text") if result.caption_path else None
    if caption_text:
        caption_html = f"<h2>Caption</h2><pre>{html.escape(caption_text)}</pre>"

    download_html = ""
    if result.package and result.package.get("review_zip"):
        download_html = (
            f'<a class="download" href="{html.escape(result.package["review_zip"])}" download>'
            "Download all slides</a>"
        )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(label)} · review</title>
<style>
body{{margin:0;background:#0f2f24;color:#f4ead7;font-family:Arial,sans-serif}}
main{{max-width:1500px;margin:auto;padding:24px}}
h1{{margin-bottom:8px}}
h2{{margin-top:32px}}
h3{{color:#d8b45f;margin:24px 0 8px}}
.meta{{color:#cbbf9f;margin-bottom:18px;line-height:1.55}}
.status{{display:inline-block;padding:4px 10px;border-radius:4px;font-weight:700;margin-bottom:18px}}
.status.PASS{{background:#1f4a37;color:#8fe3b8}}
.status.FAIL{{background:#4a1f1f;color:#e38f8f}}
.download{{display:inline-block;margin:0 0 24px;padding:12px 18px;background:#d8b45f;color:#0f2f24;text-decoration:none;font-weight:700;border-radius:6px}}
.sheet{{width:100%;max-width:900px;height:auto;display:block;margin:0 0 12px}}
figure{{margin:0 0 24px}}
figcaption{{color:#cbbf9f;margin-top:6px}}
table{{border-collapse:collapse;width:100%;margin:12px 0 24px}}
td{{padding:6px 10px;border-bottom:1px solid #1f4a37}}
tr.fail td{{color:#e38f8f}}
.slides{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}
.slides img{{width:100%;height:auto;border-radius:4px}}
pre{{white-space:pre-wrap;background:#173d30;padding:18px;border-radius:8px}}
@media(max-width:1000px){{.slides{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:650px){{.slides{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<main>
<h1>{html.escape(label)}</h1>
<div class="meta">{meta_html}</div>
<div class="status {overall_status}">QA: {overall_status}</div><br>
{download_html}
<h2>Contact sheets</h2>
{contact_sheets_html or "<p>No contact sheets.</p>"}
<h2>QA checks</h2>
<table>{checks_rows or "<tr><td>No declarative checks were run.</td></tr>"}</table>
<h2>Slides</h2>
{groups_html}
{caption_html}
</main>
</body>
</html>"""
