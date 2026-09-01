from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import textwrap
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import boto3
import yaml
from PIL import Image, ImageDraw, ImageFont

from instagram.factory.periods import MonthlyPeriod, resolve_monthly_period

S3_BUCKET = "eirepolitic-data"
MEMBER_KEY = "raw/members/oireachtas_members_34th_dail.csv"
CLASSIFIED_KEY = "processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv"
PRESENTATION_LABELS_PATH = Path("instagram/reference/issue_presentation_labels.yml")
CORNER_DIR = Path("instagram/templates/assets")
PROJECT_ID = "party_issue_monthly_profile_v1"

W, H = 1080, 1350
BG = "#0f2f24"
TEXT = "#f4ead7"
ACCENT = "#d8b45f"
MUTED = "#c8bda8"
TITLE_RULE_Y = 174
CHART_MEDIA_Y = 184

GLOSSARY = [
    ("Issues", "Dáil speech segments are grouped into political issue categories based on what each segment is mainly about."),
    ("Classified Speeches", "Speech segments assigned to one of the issue categories. Counts show how often an issue was discussed, not a party’s position on it."),
    ("Average Party", "We calculate the result for each party and then take the simple average across parties, so larger parties do not count more heavily."),
    ("Per TD", "The number of classified speeches is divided by the number of TDs in that party. This adjusts the comparison for party size."),
    ("Points vs Average", "Shows how far above the average party a result is. For example, +6.1 pts means that issue made up 6.1 percentage points more of a party’s speeches than average."),
]


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"


def _field(row: dict[str, str], names: list[str]) -> str:
    lowered = {str(k).lower().strip(): k for k in row}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            return (row.get(key) or "").strip()
    return ""


def _parse_date(value: str) -> date | None:
    value = (value or "").strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value[:19], fmt).date()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _csv_rows(body: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(body.decode("utf-8-sig"))))


def _load_labels() -> dict[str, str]:
    data = yaml.safe_load(PRESENTATION_LABELS_PATH.read_text(encoding="utf-8")) or {}
    return {str(k): str(v) for k, v in (data.get("labels") or {}).items()}


def _s3_csv(s3: Any, key: str, version_id: str | None = None) -> tuple[list[dict[str, str]], dict[str, Any]]:
    kwargs: dict[str, Any] = {"Bucket": S3_BUCKET, "Key": key}
    if version_id:
        kwargs["VersionId"] = version_id
    obj = s3.get_object(**kwargs)
    rows = _csv_rows(obj["Body"].read())
    return rows, {
        "bucket": S3_BUCKET,
        "key": key,
        "version_id": obj.get("VersionId") or version_id,
        "last_modified": obj.get("LastModified").isoformat() if obj.get("LastModified") else None,
        "etag": str(obj.get("ETag") or "").strip('"'),
        "row_count": len(rows),
    }


def _member_snapshot_for_period(s3: Any, period: MonthlyPeriod) -> tuple[list[dict[str, str]], dict[str, Any]]:
    earliest = period.end
    latest = period.end + timedelta(days=7)
    head = s3.head_object(Bucket=S3_BUCKET, Key=MEMBER_KEY)
    head_modified = head["LastModified"].date()
    if earliest <= head_modified <= latest:
        return _s3_csv(s3, MEMBER_KEY, head.get("VersionId"))

    candidates: list[dict[str, Any]] = []
    key_marker = None
    version_marker = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": S3_BUCKET, "Prefix": MEMBER_KEY, "MaxKeys": 1000}
        if key_marker:
            kwargs["KeyMarker"] = key_marker
        if version_marker:
            kwargs["VersionIdMarker"] = version_marker
        page = s3.list_object_versions(**kwargs)
        for item in page.get("Versions", []):
            if item.get("Key") != MEMBER_KEY:
                continue
            modified = item["LastModified"].date()
            if earliest <= modified <= latest:
                candidates.append(item)
        if not page.get("IsTruncated"):
            break
        key_marker = page.get("NextKeyMarker")
        version_marker = page.get("NextVersionIdMarker")

    if not candidates:
        raise RuntimeError(
            f"No period-appropriate member snapshot found for {period.key}. Expected {MEMBER_KEY} dated between {earliest} and {latest}; current object is {head_modified}."
        )
    chosen = max(candidates, key=lambda item: item["LastModified"])
    return _s3_csv(s3, MEMBER_KEY, chosen.get("VersionId"))


def _font(size: int, bold: bool = False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, start: int, minimum: int, bold: bool = True):
    size = start
    while size > minimum:
        f = _font(size, bold)
        if draw.textbbox((0, 0), text, font=f)[2] <= max_width:
            return f
        size -= 2
    return _font(minimum, bold)


def _base_slide() -> Image.Image:
    image = Image.new("RGB", (W, H), BG)
    for filename, pos in [
        ("corner_tl.png", (0, 0)), ("corner_tr.png", (925, 0)),
        ("corner_bl.png", (0, 1195)), ("corner_br.png", (925, 1195)),
    ]:
        corner = Image.open(CORNER_DIR / filename).convert("RGBA").resize((155, 155), Image.Resampling.LANCZOS)
        image.paste(corner, pos, corner)
    return image


def _draw_title(image: Image.Image, title_lines: list[str]) -> None:
    draw = ImageDraw.Draw(image)
    if len(title_lines) == 1:
        font = _fit_font(draw, title_lines[0], 820, 56, 40)
        draw.text((540, 100), title_lines[0], font=font, fill=TEXT, anchor="mm")
    else:
        font = _font(45, True)
        draw.text((540, 76), title_lines[0], font=font, fill=TEXT, anchor="mm")
        draw.text((540, 128), title_lines[1], font=font, fill=TEXT, anchor="mm")
    draw.rectangle((112, TITLE_RULE_Y, 968, TITLE_RULE_Y + 4), fill=ACCENT)


def _period_dates(period: MonthlyPeriod) -> str:
    return f"{period.start.day} {period.start.strftime('%b')} – {period.end.day} {period.end.strftime('%b')} {period.end.year}"


def _render_cover(path: Path, party: str, speech_count: int, td_count: int, period: MonthlyPeriod) -> None:
    image = _base_slide(); _draw_title(image, [party]); draw = ImageDraw.Draw(image)
    cx, cy, radius = 540, 545, 225
    draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=ACCENT, width=10)
    words = party.split(); split = max(1, len(words)//2)
    lines = [party] if len(words) == 1 else [" ".join(words[:split]), " ".join(words[split:])]
    party_font = _font(68 if max(map(len, lines)) <= 14 else 54, True)
    if len(lines) == 1:
        draw.text((cx, cy), lines[0], font=party_font, fill=TEXT, anchor="mm")
    else:
        draw.text((cx, cy-46), lines[0], font=party_font, fill=TEXT, anchor="mm")
        draw.text((cx, cy+46), lines[1], font=party_font, fill=TEXT, anchor="mm")
    number_font, label_font, small_font = _font(72, True), _font(25, True), _font(24)
    avg = speech_count / td_count if td_count else 0.0
    draw.text((294, 955), f"{speech_count:,}", font=number_font, fill=TEXT, anchor="mm")
    draw.text((294, 1022), "CLASSIFIED SPEECHES", font=label_font, fill=ACCENT, anchor="mm")
    draw.text((786, 955), f"{avg:.1f}", font=number_font, fill=TEXT, anchor="mm")
    draw.text((786, 1022), "AVG SPEECHES PER TD", font=label_font, fill=ACCENT, anchor="mm")
    draw.line((239, 1115, 841, 1115), fill=ACCENT, width=3)
    draw.text((540, 1170), period.label.upper(), font=label_font, fill=ACCENT, anchor="mm")
    draw.text((540, 1218), _period_dates(period), font=small_font, fill=TEXT, anchor="mm")
    path.parent.mkdir(parents=True, exist_ok=True); image.save(path)


def _render_chart(path: Path, party: str, period: MonthlyPeriod, title_lines: list[str], supporting: str, rows: list[dict[str, Any]], value_mode: str) -> None:
    image = _base_slide(); _draw_title(image, title_lines); draw = ImageDraw.Draw(image)
    meta_font, support_font, label_font, value_font, source_font = _font(22, True), _font(24), _font(27, True), _font(25, True), _font(18)
    draw.text((86, CHART_MEDIA_Y+18), f"{party.upper()} · {period.label.upper()}", font=meta_font, fill=ACCENT, anchor="la")
    draw.text((86, CHART_MEDIA_Y+57), supporting, font=support_font, fill=TEXT, anchor="la")
    chart_top, chart_bottom = CHART_MEDIA_Y+105, 1152
    row_h = (chart_bottom-chart_top)/max(1, len(rows)); max_value = max((float(r["value"]) for r in rows), default=1.0) or 1.0
    label_x, bar_x, bar_max, value_x = 86, 465, 345, 950
    for idx, row in enumerate(rows):
        cy = chart_top + row_h*(idx+0.5); label = str(row["label"]); wrapped = textwrap.wrap(label, width=23)[:2] or [label]
        if len(wrapped) == 1: draw.text((label_x, cy), wrapped[0], font=label_font, fill=TEXT, anchor="lm")
        else:
            draw.text((label_x, cy-17), wrapped[0], font=label_font, fill=TEXT, anchor="lm"); draw.text((label_x, cy+17), wrapped[1], font=label_font, fill=TEXT, anchor="lm")
        value = float(row["value"]); width = max(4, int(bar_max*value/max_value))
        draw.rounded_rectangle((bar_x, int(cy-14), bar_x+width, int(cy+14)), radius=8, fill=ACCENT)
        value_text = f"{int(round(value)):,} speeches" if value_mode == "count" else (f"+{value:.1f} pts vs avg" if value_mode == "share_pp" else f"+{value:.2f} per TD vs avg")
        draw.text((value_x, cy), value_text, font=value_font, fill=TEXT, anchor="rm")
    draw.text((540, 1240), "Dáil speeches · Houses of the Oireachtas / Eirepolitic classification", font=source_font, fill=MUTED, anchor="mm")
    path.parent.mkdir(parents=True, exist_ok=True); image.save(path)


def _render_glossary(path: Path) -> None:
    image = _base_slide(); _draw_title(image, ["Glossary"]); draw = ImageDraw.Draw(image)
    term_font, body_font = _font(29, True), _font(23); y = 225
    for term, body in GLOSSARY:
        draw.text((135, y), term, font=term_font, fill=TEXT, anchor="la")
        bbox = draw.textbbox((135, y), term, font=term_font, anchor="la"); underline_y = bbox[3]+8
        draw.line((bbox[0], underline_y, bbox[2], underline_y), fill=ACCENT, width=2)
        lines = textwrap.wrap(body, width=79); body_y = underline_y+22
        for line in lines:
            draw.text((135, body_y), line, font=body_font, fill=TEXT, anchor="la"); body_y += 34
        y = body_y+42
    path.parent.mkdir(parents=True, exist_ok=True); image.save(path)


def _contact_sheet(paths: list[tuple[str, Path]], out_path: Path, columns: int = 4) -> None:
    thumb_w, thumb_h, label_h, gap = 250, 312, 34, 18; rows = math.ceil(len(paths)/columns)
    canvas = Image.new("RGB", (columns*(thumb_w+gap)+gap, rows*(thumb_h+label_h+gap)+gap), BG); draw = ImageDraw.Draw(canvas); label_font = _font(18, True)
    for idx, (label, path) in enumerate(paths):
        r, c = divmod(idx, columns); x = gap+c*(thumb_w+gap); y = gap+r*(thumb_h+label_h+gap)
        im = Image.open(path).convert("RGB"); im.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS); canvas.paste(im, (x+(thumb_w-im.width)//2, y))
        draw.text((x+thumb_w//2, y+thumb_h+20), label, font=label_font, fill=TEXT, anchor="mm")
    out_path.parent.mkdir(parents=True, exist_ok=True); canvas.save(out_path, quality=92)


def _carousel_sheet(items: list[tuple[str, list[Path]]], out_path: Path) -> None:
    thumb_w, thumb_h, row_h, left = 162, 203, 225, 210
    canvas = Image.new("RGB", (left+5*thumb_w+35, 30+len(items)*row_h), BG); draw = ImageDraw.Draw(canvas); label_font = _font(20, True)
    for row_idx, (party, paths) in enumerate(items):
        y = 20+row_idx*row_h; draw.text((20, y+thumb_h//2), party, font=label_font, fill=TEXT, anchor="lm")
        for col_idx, path in enumerate(paths):
            im = Image.open(path).convert("RGB"); im.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS); canvas.paste(im, (left+col_idx*thumb_w, y))
    out_path.parent.mkdir(parents=True, exist_ok=True); canvas.save(out_path, quality=92)


def build(period_value: str, output_root: Path) -> Path:
    period = resolve_monthly_period(period_value); labels = _load_labels(); s3 = boto3.client("s3")
    speeches, speech_source = _s3_csv(s3, CLASSIFIED_KEY); members, member_source = _member_snapshot_for_period(s3, period)
    coverage_dates = [d for row in speeches if (d := _parse_date(_field(row, ["Debate Date", "debate_date", "Date"])))]
    if not coverage_dates: raise RuntimeError("Classified speech source contains no parseable debate dates")
    coverage_end = max(coverage_dates)
    if coverage_end < period.end: raise RuntimeError(f"Classifier coverage ends {coverage_end}; requested period ends {period.end}")

    member_party: dict[str, str] = {}; party_members: dict[str, set[str]] = defaultdict(set)
    for row in members:
        name = _field(row, ["Full Name", "Member Name", "Name", "full_name"]); party = _field(row, ["Party", "party", "Party Name"])
        if name and party:
            key = name.casefold(); member_party[key] = party; party_members[party].add(key)
    if not party_members: raise RuntimeError("Member snapshot produced no party memberships")

    party_category: dict[str, Counter[str]] = defaultdict(Counter); party_totals: Counter[str] = Counter(); month_rows = 0; classified_month_rows = 0; unmatched: Counter[str] = Counter(); categories: set[str] = set()
    for row in speeches:
        d = _parse_date(_field(row, ["Debate Date", "debate_date", "Date"]))
        if not d or not (period.start <= d <= period.end): continue
        month_rows += 1; category = _field(row, ["PoliticalIssues", "issue_label", "Issue Category", "issue_category", "Category", "classification"])
        if not category or category.upper() == "NONE": continue
        classified_month_rows += 1; speaker = _field(row, ["Speaker Name", "speaker_name", "Speaker", "Member Name"]); party = member_party.get(speaker.casefold()) if speaker else None
        if not party: unmatched[speaker or "<blank>"] += 1; continue
        categories.add(category); party_category[party][category] += 1; party_totals[party] += 1
    if month_rows == 0 or classified_month_rows == 0: raise RuntimeError(f"No completed classified data found for {period.key}")
    if unmatched:
        preview = ", ".join(f"{k} ({v})" for k, v in unmatched.most_common(10)); raise RuntimeError(f"Member join is not publication-ready: {sum(unmatched.values())} unmatched classified rows: {preview}")

    parties = sorted(party_members); categories_sorted = sorted(categories); share_baseline: dict[str, float] = {}; per_td_baseline: dict[str, float] = {}
    for category in categories_sorted:
        shares, rates = [], []
        for party in parties:
            total = party_totals[party]; td_count = len(party_members[party]); shares.append(party_category[party][category]/total if total else 0.0); rates.append(party_category[party][category]/td_count if td_count else 0.0)
        share_baseline[category] = sum(shares)/len(shares); per_td_baseline[category] = sum(rates)/len(rates)

    period_root = output_root/f"period={period.key}"; parties_root = period_root/"parties"; qa_rows=[]; cover_paths=[]; raw_paths=[]; share_paths=[]; per_td_paths=[]; carousel_items=[]; party_manifests=[]
    for party in parties:
        key = slugify(party); td_count = len(party_members[party]); total = int(party_totals[party])
        raw_rows = [{"canonical_label": cat, "label": labels.get(cat, cat), "value": int(count)} for cat, count in party_category[party].most_common(7)]
        share_rows=[]; per_td_rows=[]
        for cat in categories_sorted:
            count=int(party_category[party][cat]); actual_share=count/total if total else 0.0; share_delta=(actual_share-share_baseline[cat])*100.0
            if share_delta>0: share_rows.append({"canonical_label":cat,"label":labels.get(cat,cat),"value":share_delta,"raw_count":count})
            actual_rate=count/td_count if td_count else 0.0; rate_delta=actual_rate-per_td_baseline[cat]
            if rate_delta>0: per_td_rows.append({"canonical_label":cat,"label":labels.get(cat,cat),"value":rate_delta,"raw_count":count})
        share_rows=sorted(share_rows,key=lambda r:r["value"],reverse=True)[:7]; per_td_rows=sorted(per_td_rows,key=lambda r:r["value"],reverse=True)[:7]
        if not raw_rows or not share_rows or not per_td_rows: raise RuntimeError(f"{party} does not have data for all three analytical slides")
        slides_dir=parties_root/key/"slides"; paths=[slides_dir/f"0{i}_{name}.png" for i,name in enumerate(["cover","most_discussed_issues","more_than_average","more_per_td","glossary"],start=1)]
        _render_cover(paths[0],party,total,td_count,period); _render_chart(paths[1],party,period,["Most Discussed Issues"],"Total classified speeches",raw_rows,"count"); _render_chart(paths[2],party,period,["Issues Discussed","More Than Average"],"Compared with the average party",share_rows,"share_pp"); _render_chart(paths[3],party,period,["Issues Discussed","More Per TD"],"Adjusted for party size",per_td_rows,"per_td"); _render_glossary(paths[4])
        party_manifest={"party":party,"party_key":key,"period":period.key,"td_count":td_count,"classified_speeches":total,"avg_speeches_per_td":round(total/td_count,4) if td_count else None,"raw_counts":raw_rows,"share_vs_average":share_rows,"per_td_vs_average":per_td_rows,"slides":[str(p.relative_to(period_root)) for p in paths],"review_state":"pending_human_review","publication_enabled":False}
        manifest_path=parties_root/key/"manifest.json"; manifest_path.parent.mkdir(parents=True,exist_ok=True); manifest_path.write_text(json.dumps(party_manifest,indent=2,ensure_ascii=False),encoding="utf-8"); party_manifests.append(party_manifest)
        for slide_no,p in enumerate(paths,start=1):
            with Image.open(p) as im: ok=im.size==(W,H)
            qa_rows.append({"party":party,"party_key":key,"slide":slide_no,"path":str(p.relative_to(period_root)),"dimensions_ok":ok,"status":"PASS" if ok else "FAIL"})
        cover_paths.append((party,paths[0])); raw_paths.append((party,paths[1])); share_paths.append((party,paths[2])); per_td_paths.append((party,paths[3])); carousel_items.append((party,paths))
    if any(row["status"]!="PASS" for row in qa_rows): raise RuntimeError("Rendered-slide QA failed")
    contact=period_root/"contact_sheets"; _contact_sheet(cover_paths,contact/"covers.jpg"); _contact_sheet(raw_paths,contact/"most_discussed_issues.jpg"); _contact_sheet(share_paths,contact/"more_than_average.jpg"); _contact_sheet(per_td_paths,contact/"more_per_td.jpg"); _carousel_sheet(carousel_items,contact/"five_slide_overview.jpg")
    qa_path=period_root/"qa_summary.csv"
    with qa_path.open("w",newline="",encoding="utf-8") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(qa_rows[0])); writer.writeheader(); writer.writerows(qa_rows)
    lineage={"project_id":PROJECT_ID,"generated_at":datetime.now(timezone.utc).isoformat(),"period":{"key":period.key,"start":period.start.isoformat(),"end":period.end.isoformat()},"readiness":{"month_complete":period.end<date.today(),"classifier_coverage_end":coverage_end.isoformat(),"classifier_covers_period_end":coverage_end>=period.end,"month_source_rows":month_rows,"classified_month_rows":classified_month_rows,"matched_classified_rows":sum(party_totals.values()),"unmatched_classified_rows":sum(unmatched.values()),"party_count":len(parties),"category_count":len(categories_sorted)},"sources":{"classified_speeches":speech_source,"member_snapshot":member_source},"calculation":{"raw_counts":"category classified speeches for party in period; top 7","share_vs_average":"party category share minus unweighted mean category share across all parties, including zero shares; positive top 7","per_td_vs_average":"party category speeches per TD minus unweighted mean party category rate per TD across all parties, including zero rates; positive top 7"},"presentation_labels":str(PRESENTATION_LABELS_PATH),"chart_geometry":{"title_rule_y":TITLE_RULE_Y,"chart_media_y":CHART_MEDIA_Y,"previous_main_media_y":190},"parties":party_manifests,"qa":{"slide_count":len(qa_rows),"passed":len(qa_rows),"failed":0},"review_state":"pending_human_review","publication_enabled":False}
    (period_root/"run_manifest.json").write_text(json.dumps(lineage,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"status":"PASS","period":period.key,"parties":len(parties),"slides":len(qa_rows),"output":str(period_root)},indent=2)); return period_root


def main() -> None:
    parser=argparse.ArgumentParser(description="Render the reusable monthly party issue profile carousel batch"); parser.add_argument("--period",default="last_completed_month",help="YYYY-MM or last_completed_month"); parser.add_argument("--output-root",default=f"instagram/commissioning/output/{PROJECT_ID}"); args=parser.parse_args(); build(args.period,Path(args.output_root))


if __name__ == "__main__": main()
