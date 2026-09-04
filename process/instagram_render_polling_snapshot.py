from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from instagram.media_generators.horizontal_bar_chart.generator import render as render_bar_chart
from instagram.renderer.attribution import required_footer_text, resolve_attributions
from instagram.renderer.template_renderer import render_template_file

DEFAULT_REGION = "ca-central-1"
PARTY_LABELS = {
    "FF": "Fianna Fáil",
    "FG": "Fine Gael",
    "SF": "Sinn Féin",
    "LAB": "Labour",
    "GP": "Green Party",
    "PD": "Progressive Democrats",
    "WP": "Workers' Party",
    "DL": "Democratic Left",
    "SD": "Social Democrats",
    "SPBP": "PBP-Solidarity",
    "AU": "Aontú",
    "II": "Independent Ireland",
    "OTH": "Other",
}


def read_csv(path: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        _, _, rest = path.partition("s3://")
        bucket, _, key = rest.partition("/")
        s3 = boto3.client("s3", region_name=DEFAULT_REGION)
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    return pd.read_csv(path)


def _latest_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        raise RuntimeError("Polling indicator source is empty")
    if "date" not in df.columns or "cycle" not in df.columns:
        raise RuntimeError("Polling indicator source must contain date and cycle")
    parsed = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any():
        raise RuntimeError("Polling indicator source contains invalid dates")
    latest_date = parsed.max()
    latest = df.loc[parsed.eq(latest_date)].copy()
    if len(latest) != 1:
        raise RuntimeError(f"Expected one modeled row on latest date {latest_date.date()}, found {len(latest)}")
    return latest.iloc[0]


def build_chart_rows(row: pd.Series, party_codes: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for code in party_codes:
        if code not in row.index:
            continue
        estimate = pd.to_numeric(pd.Series([row.get(code)]), errors="coerce").iloc[0]
        lower = pd.to_numeric(pd.Series([row.get(f"{code}_lo")]), errors="coerce").iloc[0]
        upper = pd.to_numeric(pd.Series([row.get(f"{code}_hi")]), errors="coerce").iloc[0]
        if pd.isna(estimate):
            continue
        if pd.isna(lower) or pd.isna(upper):
            raise RuntimeError(f"Latest IPI row has incomplete uncertainty interval for {code}")
        rows.append(
            {
                "party_code": code,
                "label": PARTY_LABELS.get(code, code),
                "value": round(float(estimate) * 100, 1),
                "low": round(float(lower) * 100, 1),
                "high": round(float(upper) * 100, 1),
            }
        )
    return sorted(rows, key=lambda item: item["value"], reverse=True)


def render_polling_snapshot(spec_path: str | Path) -> dict[str, Any]:
    spec = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    if spec.get("campaign") != "ipi_polling_snapshot_v1":
        raise RuntimeError("Campaign must be ipi_polling_snapshot_v1")

    source_ids = spec.get("data", {}).get("source_ids", [])
    attributions = resolve_attributions(source_ids)
    if not any(item["source_id"] == "irish_polling_indicator" for item in attributions):
        raise RuntimeError("IPI polling campaign must declare source_id irish_polling_indicator")
    footer = required_footer_text(attributions)
    if not footer:
        raise RuntimeError("IPI attribution footer must not be empty")

    df = read_csv(spec["data"]["source_table"])
    row = _latest_row(df)
    party_codes = list(spec.get("variation", {}).get("party_codes", PARTY_LABELS.keys()))
    chart_rows = build_chart_rows(row, party_codes)
    limit = int(spec.get("variation", {}).get("limit", 8))
    chart_rows = chart_rows[:limit]
    if not chart_rows:
        raise RuntimeError("No party estimates available for polling snapshot")

    latest_date = str(row["date"])
    pretty_date = pd.Timestamp(latest_date).strftime("%d %b %Y")
    output_root = Path(spec.get("render", {}).get("output_root", "generated_posts/ipi_polling_snapshot_v1"))
    media_dir = output_root / "media"
    png_dir = output_root / "png"
    metadata_dir = output_root / "metadata"
    media_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    chart_spec = {
        "input": {"rows": chart_rows},
        "params": {
            "max_items": limit,
            "sort": "descending",
            "width": 920,
            "height": 820,
            "palette": spec.get("render", {}).get("palette", "eirepolitic_dark"),
            "title": "Modelled party support",
            "subtitle": f"IPI estimate · {pretty_date} · whiskers show uncertainty range",
            "value_suffix": "%",
        },
        "output": {},
    }
    chart_manifest = render_bar_chart(chart_spec, media_dir)
    media_path = Path(chart_manifest["output_path"])

    bindings = {
        "bindings": {
            "slide_title": spec.get("copy", {}).get("title", "Where the parties stand now"),
            "main_media": str(media_path),
            "footer_text": footer,
        }
    }
    bindings_path = metadata_dir / "bindings.yml"
    bindings_path.write_text(yaml.safe_dump(bindings, sort_keys=False, allow_unicode=True), encoding="utf-8")

    output_path = png_dir / "latest-party-support.png"
    render_manifest = render_template_file(
        spec.get("render", {}).get("template", "instagram/templates/layouts/big_media_title_v1.json"),
        bindings_path,
        output_path,
        spec.get("render", {}).get("palette", "eirepolitic_dark"),
    )

    with Image.open(output_path) as image:
        dimensions = list(image.size)
    if dimensions != [1080, 1350]:
        raise RuntimeError(f"Unexpected Instagram output dimensions: {dimensions}")

    source = next(item for item in attributions if item["source_id"] == "irish_polling_indicator")
    caption_lines = [
        spec.get("copy", {}).get("caption_intro", "Latest modelled Irish party-support estimates from the Irish Polling Indicator."),
        "",
        f"Model date: {pretty_date}.",
        "The bars show the central model estimate; whiskers show the published uncertainty range.",
        "This is a modelled polling indicator, not the result of a single opinion poll or an election forecast.",
        "",
        f"Source: {source['display_name']}",
        source["reference_url"],
    ]
    caption = "\n".join(caption_lines).strip() + "\n"
    caption_path = output_root / "caption.txt"
    caption_path.write_text(caption, encoding="utf-8")

    context = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign": "ipi_polling_snapshot_v1",
        "source_table": spec["data"]["source_table"],
        "source_attributions": attributions,
        "model_date": latest_date,
        "cycle": str(row["cycle"]),
        "chart_rows": chart_rows,
        "output_file": str(output_path),
        "caption_file": str(caption_path),
        "render_warnings": render_manifest.get("warnings", []),
        "chart_warnings": chart_manifest.get("warnings", []),
        "dimensions": dimensions,
        "publish_ready": False,
        "review_required": True,
    }
    context_path = metadata_dir / "post_context.json"
    context_path.write_text(json.dumps(context, indent=2, ensure_ascii=False), encoding="utf-8")

    review = {
        "success": True,
        "campaign": context["campaign"],
        "model_date": latest_date,
        "output_file": str(output_path),
        "caption_file": str(caption_path),
        "post_context": str(context_path),
        "visible_source_footer": footer,
        "source_reference_in_caption": source["reference_url"] in caption,
        "dimensions": dimensions,
        "publish_ready": False,
        "review_required": True,
        "checks": [
            "Confirm source footer is visible and readable.",
            "Confirm caption source reference is present.",
            "Confirm values and uncertainty ranges match post_context.json.",
            "Confirm modeled estimate is not described as a single poll or forecast.",
        ],
    }
    review_path = output_root / "review_manifest.json"
    review_path.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
    return review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render latest Irish Polling Indicator Instagram snapshot")
    parser.add_argument("--campaign", required=True, help="Path to polling campaign render_spec.yml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(render_polling_snapshot(args.campaign), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
