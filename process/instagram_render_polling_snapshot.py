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

from instagram.polling.editorial_renderer import render_change, render_latest_support, render_trend
from instagram.renderer.attribution import required_footer_text, resolve_attributions

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


def _prepare_indicator(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise RuntimeError("Polling indicator source is empty")
    if "date" not in df.columns or "cycle" not in df.columns:
        raise RuntimeError("Polling indicator source must contain date and cycle")
    result = df.copy()
    result["_parsed_date"] = pd.to_datetime(result["date"], format="%Y-%m-%d", errors="coerce")
    if result["_parsed_date"].isna().any():
        raise RuntimeError("Polling indicator source contains invalid dates")
    return result


def _latest_row(df: pd.DataFrame) -> pd.Series:
    latest_date = df["_parsed_date"].max()
    latest = df.loc[df["_parsed_date"].eq(latest_date)].copy()
    if len(latest) != 1:
        raise RuntimeError(f"Expected one modeled row on latest date {latest_date.date()}, found {len(latest)}")
    return latest.iloc[0]


def _previous_row_same_cycle(df: pd.DataFrame, latest: pd.Series) -> pd.Series:
    cycle = str(latest["cycle"])
    latest_date = latest["_parsed_date"]
    candidates = df.loc[df["cycle"].astype(str).eq(cycle) & df["_parsed_date"].lt(latest_date)].copy()
    if candidates.empty:
        raise RuntimeError(f"No previous modeled IPI date is available within cycle {cycle}")
    previous_date = candidates["_parsed_date"].max()
    previous = candidates.loc[candidates["_parsed_date"].eq(previous_date)]
    if len(previous) != 1:
        raise RuntimeError(f"Expected one previous modeled row on {previous_date.date()} in cycle {cycle}, found {len(previous)}")
    return previous.iloc[0]


def _numeric(row: pd.Series, column: str) -> float | None:
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def build_chart_rows(row: pd.Series, party_codes: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for code in party_codes:
        estimate = _numeric(row, code)
        if estimate is None:
            continue
        lower = _numeric(row, f"{code}_lo")
        upper = _numeric(row, f"{code}_hi")
        if lower is None or upper is None:
            raise RuntimeError(f"Latest IPI row has incomplete uncertainty interval for {code}")
        rows.append(
            {
                "party_code": code,
                "label": PARTY_LABELS.get(code, code),
                "value": round(estimate * 100, 1),
                "low": round(lower * 100, 1),
                "high": round(upper * 100, 1),
            }
        )
    return sorted(rows, key=lambda item: item["value"], reverse=True)


def build_change_rows(latest: pd.Series, previous: pd.Series, party_codes: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for code in party_codes:
        current = _numeric(latest, code)
        prior = _numeric(previous, code)
        if current is None or prior is None:
            continue
        rows.append(
            {
                "party_code": code,
                "label": PARTY_LABELS.get(code, code),
                "current_pct": round(current * 100, 2),
                "previous_pct": round(prior * 100, 2),
                "value": round((current - prior) * 100, 2),
            }
        )
    return sorted(rows, key=lambda item: item["value"], reverse=True)


def build_trend_series(
    df: pd.DataFrame,
    latest: pd.Series,
    party_codes: list[str],
    *,
    days: int,
    party_limit: int,
) -> list[dict[str, Any]]:
    cycle = str(latest["cycle"])
    latest_date = latest["_parsed_date"]
    start_date = latest_date - pd.Timedelta(days=max(days, 1))
    window = df.loc[
        df["cycle"].astype(str).eq(cycle)
        & df["_parsed_date"].between(start_date, latest_date, inclusive="both")
    ].sort_values("_parsed_date")
    if window.empty:
        raise RuntimeError("No modeled IPI rows available in trend window")

    ranked: list[tuple[str, float]] = []
    for code in party_codes:
        estimate = _numeric(latest, code)
        if estimate is not None:
            ranked.append((code, estimate))
    ranked_codes = [code for code, _ in sorted(ranked, key=lambda pair: pair[1], reverse=True)[:party_limit]]

    series: list[dict[str, Any]] = []
    for code in ranked_codes:
        points: list[dict[str, Any]] = []
        for _, row in window.iterrows():
            value = _numeric(row, code)
            if value is None:
                continue
            points.append({"date": str(row["date"]), "value": round(value * 100, 2)})
        if points:
            series.append({"party_code": code, "label": PARTY_LABELS.get(code, code), "points": points})
    return series


def _validate_slide(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Polling slide was not created: {path}")
    with Image.open(path) as image:
        if image.size != (1080, 1350):
            raise RuntimeError(f"Unexpected Instagram output dimensions for {path.name}: {image.size}")


def render_polling_snapshot(spec_path: str | Path) -> dict[str, Any]:
    spec = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    if spec.get("campaign") != "ipi_polling_snapshot_v1":
        raise RuntimeError("Campaign must be ipi_polling_snapshot_v1")

    attributions = resolve_attributions(spec.get("data", {}).get("source_ids", []))
    if not any(item["source_id"] == "irish_polling_indicator" for item in attributions):
        raise RuntimeError("IPI polling campaign must declare source_id irish_polling_indicator")
    footer = required_footer_text(attributions)
    if not footer:
        raise RuntimeError("IPI attribution footer must not be empty")

    df = _prepare_indicator(read_csv(spec["data"]["source_table"]))
    latest = _latest_row(df)
    previous = _previous_row_same_cycle(df, latest)
    party_codes = list(spec.get("variation", {}).get("party_codes", PARTY_LABELS.keys()))
    limit = int(spec.get("variation", {}).get("limit", 8))
    trend_days = int(spec.get("variation", {}).get("trend_days", 90))
    trend_party_limit = int(spec.get("variation", {}).get("trend_party_limit", 5))

    latest_rows = build_chart_rows(latest, party_codes)[:limit]
    change_rows = build_change_rows(latest, previous, party_codes)[:limit]
    trend_series = build_trend_series(
        df,
        latest,
        party_codes,
        days=trend_days,
        party_limit=trend_party_limit,
    )
    if not latest_rows or not change_rows or not trend_series:
        raise RuntimeError("Polling carousel requires latest, change, and trend data")

    latest_date = str(latest["date"])
    previous_date = str(previous["date"])
    pretty_latest = pd.Timestamp(latest_date).strftime("%-d %b %Y")
    pretty_previous = pd.Timestamp(previous_date).strftime("%-d %b %Y")
    trend_start_date = min(point["date"] for item in trend_series for point in item["points"])
    pretty_trend_start = pd.Timestamp(trend_start_date).strftime("%-d %b %Y")

    output_root = Path(spec.get("render", {}).get("output_root", "generated_posts/ipi_polling_snapshot_v1"))
    png_dir = output_root / "png"
    metadata_dir = output_root / "metadata"
    png_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    slides = [
        {
            "index": 1,
            "kind": "latest_support",
            "output": png_dir / "slide-01-latest-party-support.png",
        },
        {
            "index": 2,
            "kind": "change_since_previous_model_date",
            "output": png_dir / "slide-02-change.png",
        },
        {
            "index": 3,
            "kind": "trend",
            "output": png_dir / "slide-03-trend.png",
        },
    ]

    render_latest_support(
        latest_rows,
        title=spec.get("copy", {}).get("slide_1_title", "Where the parties stand now"),
        model_date=pretty_latest,
        source_text=footer,
        output_path=slides[0]["output"],
    )
    render_change(
        change_rows,
        title=spec.get("copy", {}).get("slide_2_title", "What changed since the previous model"),
        previous_date=pretty_previous,
        latest_date=pretty_latest,
        source_text=footer,
        output_path=slides[1]["output"],
    )
    render_trend(
        trend_series,
        title=spec.get("copy", {}).get("slide_3_title", "The recent polling trend"),
        start_date=pretty_trend_start,
        latest_date=pretty_latest,
        source_text=footer,
        output_path=slides[2]["output"],
    )

    for slide in slides:
        _validate_slide(slide["output"])

    source = next(item for item in attributions if item["source_id"] == "irish_polling_indicator")
    caption_lines = [
        spec.get("copy", {}).get("caption_intro", "Latest modelled Irish party-support estimates from the Irish Polling Indicator."),
        "",
        f"Latest model date: {pretty_latest}.",
        "Slide 1 shows the central model estimate with the published uncertainty range.",
        f"Slide 2 shows the percentage-point change from the previous modeled date ({pretty_previous}); it is not a comparison of two individual polls.",
        f"Slide 3 shows the last {trend_days} days within the same election cycle for the leading parties by latest model estimate.",
        "",
        "The Irish Polling Indicator is a modelled polling series, not a single opinion poll or an election forecast.",
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
        "visual_reference": "workflow 33894430571 / party_issue_monthly_profile_v2",
        "source_table": spec["data"]["source_table"],
        "source_attributions": attributions,
        "cycle": str(latest["cycle"]),
        "latest_model_date": latest_date,
        "previous_model_date": previous_date,
        "trend_start_date": trend_start_date,
        "trend_days": trend_days,
        "latest_rows": latest_rows,
        "change_rows": change_rows,
        "trend_series": trend_series,
        "slides": [
            {"index": slide["index"], "kind": slide["kind"], "output_file": str(slide["output"])}
            for slide in slides
        ],
        "caption_file": str(caption_path),
        "render_style": "approved_editorial_v1",
        "render_warnings": [],
        "dimensions": [1080, 1350],
        "publish_ready": False,
        "review_required": True,
    }
    context_path = metadata_dir / "post_context.json"
    context_path.write_text(json.dumps(context, indent=2, ensure_ascii=False), encoding="utf-8")

    review = {
        "success": True,
        "campaign": context["campaign"],
        "visual_reference": context["visual_reference"],
        "render_style": context["render_style"],
        "latest_model_date": latest_date,
        "previous_model_date": previous_date,
        "slide_count": 3,
        "slide_files": [str(slide["output"]) for slide in slides],
        "caption_file": str(caption_path),
        "post_context": str(context_path),
        "visible_source_footer": footer,
        "source_reference_in_caption": source["reference_url"] in caption,
        "dimensions": [1080, 1350],
        "publish_ready": False,
        "review_required": True,
        "checks": [
            "Confirm the redesign matches the approved July visual system and spacing.",
            "Confirm IPI source footer is visible on all three slides.",
            "Confirm caption source reference is present.",
            "Confirm latest estimates and uncertainty ranges match post_context.json.",
            "Confirm slide 2 is described as change between modeled dates, not individual polls.",
            "Confirm slide 3 stays within the current election cycle and configured trend window.",
            "Confirm the carousel does not describe the modeled series as an election forecast.",
        ],
    }
    review_path = output_root / "review_manifest.json"
    review_path.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
    return review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Irish Polling Indicator Instagram carousel")
    parser.add_argument("--campaign", required=True, help="Path to polling campaign render_spec.yml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(render_polling_snapshot(args.campaign), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
