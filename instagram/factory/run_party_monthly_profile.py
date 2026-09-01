from __future__ import annotations

import boto3
from PIL import Image, ImageDraw

from instagram.factory import party_monthly_profile as profile
from instagram.factory.oireachtas_production import resolve_production_key


def _name_aliases(value: str) -> set[str]:
    """Conservative aliases for Oireachtas display-name punctuation/title variants."""
    name = " ".join(str(value or "").strip().split())
    if not name:
        return set()
    straight = name.replace("’", "'").replace("‘", "'")
    curly = straight.replace("'", "’")
    aliases = {name, straight, curly}
    for variant in (name, straight, curly):
        if not variant.casefold().startswith("deputy "):
            aliases.add(f"Deputy {variant}")
    return aliases


def _install_member_name_aliases() -> None:
    original = profile._member_snapshot_for_period

    def wrapped(s3, period):
        rows, source = original(s3, period)
        augmented = list(rows)
        for row in rows:
            name = profile._field(row, ["Full Name", "Member Name", "Name", "full_name"])
            if not name:
                continue
            for alias in _name_aliases(name):
                if alias == name:
                    continue
                alias_row = dict(row)
                if "full_name" in alias_row:
                    alias_row["full_name"] = alias
                elif "Full Name" in alias_row:
                    alias_row["Full Name"] = alias
                else:
                    alias_row["full_name"] = alias
                augmented.append(alias_row)
        source = {**source, "name_alias_policy": "deputy-prefix-and-apostrophe-normalization"}
        return augmented, source

    profile._member_snapshot_for_period = wrapped


def _install_party_display_names() -> None:
    """Apply public-facing grouping names without changing the source data in S3."""
    original = profile._member_snapshot_for_period

    def wrapped(s3, period):
        rows, source = original(s3, period)
        renamed = []
        for row in rows:
            updated = dict(row)
            for key in ("party", "Party", "Party Name"):
                value = updated.get(key)
                if str(value or "").strip().casefold() == "independent":
                    updated[key] = "Independents"
            renamed.append(updated)
        source = {**source, "party_display_aliases": {"Independent": "Independents"}}
        return renamed, source

    profile._member_snapshot_for_period = wrapped


def _install_measured_context_centering() -> None:
    """Center the two lines below the title rule by their rendered pixel bounds."""
    original = profile._render_chart

    def wrapped(path, party, period, title_lines, supporting, rows, value_mode):
        original(path, party, period, title_lines, supporting, rows, value_mode)

        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image)
        region_top = profile.TITLE_RULE_Y + 5
        chart_top = profile.CHART_MEDIA_Y + 105

        # Clear only the context-text band. The title/rule and chart remain untouched.
        draw.rectangle((0, region_top, profile.W, chart_top - 1), fill=profile.BG)

        meta_text = f"{party.upper()} · {period.label.upper()}"
        support_text = supporting
        meta_font = profile._font(22, True)
        support_font = profile._font(24)

        meta_box = draw.textbbox((0, 0), meta_text, font=meta_font, anchor="lt")
        support_box = draw.textbbox((0, 0), support_text, font=support_font, anchor="lt")
        meta_w = meta_box[2] - meta_box[0]
        meta_h = meta_box[3] - meta_box[1]
        support_w = support_box[2] - support_box[0]
        support_h = support_box[3] - support_box[1]

        line_gap = 10
        block_h = meta_h + line_gap + support_h
        block_top = region_top + ((chart_top - region_top - block_h) // 2)

        meta_x = (profile.W - meta_w) // 2
        support_x = (profile.W - support_w) // 2
        draw.text((meta_x, block_top), meta_text, font=meta_font, fill=profile.ACCENT, anchor="lt")
        draw.text(
            (support_x, block_top + meta_h + line_gap),
            support_text,
            font=support_font,
            fill=profile.TEXT,
            anchor="lt",
        )
        image.save(path)

    profile._render_chart = wrapped


def main() -> None:
    s3 = boto3.client("s3", region_name="ca-central-1")
    resolved_key, pointer = resolve_production_key(
        s3,
        bucket=profile.S3_BUCKET,
        production_key=profile.CLASSIFIED_KEY,
    )
    print(f"Resolved unified production batch: {pointer.get('batch_id') or pointer.get('mode')}")
    print(f"Resolved classified source: {resolved_key}")
    profile.CLASSIFIED_KEY = resolved_key
    _install_member_name_aliases()
    _install_party_display_names()
    _install_measured_context_centering()
    profile.main()


if __name__ == "__main__":
    main()
