from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

DEFAULT_BUCKET = "eirepolitic-data"
DEFAULT_REGION = "ca-central-1"
OUTPUT_ROOT = "generated_posts"

DATASET_CANDIDATES: Dict[str, list[str]] = {
    "members": ["raw/members/oireachtas_members_34th_dail.csv"],
    "member_summaries": ["processed/members/members_summaries.csv"],
    "member_photos": [
        "processed/members/member_photos/members_photo_urls.csv",
        "processed/members/members_photo_urls.csv",
    ],
    "debate_issues": ["processed/debates/debate_speeches_classified.csv"],
    "constituency_images": ["processed/constituencies/constituency_images.csv"],
}

LOCAL_DATASET_FILENAMES: Dict[str, str] = {
    "members": "members.csv",
    "member_summaries": "member_summaries.csv",
    "member_photos": "member_photos.csv",
    "debate_issues": "debate_issues.csv",
    "constituency_images": "constituency_images.csv",
}

FONT_CANDIDATES: Dict[str, list[str]] = {
    "regular": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ],
    "bold": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ],
}

DEFAULT_PALETTE = {
    "bg": "#355E2B",
    "panel": "#355E2B",
    "panel_alt": "#446f39",
    "text": "#F3EFE6",
    "muted": "#E5DED1",
    "accent": "#F3EFE6",
    "accent_2": "#9EC5A2",
    "border": "#F3EFE6",
    "chart_bar": "#F3EFE6",
    "chart_grid": "#9EC5A2",
    "chart_label": "#F3EFE6",
    "chart_tick": "#E5DED1",
    "warning": "#D6C26E",
}

DEFAULT_SIZE = {"width": 1080, "height": 1350}


@dataclass(frozen=True)
class SlideGeometry:
    width: int = 1080
    height: int = 1350
    margin_x: int = 72
    margin_top: int = 72
    margin_bottom: int = 72
    panel_radius: int = 34
    gap: int = 24
