from __future__ import annotations

import io
import math
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def chart_axis_max(max_value: int) -> int:
    if max_value <= 0:
        return 1
    rough_step = max(1, math.ceil(max_value / 4))
    magnitude = 10 ** max(0, len(str(rough_step)) - 1)
    step = max(1, math.ceil(rough_step / magnitude) * magnitude)
    return step * math.ceil(max_value / step)


def build_bar_chart(
    rows: Iterable[dict],
    width: int,
    height: int,
    palette: dict[str, str],
    title: str,
) -> Image.Image:
    rows = list(rows)
    labels = [row["label"] for row in rows]
    values = [row["count"] for row in rows]

    fig = plt.figure(figsize=(width / 150, height / 150), dpi=150)
    ax = fig.add_subplot(111)
    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0))

    if values:
        ax.barh(range(len(values)), values, color=palette["chart_bar"])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=11)
        ax.invert_yaxis()
        axis_max = chart_axis_max(max(values))
        ax.set_xlim(0, axis_max)
        ax.xaxis.grid(True, color=palette["chart_grid"], alpha=0.4)
        ax.set_axisbelow(True)
    else:
        ax.text(0.5, 0.5, "No classified issue data available", ha="center", va="center", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

    ax.tick_params(axis="x", colors=palette["chart_tick"], labelsize=10)
    ax.tick_params(axis="y", colors=palette["chart_label"], labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(palette["chart_grid"])
    ax.spines["left"].set_visible(False)
    ax.set_title(title, fontsize=15, color=palette["chart_label"], loc="left", pad=12)

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", transparent=True, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGBA")
    return image.resize((width, height), Image.Resampling.LANCZOS)
