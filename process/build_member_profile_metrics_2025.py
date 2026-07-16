"""Backward-compatible entry point for the generic member profile metrics builder."""

from __future__ import annotations

import os

os.environ.setdefault("TARGET_YEAR", "2025")

from process.build_member_profile_metrics import main


if __name__ == "__main__":
    raise SystemExit(main())
