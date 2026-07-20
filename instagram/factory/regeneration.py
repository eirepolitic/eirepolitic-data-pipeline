from __future__ import annotations

from .generic_regeneration import regenerate_project_items


def regenerate_selected(*args, **kwargs):
    """Backward-compatible entry point for generic targeted regeneration."""
    return regenerate_project_items(*args, **kwargs)
