"""Deterministic Instagram infographic renderer."""


def main() -> None:
    """Run the legacy renderer without importing its heavy dependencies at package import time."""
    from .render import main as render_main

    render_main()


__all__ = ["main"]
