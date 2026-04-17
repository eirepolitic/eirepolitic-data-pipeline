from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest import TestCase


class InstagramRendererTest(TestCase):
    def test_local_fixture_render(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        output_root = repo_root / "test_output"
        if output_root.exists():
            for path in sorted(output_root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass

        cmd = [
            sys.executable,
            "process/instagram_render_post.py",
            "--spec",
            "instagram/specs/constituency_test_post.yml",
            "--data-source",
            "local",
            "--data-root",
            "tests/fixtures/instagram",
            "--output-dir",
            "test_output",
        ]
        subprocess.run(cmd, cwd=repo_root, check=True)

        post_dir = output_root / "constituency-test"
        png_dir = post_dir / "png"
        expected = [
            png_dir / "01_overview.png",
            png_dir / "02_member_profile.png",
            png_dir / "03_top_issues.png",
            png_dir / "04_member_top_issues.png",
            png_dir / "05_glossary.png",
        ]
        for path in expected:
            self.assertTrue(path.exists(), msg=f"Missing output: {path}")

        context = json.loads((post_dir / "post_context.json").read_text(encoding="utf-8"))
        self.assertEqual(context["constituency"]["name"], "Wicklow-Wexford")
        self.assertEqual(context["member"]["full_name"], "Aoife Byrne")
