# Progress Log

## 2026-05-04

- Created `feature/instagram-content-factory-foundation` branch.
- Added first JSON template assets and palettes.
- Added Pillow-based JSON template renderer and CLI.
- Added horizontal bar chart generator, fake-data cases, and limits doc.
- Added manual GitHub Actions workflows for template rendering and media generator tests.
- Started `member_profile_batch_v1` campaign documentation.

## 2026-05-04 — v2 campaign fixes

- Reviewed merged `main` after foundation PR was merged.
- Found campaign renderer crash risk: `process/instagram_render_campaign.py` imported `render_template_file`, but `instagram/renderer/template_renderer.py` did not expose it.
- Added `render_template_file` wrapper for campaign rendering from JSON template + YAML/JSON bindings.
- Added support for template palette override in campaign renders.
- Added support for both `stroke` and `outline` rectangle keys.
- Added image placeholder background rendering before missing-image fallback marks.
- Added second media generator: `ranking_table` with example spec, fake-data test cases, and limits documentation.
- Registered `ranking_table` in `process/instagram_generate_media.py`.

## 2026-05-04 — workflow validation fixes

- Initial template render and media generator workflow runs failed at the command step after dependency installation.
- Added `--palette` support to `process/instagram_render_template.py` because the template workflow passes a palette override.
- Hardened template and media workflows so blank workflow-dispatch inputs fall back to default values.
- Added debug log artifact capture to both workflows for future failures.
- Fixed imports in `process/instagram_render_template.py`, `process/instagram_generate_media.py`, and `process/instagram_render_campaign.py` by inserting the repo root into `sys.path` before importing `instagram.*` modules. This matters when scripts are run as `python process/<script>.py` in GitHub Actions.
- Validation passed after the import-path fix:
  - Template render workflow run `25361718752` passed.
  - Media generator workflow run `25361720272` passed.
