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

## 2026-05-04 — fixture campaign validation

- Added local fixture input data at `instagram/campaigns/member_profile_batch_v1/fixtures/member_profile_metrics_fixture.csv`.
- Added local fixture render spec at `instagram/campaigns/member_profile_batch_v1/render_spec_fixture.yml`.
- Changed the campaign workflow default to render the fixture spec first, so campaign rendering can be validated without S3 or AWS secrets.
- Kept production S3 rendering available by setting `spec_file=render_spec.yml` when manually dispatching the workflow.
- Added `all` generator mode to `process/instagram_generate_media.py`.
- Changed the media workflow default to `generator=all`, so all registered generators are tested by default.
- Validation passed:
  - All-generators media workflow run `25361969750` passed.
  - Fixture campaign render workflow run `25361971238` passed.

## 2026-05-04 — review pack upgrade

- Upgraded `process/instagram_render_campaign.py` review outputs.
- `review_table.csv` now includes review workflow columns: `review_status`, `review_notes`, `publish_ready`, `needs_photo_check`, and `has_render_warnings`.
- `review_table.csv` now includes relative output, bindings, and render-manifest paths.
- Added campaign-level `review_manifest.json` with item metadata and a review checklist.
- Improved `review_index.html` with metric details, warning display, binding/manifest references, and clearer human-review wording.
- Validation passed:
  - Fixture campaign render with upgraded review pack run `25362379007` passed.

## 2026-05-04 — copy pack v1

- Added deterministic copy-pack builder at `process/instagram_build_copy_pack.py`.
- The copy-pack builder reads a campaign `review_table.csv` and writes:
  - `copy/captions.csv`
  - one `.caption.txt` file per post
  - one `.alt_text.txt` file per post
  - `copy/copy_manifest.json`
- Captions and alt text are deterministic drafts only; they preserve the existing human-review gate by carrying review status, publish readiness, and safety notes from the review table.
- Updated the campaign render workflow to run the copy-pack builder after campaign rendering and include copy outputs in the artifact.
- Validation passed:
  - Campaign render plus copy-pack workflow run `25362929727` passed.

## 2026-05-05 — gated publish queue v1

- Added review-gated publish queue builder at `process/instagram_build_publish_queue.py`.
- The queue builder reads `copy/captions.csv` and writes:
  - `queue/publish_queue.csv`
  - `queue/blocked_items.csv`
  - `queue/publish_queue_manifest.json`
- Queue rules require all of the following before an item is queued:
  - `publish_ready` is yes/true/1
  - `review_status` is approved/ready/ready_to_publish/publish_ready
  - `safety_notes` is empty
- Fixture runs are expected to produce an empty queue and blocked rows because generated review tables default to `needs_review` and `publish_ready=no`.
- Updated the campaign render workflow to run render, copy pack, and gated queue in one validation chain.
- Validation passed:
  - Campaign render plus copy pack plus gated queue workflow run `25388486336` passed.
