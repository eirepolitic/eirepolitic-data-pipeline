# Progress log

## 2026-07-19 — pilot implemented

- Added the constituency project specification and joined constituency issue metric.
- Reused `horizontal_bar_draft_v1` and `title_text_media_v1` without duplicating renderer logic.
- Added deterministic minimum, maximum, and real-example scenario generation.
- Added complete-slide rendering, scenario contact sheets, slide contact sheets, and provenance manifests.
- Added local fixtures and live S3 workflow validation.

## Validation evidence

- Local complete-slide validation: workflow run `29703058481` — succeeded.
- Initial live S3 validation: workflow run `29703087934` — succeeded.
- Final live S3 validation after quality corrections: workflow run `29703335986` — succeeded.
- Preview branch: `instagram-preview-output`
- Preview root: `preview/factory/projects/constituency_issue_profile_v1/`

## Final live validation summary

- active production batch: `current-government-backfill-20260716-1`
- AWS region: `ca-central-1`
- current member rows: 176
- classified speech rows: 47,275
- matched speeches: 29,233
- constituencies represented: 43
- real example: Kildare North
- review state: `needs_review`
- approved: `false`

## Current status

Technical pilot validation is complete. Visual quality and factual interpretation require explicit human review before any approval or batch-generation work.
