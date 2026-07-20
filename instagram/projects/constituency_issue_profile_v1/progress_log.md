# Progress log

## 2026-07-19 — pilot implemented

- Added the constituency project specification and joined constituency issue metric.
- Reused `horizontal_bar_draft_v1` and `title_text_media_v1` without duplicating renderer logic.
- Added deterministic minimum, maximum, and real-example scenario generation.
- Added complete-slide rendering, scenario contact sheets, slide contact sheets, and provenance manifests.
- Added local fixtures and live S3 workflow validation.

## Phase 2 validation evidence

- Local complete-slide validation: workflow run `29703058481` — succeeded.
- Initial live S3 validation: workflow run `29703087934` — succeeded.
- Final live S3 validation after quality corrections: workflow run `29703335986` — succeeded.
- Corrected scenario-contract validation and preview publication: workflow run `29704116144` — succeeded.
- Preview branch: `instagram-preview-output`
- Preview root: `preview/factory/projects/constituency_issue_profile_v1/`

## Phase 2 live validation summary

- active production batch: `current-government-backfill-20260716-1`
- AWS region: `ca-central-1`
- current member rows: 176
- classified speech rows: 47,275
- matched speeches: 29,233
- constituencies represented: 43
- real example: Kildare North
- review state: `needs_review`
- approved: `false`

## 2026-07-20 — Phase 3 batch generation implemented

- Added deterministic constituency batch generation through `instagram/factory/constituency_batch.py`.
- Added the `generate-batch` CLI command.
- Added stable run identities from project version, production batch ID, and Git SHA.
- Added item folders, complete slides, hashes, visual manifests, item manifests, run manifests, review-state initialization, and batch review samples.
- Added isolated item failure handling.
- Added live S3 project-run storage and `latest.json` update.
- Added regression tests for deterministic, review-only batch generation.
- Restricted documentation-only changes from triggering redundant batch runs.

## Phase 3 validation evidence

- First successful live batch workflow: `29711056766`.
- Generated artifact: `constituency-factory-batch-constituency_issue_profile_v1-v1-1cd892b3cf6f`.
- Artifact size: 11,561,431 bytes.
- S3 project root: `s3://eirepolitic-data/processed/instagram_factory/projects/constituency_issue_profile_v1/`.
- All workflow gates passed: catalogue validation, project validation, unit tests, live batch rendering, immutable S3 upload, artifact upload, and review-state initialization.

## Current status

The reusable constituency design is approved as the batch baseline. Phase 3 batch generation is technically implemented and validated, but every generated item remains `unreviewed`, `approved: false`, and `publishing_allowed: false`. No automatic publishing or scheduling is enabled.

- 2026-07-19: Corrected synthetic scenario selection: shortest/longest constituency names now independently pair with smallest/largest constituency result sets; provenance is explicit in manifests and covers.
- 2026-07-19: User agreed synthetic minimum/maximum examples should remain QA fixtures.
- 2026-07-20: User approved moving beyond example refinement and into review-only batch generation.
