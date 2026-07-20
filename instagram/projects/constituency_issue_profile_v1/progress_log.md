# Progress log

## Phase 2

The constituency pilot established deterministic minimum, maximum, and real-example complete-slide validation using the active Oireachtas production data.

Validation runs:

- `29703058481` — local complete-slide validation
- `29703087934` — initial live S3 validation
- `29703335986` — final live S3 validation
- `29704116144` — corrected scenario-contract validation

## Phase 3

Implemented deterministic constituency batch generation with complete slides, hashes, provenance, item/run manifests, review-state initialization, isolated failure handling, immutable S3 storage, and artifact upload.

Validation evidence:

- live batch workflow: `29711056766`
- artifact: `constituency-factory-batch-constituency_issue_profile_v1-v1-1cd892b3cf6f`
- S3 root: `s3://eirepolitic-data/processed/instagram_factory/projects/constituency_issue_profile_v1/`

All generated items remain unreviewed, unapproved, and non-publishable.

## Phase 4 — 2026-07-20

Implemented review-state management and targeted regeneration:

- item-level and slide-level review transitions
- statuses: `unreviewed`, `approved`, `changes_requested`, and `rejected`
- append-only review history
- derived-run creation preserving the source run
- selective constituency and slide regeneration
- unaffected files copied byte-for-byte
- regenerated slide hashes and provenance
- manual S3 workflow for review updates and selective regeneration

Validation evidence:

- Phase 4 workflow: `29711566533`
- Phase 1–4 tests passed

## Phase 5 — 2026-07-20

Implemented recurring-generation readiness controls:

- production source resolution through the active unified-data pointer
- source, join, and constituency-count readiness checks
- duplicate production-batch prevention using the project `latest.json`
- `check-readiness` CLI command
- manual recurring draft workflow
- clean no-op behavior when no new production batch is available
- immutable draft generation only when readiness passes
- automatic publishing, approval, and scheduling remain disabled

Validation evidence:

- Phase 1–5 validation workflow: `29711883200`
- live source batch: `current-government-backfill-20260716-1`
- member rows: 176
- speech rows: 47,275
- matched speeches: 29,233
- constituencies: 43
- readiness result: blocked as expected because the source batch had already been generated
- no duplicate draft batch was created

## Current status

Phases 1–5 are implemented for the constituency project. Recurring generation is available manually and protected by data-readiness and duplicate-batch gates. No cron schedule, automatic approval, scheduling, or Instagram publishing is enabled. Generic multi-project orchestration remains future work.
