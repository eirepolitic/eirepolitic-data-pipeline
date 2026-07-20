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
- append-only review history with reviewer, timestamp, note, item, slide, and status
- overall review-state calculation
- publishing remains disabled regardless of review state
- derived-run creation preserving the complete source run
- selective constituency and slide regeneration
- unaffected files copied byte-for-byte into the derived run
- regenerated slide hashes and provenance recorded
- parent-run linkage and regeneration reason recorded
- manual S3 workflow for review updates and selective regeneration
- no mutation of the source run during regeneration

Validation evidence:

- Phase 4 validation workflow: `29711566533`
- catalogue validation: passed
- project validation: passed
- Phase 1–4 unit tests: passed
- regression test confirms an unaffected cover-slide hash remains unchanged when only the issue-profile slide is regenerated

## Current status

Phases 1–4 are implemented for the constituency project. Review and regeneration operations are available, but no automatic publishing, scheduling, or approval is enabled. Generic multi-project orchestration remains future work.
