from pathlib import Path

path = Path("instagram/CONTENT_FACTORY_PLAN.md")
text = path.read_text(encoding="utf-8")

old_phase3 = '''### Phase 3 — batch generation and S3 project storage

Build:

- granularity enumerator
- item context builder
- complete batch renderer
- run/item/slide manifests
- S3 uploader
- review index

Acceptance:

- one complete post set is generated for every selected item
- output structure matches the project S3 contract
- partial failures are isolated and documented
'''
new_phase3 = '''### Phase 3 — batch generation and S3 project storage

**Status: constituency implementation completed and live-validated (2026-07-20).**

Implemented for `constituency_issue_profile_v1`:

- deterministic constituency enumeration from the active production dataset
- one complete two-slide post set per constituency
- stable run IDs derived from project version, source batch ID, and Git SHA
- generated item folders with slide hashes and visual provenance
- run, item, slide, visual, and review-state manifests
- isolated item failure handling
- batch review sample generation
- immutable S3 run upload under the existing project prefix
- project `latest.json` update
- GitHub Actions artifact upload
- all outputs remain unreviewed, unapproved, and non-publishable

Validation evidence:

- live workflow run: `29711056766`
- artifact: `constituency-factory-batch-constituency_issue_profile_v1-v1-1cd892b3cf6f`
- S3 project root: `s3://eirepolitic-data/processed/instagram_factory/projects/constituency_issue_profile_v1/`
- all catalogue, project, unit-test, render, S3-upload, artifact, and review-state gates passed

This validates Phase 3 for one constituency project. Generic multi-project batch orchestration and review UI remain future work.

Acceptance:

- one complete post set is generated for every selected item
- output structure matches the project S3 contract
- partial failures are isolated and documented
'''
if old_phase3 not in text:
    raise SystemExit("Phase 3 block not found")
text = text.replace(old_phase3, new_phase3, 1)

old_immediate = '''## 18. Immediate next development tasks

Phase 1 is complete. The next milestone is Phase 2 complete-slide testing.

Recommended order for the next chat instance:

1. Read this file and the architecture/system docs.
2. Inspect current live repo state; do not assume this plan is perfectly current.
3. Inspect the completed catalogues, validators, and project template; do not duplicate them.
4. Choose one simple pilot project, preferably constituency or member grain with two slides.
5. Implement the min/max/real scenario builder for the pilot.
6. Integrate standalone visual assets into complete post-layout slots.
7. Render complete-slide test sets and contact sheets.
8. Add factual real-example validation manifests.
9. Review and refine before building batch mode.
10. Add S3 project storage only after the pilot test flow is approved.

Do not jump directly to full batch generation before complete-slide validation is working.
'''
new_immediate = '''## 18. Immediate next development tasks

Phases 1 and 2 are complete for the constituency pilot. Phase 3 batch generation and S3 storage are implemented and live-validated for that project.

Recommended order for the next chat instance:

1. Read this file and the architecture/system docs.
2. Inspect current live repo state and the latest constituency batch manifest.
3. Review generated constituency items without changing unaffected outputs.
4. Implement Phase 4 review-state commands and targeted regeneration.
5. Add per-item/per-slide review status transitions.
6. Preserve immutable prior runs during selective regeneration.
7. Build a review index or equivalent navigable output.
8. Generalize batch orchestration only after the constituency review loop is validated.

Do not enable automatic publishing, scheduling, or approval.
'''
if old_immediate not in text:
    raise SystemExit("Immediate tasks block not found")
text = text.replace(old_immediate, new_immediate, 1)

path.write_text(text, encoding="utf-8")
