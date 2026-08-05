# Oireachtas speech classification v2 branch audit

Audit date: 2026-08-05

## Verified revisions

- `main`: `14b57004ef3dbf75b796bd5b7d33a3c6d2c5beba`
- `feature/oireachtas-speech-classification-v2`: `30767b7976bcc7b7ff51747612f1959c56287782`

The observed handoff revisions remain current. No new `main` drift was present during this audit.

## Branch difference from main

The branch differs from `main` in exactly four artifacts:

1. `process/oireachtas_speech_issue_classifier.py`
2. `.github/workflows/oireachtas_speech_issue_classifier_v2.yml`
3. `tests/test_oireachtas_speech_issue_classifier.py`
4. `archive/legacy_speech_classification/README.md`

No core Oireachtas build, validation, promotion, schema, contract, or compatibility file has been changed.

## Completed implementation

The branch currently provides:

- Production-pointer resolution for `silver_speeches`.
- Delta selection by `speech_id` and `speech_text_hash`.
- Selection of new, changed, failed, and explicitly forced rows.
- Immutable classification run prefixes and run manifests.
- OpenAI Batch JSONL generation using `/v1/responses`.
- Strict JSON-schema output constrained to the fixed taxonomy.
- Separate prepare, submit, status, collect, and publish commands.
- Model availability checks against the configured OpenAI account before submission.
- Candidate CSV and Parquet outputs.
- A separately guarded classification production pointer and previous pointer.
- Source-batch checks before publication.
- Compatibility output generation from `silver_speeches` plus v2 labels.
- Taxonomy, delta, merge, parsing, candidate validation, compatibility, and basic evaluation tests.
- Documentation of legacy cutover gates without moving legacy executables.

## Active legacy dependencies

The following dependencies prevent archival now:

- `configs/oireachtas/downstream_contracts.yml` defines the active `debate_issue_labels` contract at `processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv`.
- `process/oireachtas_stage_downstream_contracts.py` stages `debate_issue_labels` into every selected immutable core candidate batch.
- `process/build_member_profile_metrics.py` reads the compatibility debate issue dataset.
- `process/oireachtas_consumer_smoke.py` validates and supplies the compatibility dataset to the Instagram consumer.
- `process/instagram_render_post.py` still defaults directly to the legacy `processed/debates/debate_speeches_classified.csv` unless an environment override is supplied.
- `.github/workflows/oireachtas_enrichment_speech_issue_labels_trial.yml` runs the legacy reshaping builder.
- `extract/oireachtas/enrichment_speech_issue_labels.py` still overwrites the active compatibility key from the legacy classified CSV.
- `.github/workflows/speech_issue_classifier.yml`, `process/speech_issue_classifier.py`, and `process/debate_speeches_csv_to_parquet.py` remain executable and maintain the legacy classified outputs.

The GitHub code-search endpoint returned incomplete empty results during this audit, so dependency identification was completed by tree comparison and direct inspection of the active Oireachtas, compatibility, consumer, workflow, and legacy files. A local repository grep remains required before archival.

## Confirmed defects and missing controls

### Data integrity and resumability

- Existing enrichment keys are not normalized consistently to strings before delta comparison.
- Merging retains classifications for speeches removed from a later active `silver_speeches` batch. Candidate validation then rejects those rows, which can block future publication.
- If the source production batch changes while an OpenAI Batch is running, collection raises before preserving a stale-but-auditable candidate and its validation evidence.
- OpenAI Batch error files are recorded but not collected or attached to row-level failure evidence.
- There is no explicit retry-attempt count, retry ceiling, or retry eligibility timestamp.
- Prepared request files and collected outputs do not yet have recorded checksums.
- Publication does not compare the classification pointer observed during preparation with the pointer current at publication, so an external concurrent publisher could overwrite newer enrichment state.

### Workflow behavior

- A successful orchestrator workflow run can trigger the classifier even when no core production promotion occurred.
- Automatic execution does not explicitly verify the orchestrator's promoted batch ID.
- Automatic continuation is incomplete: submitted batches require a later manual status and collect operation.
- Candidate generation does not have its own repository switch separate from automatic execution, automatic submission, publication, and backfill.
- Cost estimation exists only as a helper and is not emitted before submission.
- The workflow has no explicit per-run retry limit.
- The automatic model variable can be blank or unavailable; availability is checked only at paid submission time.

### Contracts and compatibility

- The enrichment table is not represented by a dedicated schema/manifest contract.
- It is not registered in a safe enrichment-specific registry or write-policy definition.
- The active compatibility key can still be overwritten by the legacy trial builder.
- The direct Instagram renderer still defaults to the legacy classified dataset.
- Compatibility output lacks a cutover-mode guard that prevents incomplete historical classifications from becoming the active consumer dataset.

### Evaluation

- There is no reviewed, versioned evaluation fixture.
- The current evaluator does not produce model agreement, latency, Batch completion success, total token usage, or projected backfill and recurring costs in one comparison report.
- Candidate model IDs are not pinned as an approved permanent choice.
- No limited paid comparison has been run.

### Tests and validation evidence

Missing coverage includes:

- Fake-S3 prepare, collect, publish, rollback, corruption, and permission tests.
- Candidate-only pointer non-mutation tests.
- Workflow YAML parsing and trigger-condition tests.
- Publication compare-and-swap tests.
- Batch error-file and retry tests.
- Removed-source-speech regression tests.
- Stale-source collection tests.
- Core Oireachtas independence regression tests.
- Full Oireachtas test-suite results.
- Candidate-only S3 evidence.
- Limited live model evidence.
- Historical backfill evidence.
- Automatic post-refresh delta evidence.

The branch's tests have not yet been executed in a checked-out runtime during this work.

## Architecture conclusions

- The classification system must remain outside the core Oireachtas batch promotion transaction.
- The separate enrichment production pointer is the correct boundary.
- The active source batch ID must be retained in every run and every row.
- Stale Batch results should be collected into immutable run artifacts but marked unpublishable rather than discarded.
- The current core table registry should not be extended blindly if doing so would make enrichment a required core refresh table. A dedicated enrichment schema/contract is safer unless repository tests demonstrate otherwise.
- Legacy files must remain active until the v2 compatibility key is validated, all consumers use it, a full backfill succeeds, and one automatic delta succeeds.

## Ordered remediation

1. Correct source-row reconciliation, key normalization, stale-run collection, checksums, retries, and publication concurrency guards.
2. Add fake-S3 lifecycle tests.
3. harden and test workflow trigger behavior and independent switches.
4. Add enrichment-specific schema, manifest, and write-policy contracts without coupling core promotion.
5. Cut compatibility generation over behind validation guards and update direct consumers.
6. Add the reviewed evaluation fixture and comparison report.
7. Run all local and CI validation.
8. Perform candidate-only S3 validation.
9. Request approval for limited paid model comparison.
10. Request approval for model selection, backfill, publication, cutover, archival, and merge.
