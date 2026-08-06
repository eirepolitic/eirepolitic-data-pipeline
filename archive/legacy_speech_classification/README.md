# Legacy speech classification archive

The legacy classifier remains executable only until the Oireachtas speech issue classifier v2 completes model evaluation, historical backfill, and one successful automatic delta run.

## Legacy assets to archive at cutover

- `.github/workflows/speech_issue_classifier.yml`
- `process/speech_issue_classifier.py`
- `process/debate_speeches_csv_to_parquet.py`
- `.github/workflows/oireachtas_enrichment_speech_issue_labels_trial.yml`
- `extract/oireachtas/enrichment_speech_issue_labels.py`

## Legacy S3 inputs and outputs

These objects must remain read-only during the rollback window:

- `raw/debates/debate_speeches_extracted.csv`
- `processed/debates/debate_speeches_classified.csv`
- `processed/debates/parquets/debate_speeches_classified.parquet`
- `processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv`

## Cutover gates

1. Unit and YAML validation pass.
2. At least two candidate models are evaluated against the same reviewed sample.
3. A model and prompt version are pinned.
4. A candidate-only S3 run passes data-quality checks.
5. Historical `silver_speeches` backfill completes.
6. Compatibility consumers are rebuilt from `enrichment_speech_issue_labels`.
7. One post-refresh automatic delta run succeeds.
8. Legacy workflows are moved out of `.github/workflows` and legacy scripts are moved into this archive.
9. Legacy S3 outputs are retained read-only for the agreed rollback period.

Automatic execution is guarded by `OIREACHTAS_SPEECH_CLASSIFIER_ENABLED`. Publication is separately guarded by `OIREACHTAS_SPEECH_CLASSIFIER_PUBLISH_ENABLED`.
