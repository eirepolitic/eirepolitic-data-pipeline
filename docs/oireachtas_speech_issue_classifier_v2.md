# Unified Oireachtas speech issue classifier v2

Status: implementation complete on `feature/unified-speech-issue-classifier`; live testing and historical backfill not yet run.

## Goal

Replace the stale legacy classifier path with a batch-aware classifier that consumes unified `silver_speeches`, reuses existing labels safely, classifies only genuinely new or changed speeches, and emits the compatibility dataset used by existing consumers.

## Root cause of stale classifications

The legacy classifier reads:

```text
raw/debates/debate_speeches_extracted.csv
```

The unified Oireachtas refresh now writes current speech data to:

```text
processed/oireachtas_unified/latest/csv/silver_speeches.csv
```

The previous enrichment trial only reshaped the legacy classified CSV; it did not classify new unified speeches. This allowed the wider Oireachtas pipeline to refresh while issue labels remained frozen.

## New classifier

Entry point:

```text
process/oireachtas_speech_issue_classifier.py
```

Default model:

```text
gpt-5.6-luna
reasoning.effort = low
text.verbosity = low
```

The classifier uses the Responses API with strict JSON-schema Structured Outputs. `issue_label` is constrained to the existing 25-category taxonomy.

## Unified enrichment table

Logical table:

```text
enrichment_speech_issue_labels
```

Primary key:

```text
speech_id
```

Logical production keys:

```text
processed/oireachtas_unified/latest/csv/enrichment_speech_issue_labels.csv
processed/oireachtas_unified/latest/parquet/enrichment_speech_issue_labels.parquet
```

Compatibility outputs:

```text
processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv
processed/oireachtas_unified/compat/debates/parquets/debate_speeches_classified_compat.parquet
```

Because the enrichment uses unified `latest/*` and `compat/*` logical keys, the existing Oireachtas IO layer automatically redirects reads/writes to the active candidate batch when `OIREACHTAS_BATCH_ID` is set.

## Label reuse order

For each current `silver_speeches` row:

1. Reuse an existing unified enrichment label only when `speech_id` and `source_speech_text_hash` both still match.
2. Otherwise migrate a legacy label when debate date, speech order, speaker name, and exact speech-text hash match uniquely.
3. Otherwise migrate a legacy label when debate date plus exact speech-text hash is unique in the legacy data.
4. Speech text below 20 words is assigned `NONE` using the existing short-text rule without an OpenAI call.
5. Remaining rows are marked `pending` and are the only rows eligible for model classification.

Legacy labels are never trusted solely by row position or legacy speech ID.

## Classification statuses

```text
classified
none
skipped_short_text
pending
failed
```

`pending` is valid for readiness and capped test runs, but complete candidate writes require zero `pending` and zero `failed` rows.

## Safety controls

- `--mode readiness` performs no OpenAI calls and no writes.
- `--mode dry-run` builds the migration/classification plan with no OpenAI calls and no writes.
- Capped model tests use `--mode classify --max-model-rows N` and cannot be combined with `--write`.
- `--write` is accepted only for a full model run (`max_model_rows=0`).
- Writes require an active candidate batch and candidate publishing mode.
- The classifier records its table entry in the candidate batch manifest.
- The classifier does not promote a candidate batch.
- The scheduled refresh hook exists but `classify_speeches` defaults to `false`.
- The production orchestrator has not been changed to enable classification automatically yet.

## OpenAI SDK isolation

The repository-wide requirement remains `openai>=1.99.2` to avoid changing unrelated LLM workflows.

Classifier workflows install:

```text
openai>=3.6.0,<4
```

immediately before classifier execution.

## Manual workflow

Workflow:

```text
.github/workflows/speech_issue_classifier.yml
```

Modes:

- `readiness`: inventory production data and migration coverage; no OpenAI.
- `dry-run`: build the full planned enrichment; no OpenAI.
- `classify` with `max_model_rows > 0`: capped Luna review sample; no S3 writes.
- `classify` with `max_model_rows = 0` and `write_candidate = true`: seed a complete candidate batch, classify all pending rows, write enrichment + compatibility outputs, and assemble the candidate. It still does not promote it.

## Dormant refresh integration

`oireachtas_refresh_reusable.yml` now accepts:

```text
classify_speeches: false
speech_classifier_model: gpt-5.6-luna
```

When enabled it requires candidate publishing and `silver_speeches` in the refresh table set, then runs the complete classifier before candidate manifest assembly and requires `enrichment_speech_issue_labels` in that manifest.

Do not enable this in the scheduled orchestrator until the readiness and Luna quality tests below pass.

## Required test sequence before historical backfill

### Test 1 — code/unit validation

Run:

```text
python -m unittest tests.test_oireachtas_speech_issue_classifier tests.test_oireachtas_speech_classifier_wiring -v
```

Expected: all tests pass.

### Test 2 — production readiness inventory

Run the manual workflow in `readiness` mode.

Review:

- `silver_rows`
- `silver_min_date` / `silver_max_date`
- `legacy_min_date` / `legacy_max_date`
- `legacy_valid_labels`
- `migrated_legacy_exact`
- `migrated_legacy_date_hash_unique`
- `legacy_migration_pct`
- `pending_model`
- `pending_min_date` / `pending_max_date`

If historical migration coverage is materially below expectations, stop and investigate matching before spending on reclassification.

### Test 3 — Luna quality sample

Run `classify` with a small capped row count such as 25 and `write_candidate=false`.

Review each returned sample against the speech excerpt. Compare a representative subset against GPT-4.1 mini if Luna quality is uncertain. No S3 classifier outputs are written by this test.

### Test 4 — larger non-writing sample

If Test 3 is satisfactory, run 100-250 pending rows and review category distribution, `NONE` rate, failures, and obvious taxonomy drift.

### Test 5 — historical backfill candidate

Only after Tests 1-4 pass:

- run `classify`
- `max_model_rows=0`
- `write_candidate=true`

This creates a complete candidate batch containing the migrated historical labels plus model classifications for outstanding speeches. It does not promote the batch.

### Test 6 — downstream validation and promotion

Run the standard candidate validation against the completed classifier batch, including member metrics and Instagram consumers. Promote only after downstream validation passes.

### Test 7 — enable scheduled classification

After the first classified enrichment batch is promoted, update the production orchestrator so scheduled weekly refreshes pass:

```text
classify_speeches: true
```

At that point candidate seeding will carry forward the existing enrichment and weekly classification should only call the model for new or changed speeches.

## Historical artifacts retained

The legacy implementation remains in the repository for audit/reference:

```text
process/speech_issue_classifier.py
extract/oireachtas/enrichment_speech_issue_labels.py
```

No active classifier workflow on this branch invokes either legacy path.
