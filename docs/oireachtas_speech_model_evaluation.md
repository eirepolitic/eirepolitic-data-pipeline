# Oireachtas speech issue model evaluation

This evaluation must use the same taxonomy, prompt version, and reviewed sample for every candidate model.

## Files

- Fixture: `tests/fixtures/oireachtas_speech_issue_evaluation_v1.csv`
- Harness: `process/oireachtas_speech_model_evaluation.py`
- Pricing template: `configs/oireachtas/speech_model_pricing_template.csv`

The included fixture is a synthetic seed and is intentionally marked `pending`. It is not approved ground truth and must not be used for paid model selection until every row has been reviewed.

## Review process

For every fixture row:

1. Replace the synthetic text with a representative, legally usable excerpt or internally reviewed paraphrase from `silver_speeches`.
2. Confirm the single dominant category using the fixed taxonomy.
3. Set `review_status` to `approved` only after review.
4. Record the reviewer name or identifier and an ISO-8601 UTC timestamp.
5. Record ambiguous alternatives or rationale in `review_notes`.
6. Add more than one example for high-volume categories and for `NONE` before final model selection.

Rows marked `pending` or `rejected` cause `--require-reviewed` validation to fail.

## Validate the fixture

```bash
python process/oireachtas_speech_model_evaluation.py validate-fixture \
  --fixture tests/fixtures/oireachtas_speech_issue_evaluation_v1.csv
```

Before any paid run:

```bash
python process/oireachtas_speech_model_evaluation.py validate-fixture \
  --fixture tests/fixtures/oireachtas_speech_issue_evaluation_v1.csv \
  --require-reviewed
```

## Prepare identical Batch requests

```bash
python process/oireachtas_speech_model_evaluation.py prepare-requests \
  --fixture tests/fixtures/oireachtas_speech_issue_evaluation_v1.csv \
  --models MODEL_A,MODEL_B \
  --output-dir evaluation_runs/candidate_models \
  --require-reviewed
```

This creates one JSONL request file and one custom-ID mapping file per model. The manifest records fixture, prompt, taxonomy, request, and mapping checksums.

## Convert Batch results

```bash
python process/oireachtas_speech_model_evaluation.py convert-batch-results \
  --mapping evaluation_runs/candidate_models/MODEL_A.mapping.csv \
  --batch-output evaluation_runs/candidate_models/MODEL_A.output.jsonl \
  --model MODEL_A \
  --batch-id BATCH_ID \
  --batch-status completed \
  --latency-seconds 120 \
  --output evaluation_runs/candidate_models/MODEL_A.results.csv
```

Use actual Batch status and elapsed wall-clock time. Failed or missing request rows remain visible in the result file.

## Compare models

Copy verified official Batch prices into a run-specific pricing file based on the template. Do not commit assumed or stale prices as current truth.

```bash
python process/oireachtas_speech_model_evaluation.py compare-results \
  --fixture tests/fixtures/oireachtas_speech_issue_evaluation_v1.csv \
  --results evaluation_runs/candidate_models/MODEL_A.results.csv \
  --results evaluation_runs/candidate_models/MODEL_B.results.csv \
  --pricing evaluation_runs/candidate_models/pricing.csv \
  --backfill-rows BACKFILL_ROW_COUNT \
  --recurring-rows EXPECTED_RECURRING_ROWS \
  --require-reviewed \
  --output evaluation_runs/candidate_models/comparison.json
```

The report includes:

- Overall accuracy.
- `NONE` precision and recall.
- Per-category precision and recall.
- Invalid-output rate.
- Classification failure rate.
- Pairwise agreement.
- Input and output token totals.
- Mean, p50, p95, and maximum latency.
- Batch completion success.
- Evaluation, backfill, and recurring cost estimates when pricing is supplied.

The report never selects a model automatically. `selection_status` remains `requires_human_approval` and `selected_model` remains null.

## Approval gate

Do not start the complete historical backfill until:

1. The fixture is fully reviewed.
2. At least two account-available candidate models complete the same sample.
3. Comparison results and projected costs are presented.
4. The user approves the selected model and prompt version.
