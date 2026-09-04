# Irish Polling Indicator ingestion implementation

## Status

Production-quality ingestion code is integrated in the repository under `extract/polling/`, with CI and a scheduled/manual GitHub Actions workflow.

The publication workflow is intentionally **rights-gated**. It will not write IPI data to S3 until repository variable `IPI_REUSE_CONFIRMED=true` is set after the source's reuse permission/licensing is documented.

## Why the publication gate exists

The upstream IPI repository is publicly readable and asks users to cite the source, but the production implementation review did not identify a clear open-data licence granting general redistribution rights.

The code therefore supports full live validation immediately while preventing accidental production redistribution. This is an operational/legal guard, not a technical limitation.

## Files

- `extract/polling/__init__.py`
- `extract/polling/ipi.py`
- `tests/test_ipi_ingestion.py`
- `.github/workflows/ipi_ingestion_ci.yml`
- `.github/workflows/ipi_ingestion.yml`

## Upstream source

Repository:

`https://github.com/Irish-Polling-Indicator/ipi-data`

Files:

- `data_polls.csv`
- `data_pollingindicator.csv`

The extractor resolves the repository's current default branch to an immutable 40-character commit SHA, then downloads both files using that exact SHA. The SHA and file hashes are written into the manifest.

An optional `IPI_UPSTREAM_REF` / workflow `upstream_ref` input can pin a branch, tag or commit for recovery/debugging.

## Source semantics preserved

### Raw polls

`data_polls.csv` values remain in **percentage points (0–100)**.

The normalized output adds:

- `source_row_number`;
- `value_unit=percentage_points`;
- `quality_flags`;
- source commit/branch/file/URL/retrieval provenance.

Historical source anomalies are flagged rather than silently changed, including:

- exact duplicate source rows;
- duplicate poll composite keys;
- fieldwork start after end;
- fieldwork midpoint outside the stated range;
- fieldwork ending after publication;
- negative source party values.

Values above 100, values below the known historical `-1` floor, invalid dates, missing core fields, non-positive sample sizes or schema changes block publication.

### Modelled polling indicator

`data_pollingindicator.csv` values remain **proportions (0–1)**.

The extractor requires:

- exact 41-column schema;
- valid ISO dates;
- non-missing election cycle;
- unique `(date, cycle)` keys;
- complete estimate/lower/upper triplets when a party is present;
- `0 <= lower <= estimate <= upper <= 1`.

Duplicate calendar dates across two election cycles are expected at election boundaries and are retained with `cycle_boundary_duplicate_calendar_date` quality flags.

## S3 layout

Default bucket:

`eirepolitic-data`

### Immutable raw snapshots

```text
raw/polling/irish_polling_indicator/by_commit/<sha>/data_polls.csv
raw/polling/irish_polling_indicator/by_commit/<sha>/data_pollingindicator.csv
```

### Immutable normalized snapshots

```text
processed/polling/irish_polling_indicator/by_commit/<sha>/csv/polls.csv
processed/polling/irish_polling_indicator/by_commit/<sha>/csv/polling_indicator.csv
processed/polling/irish_polling_indicator/by_commit/<sha>/parquet/polls.parquet
processed/polling/irish_polling_indicator/by_commit/<sha>/parquet/polling_indicator.parquet
processed/polling/irish_polling_indicator/by_commit/<sha>/manifest.json
```

### Stable latest paths

```text
processed/polling/irish_polling_indicator/latest/csv/polls.csv
processed/polling/irish_polling_indicator/latest/csv/polling_indicator.csv
processed/polling/irish_polling_indicator/latest/parquet/polls.parquet
processed/polling/irish_polling_indicator/latest/parquet/polling_indicator.parquet
processed/polling/irish_polling_indicator/latest/manifest.json
```

The stable `latest/manifest.json` is always the final S3 write and acts as the publication marker.

## Manifest

The manifest records:

- upstream repository, branch/ref and commit SHA;
- retrieval timestamp;
- source URLs, byte counts and SHA-256 hashes;
- licence status and citation reminder;
- row counts/date coverage;
- raw poll anomaly counts;
- indicator cycle-boundary dates;
- units;
- normalized CSV/Parquet hashes;
- all S3 object keys.

## Running locally or in CI

Validation only; no AWS credentials or writes:

```bash
python -m extract.polling.ipi
```

Run unit tests:

```bash
python -m unittest tests.test_ipi_ingestion -v
```

## Production publication

The production workflow runs daily at `07:17 UTC` and can also be triggered manually.

It only runs when the repository Actions variable is:

```text
IPI_REUSE_CONFIRMED=true
```

The script also checks the same environment value before constructing the S3 client, giving two independent publication guards.

Once rights are confirmed, the existing AWS secrets used by the repo are sufficient; no new paid service or infrastructure is required.

## Test coverage

Tests cover:

- exact schema enforcement;
- duplicate raw poll handling;
- preservation/flagging of the known negative-value class;
- fieldwork date anomaly flags;
- percentage values above 100 blocking publication;
- cycle-boundary duplicate calendar dates being allowed;
- duplicate `(date, cycle)` keys blocking publication;
- invalid model intervals blocking publication;
- S3 latest manifest being written last;
- explicit reuse-rights publication gate.

CI also fetches and validates the current live upstream source without writing to S3.

## Operational next step

1. Merge only after CI passes against the live source.
2. Obtain/document explicit IPI production/reuse permission or a clear licence.
3. Set `IPI_REUSE_CONFIRMED=true` in GitHub Actions repository variables.
4. Manually dispatch `Irish Polling Indicator ingestion` once and verify the published manifest/S3 objects.
5. Leave the daily schedule enabled after the first successful production run.
