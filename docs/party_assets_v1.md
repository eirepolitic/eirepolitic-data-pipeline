# Party asset layer v1

Status: implementation in progress on `feature/party-asset-layer-v1`.

## Scope

Reusable party identity and image/logo assets for all party groupings currently emitted by the Oireachtas/Instagram commissioning pipeline. This work does not redesign Instagram layouts and does not publish posts.

## Identity contract

Source party labels are resolved to a stable `party_key` through `process/party_assets.py` and `configs/reference/party_assets_v1.csv`.

Current commissioning keys:

- `100-rdr`
- `aontu`
- `fianna-fail`
- `fine-gael`
- `green-party`
- `independent`
- `independent-ireland`
- `labour-party`
- `people-before-profit-solidarity`
- `sinn-fein`
- `social-democrats`

Aliases are explicit. Examples include `100% Redress Party -> 100-rdr`, `Non-Party -> independent`, and `Solidarity - People Before Profit -> people-before-profit-solidarity`.

## Target S3 structure

```text
s3://eirepolitic-data/processed/reference/party_assets/v1/
  assets/
    {party_key}/
      source.svg|png
      logo.png
  party_assets.csv
  party_assets.parquet
  manifest.json
  contact_sheet.png
```

Canonical consumer asset specification:

- transparent PNG derivative
- safely containable in a square box without cropping or stretching
- target master canvas 1600x1600 px where source quality permits
- original SVG/PNG retained as `source.*` when legally and technically appropriate
- deterministic filename: `assets/{party_key}/logo.png`
- source provenance and usage/licensing note retained in the mapping dataset

## Source policy

Priority order:

1. official party brand/media source;
2. Electoral Commission registered party emblem;
3. explicit approved fallback.

Do not fabricate an official-looking logo when an authoritative emblem exists. Do not automatically combine constituent party logos for non-standard groupings.

`Independent` is explicitly `approved_fallback` with `fallback_type=no_party_logo`; it has no party-logo S3 URI. Consumers must use a neutral non-party treatment and must not imply official Independent branding.

## Asset states

- `source_identified_pending_ingest`: authoritative source identified, but normalized S3 asset has not yet been validated as present.
- `pending_review`: asset exists or has been prepared but requires human/licensing review.
- `approved`: normalized asset exists, validates technically, and is approved for use.
- `approved_fallback`: no party logo should be used; an explicit fallback policy applies.

No row may silently fall through to an invented wordmark or guessed logo.

## Validation

`process/party_assets.py` validates:

- unique `party_key` values;
- deterministic aliases and collision detection;
- required provenance and usage fields;
- S3 URI requirement for non-fallback rows;
- explicit fallback type for fallback rows;
- read-only S3 existence checks for expected objects;
- read-only bucket scan for pre-existing image candidates matching party names/aliases.

Run locally:

```bash
python -m unittest tests.test_party_assets -v
python process/party_assets.py
```

Read-only S3 audit (requires configured AWS read access):

```bash
python process/party_assets.py --audit-s3 --output party_asset_audit.json
```

The manual workflow `.github/workflows/party_assets_audit.yml` performs the same read-only audit and stores the report as a temporary GitHub Actions artifact. It does not write to S3.

## Source audit snapshot — 2026-08-31

Authoritative sources have been identified for all ten registered-party/grouping rows in the commissioning set. The registry records official party websites where appropriate, the Green Party brand page where downloadable artwork guidance is published, and Electoral Commission registered-emblem sources for 100% RDR, Independent Ireland, and the combined People Before Profit-Solidarity grouping.

No open redistribution licence has been assumed. Rows without explicit published reuse terms retain a conservative editorial-identification usage note and remain `source_identified_pending_ingest` until the exact source asset and terms are reviewed.

## Remaining implementation sequence

1. Run the read-only S3 audit and capture any existing party-logo candidates.
2. Compare candidates with authoritative sources and reject stale/unofficial duplicates.
3. Retrieve exact source artwork for each non-fallback row.
4. Normalize approved derivatives and perform technical validation.
5. Generate the contact sheet for human review.
6. Only after approval, write versioned assets plus CSV/Parquet/manifest under the new `processed/reference/party_assets/v1/` prefix.
7. Wire the Instagram Content Factory to resolve the approved party asset by `party_key`, without redesigning the post.

The classified-debate cutoff/backfill issue is out of scope for this work.
