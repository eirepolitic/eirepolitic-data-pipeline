# Party asset layer v1

Status: implementation in progress on `feature/party-asset-layer-v1`.

## Scope

Reusable party identity and image/logo assets for party groupings emitted by the Oireachtas/Instagram pipeline. This work does not redesign Instagram layouts and does not publish posts.

## Identity contract

Source party labels resolve to stable `party_key` values through `process/party_assets.py` and `configs/reference/party_assets_v1.csv`.

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

Aliases are explicit. Examples: `100% Redress Party -> 100-rdr`, `Non-Party -> independent`, and `Solidarity - People Before Profit -> people-before-profit-solidarity`.

## Target S3 structure

```text
s3://eirepolitic-data/processed/reference/party_assets/v1/
  assets/
    {party_key}/
      source.svg|png|jpg|webp
      logo.png
  party_assets.csv
  party_assets.parquet
  manifest.json
  contact_sheet.png
```

Canonical consumer asset specification:

- transparent PNG derivative
- safely contained without cropping or stretching
- 1600x1600 px normalized canvas
- 160 px safe margin
- original authoritative SVG/raster retained as `source.*` where appropriate
- deterministic consumer filename: `assets/{party_key}/logo.png`
- SHA-256 recorded in the build manifest
- provenance and licensing/usage note retained in the mapping dataset

## Source policy

Priority:

1. exact logo/brand asset served by an official party site;
2. exact registered-emblem asset published by the Electoral Commission;
3. authoritative register/brand page pending exact-file verification;
4. explicit approved fallback.

Do not scrape a generic webpage and guess which image is the logo. Do not fabricate an official-looking logo. Do not synthesize a combined emblem for non-standard groupings.

`Independent` is `approved_fallback` with `fallback_type=no_party_logo`; it has no party-logo S3 URI. Consumers must use an explicitly neutral non-party treatment.

## Source audit snapshot — 2026-08-31

Exact direct authoritative asset URLs are pinned for:

- `100-rdr` — Electoral Commission registered emblem
- `aontu` — official Aontú logo asset
- `fine-gael` — official Fine Gael logo asset; current pinned variant uses light/white text and requires visual review
- `green-party` — official Green Party SVG
- `independent-ireland` — Electoral Commission registered emblem
- `labour-party` — official Labour RGB SVG mark
- `people-before-profit-solidarity` — Electoral Commission registered grouping emblem

Still intentionally unresolved at exact-file level:

- `fianna-fail`
- `sinn-fein`
- `social-democrats`

All three have authoritative official-party / Electoral Commission evidence for their identity and registered emblem, but the registry does not yet point at a standalone direct reusable source file. PDF extraction, screenshot-derived artwork, or guessed site assets have not been substituted.

No open redistribution licence is assumed where none is explicitly published. Usage notes remain conservative and assets stay unapproved until human review.

## Asset states

- `source_identified_pending_ingest`: authoritative source identified, normalized approved S3 asset not yet present.
- `pending_review`: prepared asset requires human/source/licensing review.
- `approved`: normalized asset exists, validates technically, and is approved for use.
- `approved_fallback`: no party logo should be used and an explicit fallback applies.

No recognized party may silently fall through to invented branding.

## Tooling

### Registry / resolver

`process/party_assets.py`

Validates unique keys, aliases, provenance, fallback state, and can perform a read-only S3 candidate audit.

```bash
python -m unittest tests.test_party_assets -v
python -m process.party_assets
```

### Safe source fetch

`process/party_assets_fetch.py`

Fetches only direct HTTPS image/SVG URLs with an approved direct-asset `source_type`. Generic party webpages remain `unresolved_source` and are never scraped for guessed assets.

```bash
python -m process.party_assets_fetch \
  --staging-root generated_party_asset_review/sources \
  --allow-unresolved \
  --output generated_party_asset_review/fetch_report.json
```

### Normalize / contact sheet

`process/party_assets_build.py`

Supports PNG/JPEG/WebP and SVG source masters, normalizes them to the 1600x1600 transparent consumer format, validates output, hashes files, writes a manifest, and creates a contact sheet. The contact sheet previews logos across both light and dark backgrounds.

```bash
python -m process.party_assets_build \
  --source-root generated_party_asset_review/sources \
  --output-root generated_party_asset_review/build
```

### Guarded S3 upload

`process/party_assets_upload.py`

Dry-run by default; refuses an unsuccessful build and refuses to overwrite an existing object. No upload has been executed during this implementation.

```bash
python -m process.party_assets_upload --build-root generated_party_asset_review/build
# --apply is required for a real upload and must only be used after approval.
```

## CI / review

`.github/workflows/party_assets_review.yml` performs a no-S3-write review build:

1. run all party-asset unit tests;
2. fetch direct authoritative assets only;
3. normalize available assets;
4. generate manifest and light/dark contact sheet;
5. expose per-party fetch states;
6. retain a packaged review artifact and standalone contact sheet.

The workflow deliberately remains incomplete/red while any recognized non-fallback party lacks a reviewed source.

`.github/workflows/party_assets_audit.yml` performs a read-only S3 inventory. Registry/build/upload tests pass, but the configured GitHub Actions AWS principal currently receives `AccessDenied` for bucket enumeration.

Required discovery permission is read-only:

```text
Action:   s3:ListBucket
Resource: arn:aws:s3:::eirepolitic-data
```

No `s3:PutObject`, delete, or production-pointer permission is required for the discovery audit. IAM has not been changed by this work.

## Remaining sequence

1. Review the generated contact sheet and source provenance for fetched assets.
2. Resolve exact standalone authoritative files for Fianna Fáil, Sinn Féin, and Social Democrats (or explicitly approve a different authoritative extraction strategy).
3. Grant/execute read-only `s3:ListBucket` audit to determine whether equivalent party assets already exist elsewhere in the bucket.
4. Compare any existing S3 candidates against the authoritative sources and reject stale/unofficial duplicates.
5. Mark reviewed rows `pending_review`/`approved` as appropriate.
6. Only after explicit human approval, use the guarded uploader for the new versioned S3 prefix and store CSV/Parquet/manifest/contact-sheet outputs.
7. In a separate rendering change, replace the temporary party-name wordmark with the shared `party_key -> asset` resolver without redesigning the post.

The classified-debate cutoff/backfill issue remains out of scope.
