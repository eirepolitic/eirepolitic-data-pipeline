# Party asset layer v1

Status: implementation paused pending manual logo sourcing for the remaining unresolved parties.

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
3. authoritative manually supplied party artwork after provenance review;
4. explicit EirePolitic-generated stand-in where no official shared identity exists.

Do not scrape a generic webpage and guess which image is the logo. Do not fabricate an official-looking party logo. Do not synthesize a combined emblem for non-standard groupings.

`Independent` uses an EirePolitic-generated neutral `IND / INDEPENDENT` stand-in. It is explicitly marked `eirepolitic_generated_standin`, is not official branding, and must never be presented as an official Independent party emblem.

## Source audit snapshot — 2026-08-31

Exact direct authoritative asset URLs are pinned and technically working for:

- `100-rdr` — Electoral Commission registered emblem
- `aontu` — official Aontú logo asset
- `fine-gael` — official Fine Gael logo asset
- `green-party` — official Green Party SVG
- `independent-ireland` — Electoral Commission registered emblem
- `labour-party` — official Labour RGB SVG mark
- `people-before-profit-solidarity` — Electoral Commission registered grouping emblem

Generated stand-in:

- `independent` — neutral EirePolitic-created stand-in; not official branding

### Deferred manual sourcing

The following remain intentionally unresolved and are now deferred for manual sourcing:

- `fianna-fail`
- `sinn-fein`
- `social-democrats`

Automated official-site discovery and official-publication extraction were attempted. The resulting candidates were reviewed and rejected as not sufficiently correct for canonical use. **None of those publication-derived candidates are approved or canonical.**

When work resumes, manually supplied logo files for these parties should be placed through the same provenance/normalization/review pipeline rather than bypassing the asset layer.

No open redistribution licence is assumed where none is explicitly published. Usage notes remain conservative and assets stay unapproved until human review.

## Asset states

- `source_identified_pending_ingest`: authoritative source identified, normalized approved S3 asset not yet present.
- `pending_review`: prepared/generated asset requires human/source/licensing review.
- `approved`: normalized asset exists, validates technically, and is approved for use.
- `approved_fallback`: explicit no-logo treatment where applicable.

No recognized party may silently fall through to invented branding.

## Tooling

### Registry / resolver

`process/party_assets.py`

Validates unique keys, aliases, provenance, generated stand-ins, fallback state, and can perform a read-only S3 candidate audit.

```bash
python -m unittest tests.test_party_assets -v
python -m process.party_assets
```

### Source staging

`process/party_assets_fetch.py`

Fetches only approved direct HTTPS image/SVG sources and generates registered EirePolitic stand-ins locally. Generic party webpages remain `unresolved_source` and are never scraped for guessed canonical assets.

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

`.github/workflows/party_assets_review.yml` performs a no-S3-write review build and publishes browser-accessible review sheets on the feature branch.

`.github/workflows/party_assets_publication_review.yml` is review-only tooling for extracting candidates from official party publications. Its candidates are advisory only and are not canonical. Current publication-derived candidates were rejected and should not be promoted.

`.github/workflows/party_assets_audit.yml` performs a read-only S3 inventory. Registry/build/upload tests pass, but the configured GitHub Actions AWS principal currently receives `AccessDenied` for bucket enumeration.

Required discovery permission is read-only:

```text
Action:   s3:ListBucket
Resource: arn:aws:s3:::eirepolitic-data
```

No `s3:PutObject`, delete, or production-pointer permission is required for the discovery audit. IAM has not been changed by this work.

## Resume point

When manual logo files for Fianna Fáil, Sinn Féin, and Social Democrats are available:

1. supply/upload the three files and identify their source/provenance where known;
2. add them to the party asset staging/review flow;
3. normalize and regenerate the complete contact sheet;
4. review/approve the complete asset set, including the Independent stand-in;
5. run the read-only S3 inventory when `s3:ListBucket` is available;
6. compare existing S3 candidates and reject stale/unofficial duplicates;
7. only after explicit approval, perform the guarded versioned S3 upload;
8. integrate the shared resolver into Instagram in a separate rendering change without redesigning the post.

The classified-debate cutoff/backfill issue remains out of scope.
