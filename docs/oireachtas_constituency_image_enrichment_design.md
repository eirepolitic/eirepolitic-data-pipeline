# Oireachtas constituency image enrichment trial design

**Status:** design complete  
**Last updated:** 2026-06-30

## Purpose

Design a side-by-side unified enrichment table for constituency image metadata without overwriting the existing legacy constituency image index.

## Existing legacy workflow

Workflow:

```text
.github/workflows/constituency_images_index.yml
```

Script:

```text
process/constituency_images_indexer.py
```

Current outputs:

```text
processed/constituencies/constituency_images.csv
processed/constituencies/parquets/constituency_images.parquet
```

## Proposed unified enrichment table

Table name:

```text
enrichment_constituency_images
```

Proposed outputs:

```text
processed/oireachtas_unified/enrichment/media/constituency_images/constituency_images_trial.csv
processed/oireachtas_unified/enrichment/media/constituency_images/parquets/constituency_images_trial.parquet
```

Proposed compatibility outputs:

```text
processed/oireachtas_unified/compat/media/constituency_images_compat.csv
processed/oireachtas_unified/compat/media/parquets/constituency_images_compat.parquet
```

## Proposed columns

```text
record_id
constituency
image_key
image_url
media_type
source_key
source_system
source_hash
retrieved_at_utc
review_status
run_id
```

Optional fields if present in the legacy source:

```text
caption
alt_text
license
source_url
width
height
```

## DQ checks

Required:

```text
row_count_gt_zero
record_id_unique
constituency_populated
image_locator_populated
row_count_expected
```

Informational:

```text
image_key_populated_count
image_url_populated_count
missing_locator_count
```

## Implementation approach

The first builder should not download or generate new images. It should reshape the existing legacy image index into a unified enrichment shape.

Recommended module:

```text
extract/oireachtas/enrichment_constituency_images.py
```

Recommended workflow:

```text
.github/workflows/oireachtas_constituency_image_enrichment_trial.yml
```

## Consumer rollout

Do not repoint Instagram rendering directly to the enrichment table first.

Use the compatibility output first:

```text
processed/oireachtas_unified/compat/media/constituency_images_compat.csv
```

Then run Instagram constituency and campaign render trials with:

```text
INSTAGRAM_CONSTITUENCY_IMAGES_DATASET_KEYS=processed/oireachtas_unified/compat/media/constituency_images_compat.csv
```

## Current decision

Proceed to implementation after member photo consumer validation is complete. This is lower risk than member summary replacement because it is media metadata, not generated text.
