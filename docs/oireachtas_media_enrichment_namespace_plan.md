# Oireachtas media and enrichment namespace follow-up

**Status:** planning complete  
**Last updated:** 2026-06-30

## Purpose

This document decides how to handle the remaining enrichment/media datasets after the classified-issue enrichment trial.

## Remaining non-deterministic or media datasets

| Dataset | Current key | Current role | Recommendation |
|---|---|---|---|
| Member photo URLs | `processed/members/member_photos/members_photo_urls.csv` | Supplies member images for profile/campaign rendering. | Create side-by-side unified enrichment namespace before any consumer cutover. |
| Legacy member photo fallback | `processed/members/members_photo_urls.csv` | Fallback path used by member profile metrics. | Keep as fallback until unified photo index is validated. |
| Member summaries | `processed/members/members_summaries.csv` | Supplies short member background text. | Keep legacy until provenance/review fields exist. |
| Constituency images | `processed/constituencies/constituency_images.csv` | Supplies constituency media index. | Move later into media enrichment namespace, not Oireachtas silver/gold. |

## Recommended namespace

Do not put these in deterministic silver/gold Oireachtas tables. Use:

```text
processed/oireachtas_unified/enrichment/media/member_photo_urls/
processed/oireachtas_unified/enrichment/text/member_summaries/
processed/oireachtas_unified/enrichment/media/constituency_images/
```

## Candidate future tables

```text
enrichment_member_photo_urls
enrichment_member_summaries
enrichment_constituency_images
```

## Minimum common columns

All enrichment/media outputs should include:

```text
record_id
source_key
source_system
source_url
source_hash
retrieved_at_utc
review_status
run_id
```

Dataset-specific keys:

```text
member_code
full_name
constituency
photo_url
summary_text
image_key
image_url
media_type
```

## Rollout order

Recommended order:

1. `enrichment_member_photo_urls`
2. `enrichment_constituency_images`
3. `enrichment_member_summaries`

Reason:

- Photo and constituency image indexes are mostly deterministic media metadata.
- Member summaries are text generation/enrichment and need stronger provenance/review controls.

## Consumer rule

Do not repoint Instagram or member-profile consumers directly to new enrichment files first. Build compatibility outputs first:

```text
processed/oireachtas_unified/compat/media/members_photo_urls_compat.csv
processed/oireachtas_unified/compat/media/constituency_images_compat.csv
processed/oireachtas_unified/compat/text/members_summaries_compat.csv
```

Then validate consumer artifacts side by side.

## Current decision

No production consumer repointing is recommended yet. The next enrichment implementation after classified issues should be a member photo URL trial builder because it is lower risk than generated member summaries.
