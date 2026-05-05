# Member Profile Batch v1 — Campaign Brief

## Goal
Generate deterministic Instagram profile cards for selected Dáil members using existing eirepolitic data products.

## Intended audience
People who want quick, visual summaries of TD activity and profile context.

## Post type
Single-image member profile cards. Future versions may become carousel posts.

## Variation grain
Member. Version 1 renders one output image per selected TD.

## Data sources
- `raw/members/oireachtas_members_34th_dail.csv`
- `processed/members/members_photo_urls.csv`
- `processed/members/member_photos/members_photo_urls.csv` fallback
- `processed/members/member_profile_metrics_2025.csv`
- `processed/members/members_summaries.csv` when background text is included

## Templates
- `instagram/templates/layouts/profile_card_v1.json`

## Media generators
None required for the first single-card render. Later versions can add issue charts or rankings.

## Review requirements
Every generated image must be checked for name/photo correctness, text overflow, missing photos, misleading metrics, and visual legibility before publishing.

## Scheduling
Manual only for v1. No cron schedule until reviewed outputs are approved.
