# Member Profile Batch v1 — Campaign Brief

## Goal

Generate one deterministic Instagram profile image per selected TD using existing 2025 member metrics.

## Intended audience

People who want quick, factual summaries of Dáil member activity.

## Post type

Single-image member profile card first. Carousel variants can be added later.

## Variation grain

`member`

## Data sources

- `processed/members/member_profile_metrics_2025.csv`
- `raw/members/oireachtas_members_34th_dail.csv`
- `processed/members/members_photo_urls.csv`
- `processed/debates/debate_speeches_classified.csv`
- `processed/votes/dail_vote_member_records.csv`

## Templates

- `instagram/templates/layouts/profile_card_v1.json`

## Media generators

- None required for the first single-card render.
- Horizontal bar chart generator is available for future carousel slides.

## Review requirements

Manually check every generated card for name, party, constituency, metrics, missing images, text fit, and source warnings before publication.

## Scheduling

Manual only for v1.
