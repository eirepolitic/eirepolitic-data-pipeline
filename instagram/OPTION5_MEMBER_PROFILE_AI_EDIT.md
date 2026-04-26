# Option 5: Template-Based Member Profile AI Edit Test

This test shifts option 5 away from full-slide generation.

## Goal

Use an existing member-profile slide image as the master template and ask the model to replace:
- member portrait
- member name
- constituency
- party
- top issue
- vote participation %
- speech rank

The model should preserve the original layout and styling as closely as possible.

## Why this is safer than full-slide generation

- the visual direction already exists
- the pipeline can preserve exact source values in sidecar metadata
- review focuses on edit fidelity instead of invention
- the vote and speech metrics are computed deterministically before the edit call

## New pipeline pieces

### Vote member records

Script:
- `process/build_dail_votes_member_records.py`

Outputs:
- `s3://eirepolitic-data/processed/votes/dail_vote_divisions.csv`
- `s3://eirepolitic-data/processed/votes/parquets/dail_vote_divisions.parquet`
- `s3://eirepolitic-data/processed/votes/dail_vote_member_records.csv`
- `s3://eirepolitic-data/processed/votes/parquets/dail_vote_member_records.parquet`

Design notes:
- source endpoint: Oireachtas votes API
- chamber fixed to Dáil
- house fixed to 34
- committee votes excluded when `committeeCode` is non-empty
- member rows are expanded from `taVotes`, `nilVotes`, and `staonVotes`
- `unique_vote_id` is built as `voteId_date`

### Member profile metrics 2025

Script:
- `process/build_member_profile_metrics_2025.py`

Outputs:
- `s3://eirepolitic-data/processed/members/member_profile_metrics_2025.csv`
- `s3://eirepolitic-data/processed/members/parquets/member_profile_metrics_2025.parquet`

Columns:
- `member_code`
- `full_name`
- `constituency`
- `party`
- `photo_url`
- `top_issue_2025`
- `top_issue_count_2025`
- `vote_participation_pct_2025`
- `distinct_votes_participated_2025`
- `all_distinct_vote_ids_2025`
- `speech_count_2025`
- `speech_rank_2025`

Metric logic:
- vote participation % = distinct vote IDs the TD participated in / all distinct vote IDs in 2025
- speech rank = dense rank by 2025 classified speech count descending
- top issue = most common non-`NONE` classified issue in 2025

### AI template edit runner

Script:
- `process/render_member_profile_ai_edit.py`

Spec:
- `instagram/specs/member_profile_ai_test.yml`

It:
- loads the 2025 member metrics table
- picks a member with a photo
- downloads the member photo
- uses the template slide image as the master reference
- calls the image edit model with the template plus the new portrait
- writes sidecar truth metadata and the exact prompt used

## Workflow

Manual workflow:
- `Generate Instagram Option 5 Member Profile AI Edit Test (Manual)`

It runs:
1. vote member record extraction
2. 2025 member metrics build
3. member-profile AI edit generation

## Required manual asset

The workflow expects a template image file at:
- `instagram/reference/member_profile_template.png`

That file is not generated automatically. It should be added to the repo before running the member-profile AI edit workflow.

## Review rule

Do not trust the edited output just because it resembles the template.

Check the generated image against:
- `source_values.json`
- the selected member photo
- the exact visible values expected on the slide
