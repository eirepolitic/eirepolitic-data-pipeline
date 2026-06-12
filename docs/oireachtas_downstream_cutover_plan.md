# Oireachtas downstream cutover plan

**Status:** planning only  
**Last updated:** 2026-06-12  
**Do not execute cutover without explicit approval.**

This document maps legacy S3 inputs to unified Oireachtas outputs and gives a reversible trial plan. It does not change downstream consumers, workflows, or legacy pipelines.

## Current downstream consumers observed

| Consumer area | Current use | Source evidence |
|---|---|---|
| Instagram template/render context | Loads S3-backed member, photo, debate/classification, and constituency image data. | `instagram/README.md` |
| Member profile metric build | Loads members, photos, classified debates, and member vote records; writes `processed/members/member_profile_metrics_2025.csv`. | `process/build_member_profile_metrics_2025.py` |
| Legacy monthly extraction | Still produces old member/debate inputs. | `.github/workflows/monthly_extract.yml` |
| Legacy member metric workflow | Still produces old 2025 member profile metrics. | `.github/workflows/build_member_profile_metrics_2025.yml` |

## Legacy-to-unified replacement map

| Legacy S3 key | Unified candidate key | Readiness | Notes |
|---|---|---|---|
| `raw/members/oireachtas_members_34th_dail.csv` | `processed/oireachtas_unified/latest/csv/gold_current_members.csv` | trial-ready | Better downstream shape for current roster. Existing old file has 176 rows; latest test gold output currently has 10 rows because test workflows use `limit=10`. |
| `raw/members/oireachtas_members_34th_dail.csv` | `processed/oireachtas_unified/latest/csv/silver_members.csv` | trial-ready | Stable identity table, but downstream code expecting `party`/`constituency` will need mapping from `latest_party_name` and `latest_constituency_name` or use `gold_current_members`. |
| `processed/votes/dail_vote_member_records.csv` | `processed/oireachtas_unified/latest/csv/silver_member_votes.csv` | trial-ready | Unified field names differ from legacy: `memberCode`/`unique_vote_id` are legacy; unified has `member_code`, `division_id`, and `vote_id`. Metric builder logic needs a compatibility adapter before direct replacement. |
| `processed/votes/dail_vote_divisions.csv` | `processed/oireachtas_unified/latest/csv/silver_divisions.csv` | trial-ready | Unified preserves committee rows and different metadata columns. Use downstream filters rather than changing base table. |
| `processed/members/member_profile_metrics_2025.csv` | `processed/oireachtas_unified/latest/csv/gold_member_activity_yearly.csv` | partial | Unified table covers speech/vote activity metrics, not photo URL, top classified issue, or old exact output schema. |
| `processed/members/parquets/member_profile_metrics_2025.parquet` | `processed/oireachtas_unified/latest/parquet/gold_member_activity_yearly.parquet` | partial | Same schema caveat as CSV. |
| `processed/debates/debate_speeches_classified.csv` | no deterministic unified replacement | not ready | Unified `silver_speeches` is deterministic only and intentionally excludes LLM issue classifications. Keep legacy classified debate input until enrichment layer is built. |
| `processed/members/members_summaries.csv` | no deterministic unified replacement | not ready | Summaries are enrichment content, not deterministic Oireachtas extraction. |
| `processed/members/member_photos/members_photo_urls.csv` | keep existing | keep | Photo data is outside Oireachtas deterministic model. |
| `processed/members/members_photo_urls.csv` | keep existing | keep | Existing fallback remains valid. |
| `processed/constituencies/constituency_images.csv` | keep existing | keep | Image data is outside Oireachtas deterministic model. |

## Safe trial strategy

Use environment variables and new trial output keys first. Do not overwrite legacy outputs.

### Trial 1 — member roster adapter

Goal: prove that unified current members can feed downstream context building.

Recommended adapter output:

```text
processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv
```

Suggested compatibility columns:

| Compat column | Source |
|---|---|
| `member_code` | `gold_current_members.member_code` |
| `full_name` | `gold_current_members.full_name` |
| `constituency` | `gold_current_members.constituency_name` |
| `party` | `gold_current_members.party_name` |
| `house_no` | `gold_current_members.house_no` |
| `source` | literal `oireachtas_unified` |

### Trial 2 — member vote adapter

Goal: prove unified votes can feed the old profile metric builder without changing its logic.

Recommended adapter output:

```text
processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv
```

Suggested compatibility columns:

| Compat column | Source |
|---|---|
| `memberCode` | `silver_member_votes.member_code` |
| `member_name` | `silver_member_votes.member_name` |
| `unique_vote_id` | `silver_member_votes.division_id` fallback `vote_id` |
| `date` | `silver_member_votes.division_date` |
| `vote` | `silver_member_votes.vote_label` |
| `party` | `silver_member_votes.party_name_at_vote` |
| `constituency` | `silver_member_votes.constituency_name_at_vote` |

### Trial 3 — metrics side-by-side build

Run `process/build_member_profile_metrics_2025.py` with environment-variable overrides and a trial output location:

```bash
MEMBERS_INPUT_KEY=processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv \
MEMBER_VOTES_INPUT_KEY=processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv \
MEMBER_PROFILE_METRICS_OUTPUT_CSV_KEY=processed/oireachtas_unified/compat/members/member_profile_metrics_2025_trial.csv \
MEMBER_PROFILE_METRICS_OUTPUT_PARQUET_KEY=processed/oireachtas_unified/compat/members/parquets/member_profile_metrics_2025_trial.parquet \
python process/build_member_profile_metrics_2025.py
```

Keep these unchanged during the first trial:

```text
processed/debates/debate_speeches_classified.csv
processed/members/member_photos/members_photo_urls.csv
processed/members/members_photo_urls.csv
```

Reason: issue classification and photo inputs are not replaced by deterministic unified Oireachtas tables.

## Cutover gates

Do not repoint default workflow inputs until all of these are true:

1. Unified scheduled weekly/monthly/yearly workflows have completed at least one non-test run with production-appropriate limits.
2. Compatibility adapters exist and are validated with review samples.
3. Trial profile metrics output exists beside legacy output.
4. Comparison report shows expected or documented differences.
5. Instagram preview/render workflow succeeds using trial metrics and member roster.
6. Rollback is documented and tested by switching environment variables back to old keys.
7. User explicitly approves the cutover.

## Rollback plan

Rollback is environment/config-only if the trial strategy is followed:

1. Set `MEMBERS_INPUT_KEY` back to `raw/members/oireachtas_members_34th_dail.csv`.
2. Set `MEMBER_VOTES_INPUT_KEY` back to `processed/votes/dail_vote_member_records.csv`.
3. Set `MEMBER_PROFILE_METRICS_OUTPUT_CSV_KEY` back to `processed/members/member_profile_metrics_2025.csv`.
4. Set `MEMBER_PROFILE_METRICS_OUTPUT_PARQUET_KEY` back to `processed/members/parquets/member_profile_metrics_2025.parquet`.
5. Keep old workflows active until one full scheduled cycle succeeds after any approved cutover.

## Current recommendation

Do not cut over yet. Build compatibility adapters first, then run a side-by-side profile metric trial. Current unified latest outputs are validated, but many are limited test outputs, so they are not yet production replacements by row count.
