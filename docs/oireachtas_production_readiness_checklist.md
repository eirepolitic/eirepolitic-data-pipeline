# Oireachtas production readiness checklist

**Status:** approval checklist  
**Last updated:** 2026-06-17  
**No downstream cutover is approved by this document.**

Use this checklist before repointing Instagram, member profile metrics, or any other downstream consumer to unified Oireachtas outputs.

## Approval gate

Do not repoint consumers or disable legacy workflows until all required checks below are complete and the user explicitly approves the change.

Required approval phrase to record in a future commit or ticket:

```text
Approved: cut over <consumer name> from legacy Oireachtas keys to unified compatibility outputs.
```

## Required checks

| Area | Check | Status |
|---|---|---|
| Table validation | Required silver/gold/control tables have DQ pass. | done |
| Registry | Required tables are marked `confirmed` in `configs/oireachtas/tables.yml`. | done |
| Latest publishing | `mode=test` cannot overwrite `processed/oireachtas_unified/latest/*`. | done, run `27431598142` |
| Scheduled windows | Weekly/monthly/yearly scheduled workflows use dynamic date windows. | done |
| Production-sized refresh | Target consumer tables refreshed with non-test mode and latest publishing. | done, run `27661934424` |
| Compatibility outputs | Required compat CSVs exist under `processed/oireachtas_unified/compat/...`. | done, run `27661982505` |
| Side-by-side trial | Member profile metric trial writes to non-legacy output keys. | done, run `27661985049` |
| Adapter comparison | Legacy inputs are compared to compatibility outputs. | done, run `27661990358` |
| Consumer smoke test | Target downstream workflow runs using trial/compat keys. | done, run `27661992188` |
| Mismatch review | Remaining roster/profile member-code mismatches are identified. | pending P13 validation |
| Rollback | Consumer can be switched back to legacy environment variables or config. | documented |
| User approval | Explicit approval exists for each consumer cutover. | pending |

## Current safe outputs

These outputs are safe to inspect and trial because they do not overwrite legacy keys:

```text
processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv
processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv
processed/oireachtas_unified/compat/members/member_profile_metrics_2025_trial.csv
processed/oireachtas_unified/compat/members/parquets/member_profile_metrics_2025_trial.parquet
```

## Current legacy keys to keep active

Keep these in place until a consumer-specific cutover is approved:

```text
raw/members/oireachtas_members_34th_dail.csv
processed/votes/dail_vote_member_records.csv
processed/debates/debate_speeches_classified.csv
processed/members/member_profile_metrics_2025.csv
processed/members/parquets/member_profile_metrics_2025.parquet
processed/members/member_photos/members_photo_urls.csv
processed/members/members_photo_urls.csv
processed/constituencies/constituency_images.csv
```

## Latest validation evidence

| Check | Run | Result | Notes |
|---|---:|---|---|
| Production-sized refresh | `27661934424` | success / DQ pass | `gold_current_members` 174 rows; `silver_member_votes` 29,805 rows. |
| Compatibility adapters | `27661982505` | success / DQ pass | Roster compat 174 rows; member-votes compat 29,805 rows. |
| Member profile trial | `27661985049` | success / DQ pass | Trial metrics 174 rows; 172 matched legacy member codes. |
| Compatibility adapter comparison | `27661990358` | success / DQ pass | Roster 176 legacy vs 174 compat; votes 30,968 legacy vs 29,805 compat. |
| Instagram consumer smoke | `27661992188` | success | Artifact-only render confirmed compat members key. |

Review outputs:

```text
review/member_profile_metrics_trial/latest/{manifest.json,sample.csv,dq.json,report.md}
review/compat_adapter_comparison/latest/{manifest.json,sample.csv,dq.json,report.md}
review/compat_downstream_adapters/latest/{manifest.json,sample.csv,dq.json}
```

## Recommended cutover order

1. Complete remaining mismatch review.
2. Review row/key differences and rendered smoke artifact.
3. Approve one consumer at a time using the exact approval phrase.
4. Repoint that consumer through config/environment variables only.
5. Keep legacy workflow active for at least one complete scheduled cycle.
6. Roll back immediately if consumer output regresses.

## Rollback

Rollback should be config-only if the trial approach is followed.

For Instagram constituency rendering:

```bash
unset INSTAGRAM_MEMBERS_DATASET_KEYS
```

For member profile metrics:

```bash
MEMBERS_INPUT_KEY=raw/members/oireachtas_members_34th_dail.csv
MEMBER_VOTES_INPUT_KEY=processed/votes/dail_vote_member_records.csv
MEMBER_PROFILE_METRICS_OUTPUT_CSV_KEY=processed/members/member_profile_metrics_2025.csv
MEMBER_PROFILE_METRICS_OUTPUT_PARQUET_KEY=processed/members/parquets/member_profile_metrics_2025.parquet
```

## Known caveats before approval

- Roster comparison still has 2 legacy-only member codes after the refresh.
- Member profile metrics comparison has 2 legacy-only and 2 trial-only member codes after the refresh.
- Deterministic unified outputs do not replace LLM/classified issue data such as `processed/debates/debate_speeches_classified.csv`.
- Photo URLs, member summaries, and constituency image indexes remain outside the deterministic Oireachtas model.

## Decision

Current recommendation: **do not cut over yet**. Complete P13 mismatch review and wait for explicit consumer-specific approval.
