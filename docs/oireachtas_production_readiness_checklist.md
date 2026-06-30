# Oireachtas production readiness checklist

**Status:** pre-production cutover validated  
**Last updated:** 2026-06-30

This checklist tracks readiness and validation for repointing downstream consumers to unified Oireachtas compatibility outputs.

## Current decision

The first two pre-production consumer cutovers have been applied and validated:

```text
Instagram constituency renderer
Member profile metrics
```

These systems are not yet in active use, so the cutovers were applied without requiring the earlier explicit approval phrase.

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
| Adapter comparison | Legacy inputs are compared to compatibility outputs. | done, latest run `28414819264` |
| Consumer smoke test | Target downstream workflow runs using trial/compat keys. | done, run `27661992188` |
| Mismatch review | Remaining roster/profile member-code mismatches are identified. | done, latest run `28414847238` |
| Instagram cutover validation | Instagram constituency renderer runs with compat roster default. | done, run `28414647932` |
| Member profile metrics cutover validation | Member profile metrics runs with compat roster/vote defaults. | done, run `28414678714` |
| Rollback | Consumer can be switched back to legacy environment variables or config. | documented |

## Applied consumer changes

### Instagram constituency renderer

Workflow:

```text
.github/workflows/instagram_constituency_test.yml
```

Applied default:

```yaml
      INSTAGRAM_MEMBERS_DATASET_KEYS: "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv"
```

Validation:

```text
Run: 28414647932
Status: success
Artifact: instagram-constituency-test
Artifact ID: 7968986389
```

### Member profile metrics

Workflow:

```text
.github/workflows/build_member_profile_metrics_2025.yml
```

Applied defaults:

```yaml
      MEMBERS_INPUT_KEY: "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv"
      MEMBER_VOTES_INPUT_KEY: "processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv"
```

The legacy vote-record rebuild step was removed from this workflow because the metrics build now reads the unified compatibility vote file directly.

Validation:

```text
Failed pre-correction run: 28414649704
Corrected successful run: 28414678714
```

## Latest validation evidence

| Check | Run | Result | Notes |
|---|---:|---|---|
| Production-sized refresh | `27661934424` | success / DQ pass | `gold_current_members` 174 rows; `silver_member_votes` 29,805 rows. |
| Compatibility adapters | `27661982505` | success / DQ pass | Roster compat 174 rows; member-votes compat 29,805 rows. |
| Instagram cutover validation | `28414647932` | success | Artifact ID `7968986389`. |
| Member profile metrics cutover validation | `28414678714` | success | Production metrics workflow completed with compat inputs. |
| Post-cutover adapter comparison | `28414819264` | success / DQ pass | Roster 176 legacy vs 174 compat; votes 30,968 legacy vs 29,805 compat. |
| Post-cutover mismatch review | `28414847238` | success / DQ pass | Profile metrics now 174/174 matched; roster has 2 legacy-only members. |

Review outputs:

```text
review/member_profile_metrics_trial/latest/{manifest.json,sample.csv,dq.json,report.md}
review/compat_adapter_comparison/latest/{manifest.json,sample.csv,dq.json,report.md}
review/compat_downstream_adapters/latest/{manifest.json,sample.csv,dq.json}
review/member_code_mismatch_review/latest/{manifest.json,sample.csv,dq.json,report.md}
```

## Current safe outputs

```text
processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv
processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv
processed/oireachtas_unified/compat/members/member_profile_metrics_2025_trial.csv
processed/oireachtas_unified/compat/members/parquets/member_profile_metrics_2025_trial.parquet
```

## Latest mismatch baseline

| Dataset | Legacy members | Unified members | Matched | Legacy-only | Unified-only |
|---|---:|---:|---:|---:|---:|
| roster | 176 | 174 | 174 | 2 | 0 |
| member_profile_metrics | 174 | 174 | 174 | 0 | 0 |

Remaining mismatches:

```text
Catherine Connolly — Independent — Galway West
Paschal Donohoe — Fine Gael — Dublin Central
```

## Rollback

Rollback is workflow-config only.

For Instagram constituency rendering, remove:

```yaml
      INSTAGRAM_MEMBERS_DATASET_KEYS: "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv"
```

For member profile metrics, restore legacy input keys:

```yaml
      MEMBERS_INPUT_KEY: "raw/members/oireachtas_members_34th_dail.csv"
      MEMBER_VOTES_INPUT_KEY: "processed/votes/dail_vote_member_records.csv"
```

If legacy vote records should be rebuilt inside the same workflow, restore the removed `Build Dail vote member records` step.

## Remaining caveats

- Deterministic unified outputs do not replace LLM/classified issue data such as `processed/debates/debate_speeches_classified.csv`.
- Photo URLs, member summaries, and constituency image indexes remain outside the deterministic Oireachtas model.
- Roster still has 2 legacy-only member codes relative to the current unified compat roster.

## Decision

Current recommendation: keep the pre-production cutovers in place and monitor using `docs/oireachtas_post_cutover_monitoring_plan.md`.
