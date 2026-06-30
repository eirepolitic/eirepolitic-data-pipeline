# Oireachtas post-cutover monitoring plan

**Status:** active for pre-production cutover validation  
**Last updated:** 2026-06-30

## Cutovers applied

| Consumer | Change | Validation run | Result |
|---|---|---:|---|
| Instagram constituency renderer | Default roster input now uses `processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv`. | `28414647932` | success |
| Member profile metrics | Default member and vote inputs now use unified compatibility CSVs. Legacy vote-record rebuild step removed from this workflow. | `28414678714` | success |

## Rollback commands / patches

### Instagram rollback

Remove this workflow environment variable from `.github/workflows/instagram_constituency_test.yml`:

```yaml
      INSTAGRAM_MEMBERS_DATASET_KEYS: "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv"
```

### Member profile metrics rollback

Restore these legacy input keys in `.github/workflows/build_member_profile_metrics_2025.yml`:

```yaml
      MEMBERS_INPUT_KEY: "raw/members/oireachtas_members_34th_dail.csv"
      MEMBER_VOTES_INPUT_KEY: "processed/votes/dail_vote_member_records.csv"
```

If legacy vote records should continue to be rebuilt inside the same workflow, restore the removed step:

```yaml
      - name: Build Dail vote member records
        run: |
          python process/build_dail_votes_member_records.py \
            --date-start "${{ inputs.date_start }}" \
            --date-end "${{ inputs.date_end }}"
```

## Monitoring checks

After each manual or scheduled run, check:

1. Workflow conclusion is `success`.
2. Instagram render artifact is uploaded and visually plausible.
3. Member profile metrics output row count remains near the post-refresh trial count of 174 rows.
4. No missing required columns in generated CSV or parquet outputs.
5. Known member-code differences remain limited to the P13 mismatch list unless source data changes.

## Known mismatch baseline

From P13 run `27662884471`:

| Dataset | Side | Members |
|---|---|---|
| roster | legacy-only | Catherine Connolly, Paschal Donohoe |
| member profile metrics | legacy-only | Catherine Connolly, Paschal Donohoe |
| member profile metrics | trial-only | Daniel Ennis, Seán Kyne |

## Next recommended validation

Run the following after any future unified refresh:

1. `Oireachtas Downstream Compatibility Adapters (Manual)`
2. `Oireachtas Compatibility Adapter Comparison (Manual)`
3. `Generate Instagram Constituency Test Post (Manual)`
4. `Build Member Profile Metrics 2025 (Manual)`

Keep the rollback instructions in this document until the workflows have completed at least one full operational cycle without regression.
