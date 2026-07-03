# Enrichment member summaries trial

- Status: `success`
- DQ status: `pass`
- Run ID: `enrichment_member_summaries_20260703T163359Z`
- Source key: `processed/members/members_summaries.csv`
- Source rows: `174`
- Row limit: `0`
- Trial rows: `174`
- Compat rows: `174`
- Summary text populated: `174`
- Summary text missing: `0`

## Outputs

- Trial CSV: `processed/oireachtas_unified/enrichment/text/member_summaries/member_summaries_trial.csv`
- Trial parquet: `processed/oireachtas_unified/enrichment/text/member_summaries/parquets/member_summaries_trial.parquet`
- Compat CSV: `processed/oireachtas_unified/compat/text/members_summaries_compat.csv`
- Compat parquet: `processed/oireachtas_unified/compat/text/parquets/members_summaries_compat.parquet`

This trial does not call OpenAI and does not overwrite legacy member summary keys.
