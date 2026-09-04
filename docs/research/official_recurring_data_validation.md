# Official recurring political-adjacent data validation

## Purpose

Continue the Irish political data-source research with two high-value official recurring datasets that narrowly missed the original top five:

1. eTenders procurement open data.
2. Department of Finance monthly Exchequer/tax-receipt data.

This is research-only. No production architecture, schemas, pipelines, jobs, secrets, infrastructure or production data may be changed.

## Questions to answer

### eTenders

- What exact files/resources are currently published and how often are they updated?
- Are awards, buyers, suppliers, values, procedures and dates consistently machine-readable?
- What coverage/completeness caveats exist across years?
- Are there duplicate/null/entity-normalisation issues that materially affect recurring charts?
- Is CKAN datastore/API access available, or is direct CSV the better path?
- What minimum validation rules are needed before supplier/public-body rankings are trusted?

### Department of Finance

- What exact recurring monthly tax-receipt file/API is published now?
- What tax categories, periods and units are present?
- Is history truly continuous from 1984 to present in one schema?
- Are revisions, cumulative-vs-monthly semantics or category changes documented in the data?
- Is the data directly suitable for recurring month-on-month/year-on-year graphics?
- What validation/version metadata should a later loader preserve?

## Guardrails

- Read-only official access only.
- Prefer CKAN/API/static files over scraping.
- Do not create production connectors.
- Preserve licence/attribution and source caveats.
- Temporary validation workflows/reports must be removed before merge.
- Final merge contains documentation only.

## Planned sequence

1. Validate eTenders access, schema, update cadence and data-quality caveats.
2. Validate Department of Finance monthly tax/Exchequer access, schema and time-series semantics.
3. Compare editorial value, automation, maintenance and source-risk.
4. Update the broader Irish political-data research ranking if warranted.

## Research log

### 2026-09-03 — plan

- Created the near-next official recurring-data validation plan.
- Production changes: none.
- Next: inspect the current official eTenders resource and exact access path.

## Living next step

Validate eTenders first, then Department of Finance monthly tax/Exchequer data, keeping all checks read-only and documentation-only.