# Official recurring political-adjacent data validation

## Purpose

Continue the Irish political data-source research with two high-value official recurring datasets that narrowly missed the original top five:

1. eTenders procurement open data.
2. Department of Finance monthly Exchequer/tax-receipt data.

This remained research-only. No production architecture, schemas, pipelines, jobs, secrets, infrastructure or production data were changed.

## Final conclusion

### eTenders

**Status: Ready for a later bounded non-production prototype.**

The current official procurement CSV is large but structurally usable: **87,427 rows and 31 columns**, no exact duplicate rows, direct public CSV access, and CC BY 4.0 licensing. The main work is not connectivity; it is entity/value interpretation and completeness-aware analysis.

Strong recurring uses include:

- largest awards by contracting authority;
- supplier concentration;
- procurement by procedure/category;
- SME bid/award participation;
- quarterly public-procurement scorecards.

The source should not be used for naive league tables without handling missing awarded values, supplier-name variants, framework/mini-competition relationships, cancelled notices, and the post-2023 publication-rule change.

### Department of Finance monthly tax receipts

**Status: High editorial value, but automated CSV access needs remediation/owner clarification before prototyping.**

The official service remains current and states that monthly Exchequer tax receipts run from January 1984 to the present and are published on the second working day of the following month. The dataset is CC BY 4.0 and is explicitly intended as a monthly recurring series.

However, the official open-data record still advertises a legacy `databank.finance.gov.ie` CSV endpoint. In the bounded automation environment, both the HTTPS path and the HTTP path failed certificate verification before machine-readable CSV could be retrieved. This is an operational access problem, not evidence that the underlying data/service is unavailable.

The separate `FIQ02 - Exchequer Account (Historical Series)` resource is technically cleaner (CSV/JSON-stat/PX/XLSX via PxStat), but its catalogue update date is historical and it is not a safe substitute for the current monthly tax-receipts feed.

**Recommendation:** keep monthly tax receipts on the roadmap, but do not build against the legacy CSV URL until the Department/data.gov.ie provides or confirms a stable TLS-valid machine-readable endpoint.

# 1. eTenders procurement open data

## Official source

Dataset:

https://data.gov.ie/dataset/contract-notices-published-on-etenders

Current direct CSV inspected:

```text
https://assets.gov.ie/static/documents/4d482e0e/Public_Procurement_Opendata_Dataset.csv
```

Publisher: Office of Government Procurement.

Licence: **CC BY 4.0**.

Update cadence: **quarterly**.

Current catalogue coverage during this validation: **1 January 2013 through 30 June 2026**.

The catalogue notes that Circular 05/2023 strengthened publication requirements, including publication of contract-award details over €25,000 excluding VAT even where procurement occurred outside eTenders. That change improves completeness but creates a structural coverage break when comparing earlier years with later years.

## Exact file diagnostics

- rows: **87,427**;
- columns: **31**;
- encoding: CP1252;
- file size: approximately **34.7 MB**;
- SHA-256: `0c7a52980d381ad82f7e52f1b0e21bed78a2e9ad1f22db09881e6d7c786f8cbc`;
- exact duplicate rows: **0**.

Exact header:

```text
Tender ID
Parent Agreement ID
Contracting Authority
Name of Client Contracting Authority
Agreement Owner
Tender/Contract Name
Notice Published Date / Contract Created Date
Directive
Competition Type
Main Cpv Code
Main Cpv Code Description
Additional CPV Codes on CFT
Spend Category
Contract Type
Threshold Level
Procedure
Tender Submission Deadline
Evaluation Type
Sum of Notice Estimated Value (€)
Sum of Contract Duration (Months)
Cancelled Date
Award Published
Sum of Awarded Value (€)
Sum of No of Bids Received
Sum of No of SMEs Bids Received
Awarded Suppliers
Sum of No of Awarded SMEs
TED Notice Link
TED CAN Link
Platform
Source
```

## Important missingness

Many fields are intentionally sparse because a procurement notice may not yet be awarded, may use a different workflow, or may not expose every optional attribute.

Notable missing counts in the tested snapshot:

| Field | Missing rows |
| --- | ---: |
| Tender ID | 3,910 |
| Parent Agreement ID | 83,708 |
| Contracting Authority | 33 |
| Notice/contract-created date | 1,543 |
| Main CPV code/description | 51,767 |
| Spend Category | 13,121 |
| Contract Type | 13,341 |
| Procedure | 3,915 |
| Tender Submission Deadline | 15,261 |
| Evaluation Type | 62,668 |
| Estimated Value | 55,911 |
| Award Published | 41,482 |
| Awarded Value | 56,737 |
| Awarded Suppliers | 42,576 |
| TED Notice Link | 59,399 |
| TED CAN Link | 70,718 |

A later analytical model must distinguish:

- not applicable;
- not yet awarded;
- not reported;
- genuinely missing/invalid.

Do not fill these fields with zero or “no award” without source semantics.

## Contracting-authority normalization risk

There are **5,931 distinct non-empty `Contracting Authority` strings** in the file.

The source includes obvious naming variants and historical names. The separate `Name of Client Contracting Authority` field also shows variants such as:

- `Health Service Executive (HSE)`;
- `HEALTH SERVICE EXECUTIVE`;
- `HSE`.

Therefore public-body rankings should use:

1. raw source name preserved unchanged;
2. a separate derived normalized authority identifier/name;
3. versioned mapping rules with manual review for high-value entities.

Never overwrite the source label with the normalized label.

## Supplier normalization risk

`Awarded Suppliers` contains **16,696 distinct non-empty strings** in the tested snapshot.

Examples show legal-name/encoding/format variants such as:

- `Mazars`;
- `Deloitte Ireland LLP`;
- `Ernst &amp; Young`;
- supplier strings prefixed with delimiter characters in some rows.

Supplier-concentration graphics therefore require a separate entity-resolution layer. Raw string grouping alone will understate concentration and split the same supplier into variants.

## Competition/procedure structure

Competition types include:

- Bespoke;
- Framework;
- Standalone Direct Invite;
- Standalone Contract;
- FW - Mini-Comp;
- DPS Tender;
- DPS/UQS;
- Simplified.

Procedures include:

- Open Procedure;
- Direct Invite – Mini-Competition;
- Competitive Procedure With Negotiation;
- Restricted Procedure;
- Simplified;
- DPS Qualification;
- Competitive Dialogue;
- Negotiated Procedure Without Prior Publication;
- UQS Qualification;
- Direct Invite – Quick Quote.

`Parent Agreement ID`, framework and mini-competition relationships mean the source is not simply one row = one independent public contract. Any total-spend analysis must define its grain carefully to avoid double counting framework-level and call-off/mini-competition records.

## Value fields

The tested snapshot contains:

- `Sum of Notice Estimated Value (€)`: 31,516 populated numeric rows;
- `Sum of Awarded Value (€)`: 30,690 populated numeric rows.

Values include zero and very large multi-billion-euro figures. These should be treated as source values requiring outlier review, not automatically interpreted as one-off cash expenditure in the publication year.

A future validation should flag:

- zero award/estimate values;
- extremely large values;
- duration/value combinations suggesting framework ceilings rather than actual spend;
- multiple awarded suppliers or aggregate values;
- awards linked to parent agreements.

**Editorial rule:** use “contract/award value recorded in eTenders” rather than implying the figure equals cash actually paid.

## Bid/SME fields

The source exposes:

- number of bids received;
- number of SME bids received;
- number of awarded SMEs.

These are potentially excellent recurring transparency metrics, but they should only be used on rows where the relevant fields are populated and where competition type/procedure makes comparison meaningful.

## Date handling

The source uses day/month/year strings. A future parser should parse dates explicitly rather than rely on locale inference.

`Tender Submission Deadline` can extend beyond the current coverage period for notices already published. This is expected and should not be treated as a date error.

## Coverage break after Circular 05/2023

The strengthened requirement for awards above €25,000 excluding VAT materially changes what is expected to appear in the dataset after Circular 05/2023.

Therefore:

- do not claim that year-on-year growth in record count or award count necessarily reflects real procurement growth;
- distinguish publication/coverage-rule effects from real economic change;
- place an annotation/series break around the rule change for historical charts.

## Recommended future source model

A later non-production proof should preserve at least:

```text
source_file_url
source_file_hash
retrieved_at
source_row_number
tender_id
parent_agreement_id
contracting_authority_raw
client_contracting_authority_raw
agreement_owner_raw
tender_contract_name
notice_created_date
competition_type
procedure
cpv_code
spend_category
contract_type
threshold_level
estimated_value_eur
awarded_value_eur
award_published
awarded_suppliers_raw
bids_received
sme_bids_received
awarded_smes
cancelled_date
platform
source
validation_flags
```

Entity normalization should live in derived fields/tables, not replace source values.

## Minimum future validation rules

1. Record file URL, retrieval timestamp and SHA-256.
2. Fingerprint all 31 columns and alert on schema changes.
3. Parse dates with explicit `DD/MM/YYYY` rules.
4. Preserve nulls; never default missing values to zero.
5. Validate numeric fields but allow zero where published.
6. Flag negative monetary/count values if they appear.
7. Separate cancelled records from active/awarded analyses.
8. Define framework/mini-competition/parent-agreement grain before summing values.
9. Normalize authorities/suppliers only in derived fields.
10. Treat supplier HTML entities/delimiters as source-cleaning issues, preserving originals.
11. Annotate the 2023 publication-rule change in historical comparisons.
12. Describe monetary fields as recorded contract/award values, not actual Exchequer cash spend.

# 2. Department of Finance monthly Exchequer tax receipts

## Official source

Open-data record:

https://data.gov.ie/dataset/monthly-exchequer-tax-receipts-1984-present

Official current service pages:

- https://www.gov.ie/en/department-of-finance/services/access-the-department-of-finance-databank/
- https://www.gov.ie/en/department-of-finance/services/view-irish-exchequer-tax-receipts-data/

Publisher: Department of Finance.

Licence: **CC BY 4.0**.

Coverage stated by the Department: **January 1984 to present**.

Update cadence: monthly, published on the **second working day of the subsequent month**, with quarterly press context.

Tax heads listed by the official dataset include:

- Customs;
- Excise Duty;
- Capital Gains Tax;
- Capital Acquisitions Tax;
- Stamps;
- Income Tax;
- Corporation Tax;
- Value Added Tax;
- Training and Employment Levy;
- Local Property Tax;
- Unallocated Tax Receipts.

## Machine-readable access problem

The data.gov.ie resource advertises:

```text
http://databank.finance.gov.ie/FinDataBank.aspx?rep=OpenDataSourceCSV
```

The bounded automation test attempted both:

```text
https://databank.finance.gov.ie/FinDataBank.aspx?rep=OpenDataSourceCSV
http://databank.finance.gov.ie/FinDataBank.aspx?rep=OpenDataSourceCSV
```

Both failed in the automated validation environment with TLS certificate verification errors after redirect/connection handling.

This investigation did **not** disable TLS verification or work around certificate validation.

### Interpretation

- The official public service is still current and explicitly advertises January 1984–present monthly data.
- The open-data machine-readable access metadata is stale/legacy and operationally unreliable for automation.
- This is a source-maintenance/access problem, not evidence that the monthly series itself has ended.

## Why FIQ02 is not a direct replacement

Official alternative:

`FIQ02 - Exchequer Account (Historical Series)`

It provides:

- CSV;
- JSON-stat;
- PX;
- XLSX;
- stable PxStat API access;
- CC BY 4.0 licensing.

However, its catalogue metadata is historical (data last updated July 2021 in the checked resource), and the table is a broader quarterly Exchequer account series rather than a verified current replacement for the Department's monthly tax-receipts feed.

Therefore FIQ02 is useful for historical fiscal context but should not silently substitute for the monthly tax dataset.

## Recommended next action before implementation

No connector should be built yet.

Before prototyping monthly tax receipts, obtain or verify one of:

1. a Department/data.gov.ie TLS-valid HTTPS CSV endpoint;
2. a CKAN datastore/API resource for the monthly series;
3. a current official replacement endpoint documented by the Department.

A simple source-maintenance query to the dataset owner/open-data team would be enough; no purchase or registration is needed.

## Editorial value once access is stable

The source remains highly attractive for recurring features:

- monthly total tax receipts;
- year-on-year tax-head changes;
- corporation tax concentration/volatility;
- income tax/VAT/corporation tax mix;
- pre/post-budget comparisons;
- long-run 1984–present tax composition.

But a future implementation must distinguish monthly values from cumulative year-to-date figures if both appear, and must retain historical tax-head changes/renames rather than assuming one unchanged schema for 40+ years.

# Comparison

| Source | Editorial value | Access | Automation | Licence | Main risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| eTenders procurement | High | Easy | High | CC BY 4.0 | entity normalization, sparse awards, coverage-rule break | Ready for bounded prototype |
| Monthly tax receipts | High | Currently unreliable machine endpoint | Low–Medium until fixed | CC BY 4.0 | legacy TLS-invalid CSV access, long-run tax-head semantics | Keep high priority; fix/confirm access first |
| FIQ02 historical Exchequer | Medium–High | Easy API | High | CC BY 4.0 | historical/quarterly, not current monthly replacement | Useful supplementary source |

# Research log

### 2026-09-03 — plan

- Created the near-next official recurring-data validation plan.
- Production changes: none.

### 2026-09-03 — eTenders source validation

- Confirmed current official quarterly CSV, CC BY 4.0 and coverage through 30 June 2026.
- Validated 87,427 rows, 31 fields and zero exact duplicate rows.
- Identified high missingness in award/value/supplier fields and substantial authority/supplier normalization work.
- Confirmed Circular 05/2023 creates a historical coverage/completeness break that must be annotated.
- Production changes: none.

### 2026-09-03 — Finance monthly tax access validation

- Confirmed the Department's service still states January 1984–present monthly coverage and second-working-day publication cadence.
- Confirmed the open-data record is CC BY 4.0.
- Automated access to the advertised legacy databank CSV endpoint failed TLS certificate validation.
- No certificate bypass was attempted.
- Confirmed FIQ02 is a clean structured historical alternative but not a verified current monthly replacement.
- Production changes: none.

## Evidence references

### eTenders

- https://data.gov.ie/dataset/contract-notices-published-on-etenders
- https://assets.gov.ie/static/documents/4d482e0e/Public_Procurement_Opendata_Dataset.csv

### Department of Finance

- https://data.gov.ie/dataset/monthly-exchequer-tax-receipts-1984-present
- https://www.gov.ie/en/department-of-finance/services/access-the-department-of-finance-databank/
- https://www.gov.ie/en/department-of-finance/services/view-irish-exchequer-tax-receipts-data/
- https://data.gov.ie/dataset/fiq02-exchequer-account-historical-series
- https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/FIQ02/JSON-stat/2.0/en

## Living next step

The near-next recurring-source validation is complete. Update the broader Irish political-data ranking to promote **eTenders** to the next prototype tier, retain **monthly tax receipts** as a high-value source blocked only by machine-access reliability, and keep **FIQ02** as a supplementary historical fiscal source rather than a substitute.