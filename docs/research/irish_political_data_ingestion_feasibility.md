# Irish political data ingestion feasibility

## Scope

Read-only follow-up to `irish_political_data_sources.md`. This document tests the five highest-priority future-ingestion candidates without implementing a connector or changing production architecture, schemas, pipelines, secrets, infrastructure, or production data.

## Exact validation result — completed 2026-09-03

All five candidates below have now been tested beyond the initial feasibility pass. No production connector, schema or pipeline was created.

| Prototype order | Source | Exact validation result | Main safeguard | Durable record |
| --- | --- | --- | --- | --- |
| 1 | Irish Polling Indicator | Ready for non-production prototype | Pin upstream commit; preserve raw/model units; audit duplicate/date anomalies | `ipi_readonly_proof.md`, `ipi_bounded_validation.md` |
| 2 | Official referendum results | Ready for non-production prototype | Prefer CKAN datastore; preserve boundary vintage; flag 2015 900-vote anomaly | `referendum_data_validation.md` |
| 3 | CSO PxStat | Ready after selecting a concrete political-use table | Validate unit, dimensions and geography for the exact table | `cso_pxstat_validation.md` |
| 4 | Official general-election count/transfer data | Ready with source-specific adapters | 2016/2020 CSV-style; 2024 workbook parser; IDs are election-scoped | `election_count_data_validation.md` |
| 5 | Irish Demographic Polling Datasets | Technically ready; rights clarification first | Do not treat weighted counts as respondent n; clarify recurring production/republication rights | `irish_demographic_polling_validation.md` |

The validation sequence found no reason to reject any of the five technically. The only gating issue is rights clarity for the demographic polling dataset.

## Initial pre-validation summary

| Rank | Source | Feasibility | Recommended future ingestion mode | Main risk |
| --- | --- | --- | --- | --- |
| 1 | Irish Polling Indicator | Ready for bounded proof | Static GitHub CSV pull; stable DOI snapshot for reproducibility | Development files are revised as new polls/model updates arrive |
| 2 | CSO PxStat | Ready for bounded proof | JSON-stat API by explicit table ID | Geography/table versions must be pinned and interpreted correctly |
| 3 | Official referendum results | Ready for bounded proof | Direct CSV per referendum + light normalization | Historical column naming varies slightly |
| 4 | Official general-election count/transfer data | Ready for bounded proof with adapters | Per-election file adapter | 2024 moved from CSV-style resources to a richer XLSX workbook |
| 5 | Irish Demographic Polling Datasets | Technically ready; legal clarification recommended | Static GitHub CSV pull | Commercial/republication rights are less explicit than technical access |

No source requires production scraping for the recommended path.

## 1. Irish Polling Indicator

### Access tested

- Project/data page: https://pollingindicator.com/
- Development repository: https://github.com/Irish-Polling-Indicator/ipi-data
- Stable dataset DOI: 10.7910/DVN/8YVVYX

The live project page states that the development dataset is updated after each new poll. Raw polls are offered in CSV/XLSX/Stata/R and modeled daily estimates in CSV/Stata/R. Stable releases are published after an election cycle with a DOI.

### Fields and identifiers

The raw polling dataset exposes poll/publication date, fieldwork start/end/midpoint, pollster, sample size and party-support values. These are sufficient to create a stable source-level poll identity using a composite of pollster + fieldwork/publication dates, subject to later duplicate testing.

### Versioning/update behaviour

- Development files are mutable and expected to change as new polls arrive.
- Modeled historical estimates can be recalculated/revised.
- Stable election-cycle releases are the reproducible historical anchor.

### Recommended future design

Use two source concepts in any later proof:

1. **Development snapshot** for current recurring graphics.
2. **Stable DOI release** for reproducible historical graphics/audits.

A future ingest should record retrieval timestamp and source version/commit where possible. It should not treat development modeled estimates as immutable facts.

### Failure/monitoring risks

- party columns can change between electoral cycles as party relevance changes;
- modeled historical values may change when the model/data change;
- pollster methodology is heterogeneous and must remain attached to raw polls;
- a static GitHub path may remain reachable while its schema changes.

### Feasibility

**Ready for bounded proof.** Easy access, no authentication, no scraping, high automation potential. This remains the recommended first polling ingestion experiment.

## 2. Irish Demographic Polling Datasets

### Access tested

Repository: https://github.com/Irish-Dem-Polling/datasets

The repository currently contains separate folders for:

- `vote-intention`;
- `government-satisfaction`;
- `party-leaders`.

Government-satisfaction and leader datasets provide paired `counts` and `prop` CSV files alongside Stata/R files. Vote-intention data are separated by source pollster (Behaviour & Attitudes and RED C).

### Representative schema evidence

The government-satisfaction CSV header exposes:

- `date`, `date_start`, `date_end`, `date_middle`, `sample_size`;
- question/result field (`satisfaction_government`);
- `total`, `male`, `female`;
- age groups (`age_18_34`, `age_35_54`, `age_55`);
- social-class groups (`class_abc1`, `class_c2de`, `class_f`);
- regions (`region_dublin`, `region_leinster`, `region_munster`, `region_connacht_ulster`);
- urban/rural;
- constituency seat magnitude (`const_seats_3`, `const_seats_4`, `const_seats_5`);
- voting-intention/likelihood fields and past/future party-vote fields.

The paired counts/proportions structure is operationally useful: proportions support graphics, while counts help assess subgroup reliability.

### Update/version behaviour

The repository has hundreds of commits and is maintained as a living dataset. The public project site reports coverage through 2025 while older repository README text still states 2011–2023, proving that prose metadata can lag the files. A future loader should derive coverage from file contents, not README text.

### Recommended future design

- Pull the raw CSV files directly.
- Preserve `counts` and `prop` datasets separately or with an explicit measure type.
- Keep source pollster and survey dates attached.
- Apply minimum subgroup-base checks before any graphic is eligible for publication.

### Failure/monitoring risks

- small subgroup counts can create unstable percentages;
- file coverage can advance before README text is updated;
- pollster-specific fields/years may not be perfectly symmetric;
- repository citation terms are clear for news/research use, but no conventional open-data licence was found and the data derive from pollster reports.

### Feasibility

**Technically ready for bounded proof; legal/republication clarification recommended before production use.** No authentication or scraping is needed.

## 3. Department of Housing — official general-election count/transfer data

### Access tested

- 2016 count details: public CSV, CC BY-SA 4.0.
- 2020 count details: public CSV/JSON/XML, CC BY-SA 4.0.
- 2024 election results/transfers: public XLSX, CC BY-SA 4.0.

### 2016/2020 schema pattern

The 2016 count-detail dictionary includes fields such as:

- Constituency Name;
- Candidate surname / Candidate First Name;
- Result;
- Count Number;
- Non_Transferable;
- Occurred On Count;
- Required To Reach Quota;
- Required To Save Deposit;
- Transfers;
- Votes;
- Total Votes;
- Constituency Number;
- Candidate Id.

The 2020 resource uses the same broad candidate-count grain and exposes similar fields. Separate candidate and constituency-detail datasets are also available.

### 2024 break in delivery format

The 2024 official result is a single XLSX workbook containing election results and transfer information compiled from returning-officer notices and supplementary material. The resource is machine-downloadable but is not published as the same simple CSV/JSON/XML structure as 2016/2020.

### Recommended future design

Do **not** assume one permanent election schema. Build a proof with source-specific adapters:

- 2016 adapter;
- 2020 adapter;
- 2024 workbook adapter.

Normalize only after preserving the original source fields and election year. Candidate identity across elections should be a separate later problem, not inferred from names during initial ingestion.

### Failure/monitoring risks

- file structure changes between elections;
- candidate naming/party labels vary;
- 2024 workbook may contain multiple sheets/grains;
- constituency boundaries and identifiers change across election cycles;
- CC BY-SA obligations must be carried into any derived data distribution decisions.

### Feasibility

**Ready for bounded proof with per-election adapters.** Strong official provenance and licence, but more transformation effort than polling CSVs or referendum CSVs.

## 4. CSO PxStat / data.cso.ie

### Access tested

Official user guidance confirms PxStat is intended for automation through APIs and provides JSON-stat, CSV, PX and XLSX. Table IDs are exposed through data.gov.ie catalogue records.

Example table:

- Census 2022 table `F4061` — Population.
- JSON-stat endpoint: `https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/F4061/JSON-stat/2.0/en`
- Licence: CC BY 4.0.

Other Census 2022 tables follow the same table-ID pattern. The CSO also provides an official `csodata` R package which uses the PxStat API and can list/search current tables and metadata.

### Schema/identifier behaviour

PxStat is multidimensional rather than row-oriented CSV first. A future Python proof should expect JSON-stat dimensions, category codes/labels and a flattened value array. The table ID is the primary source identifier; dimension codes must be preserved as source metadata.

### Geography/versioning

CSO publishes multiple geographic grains and Census vintages. Geography is therefore the main correctness risk. A future political use must explicitly store:

- source table ID;
- census/reference year;
- geography type;
- geography code and label;
- retrieval date;
- relevant boundary vintage.

A constituency profile must never silently join statistics using a different boundary vintage from the election result being visualized.

### Recommended future design

Use the PxStat JSON-stat API rather than scraping or downloading spreadsheets manually. Start with one small, politically useful table and one geography level before broadening.

### Failure/monitoring risks

- choosing the wrong table or geography version;
- classification changes between statistical releases;
- table updates/revisions;
- dimensional schema is more complex than ordinary CSV;
- a title such as “Population” is not a stable identifier — table ID is.

### Feasibility

**Ready for bounded proof.** Excellent official API/licence and high automation potential. Operationally safer than election-workbook normalization, though analytically more complex.

## 5. Department of Housing — official referendum results

### Access tested

Multiple official referendum resources are public CSVs under CC BY-SA 4.0.

Examples checked:

- 2018 Thirty-sixth Amendment;
- 2015 Thirty-fifth Amendment;
- 1992 Thirteenth Amendment;
- 1986 Tenth Amendment.

### Schema consistency

The 2018 CSV dictionary exposes:

- Constituency;
- Electorate;
- Total Poll;
- Percentage Poll;
- Votes in Favour of proposal;
- Votes against proposal;
- Spoilt Votes.

The 2015 file uses the same structure with only capitalization differences. Older files use the same core concepts, although `Total Electorate` vs `Electorate` and capitalization/spelling vary.

### Update/version behaviour

Individual referendum datasets are effectively static after publication (`Update frequency: Never` on the checked 2018 resource). This makes historical ingestion operationally simple. A new referendum should be treated as a new source resource rather than an append to one master official file.

### Recommended future design

Use one direct CSV per referendum and normalize a very small controlled field set:

- referendum identifier/title/date;
- constituency;
- electorate;
- total poll;
- turnout percentage;
- votes for;
- votes against;
- spoilt/invalid votes;
- source resource ID/URL and licence.

Do not infer that constituency labels correspond to the same geography across decades; preserve referendum date and boundary context.

### Failure/monitoring risks

- capitalization/spelling differences in headers;
- older resources can expose fewer metadata fields;
- constituency boundaries/names change across decades;
- separate referendum datasets must be discovered/catalogued.

### Feasibility

**Ready for bounded proof.** This is the simplest official historical normalization target among the non-polling top five.

# Comparative recommendation

## Implementation-readiness order after exact validation

1. **Irish Polling Indicator** — first prototype because polling is the highest-priority EirePolitic use case and exact validation confirms the source is workable with explicit anomaly/version controls.
2. **Referendum results** — easiest fully official historical normalization; CKAN datastore access is cleaner than older raw CSVs.
3. **CSO PxStat** — excellent API, but select the exact editorial table first because unit/geography semantics are table-specific.
4. **General-election count/transfer data** — high value and validated, but 2024 requires a separate workbook adapter.
5. **Irish Demographic Polling Datasets** — technically ready, but obtain/record production/republication rights clarification first.

## What a later bounded proof should demonstrate

For each source, a proof should stop before production deployment and show only:

- successful fetch;
- source metadata/citation capture;
- row/dimension counts;
- schema fingerprint;
- date coverage;
- duplicate/null checks;
- one normalized sample output kept outside production;
- documented failure/monitoring rules.

No schema design or deployment is approved by this research.

# Near-next validation update

The two previously noted near-next recurring sources were subsequently validated:

| Source | Result | Main safeguard |
| --- | --- | --- |
| eTenders procurement | Ready for bounded prototype | Normalize authority/supplier entities in derived fields; account for missing awards and the 2023 publication-rule break |
| Department of Finance monthly tax receipts | Defer prototype until access is fixed/confirmed | Legacy advertised CSV endpoint fails TLS validation in automation; require a stable official machine endpoint |
| FIQ02 historical Exchequer | Ready as supplementary historical source | Do not treat as a current monthly tax-receipts replacement |

See `official_recurring_data_validation.md` for exact diagnostics.

# Evidence references

- Irish Polling Indicator: https://pollingindicator.com/
- Irish Polling Indicator method: https://pollingindicator.com/method
- IPI development data: https://github.com/Irish-Polling-Indicator/ipi-data
- Irish Demographic Polling Datasets: https://github.com/Irish-Dem-Polling/datasets
- General Election 2016 count details: https://data.gov.ie/dataset/general-election-2016-count-details
- General Election 2020 count details: https://data.gov.ie/dataset/general-election-2020-count-details
- 2024 Dáil results: https://data.gov.ie/dataset/34th-dail-general-election-29-november-2024-election-results
- CSO PxStat guide: https://www.cso.ie/en/databases/userguides/pxstatuserguide/
- Example CSO table F4061: https://data.gov.ie/dataset/f4061-population
- 2018 referendum results: https://data.gov.ie/dataset/referendum-results-on-the-thirty-sixth-amendment-of-the-constitution-bill-2018
- 2015 Thirty-fifth Amendment resource: https://data.gov.ie/dataset/referendum-on-the-thirty-fifth-amendment-of-the-constitution-bill-2015
- 1992 Thirteenth Amendment resource: https://data.gov.ie/dataset/referendum-on-the-thirteenth-amendment-of-the-constitution-bill-1992

# Living next step

The top-five read-only validation programme is complete. The next step requires a **separate implementation decision**, not more source discovery: build a bounded non-production Irish Polling Indicator prototype pinned to an exact upstream commit, with no production schema or pipeline changes.
