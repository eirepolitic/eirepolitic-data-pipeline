# Irish political data sources investigation

## Purpose

Identify additional Irish political data sources that could support EirePolitic posts, charts, infographics, public dashboards, historical comparisons, and recurring political-data features. Irish political polling is the highest priority.

This is a research-only investigation. No production architecture, schemas, pipelines, or production data were changed.

## Guardrails

- No production schema, architecture, connector, or production-data changes.
- No purchases, subscriptions, paid sign-ups, or bypassing access controls.
- Do not assume scraping is permitted. Record terms/licence evidence or uncertainty.
- Prefer primary sources. Treat secondary aggregators as discovery-only unless provenance and reuse rights are sufficiently clear.
- Do not reproduce copyrighted polling reports in full.
- Polls measure opinion during fieldwork; they are not election predictions.
- Record fieldwork and methodology where available and disclose comparability limits.
- Small read-only access checks are allowed only to confirm feasibility.
- Final merge contains documentation only.

## Evidence standard

For every polling source, investigate: organisation/source; URL; primary/secondary status; poll types; years; update frequency; geography; historical availability; crosstabs; constituency coverage; format; API/downloads; programmatic access; scraping requirement/permission; authentication/rate limits; licence/terms; attribution/commercial restrictions; published cost and pricing model; ingestion difficulty; maintenance burden; methodology transparency; recurring-use suitability; and historical-series safety.

Methodology evidence, where available: pollster; commissioner; fieldwork dates; sample size; sampled population; sampling method; weighting; mode; undecided treatment; margin of error; question wording.

Cost classes: **Free**, **Free with attribution**, **Free with registration/API key**, **Cheap paid**, **Moderate paid**, **Expensive/enterprise**, **Contact for pricing**, **Unknown**. Unpublished prices are not estimated.

For non-polling sources, investigate: contents; EirePolitic value; geographic/historical coverage; update frequency; format; API/download access; licence; cost; ingestion difficulty; maintenance burden; and concrete visualization ideas.

## Investigation plan and status

1. **Establish framework — complete.** Durable research record, index entry and documentation-only PR.
2. **Discover Irish polling sources — complete.** Research projects, pollsters, publishers, aggregators and paid APIs.
3. **Verify polling feasibility — complete.** Access, formats, history, licensing, pricing, methodology and read-only access checks.
4. **Assess historical-series safety — complete.** Mode, weighting, question wording, undecided treatment and pollster-era changes.
5. **Investigate other Irish political datasets — complete.** Official election/open-data sources and high-value opinion/research datasets.
6. **Rank and conclude — complete.** Final lists and top-five future ingestion candidates below.
7. **Read-only ingestion feasibility — complete.** All five priority candidates were validated with exact source-level checks; no production connector or schema was implemented.

## Read-only ingestion feasibility plan

### Scope

Test only the current top five future-ingestion candidates:

1. Irish Polling Indicator.
2. Irish Demographic Polling Datasets.
3. Department of Housing official general-election count/transfer data.
4. CSO PxStat/data.cso.ie.
5. Department of Housing official referendum results.

### Evidence to collect for each candidate

- exact machine-readable entry point used;
- whether access works without authentication;
- content type/format actually returned;
- stable identifiers and date fields;
- representative field/column names;
- whether historical files use one schema or multiple schemas;
- whether pagination/versioning is relevant;
- whether source updates overwrite files or append/version them;
- obvious null/duplicate/encoding issues visible in a bounded sample;
- whether attribution/licence metadata can be preserved alongside the data;
- minimum transformation likely required before analysis;
- expected failure modes and monitoring needs;
- recommended ingestion mode for a future implementation: static-file pull, API query, manual file acquisition, or not recommended;
- final feasibility rating: Ready for bounded proof / Needs clarification / Defer.

### Constraints

- Read-only web/file/API requests only.
- Do not add code, workflows, schemas, secrets, jobs, tables, buckets, or infrastructure.
- Do not create temporary repository tooling unless essential; any such artefact must not be merged.
- Do not infer permission beyond documented licence/terms.
- Keep poll methodology metadata attached to polling observations in any future design recommendation.

### Phase order

1. Irish Polling Indicator access/schema/versioning check.
2. Irish Demographic Polling Datasets access/schema/crosstab check.
3. Election count/transfer file consistency across recent elections.
4. CSO JSON-stat API shape and geography/versioning check.
5. Referendum results file consistency across multiple years.
6. Compare operational risk and recommend the order for later implementation.

## Research log

### 2026-09-03 — Phase 1: framework

- Created this research framework and evidence requirements.
- Added the investigation to `docs/research/README.md`.
- Opened documentation-only draft PR #63.
- Production changes: none.

### 2026-09-03 — Phase 2: polling-source discovery

A broad inventory was built across research datasets, primary pollsters, commissioning media, issue-poll projects, secondary aggregators, and commercial APIs.

Main discovery: the **Irish Polling Indicator** and **Irish Demographic Polling Datasets** are substantially stronger free structured candidates than scraping publisher pages. Direct pollsters remain essential primary evidence for methodology and special polls.

Production changes: none.

### 2026-09-03 — Phases 3–4: polling access, licensing, methodology and comparability

Read-only checks confirmed that both UCD-linked research projects expose direct machine-readable downloads without login. Terms/licensing checks narrowed the secondary candidates: visible results on a website are not automatically reusable data, and Ireland Votes explicitly prohibits automated extraction without permission.

Production changes: none.

### 2026-09-03 — Phase 5: non-polling political datasets

Official open-data sources provide especially strong election/count, referendum, CSO, public-finance and procurement datasets. The Electoral Commission's National Election and Democracy Study is unusually valuable research data but requires a manual email request. SIPO and the Register of Lobbying contain important transparency information but are currently less automation-friendly than the open-data sources.

Production changes: none.

### 2026-09-03 — Phase 6: ranking

Rankings and the top five future ingestion candidates were completed. No connectors or schemas were implemented.

Production changes: none.

### 2026-09-03 — Phase 7 plan: read-only ingestion feasibility

- Added a bounded feasibility plan for the existing top five candidates.
- No production files, schemas, code or infrastructure changed.
- Next: inspect the Irish Polling Indicator's concrete file access, schema and update/versioning behaviour.

### 2026-09-03 — Phase 7 complete: exact top-five validation

All five priority future-ingestion candidates were validated with source-level or pinned-file diagnostics. Dedicated durable records now exist for:

- `ipi_readonly_proof.md` and `ipi_bounded_validation.md`;
- `irish_demographic_polling_validation.md`;
- `election_count_data_validation.md`;
- `cso_pxstat_validation.md`;
- `referendum_data_validation.md`.

Key readiness findings:

- **Irish Polling Indicator:** technically ready; exact validation found two duplicate raw poll rows, historical date anomalies, and election-cycle boundary duplicate dates in the modeled series. These are manageable with provenance and validation flags.
- **Official referendum results:** technically ready and very easy to automate through the CKAN datastore; one 2015 Dublin Central 900-vote arithmetic anomaly must be preserved/flagged rather than silently corrected.
- **CSO PxStat:** technically ready; API quality is strong, but table unit/geography semantics must be validated per use case.
- **Official general-election count/transfer data:** technically ready with year-specific adapters; 2016/2020 are clean CSV-style tables while 2024 requires workbook parsing for full transfers.
- **Irish Demographic Polling Datasets:** technically ready and editorially valuable, but explicit production/commercial-republication permission should be clarified before ingestion.

Production changes: none.

# Priority 1 — Irish polling sources

## 1. Irish Polling Indicator (IPI)

**Locations**
- https://pollingindicator.com/
- https://pollingindicator.com/method
- https://github.com/Irish-Polling-Indicator/ipi-data
- Stable dataset DOI: 10.7910/DVN/8YVVYX

**Organisation/type:** UCD-hosted academic research project maintained by Stefan Müller, with Tom Louwerse as founding member. It curates published Irish Dáil voting-intention polls and produces a Bayesian polling indicator.

**Coverage:** public site states raw polling results **1982–2026** and daily aggregated estimates **1987–2026**. Development data update after new polls; stable releases are issued after an election cycle.

**Poll types/geography:** national Republic of Ireland Dáil voting intention/party support. Not a general issue-poll, leader-satisfaction, referendum or constituency-poll database.

**Formats/access:** raw polls in CSV, XLSX, Stata and R; estimates in CSV, Stata and R. No login/API key. Static GitHub files are machine-readable. No scraping required.

**Read-only check:** raw CSV was directly reachable and visibly includes poll/publication date, fieldwork start/end/midpoint, pollster, sample size and party results.

**Licence/reuse:** project requests citation/attribution and encourages use in work/articles. No conventional repository `LICENSE` file was identified, so classify **Free with attribution/citation**, not as unrestricted CC open data.

**Authentication/rate limit:** none documented for static files; normal host limits apply.

**Methodology:** combines polls from multiple established pollsters. The modeled indicator includes uncertainty ranges and is explicitly designed to contextualise trends rather than predict elections.

**Historical-series safety:** High if used correctly. Raw poll-by-poll series should retain pollster/fieldwork/sample/method metadata. The IPI modeled estimate should always be labelled as the IPI estimate. Development estimates can be revised; stable DOI releases are preferable for reproducible historical graphics.

**Cost:** Free with attribution. **Ingestion:** Easy. **Maintenance:** Low. **Automation:** High. **Value:** High.

**EirePolitic uses:** latest poll timeline; party support since an election; pollster comparison; IPI estimate with uncertainty band; same-point-in-cycle comparisons.

## 2. Irish Demographic Polling Datasets

**Locations**
- https://pollingindicator.com/ (related-project section)
- https://github.com/Irish-Dem-Polling/datasets

**Organisation/type:** UCD-linked research dataset assembled from published RED C and Behaviour & Attitudes reports.

**Coverage:** project site states 100+ polls published **2011–2025**. Older repository text referred to 2011–2023, so file-level dates should be checked in any later ingestion test.

**Poll types:** vote intention; government satisfaction; party-leader satisfaction. Includes all respondents and demographic/geographic subsamples such as age, gender, social class, region and district magnitude. Not a general constituency-poll database.

**Formats/access:** public CSV, Stata and R files; dashboard/subset downloads. No login. No scraping needed.

**Read-only check:** B&A and RED C vote-intention files plus government- and leader-satisfaction files were directly reachable without login.

**Licence/reuse:** project asks news/academic users to cite the dataset authors and asks users of individual survey results to cite/reference original pollster reports. No standard repository open-data licence was identified. Because source tables derive from pollster reports, legal confidence is **Medium**, especially for detailed commercial republication.

**Methodology:** weighted proportions/counts are retained where available. Underlying B&A and RED C methods and question wording can vary by wave.

**Historical-series safety:** Medium–High if used as pollster-specific series with methodology notes; not one unchanged instrument across all years.

**Cost:** Free with attribution. **Ingestion:** Easy. **Maintenance:** Low–Medium. **Automation:** High. **Value:** High.

**EirePolitic uses:** party support by age/region/gender/social class; government satisfaction by demographic; leader-satisfaction trends; subgroup gaps.

## 3. RED C Research / Business Post

**Locations**
- https://redcresearch.com/
- https://redcresearch.com/our-omnibus/

**Coverage:** recurring national voting intention plus issue, satisfaction and demographic questions depending on wave; public political-poll archive.

**Access:** public articles/reports, not a documented free structured polling API/feed. Any automated extraction from pages would require separate permission/terms review; this investigation does not approve scraping.

**Methodology:** current work uses RED C Live online polling/panel controls. Historical methodology differs materially; older polling used telephone sampling and, in some eras, different treatment of undecideds/likely voters. A long RED C series therefore requires method-era metadata.

**Published commissioned price:** Irish omnibus lists **€625 + VAT/question for data tables** and **€795 + VAT/question for full service**. This is bespoke research, not a historical feed.

**Cost:** free releases to view; commissioned research **Moderate paid / bespoke**. **Automation:** Low–Medium without a supplied feed. **Value:** High as a primary methodology/source reference.

## 4. Ireland Thinks / Sunday Independent / The Evidence

**Locations**
- https://www.irelandthinks.ie/
- https://analysis.irelandthinks.ie/
- https://analysis.irelandthinks.ie/services/

**Coverage:** recurring national voting intention; issue, election, referendum, presidential, European/local polling; demographic analysis on some projects.

**Access:** public analyses exist; no unrestricted public historical API/feed confirmed. The Evidence has data-portal/sign-in elements; no paid sign-up was attempted.

**Methodology:** project-specific online, face-to-face and telephone methods are documented. A specific 2024 election-day project said its data were free to use with attribution, but that permission cannot safely be generalized to all monthly polls.

**Published commissioned prices:** monthly omnibus **€800 + VAT** first question and **€450 + VAT** subsequent questions; rapid polling **€2,000 + VAT** first question and **€900 + VAT** additional questions; video-poll **€1,200 + VAT/question**.

**Cost:** free public releases; commissioned polling **Moderate paid/bespoke**; data-portal pricing Unknown. **Automation:** Low–Medium unless structured access is agreed.

## 5. Ipsos B&A / The Irish Times

**Locations**
- https://www.ipsosbanda.ie/news-polls/
- https://www.ipsosbanda.ie/research-approaches/omnibus/

**Coverage:** national party support, government/leader satisfaction, issues and referendums; occasional constituency/byelection and European-election polling.

**Access:** public pages/reports; no public polling API or structured historical feed confirmed. Omnibus available by contact; no public price found.

**Reuse evidence:** Irish Times poll methodology notes state that extracts may be quoted/published with acknowledgement to The Irish Times and Ipsos B&A. This does not imply a right to bulk-republish complete reports/crosstabs.

**Methodology:** recent national Irish Times polling has used representative in-home samples; individual projects may use telephone. Historical MRBI/Ipsos/B&A lineage and changes to voting-likelihood/party-support treatment make long trends method-sensitive.

**Cost:** public releases free to view; commissioned polling **Contact for pricing**. **Automation:** Low–Medium.

## 6. Opinions / The Sunday Times

**Locations**
- https://opinions.ie/
- https://opinions.ie/omnibus/

National voting-intention polling is confirmed in the 2024 election cycle; RTÉ's poll-of-polls methodology identified Opinions political polling as online. No clean public historical structured archive/API was found. Monthly omnibus is offered but no public price was identified.

**Cost:** free releases to view; commissioned research **Contact for pricing**. **Automation:** Low.

## 7. Amárach Research

**Locations**
- https://amarach.com/
- https://amarach.com/amarach-omnibus-survey.html

Useful for issue, referendum, constitutional, EU and public-policy surveys. The omnibus is recurring, but no general historical political-poll API/feed was identified and no sufficiently clear current reusable-data price was confirmed.

**Cost:** **Contact/unclear pricing**. **Automation:** Low unless commissioners publish structured results.

## 8. European Movement Ireland — Ireland and the EU Poll

**Location:** https://www.europeanmovement.ie/em-ireland-eu-poll/

Annual series since 2013, currently conducted by Amárach, covering attitudes to the EU in the Republic and Northern Ireland. Downloadable findings exist; raw reusable microdata/API were not identified.

**Cost:** Free to view/download reports. **Licence confidence:** Medium/Low for bulk data reuse because no explicit structured-data licence was found. **Automation:** Low–Medium. **Value:** Medium–High for annual EU-attitude graphics.

## 9. ARINS / Irish Times constitutional-future surveys

Repeated all-island constitutional-attitude surveys are valuable for United-Ireland/constitutional comparisons. Treat as project/report sources until source-by-source reuse rights and structured downloads are confirmed.

**Cost:** generally public findings. **Automation:** Low. **Value:** Medium–High for specific features.

## 10. IrelandElection.com

**Location:** https://irelandelection.com/opinion_polls_party.php

Searchable HTML tables expose pollster, commissioner, sample/margin and party values. Useful for discovery/cross-checking, but no clear reuse/scraping permission was established.

**Decision:** Discovery-only unless permission is obtained.

## 11. Ireland Votes

**Locations**
- https://www.irelandvotes.com/polling/ireland/dail
- https://www.irelandvotes.com/about/terms

Long poll series, aggregate and government-approval tracker. Published terms expressly prohibit scraping, bulk downloading, automated extraction and commercial exploitation without permission.

**Decision:** Rejected as an automated ingestion source unless explicit permission/licensing is obtained.

## 12. IrishPolitics.ie tracker

**Location:** https://www.irishpolitics.ie/tracker

Secondary tracker using established pollsters and exposing a CSV control in the interface. No sufficiently clear reuse licence/provenance terms were established.

**Decision:** Discovery/cross-checking only pending permission.

## 13. Wikipedia Irish polling tables

Useful discovery/index source for national, referendum and constituency polls with source links. Not preferred over IPI/primary sources for a recurring production series.

## 14. Europe Elects polling database

**Location:** https://europeelects.eu/data-access-request/

Ireland covered. Published **€15 per country** CSV access is for academic researchers/students for **non-commercial** work and includes no-redistribution/resale restrictions. Commercial users must contact Europe Elects.

**Decision:** the €15 tier is **not** a cheap commercial option for EirePolitic. Commercial access is **Contact for pricing**.

## 15. PolitPro API

**Location:** https://politpro.eu/en/api

**Access:** Ireland (`IE`) supported. JSON endpoints expose current trend, historical trend and latest raw voting-intention polls. Fields include fieldwork start/end, sample size, institute and party values. Webhooks supported.

**Authentication/rate limit:** Bearer token after registration; **30 requests/minute** documented.

**Use rights:** API page permits editorial use, use in own products and for customers, but not a competing polling-data platform.

**History:** historical trend data may be revised retroactively; consumers are told to re-fetch rather than assume immutability.

**Pricing:** no public API price identified. **Contact for pricing**. No account was created.

**Value:** potentially strong paid operational feed, but not preferable to free IPI until price/licence terms are known.

## 16. PoliticalAPI polling endpoint

**Location:** https://politicalapi.com/political-polls-api

Normalized REST polling with Ireland coverage. Provider states polling data are normalized from publicly available Wikipedia polling tables, making it a convenience layer rather than a primary source.

**Published ongoing prices:** Starter **€89/month** after introductory first month; Professional **€199/month**; Business **€599/month**. Introductory launch prices are not treated as ongoing cost.

**Cost:** Starter **Moderate paid**; higher tiers expensive relative to free Irish polling alternatives. **Automation:** High. **Provenance value:** Medium/Low for polling because underlying source is secondary.

## 17. PollBase / PollBase Pro

British/Great Britain focused, not a Republic of Ireland polling source.

**Decision:** Rejected as out of scope.

# Polling methodology rules for any future EirePolitic series

1. Store pollster, commissioner, fieldwork start/end, publication date, sample size and mode with every poll where available.
2. Keep sampling/weighting notes and a method-version field when a pollster changes method.
3. Do not calculate naive long-run averages across online, phone and face-to-face polls without a documented method.
4. Do not mix raw individual polls and aggregator model estimates without labeling the difference.
5. Preserve undecided/non-voter treatment; historical examples show this changes over time.
6. Prefer pollster-specific comparisons. Cross-pollster comparisons need context.
7. Keep constituency polls separate from national polls.
8. Treat issue/referendum/presidential questions as question-specific series; wording matters.
9. Retain subgroup base sizes for crosstabs where available and avoid false precision for small subgroups.
10. Every public poll graphic should state that polling measures opinion during fieldwork and is not an election prediction.

# Polling access tests

No connector was created.

| Source | Read-only check | Result |
| --- | --- | --- |
| Irish Polling Indicator | Raw development CSV | Direct structured file reachable; poll dates, fieldwork, pollster, sample and party fields visible |
| Irish Polling Indicator | Estimates | Direct CSV available; stable DOI dataset documented |
| Irish Demographic Polling Datasets | Vote-intention files | Direct GitHub CSVs reachable without login |
| Irish Demographic Polling Datasets | Government/leader satisfaction | Direct GitHub CSVs reachable without login |
| PolitPro | API documentation | Ireland supported; REST endpoints, Bearer auth, 30 req/minute and webhooks documented |
| Ireland Votes | Terms | Automated extraction prohibited without permission; no scraping attempted |

# Priority 2 — other Irish political datasets

## A. Department of Housing Open Data — general election results and count transfers

**Locations**
- https://opendata.housing.gov.ie/
- https://data.gov.ie/dataset/general-election-2020-count-details
- https://data.gov.ie/dataset/34th-dail-general-election-29-november-2024-election-results

**Contains:** official candidate, constituency and count/transfer results. The 2020 count-details dataset exposes candidate/count rows; the 2024 workbook states that election results and vote transfers were compiled from returning-officer notices/supplementary information. The portal also contains older general-election statistics/candidate/first-preference datasets.

**Coverage:** historical datasets vary by election; open-data catalogue includes 2002, 2007, 2011, 2016, 2020 and 2024 material, with different granularity by year.

**Formats:** 2020 count data in CSV/JSON/XML; 2024 detailed results in XLSX; older datasets commonly CSV.

**Programmatic access:** direct downloads plus CKAN catalogue API. No authentication.

**Licence:** election datasets checked are **CC BY-SA 4.0**. Free with attribution/share-alike.

**Difficulty:** Easy–Moderate due schema differences between election years. **Maintenance:** Low between elections. **Automation:** High for new structured releases.

**EirePolitic uses:** transfer waterfalls; “where votes moved”; first-preference vs final-seat charts; constituency turnout/quota; candidate histories; transfer efficiency; historical election comparisons.

## B. Department of Housing Open Data — referendum results

**Locations**
- https://opendata.housing.gov.ie/dataset/?tags=referendum
- example: https://data.gov.ie/dataset/referendum-results-on-the-thirty-sixth-amendment-of-the-constitution-bill-2018

**Contains:** official referendum result files, typically electorate, poll/turnout, yes/no, invalid/spoilt and constituency fields.

**Coverage:** portal search exposes dozens of referendum datasets, including historical amendments from the 1950s onward; the 2018 Thirty-sixth Amendment file is a clear modern example.

**Formats/access:** CSV; direct download and CKAN catalogue. No authentication.

**Licence:** checked modern result file is **CC BY-SA 4.0**.

**Difficulty:** Easy–Moderate because historical schemas/names may vary. **Maintenance:** Low. **Automation:** High for structured releases.

**EirePolitic uses:** referendum maps; turnout vs result; strongest/weakest constituencies; historical amendment comparisons; poll-versus-result retrospective graphics.

## C. Central Statistics Office — PxStat / data.cso.ie

**Locations**
- https://data.cso.ie/
- example JSON-stat endpoint pattern: https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/{TABLE}/JSON-stat/2.0/en
- https://data.gov.ie/organization/central-statistics-office

**Contains:** official statistics across population/demographics, housing, health, education, labour, migration, crime-related series, prices, income, local geography and many policy-relevant subjects.

**Coverage/update:** extensive historical series; update frequency depends on table (census, monthly, quarterly, annual, etc.).

**Formats/access:** JSON-stat REST endpoints plus CSV, PX and XLSX for many tables. No per-dataset scraping needed.

**Licence:** CSO/data.gov datasets checked are **CC BY 4.0**, allowing commercial reuse with attribution.

**Difficulty:** Moderate because table dimensions/codes must be understood. **Maintenance:** Low–Medium. **Automation:** High.

**EirePolitic uses:** constituency/area demographic profiles; housing affordability/supply trends; migration and population; health/education capacity; labour-market comparisons; policy-claim context; per-capita regional dashboards.

**Caution:** political interpretation should distinguish official statistical change from policy causation. Geography versions must match the election/constituency boundary vintage used in a graphic.

## D. Department of Finance — monthly Exchequer tax receipts / Exchequer history

**Locations**
- https://data.gov.ie/dataset/monthly-exchequer-tax-receipts-1984-present
- https://data.gov.ie/dataset/fiq02-exchequer-account-historical-series

**Contains:** monthly tax receipts by tax category from **1984-present**; separate historical Exchequer account series.

**Update:** monthly tax receipts are published on the second working day of the following month, with quarterly press context.

**Formats:** monthly tax receipts CSV; FIQ02 CSV/JSON-stat/PX/XLSX with a PxStat REST endpoint.

**Licence:** **CC BY 4.0**.

**Cost:** Free with attribution. **Difficulty:** Easy. **Maintenance:** Low. **Automation:** High.

**EirePolitic uses:** monthly tax-revenue scorecard; income/corporation/VAT mix; year-on-year revenue change; pre/post-budget trends; long-run revenue composition.

## E. Office of Government Procurement — eTenders open data

**Location:** https://data.gov.ie/dataset/contract-notices-published-on-etenders

**Contains:** tender notices, contract awards, procedures, suppliers and public bodies. Current open dataset covers competitions from **1 January 2013** onward; as of the review the portal described coverage through Q2 2026 and quarterly updates.

**Formats/access:** direct CSV; no authentication.

**Licence:** **CC BY 4.0**.

**Cost:** Free with attribution. **Difficulty:** Easy–Moderate due cleaning/entity normalization. **Maintenance:** Low–Medium. **Automation:** High.

**EirePolitic uses:** largest awards by department/public body; supplier concentration; procurement by sector; awards over time; geographic/company comparisons; recurring quarterly procurement dashboard.

**Caution:** publication requirements/completeness changed over time; comparisons must account for coverage rules and records entered outside the platform.

## F. Electoral Commission — National Election and Democracy Study (NEDS)

**Location:** https://www.electoralcommission.ie/what-we-do/national-election-and-democracy-study/

**Contains:** post-election/referendum voter studies, attitudes, participation, vote/non-vote behaviour and democratic engagement. General Election 2024 study used a probabilistic face-to-face sample of **1,426** plus an online complement of **1,421**. 2024 local/European, Limerick mayoral and Family/Care referendum studies also exist.

**Access:** questionnaires and slide decks public; **raw datasets and codebooks can be freely requested by email** from the Electoral Commission. No purchase is required, but access is manual rather than automated. No request was sent during this investigation.

**Format:** raw format not confirmed without requesting files. **API:** none identified.

**Licence/reuse:** public page says raw data/codebooks can be freely requested; detailed reuse licence should be confirmed when obtaining files.

**Difficulty:** Moderate. **Maintenance:** Medium because each study may require manual acquisition and schema review. **Automation:** Low–Medium. **Value:** High.

**EirePolitic uses:** who voted/non-voted; turnout motivations; political trust; campaign information sources; referendum voter reasoning; demographic participation gaps.

## G. European Social Survey (ESS)

**Locations**
- https://www.europeansocialsurvey.org/data-portal
- https://ess.sikt.no/en/api
- https://www.europeansocialsurvey.org/contact/disclaimer

**Contains:** repeated cross-European social/political attitudes including trust in politicians/institutions, democracy/government, immigration, political participation/affiliation, identity, discrimination, welfare, health and values. Ireland participates in multiple rounds; Round 11 (2023/24) includes Ireland.

**Access:** Data Portal plus beta API. API file downloads require an ESS user ID after registration; the ID is for usage statistics rather than request authentication. No registration was performed here.

**Licence:** **CC BY-NC-SA 4.0** for ESS data. This is **non-commercial**, so EirePolitic commercial/public-product reuse must be assessed carefully and may require permission or use limited to appropriately licensed derived reporting.

**Cost:** Free for permitted use. **Difficulty:** Moderate. **Automation:** High technically, but licence confidence for EirePolitic commercial use is Low/Medium.

**EirePolitic uses:** Ireland-vs-Europe political trust, immigration attitudes, democratic satisfaction, participation and values — if licensing is suitable.

## H. Eurobarometer / GESIS Eurobarometer data service

**Locations**
- https://europa.eu/eurobarometer/
- https://www.gesis.org/en/eurobarometer-data-service/data-and-documentation
- https://data.europa.eu/

**Contains:** Standard, Special and Flash Eurobarometer surveys including Irish samples and many political/public-policy topics. Some Ireland-specific referendum surveys exist; current waves include Ireland national reports.

**Access:** European Commission/data.europa.eu often exposes Excel/ZIP result tables; GESIS provides free microdata downloads (SPSS/Stata) after user registration. No account was created.

**Licence/terms:** citation of primary data is required by GESIS; questionnaires have separate copyright conditions. Exact reuse conditions should be checked per dataset/output rather than assuming a blanket commercial open licence.

**Cost:** Free. **Difficulty:** Moderate. **Maintenance:** Medium. **Automation:** Medium because file structures and release formats vary.

**EirePolitic uses:** Ireland-vs-EU sentiment; EU trust; issue salience; referendum retrospectives; long-run EU-attitude comparisons.

## I. SIPO — political donations, party accounts and election expenses

**Locations**
- https://data.sipo.ie/
- https://data.sipo.ie/en/collection/5b104-election-reports/
- https://www.sipo.ie/en/collection/76651-annual-disclosures/

**Contains:** election spending by candidates/parties, candidate expense reimbursement, donations, third-party activity, annual party donation statements, Exchequer funding and party statements of accounts.

**Coverage:** election-report collections include current and historical Dáil/Seanad/European/Presidential material; earlier reports may require contacting SIPO.

**Format/access:** primarily reports, statements and PDFs/collections rather than a clean machine-readable consolidated API/CSV. Public and free to view.

**Licence:** no general structured-data licence was confirmed for the report contents. **Cost:** Free. **Difficulty:** Hard for reliable automation. **Maintenance:** High if manually extracting reports. **Automation:** Low.

**EirePolitic uses:** candidate spending league tables; party finance trends; donations; state funding; election spending vs electoral outcome.

**Recommendation:** high editorial value, but pursue only after easier structured sources unless SIPO exposes or agrees to structured data.

## J. Register of Lobbying

**Locations**
- https://www.lobbying.ie/
- https://www.lobbying.ie/help-resources/information-for-lobbyists/guidelines-for-people-carrying-on-lobbying-activities/register-of-lobbying/

**Contains:** public registrations and lobbying returns describing who lobbied whom, about what, intended result and relevant designated public officials. Returns are made for four-month reporting periods; register is free to inspect.

**Coverage:** system operates from 2015 onward.

**Access:** public searchable register. No official documented public REST API or clearly licensed CSV export was confirmed in this investigation. An unofficial analytics site indicates periodic source CSV exports exist, but that is not sufficient evidence to recommend production ingestion.

**Licence/scraping:** no explicit automated-reuse licence/permission confirmed. **Cost:** Free to inspect. **Difficulty:** Hard until official bulk access/terms are clarified. **Maintenance:** High if dependent on HTML extraction. **Automation:** Low/Unknown.

**EirePolitic uses:** most-lobbied policy areas; organisations contacting ministers/departments; lobbying trends by reporting period; issue networks.

**Decision:** high-value future source, but request/confirm official bulk-data access before building anything.

## K. Tailte Éireann / GeoHive electoral boundaries

**Locations**
- https://data.gov.ie/ (Tailte Éireann boundary datasets)
- https://www.geohive.ie/

**Contains:** Dáil constituency, Local Electoral Area and Electoral Division boundaries in map-ready formats; current/historical boundary products are available through Tailte Éireann/GeoHive catalogues.

**Formats/access:** GeoJSON, CSV, KML, shapefile and ArcGIS-style direct downloads depending dataset. No login for checked resources.

**Licence:** metadata was inconsistent across checked catalogue records: some 2026 LEA resources are CC BY 4.0 while some constituency/ED records displayed “No licence specified.” Therefore **do not generalize one licence to every boundary resource**; verify the specific chosen distribution before ingestion.

**Cost:** Free. **Difficulty:** Easy–Moderate. **Maintenance:** Low. **Automation:** High. **Legal confidence:** Medium until a specifically CC-licensed constituency file is selected.

**EirePolitic uses:** constituency/LEA maps; joining election results to geography; constituency demographic maps; boundary-change explainers.

## L. data.gov.ie catalogue/API itself

**Locations**
- https://data.gov.ie/
- https://data.gov.ie/pages/opendatalicence

The national portal is CKAN-based and exposes catalogue APIs, making it valuable as a discovery/metadata layer for official datasets. The Irish open-data framework recommends **CC BY 4.0**, which permits commercial reuse with attribution, but the licence must still be checked on the exact dataset/resource because some catalogue records are inconsistent or use other licences such as CC BY-SA.

**Use:** automated discovery/monitoring of new official datasets rather than a single political dataset.

# Non-polling sources considered but not prioritized

- **Generic official datasets with no clear political visualization:** not retained merely because they exist. The CSO/data.gov catalogues are enormous; selection should be driven by a recurring editorial question.
- **Unstructured FOI disclosure/publication logs:** no consistent national machine-readable source was found; agency-by-agency extraction would create high maintenance.
- **Media/political-appearance datasets:** no sufficiently authoritative, reusable national dataset was identified in this pass.
- **SIPO/lobbying as immediate ingestion:** editorially valuable, but structured bulk access/licensing is weaker than the top official open-data sources.
- **Tailte constituency boundary resources with “No licence specified”:** do not ingest that exact distribution until a clear licence is verified, even though equivalent/current boundary products may exist elsewhere.

# Final rankings

## 1. Best free polling sources

| Rank | Source | Value | Access | Cost | Automation | Licensing confidence | Why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Irish Polling Indicator | High | Easy | Free with attribution | High | High/Medium | Long national series, direct structured files, fieldwork/sample metadata, stable releases |
| 2 | Irish Demographic Polling Datasets | High | Easy | Free with attribution | High | Medium | Unique demographic, government and leader-satisfaction crosstabs in CSV |
| 3 | Primary pollster public releases (RED C / Ireland Thinks / Ipsos B&A) | High | Moderate | Free to view | Low–Medium | Medium | Best methodology/source evidence and special polls, but no general reusable feed confirmed |
| 4 | European Movement Ireland EU Poll | Medium | Moderate | Free | Low–Medium | Medium/Low | Repeated annual EU-attitude series, but mainly report downloads |

## 2. Best cheap paid polling sources

**No clearly cheap commercial polling dataset/feed was confirmed.**

| Source | Status | Price evidence | Conclusion |
| --- | --- | --- | --- |
| Europe Elects | Non-commercial tier only | €15/country for academic/non-commercial users | Not a valid EirePolitic commercial-price candidate; commercial terms contact-only |
| PolitPro API | Commercial/editorial API | Public price not found | Strong candidate if quote is reasonable; Contact for pricing |
| PoliticalAPI | Commercial API | €89/month Starter after intro; €199 Pro; €599 Business | Technically easy but Moderate/Expensive relative to free IPI and polling provenance is secondary |
| RED C omnibus | Bespoke research | €625 + VAT/question data tables; €795 + VAT/question full service | Useful for commissioning original questions, not a data-feed substitute |
| Ireland Thinks omnibus | Bespoke research | €800 + VAT first question; €450 + VAT subsequent | Useful for commissioning original questions, not a historical feed |

## 3. Best non-polling political datasets

| Rank | Source | Value | Access | Cost | Automation | Licensing confidence |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Official general-election count/transfer data (DHLGH) | High | Easy–Moderate | Free | High | High |
| 2 | CSO PxStat / data.cso.ie | High | Moderate | Free | High | High |
| 3 | Official referendum result datasets (DHLGH) | High | Easy–Moderate | Free | High | High |
| 4 | eTenders procurement open data | High | Easy–Moderate | Free | High | High |
| 5 | Department of Finance tax/Exchequer data | High | Easy | Free | High | High |
| 6 | Electoral Commission NEDS | High | Moderate/manual request | Free | Low–Medium | Medium |
| 7 | Eurobarometer | Medium–High | Moderate | Free | Medium | Medium |
| 8 | Tailte/GeoHive electoral geography | High for maps | Easy–Moderate | Free | High | Medium because specific resource licences vary |
| 9 | SIPO political finance/expenses | High editorially | Hard | Free | Low | Medium/Low for structured reuse |
| 10 | Register of Lobbying | High editorially | Hard | Free | Low/Unknown | Low/Medium until bulk-access terms are clarified |
| 11 | European Social Survey | Medium–High | Moderate | Free for permitted use | High technically | Low/Medium for EirePolitic because CC BY-NC-SA is non-commercial |

## 4. High-value sources currently inaccessible, restricted or potentially too expensive

- **PolitPro API** — technically excellent; price is contact-only.
- **Europe Elects commercial polling access** — academic €15 tier is non-commercial; commercial price/permission required.
- **Ireland Votes bulk data** — useful series, but terms prohibit scraping/automated extraction without permission.
- **SIPO structured finance/expense data** — valuable information exists, but no clean consolidated machine-readable feed was verified.
- **Register of Lobbying bulk data** — high public-interest value; official bulk/API/licence path needs confirmation before use.
- **ESS for commercial EirePolitic use** — data licence is CC BY-NC-SA 4.0; commercial use is not safely assumed.

## 5. Sources investigated and rejected / discovery-only

- **PollBase / PollBase Pro** — UK/GB, out of scope.
- **Ireland Votes as an automated source** — prohibited by published terms without permission.
- **IrelandElection.com as a production source** — no sufficiently clear reuse/scraping licence; discovery only.
- **IrishPolitics.ie tracker as a production source** — licensing/provenance not sufficiently clear; discovery only.
- **Wikipedia as preferred polling source** — useful index, but better Irish curated/primary sources exist.
- **Europe Elects €15 tier as EirePolitic paid option** — non-commercial restriction excludes ordinary commercial use.
- **PoliticalAPI for polling alone** — not legally rejected, but poor value/provenance relative to free IPI because polling normalization is based on Wikipedia tables.
- **Unlicensed Tailte boundary resource variants** — do not use until a clear licence is attached to the exact chosen resource.

# Validated top-five readiness and future prototype order

No production implementation is authorized by this research task. Exact validation is now complete for all five. Recommended order for any later **non-production prototype** is:

1. **Irish Polling Indicator** — first priority because polling is the core EirePolitic goal and the source is free, structured and highly automatable. Prototype must pin the upstream commit, preserve raw/model distinction and validate duplicates/date anomalies.
2. **Official referendum results** — simplest official political-history source. Prefer CKAN datastore records, preserve boundary vintages and flag the documented 2015 arithmetic anomaly.
3. **CSO PxStat** — strong official API. Select one concrete politically useful table whose unit and geography match the intended feature before prototyping.
4. **Official general-election count/transfer data** — high value, but use source-specific adapters: 2016/2020 CSV-style and 2024 workbook parsing.
5. **Irish Demographic Polling Datasets** — technically strong and uniquely useful for crosstabs/satisfaction, but obtain/record clear recurring production/republication permission first.

**Near-next choices:** eTenders procurement and Department of Finance monthly tax receipts remain strong, openly licensed recurring-data candidates after the top-five programme.

# Example recurring EirePolitic feature ideas

- **Poll Tracker:** raw poll timeline + IPI modeled estimate and uncertainty, with fieldwork/method labels.
- **Who is moving?** demographic party-support changes from the UCD demographic datasets.
- **Leader/Government satisfaction:** pollster-specific trend cards rather than mixing unlike methods.
- **Count-by-count:** election transfer flows and elimination/election milestones by constituency.
- **Referendum map archive:** constituency results and turnout across historical referendums.
- **Constituency profile:** join election results to CSO demographic indicators and licensed boundary geography.
- **State receipts monthly:** tax category year-on-year scorecard two working days after month end.
- **Who gets public contracts?** quarterly procurement awards by public body/supplier/category with coverage caveats.
- **Why people voted:** NEDS-based election/referendum participation and motivation graphics after each study release.
- **Ireland vs Europe:** Eurobarometer comparisons on trust, EU attitudes and major issues.

# Licensing/attribution notes

- Irish open-data datasets explicitly marked **CC BY 4.0** can be reused commercially with attribution. Recommended national attribution wording is available at https://data.gov.ie/pages/opendatalicence.
- **CC BY-SA 4.0** election/referendum datasets also carry share-alike obligations; a future implementation should review how those obligations apply to derived datasets/products.
- Do not infer a licence from “publicly visible”. Pollster reports, SIPO reports, lobbying pages and secondary trackers require their own rights analysis.
- Polling dataset citation should identify the research dataset and, where required, the underlying pollster/source report.
- For Tailte/GeoHive, verify the licence of the exact boundary resource selected because catalogue records checked were inconsistent.

# Source/reference register

## Polling
- https://pollingindicator.com/
- https://pollingindicator.com/method
- https://github.com/Irish-Polling-Indicator/ipi-data
- https://github.com/Irish-Dem-Polling/datasets
- https://redcresearch.com/
- https://redcresearch.com/our-omnibus/
- https://www.irelandthinks.ie/
- https://analysis.irelandthinks.ie/
- https://analysis.irelandthinks.ie/services/
- https://www.ipsosbanda.ie/news-polls/
- https://www.ipsosbanda.ie/research-approaches/omnibus/
- https://opinions.ie/
- https://opinions.ie/omnibus/
- https://amarach.com/
- https://www.europeanmovement.ie/em-ireland-eu-poll/
- https://irelandelection.com/opinion_polls_party.php
- https://www.irelandvotes.com/polling/ireland/dail
- https://www.irelandvotes.com/about/terms
- https://www.irishpolitics.ie/tracker
- https://europeelects.eu/data-access-request/
- https://politpro.eu/en/api
- https://politicalapi.com/political-polls-api
- https://en.wikipedia.org/wiki/Opinion_polling_for_the_2024_Irish_general_election

## Official/open political data
- https://data.gov.ie/
- https://data.gov.ie/pages/opendatalicence
- https://opendata.housing.gov.ie/
- https://data.gov.ie/dataset/general-election-2020-count-details
- https://data.gov.ie/dataset/34th-dail-general-election-29-november-2024-election-results
- https://opendata.housing.gov.ie/dataset/?tags=referendum
- https://data.gov.ie/dataset/referendum-results-on-the-thirty-sixth-amendment-of-the-constitution-bill-2018
- https://data.cso.ie/
- https://data.gov.ie/dataset/monthly-exchequer-tax-receipts-1984-present
- https://data.gov.ie/dataset/fiq02-exchequer-account-historical-series
- https://data.gov.ie/dataset/contract-notices-published-on-etenders
- https://www.electoralcommission.ie/what-we-do/national-election-and-democracy-study/
- https://data.sipo.ie/
- https://data.sipo.ie/en/collection/5b104-election-reports/
- https://www.sipo.ie/en/collection/76651-annual-disclosures/
- https://www.lobbying.ie/
- https://www.geohive.ie/
- https://www.europeansocialsurvey.org/data-portal
- https://ess.sikt.no/en/api
- https://www.europeansocialsurvey.org/contact/disclaimer
- https://europa.eu/eurobarometer/
- https://www.gesis.org/en/eurobarometer-data-service/data-and-documentation
- https://data.europa.eu/

# Living next-step plan

1. **Completed:** initial source discovery, licensing, methodology, pricing and rankings.
2. **Completed:** exact read-only validation of the Irish Polling Indicator.
3. **Completed:** exact read-only validation of the Irish Demographic Polling Datasets.
4. **Completed:** exact read-only validation of official 2016/2020/2024 election count/transfer data.
5. **Completed:** exact API validation of CSO PxStat using table F4061, including unit/geography safeguards.
6. **Completed:** exact cross-year validation of official referendum results for 1986, 1992, 2015 and 2018.
7. **Completed:** implementation-readiness ranking and durable documentation.
8. **Next only under a separate implementation decision:** build a bounded non-production prototype for the Irish Polling Indicator, followed by referendum data and one deliberately selected CSO political-use table. Do not change production schemas or pipelines as part of this research record.
