# Irish political data sources investigation

## Purpose

Identify additional Irish political data sources that could support EirePolitic posts, charts, infographics, public dashboards, historical comparisons, and recurring political-data features. Irish political polling is the highest priority.

This is a research-only investigation. No production architecture, schemas, pipelines, or production data may be changed.

## Guardrails

- No production schema, architecture, connector, or production-data changes.
- No purchases, subscriptions, paid sign-ups, or bypassing access controls.
- Do not assume scraping is permitted. Record terms/licence evidence or uncertainty.
- Prefer primary sources. Treat secondary aggregators as discovery-only unless provenance and reuse rights are sufficiently clear.
- Do not reproduce copyrighted polling reports in full.
- Polls are measurements of opinion at a point in time, not election predictions.
- Record fieldwork and methodology where available and disclose comparability limits.
- Small read-only access checks are allowed only to confirm feasibility.
- Final merge must contain documentation changes only.

## Evidence standard

For every polling source, collect: organisation/source; URL; primary/secondary status; poll types; years; update frequency; geography; history; crosstabs; constituency coverage; format; API/downloads; programmatic access; scraping requirement/permission; authentication/rate limits; licence/terms; attribution/commercial restrictions; published cost and pricing model; ingestion difficulty; maintenance burden; methodology transparency; recurring-use suitability; and historical-series safety.

Methodology fields, where available: pollster; commissioner; fieldwork dates; sample size; sampled population; sampling method; weighting; mode; undecided treatment; margin of error; and question wording.

Cost classes: **Free**, **Free with attribution**, **Free with registration/API key**, **Cheap paid**, **Moderate paid**, **Expensive/enterprise**, **Contact for pricing**, **Unknown**. Never infer unpublished pricing.

For non-polling sources, collect: contents; EirePolitic use; geographic/historical coverage; update frequency; format; API/download access; licence; cost; ingestion difficulty; maintenance burden; and concrete visualization ideas.

## Investigation plan

1. **Establish framework** — create this durable record, add it to the research index, and work on a documentation-only PR.
2. **Discover Irish polling sources** — pollsters, commissioners, research projects, archives, open-data portals, aggregators and commercial feeds.
3. **Verify polling feasibility** — access, formats, history, licensing, pricing, authentication, methodology and small read-only access tests.
4. **Assess historical-series safety** — methodology changes, mode, weighting, question wording, undecided treatment and whether data can be compared safely over time.
5. **Investigate other Irish political datasets** — only datasets with credible recurring public-interest visualization value.
6. **Rank and conclude** — best free polling, best cheap paid polling, best non-polling, high-value inaccessible/expensive, rejected sources, and top five future ingestion candidates.

## Research log

### 2026-09-03 — Phase 1: framework

- Created this research framework and evidence requirements.
- Added the investigation to `docs/research/README.md`.
- Opened documentation-only draft PR #63.
- Production changes: none.

### 2026-09-03 — Phase 2: polling-source discovery

A broad inventory was built across research datasets, primary pollsters, commissioning media, issue-poll projects, secondary aggregators, and commercial APIs.

Strong structured/free leads were the **Irish Polling Indicator** and **Irish Demographic Polling Datasets**. Direct pollsters (RED C, Ireland Thinks, Ipsos B&A, Opinions and Amárach) are valuable primary evidence but do not generally expose a clean reusable public feed. Several secondary/commercial aggregators were also found and retained only where their terms and provenance could be evaluated.

Production changes: none.

### 2026-09-03 — Phases 3–4: polling access, licensing, methodology and historical-series safety

Read-only checks confirmed that the two UCD-linked research projects expose direct structured downloads without a login. Licensing/terms checks materially narrowed the other candidates: visible poll results do **not** automatically create a reusable data feed, and Ireland Votes explicitly prohibits automated extraction without permission.

Production changes: none.

## Polling sources — verified findings

### 1. Irish Polling Indicator (IPI) — strongest free national polling source

**Locations**
- https://pollingindicator.com/
- https://pollingindicator.com/method
- https://github.com/Irish-Polling-Indicator/ipi-data
- Stable dataset DOI: 10.7910/DVN/8YVVYX

**Source/type:** UCD-hosted academic research project maintained by Stefan Müller, with Tom Louwerse as founding member. It curates the underlying published Irish Dáil voting-intention polls and produces a Bayesian polling indicator.

**Coverage:** public site currently states all available raw polling results **1982–2026** and daily aggregated estimates **1987–2026**. Development data are updated after new polls; stable releases are published after an election cycle.

**Polling types:** national Dáil voting intention / party support. It is not a general issue-poll, leader-satisfaction, referendum or constituency-poll database.

**Formats/access:** raw polls in CSV, XLSX, Stata and R; estimates in CSV, Stata and R. No account, API key or authentication required. Static GitHub files are directly machine-readable. No scraping is needed.

**Read-only access check:** public raw CSV was reachable and contains fields including poll/publication date, fieldwork start/end/midpoint, pollster, sample size and party results. This is sufficient for a simple future pull from a static file; no connector was built.

**Licence/reuse evidence:** the project asks users to cite the dataset and identify the Irish Polling Indicator/maintainer when referring to the data. The project publicly encourages use in academic work and reports/articles. A conventional repository `LICENSE` file was not identified during this review, so this should be treated as **Free with attribution/citation** rather than assuming an unrestricted open-data licence.

**Cost:** Free with attribution/citation.

**Authentication/rate limits:** none documented for the static files; normal GitHub hosting constraints would apply.

**Methodology:** combines national Dáil polls from multiple pollsters. The model is expressly intended to contextualise polls and trends, not predict elections. Credible intervals are supplied for the modeled estimates. Underlying pollster methods vary over time.

**Historical-series safety:**
- **Raw poll-by-poll series:** suitable if EirePolitic retains pollster, fieldwork, sample-size and methodology caveats.
- **IPI model series:** suitable as the IPI's own modeled historical estimate, clearly labelled as such.
- Development estimates may be revised as new data/model changes are incorporated; stable DOI releases are preferable for reproducible historical graphics.
- Do not silently compare polls from different methodological eras as though they were one unchanged instrument.

**Difficulty:** Easy. **Maintenance:** Low. **Automation potential:** High. **Recurring graphics:** High value.

**Example uses:** latest poll timeline; party support since an election; pollster comparison; IPI estimate with uncertainty band; historical “same point in previous cycle” comparisons.

### 2. Irish Demographic Polling Datasets — strongest free crosstab/satisfaction source

**Locations**
- https://pollingindicator.com/ (related-project section)
- https://github.com/Irish-Dem-Polling/datasets

**Source/type:** UCD-linked research dataset assembled from published RED C and Behaviour & Attitudes polling reports.

**Coverage:** project site states more than 100 polls published **2011–2025**. An older repository README snapshot still referred to 2011–2023; the live project description is more current, so file-level dates should be treated as authoritative during any later ingestion test.

**Polling types:** vote intention; government satisfaction; party-leader satisfaction. Includes all respondents and demographic/geographic subsamples such as age, gender, social class, region and district magnitude. It does not provide general constituency-level polling.

**Formats/access:** CSV, Stata and R files in public GitHub directories. Interactive dashboard/subset downloads are also available. No authentication. No scraping needed.

**Read-only access checks:** public files for B&A vote intention, RED C vote intention, government satisfaction and party-leader satisfaction were directly reachable without login.

**Licence/reuse evidence:** project documentation asks users of the datasets in news reports or academic research to cite the dataset authors, and asks users of individual survey results to cite/reference the original pollster reports. No standard repository `LICENSE` was identified. Because the dataset is derived from pollster reports, commercial/republication rights are less explicit than a CC-licensed official dataset.

**Cost:** Free with attribution/citation.

**Legal confidence:** Medium rather than High until reuse terms are confirmed for the intended EirePolitic presentation, especially if reproducing detailed pollster crosstabs.

**Methodology:** source reports are RED C and B&A. Weighted proportions/counts are retained where available. Pollster methods and question wording vary and must stay attached to a poll/wave.

**Historical-series safety:** useful for recurring demographic/satisfaction comparisons, but not as one homogeneous instrument. Use pollster-specific series where possible and flag changes in mode, weighting or question wording.

**Difficulty:** Easy. **Maintenance:** Low–Medium. **Automation potential:** High. **Recurring graphics:** High value.

**Example uses:** support by age/region; government satisfaction by demographic; leader satisfaction trends; gender/social-class gaps in party support.

### 3. RED C Research / Business Post

**Locations**
- https://redcresearch.com/
- https://redcresearch.com/our-omnibus/

**Coverage/types:** recurring national voting intention plus issue, satisfaction and demographic questions depending on wave. Public political-poll archive exists.

**Access:** poll reports/articles rather than a documented public structured feed or API. Some reports contain detailed tables; a production process would likely require publisher-supplied files or controlled/manual extraction. Scraping permission was not established, so scraping is **not approved by this investigation**.

**Current methodology evidence:** RED C operates its RED C Live online panel and describes quota/weighting controls. Recent election work is online. Historical methodology differs materially: older polling used telephone sampling and, in some periods, treatment/reallocation of undecided respondents; likely-voter treatment has also changed. Therefore a long RED C trend requires method-era metadata.

**Published paid research:** Irish omnibus page gives **€625 + VAT per question for data tables** and **€795 + VAT per question for full service**. This is bespoke/omnibus research, not a polling feed or historical-dataset subscription.

**Cost:** public poll releases free to view; commissioned data **Moderate paid / bespoke**.

**Series safety:** strong primary-source evidence for individual polls, but not a uniform decades-long method. Do not ingest page values as a continuous series without storing methodology.

**Automation potential:** Low–Medium from public reports; potentially higher only with an agreed data supply arrangement.

### 4. Ireland Thinks / Sunday Independent / The Evidence

**Locations**
- https://www.irelandthinks.ie/
- https://analysis.irelandthinks.ie/
- https://analysis.irelandthinks.ie/services/

**Coverage/types:** recurring national voting intention; issue polling; election/referendum/presidential/European/local work; demographic analysis on some projects.

**Access:** public analyses exist, but no unrestricted public historical API/feed was confirmed. The Evidence includes a data-portal interface with sign-in/subscription elements; no public reusable-feed price was established. Do not sign up during this investigation.

**Methodology:** Ireland Thinks documents use of online, face-to-face and telephone methods depending on project and discusses representative recruitment, weighting and post-stratification. A specific 2024 election-day project stated that its data were free to use with attribution to Ireland Thinks; that permission should **not** be generalized to every monthly poll without confirmation.

**Published commissioned polling prices:** monthly omnibus first question **€800 + VAT**, subsequent questions **€450 + VAT**; rapid polling first question **€2,000 + VAT**, additional questions **€900 + VAT**; video-poll question **€1,200 + VAT**. These are commissioned-research prices, not a historical feed.

**Cost:** public releases free to view; commissioned polling Moderate paid/bespoke; data-portal pricing Unknown.

**Series safety:** keep mode, sample, weighting, fieldwork and commissioner with every observation. Individual-publication use can be strong, but no general reusable feed licence was confirmed.

### 5. Ipsos B&A / The Irish Times

**Locations**
- https://www.ipsosbanda.ie/news-polls/
- https://www.ipsosbanda.ie/research-approaches/omnibus/

**Coverage/types:** national party support, government/leader satisfaction, issues and referendums; occasional constituency/byelection and European-election polling.

**Access:** public news/poll pages and reports; no public polling API or structured historical feed confirmed. Omnibus is available by contact; public price was not found.

**Reuse evidence:** Irish Times methodology/publication notes state that extracts may be quoted/published with acknowledgement to The Irish Times and Ipsos B&A. This supports attributed publication of extracts, **not** an assumption that full reports or crosstabs can be bulk-republished as a dataset.

**Methodology:** recent national Irish Times polling has used representative adult/eligible-voter samples with in-home interviewing and constituency-spread sampling; specific projects can instead use telephone. The historical MRBI/Ipsos/B&A lineage and changes in party-support treatment matter. Irish Times methodology changed around 2010 and again in later years (including treatment of voting likelihood), so long trends need explicit method versions.

**Cost:** public releases free to view; commissioned polling **Contact for pricing**.

**Automation potential:** Low–Medium unless structured access is agreed.

### 6. Opinions / The Sunday Times

**Location:** https://opinions.ie/ and https://opinions.ie/omnibus/

**Coverage/types:** national voting intention is confirmed in the 2024 election cycle; RTÉ poll-of-polls documentation identified its political surveys as online. Issues can be commissioned through its omnibus.

**Access:** no clean public historical structured archive/API was identified. Monthly online omnibus is available, but no public price was found.

**Cost:** public poll releases free to view; commissioned research **Contact for pricing**.

**Historical-series safety:** individual polls usable with source/method metadata; insufficient verified public structure for a preferred recurring feed.

### 7. Amárach Research

**Location:** https://amarach.com/ and https://amarach.com/amarach-omnibus-survey.html

**Coverage/types:** issue, referendum, constitutional, EU and public-policy surveys for government/civil-society/commercial commissioners. Twice-monthly omnibus uses a nationally balanced Irish adult sample.

**Access:** primarily reports/commissioned outputs; no general historical political-poll API/feed identified.

**Cost:** commissioned research; public page did not yield a sufficiently clear current reusable-data price for this investigation, so **Contact/unclear pricing** rather than guessing.

**Use:** strong source for topic-specific recurring features when a commissioner publishes the underlying results with adequate methodology/reuse permission.

### 8. European Movement Ireland — Ireland and the EU Poll

**Location:** https://www.europeanmovement.ie/em-ireland-eu-poll/

**Coverage:** annual programme since 2013; current waves conducted by Amárach. Covers Republic of Ireland and Northern Ireland attitudes to the EU. A 2026 wave used an online representative sample across both jurisdictions.

**Access:** downloadable findings/reports, not an API. Raw reusable microdata were not identified.

**Cost:** Free to view/download reports.

**Reuse:** no explicit structured-data licence found in this investigation; use published findings with attribution and do not assume bulk extraction rights.

**Automation:** Low–Medium. **Value:** Medium–High for annual EU-attitude comparisons.

### 9. ARINS / Irish Times constitutional-future surveys

Repeated all-island/constitutional-attitude surveys are valuable for United-Ireland and constitutional-comparison graphics. They should be treated as project/report sources rather than a single licensed feed until source-by-source download and reuse rights are confirmed.

**Cost/access:** generally public findings; structured reusable data/licence not confirmed. **Automation:** Low. **Value:** Medium–High for topic-specific features.

### 10. IrelandElection.com

**Location:** https://irelandelection.com/opinion_polls_party.php

Searchable HTML tables expose pollster, commissioner, sample/margin and party values, including recent polls. This is useful for discovery and cross-checking, but no clear licence/scraping permission was established. **Discovery-only unless permission is obtained.**

### 11. Ireland Votes

**Locations**
- https://www.irelandvotes.com/polling/ireland/dail
- https://www.irelandvotes.com/about/terms

Displays a long poll series, aggregate and approval tracker. Its terms expressly prohibit scraping, bulk downloading, automated extraction and commercial exploitation without permission.

**Conclusion:** high-value reference/discovery site, but **rejected as an automated ingestion source unless explicit permission/licensing is obtained**.

### 12. IrishPolitics.ie poll tracker

**Location:** https://www.irishpolitics.ie/tracker

Secondary tracker using established pollsters and exposing a CSV control in the interface. No sufficiently clear reuse licence/provenance terms were established. **Discovery/cross-checking only** pending permission.

### 13. Wikipedia Irish polling tables

Useful as a discovery/index source with links to original polls, including constituency and referendum polls. It is not preferred over IPI or primary pollster evidence for a production polling series. Any reuse would also require compliance with Wikipedia licensing and careful preservation of underlying citations.

### 14. Europe Elects polling database

**Location:** https://europeelects.eu/data-access-request/

Ireland is covered. Published price of **€15 per country** applies to academic researchers/students for **non-commercial** work and includes restrictions including no redistribution/resale. Commercial entities are instructed to contact Europe Elects.

**EirePolitic conclusion:** the apparent €15 option is **not a valid cheap commercial option**. Commercial access is **Contact for pricing**.

### 15. PolitPro API

**Location:** https://politpro.eu/en/api

**Coverage/access:** Ireland (`IE`) is supported. JSON endpoints provide current trend, historical trend and raw latest voting-intention polls. API documentation supplies fieldwork start/end, sample size, institute and party values. Historical trends can be revised retroactively. Webhooks are supported.

**Authentication/rate limit:** Bearer token after registration; **30 requests/minute** documented.

**Use rights:** API page says data may be used editorially, in own products and for own customers, but not to build a competing polling-data platform.

**Pricing:** no public API price identified; **Contact for pricing**. No account was created.

**Value:** potentially excellent operational feed, but not preferable to free IPI until price/licence terms are known.

### 16. PoliticalAPI polling endpoint

**Location:** https://politicalapi.com/political-polls-api

**Access:** normalized REST API with Ireland coverage and poll metadata. The provider states the polling data are normalized from publicly available Wikipedia polling tables; therefore it is a convenience layer, not a primary polling source.

**Published standard subscription pricing:** Starter **€89/month** after introductory first month; Professional **€199/month**; Business **€599/month**. Introductory launch prices are lower for the first month only and should not be treated as the ongoing cost.

**Cost:** Starter Moderate paid; Professional/Business Expensive for this use case relative to free alternatives.

**Conclusion:** technically easy but poor value for polling alone because IPI offers better Irish historical provenance for free. Could become relevant only if EirePolitic values the provider's wider cross-country API resources.

### 17. PollBase / PollBase Pro

British/Great Britain focused, not a realistic Republic of Ireland polling source. **Rejected as out of scope.**

## Polling methodology and comparability rules for any future EirePolitic series

1. Store **pollster, commissioner, fieldwork start/end, publication date, sample size and mode** with every poll where available.
2. Store weighting/sampling notes and an explicit **method-version or methodology-note** when a pollster changes mode or treatment.
3. Do not calculate a naive long-run average across online, phone and face-to-face polls without a documented method.
4. Do not silently combine raw individual polls with an aggregator's modeled estimates. Label IPI estimates as modeled IPI estimates.
5. Preserve the treatment of undecided/non-voters. Historical RED C and Irish Times/MRBI examples show that treatment has changed over time.
6. Use pollster-specific comparisons when possible. Cross-pollster comparisons need context because sampling frames, weighting and question wording differ.
7. Keep constituency polls separate from national polls; their samples and margins are different.
8. Treat issue/referendum/presidential questions as question-specific series. Exact wording and fieldwork context can materially affect comparability.
9. For demographic crosstabs, retain subgroup base sizes/counts when available; small subsamples should not be presented with false precision.
10. Every public graphic should state that polling measures opinion during fieldwork and is not an election prediction.

## Polling access-test record

No connector was created. Read-only browser/file checks confirmed:

| Source | Check | Result |
| --- | --- | --- |
| Irish Polling Indicator | Raw development CSV | Directly reachable; structured poll dates, fieldwork, pollster, sample and party columns visible |
| Irish Polling Indicator | Estimates | Direct CSV available; stable DOI version also documented |
| Irish Demographic Polling Datasets | Vote-intention CSVs | Direct GitHub files reachable without login |
| Irish Demographic Polling Datasets | Government/leader satisfaction CSVs | Direct GitHub files reachable without login |
| PolitPro | API documentation | Ireland supported; documented REST endpoints, Bearer authentication, 30 requests/minute, webhooks |
| Ireland Votes | Terms | Automated extraction prohibited without permission; no access test attempted |

## Current polling conclusions

### Strong free candidates

- **Irish Polling Indicator** — best national historical voting-intention foundation.
- **Irish Demographic Polling Datasets** — best demographic/leader/government-satisfaction foundation; legal confidence slightly lower because there is no standard open-data licence and it derives from pollster reports.

### Direct pollsters

RED C, Ireland Thinks and Ipsos B&A remain important as the primary evidence behind individual polls, methodology and special polls. They are not currently verified as free automated feeds.

### Paid market

No **confirmed cheap commercial Irish polling feed** emerged. Europe Elects' €15 access is non-commercial only. PolitPro is promising but contact-priced. PoliticalAPI has published API pricing but starts at €89/month after the introductory period and is secondary to Wikipedia for polling provenance. Commissioning one-off omnibus questions costs hundreds of euro per question and is a different use case from acquiring a historical feed.

## Sources checked

### Polling/data projects
- https://pollingindicator.com/
- https://pollingindicator.com/method
- https://github.com/Irish-Polling-Indicator/ipi-data
- https://github.com/Irish-Dem-Polling/datasets

### Pollsters/commissioners
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
- https://amarach.com/amarach-omnibus-survey.html
- https://www.europeanmovement.ie/em-ireland-eu-poll/

### Secondary/commercial sources
- https://irelandelection.com/opinion_polls_party.php
- https://www.irelandvotes.com/polling/ireland/dail
- https://www.irelandvotes.com/about/terms
- https://www.irishpolitics.ie/tracker
- https://europeelects.eu/data-access-request/
- https://politpro.eu/en/api
- https://politicalapi.com/political-polls-api
- https://en.wikipedia.org/wiki/Opinion_polling_for_the_2024_Irish_general_election

## Rejected / constrained polling sources

- **PollBase / PollBase Pro** — out of scope; British/GB polling.
- **Ireland Votes automated ingestion** — terms prohibit automated extraction without permission.
- **Europe Elects €15 tier for EirePolitic** — non-commercial/academic restriction means this published price does not apply to EirePolitic use.
- **IrelandElection.com and IrishPolitics.ie as production sources** — licensing/scraping permission not sufficiently established; retain for discovery/cross-checking only.
- **Wikipedia as preferred production source** — useful discovery layer, but curated IPI/primary sources offer better Irish polling provenance.

## Ranked recommendations

Final ranking is pending Phase 5 non-polling research.

## Living next-step plan

1. **Completed:** establish research framework and documentation-only PR.
2. **Completed:** discover Irish polling sources.
3. **Completed:** verify polling access, licensing, pricing, methodology and read-only feasibility.
4. **Completed:** assess polling historical-series safety and comparability rules.
5. **Current:** investigate and verify high-value non-polling Irish political datasets.
6. Rank all candidates, produce the required final lists, and select the top five sources for a future ingestion investigation.
