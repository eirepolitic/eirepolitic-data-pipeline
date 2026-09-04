# Irish political data sources investigation

## Purpose

Identify additional Irish political data sources that could support EirePolitic posts, charts, infographics, public dashboards, historical comparisons, and recurring political-data features.

This is a research-only investigation. It must not change production architecture, schemas, pipelines, or production data.

The highest priority is Irish political polling data. Other datasets are considered only after polling has been investigated thoroughly.

## Guardrails

- No production schema, architecture, or connector changes.
- No purchases, subscriptions, paid sign-ups, or bypassing access controls.
- Do not assume scraping is permitted; record evidence from terms, robots guidance, licences, or publisher statements where available.
- Prefer primary sources. Treat secondary aggregators as discovery-only unless provenance, reproducibility, and reuse rights are sufficiently clear.
- Do not reproduce copyrighted polling reports in full.
- Keep poll results distinct from election results and predictions.
- Record fieldwork dates and methodology wherever available.
- Document limitations and uncertainty explicitly.
- Small read-only access checks are acceptable only to verify feasibility.
- Final merge should contain documentation changes only; temporary investigation artefacts should not be merged wholesale.

## Research questions

### Priority 1: Irish polling

Investigate realistic sources covering, where available:

- voting intention and party support;
- leader approval/satisfaction;
- government satisfaction;
- issue polling;
- referendum polling;
- election and constituency polling;
- European election polling;
- presidential polling;
- demographic/crosstab breakdowns;
- historical polling series.

### Priority 2: other Irish political datasets

After polling, investigate high-value public-interest datasets such as:

- election, constituency, transfer/count, candidate, local, European, and referendum results;
- donations, party finances, expenses, spending, lobbying, interests, and public appointments;
- procurement, spending, budgets, Estimates, and Exchequer data;
- housing, health, education, transport, crime, immigration, population, demographics, and constituency statistics;
- constituency geography and boundary data;
- opinion surveys including Eurobarometer and the European Social Survey;
- EU/OECD datasets useful for Irish political comparisons;
- legislative/government open data not already used by EirePolitic;
- structured FOI publication logs or credible media/political-appearance datasets.

Only sources with a plausible recurring visualization or public-interest analysis use case should be retained as recommendations.

## Evidence to collect for every polling source

For each polling source, record:

| Area | Evidence required |
| --- | --- |
| Identity | Organisation/source name; canonical URL/source location; primary vs secondary source |
| Coverage | Poll types; years covered; update frequency; geographic scope; historical availability; crosstabs/demographics; constituency-level data |
| Access | Data format; API; CSV/Excel/JSON/XML/RSS/static files; HTML tables; programmatic access; whether scraping is required; authentication; rate limits; pagination/identifiers/date fields where relevant |
| Legal reuse | Licence/terms; attribution; commercial-use restrictions; scraping permission evidence or uncertainty |
| Cost | Published price; pricing model; free tier; classification: Free / Free with attribution / Free with registration or API key / Cheap paid / Moderate paid / Expensive or enterprise / Contact for pricing / Unknown |
| Operations | Expected ingestion difficulty: Easy / Moderate / Hard; expected maintenance burden: Low / Medium / High; automation potential: High / Medium / Low |
| Quality | Reliability; methodology transparency; suitability for recurring EirePolitic graphics; provenance/reproducibility for aggregators |
| Poll methodology | Pollster; commissioner; fieldwork dates; sample size; sampled population; sampling method; weighting; mode; undecided treatment; margin of error where supplied; question wording where available |
| Historical-series safety | Whether the source supports a consistent historical series and what comparability caveats apply |

Paid sources must be labelled as one-off purchase, monthly subscription, annual subscription, API/feed access, bespoke research, or contact/unclear pricing. Never infer an unpublished price.

## Evidence to collect for every non-polling source

For each non-polling source, record:

- what the source contains;
- why it matters for EirePolitic;
- geographic and historical coverage;
- update frequency;
- data format;
- API/download/programmatic options;
- licence and attribution requirements;
- cost classification;
- ingestion difficulty;
- maintenance burden;
- at least one concrete chart, post, dashboard, or recurring-feature idea.

## Investigation phases

### Phase 1 — establish research framework

1. Create this durable research document.
2. Add it to `docs/research/README.md`.
3. Work only on a documentation branch/PR.
4. Record the evidence standard, guardrails, and living next steps before substantive source research.

### Phase 2 — discover Irish polling sources

1. Build a broad source inventory from pollsters, newspapers, broadcasters, universities/research institutes, official/open-data portals, GitHub/data projects, polling aggregators, and archives.
2. Prefer original pollster or commissioning-source evidence.
3. Record secondary sources separately and identify their underlying provenance.
4. Reject obvious dead ends early, but document the reason.

### Phase 3 — verify polling access, licensing, methodology, and cost

For promising sources:

1. Verify historical coverage and the exact data fields exposed.
2. Check APIs, downloadable files, static structured data, or HTML tables.
3. Inspect authentication, pagination, date identifiers, and update behaviour where applicable.
4. Check terms/licensing and whether automated retrieval or scraping appears permitted.
5. Record published pricing exactly where available.
6. Capture methodology evidence sufficient to judge comparability across time and pollsters.
7. Run small read-only access checks where useful; do not build connectors.

### Phase 4 — assess historical-series safety and recurring-use value

1. Separate raw poll releases from aggregator-derived series.
2. Identify changes in methodology, mode, weighting, question wording, undecided treatment, and population sampled.
3. Decide whether each source can safely support a consistent historical series, a poll-by-poll series with caveats, or discovery only.
4. Record candidate recurring graphics and appropriate disclaimers.

### Phase 5 — investigate other Irish political datasets

1. Prioritize official and research-institution sources with clear reuse rights.
2. Test whether each dataset enables a meaningful EirePolitic visualization or recurring feature.
3. Record access, licence, cost, ingestion effort, maintenance burden, and example uses.
4. Avoid collecting low-value datasets merely because they are available.

### Phase 6 — rank and conclude

Produce these final lists:

1. Best free polling sources.
2. Best cheap paid polling sources.
3. Best non-polling political datasets.
4. High-value sources currently inaccessible or too expensive.
5. Sources investigated and rejected.

For the strongest candidates assign:

- value to EirePolitic: High / Medium / Low;
- access difficulty: Easy / Moderate / Hard;
- cost: Free / Cheap / Moderate / Expensive;
- automation potential: High / Medium / Low;
- confidence in legality/licensing: High / Medium / Low.

Identify the top five sources recommended for a later ingestion investigation. Do not implement them in this task.

## Research log

### 2026-09-03 — Phase 1

- Created the research framework and evidence requirements.
- Production changes: none.
- Next: discover and inventory realistic Irish polling sources, then verify access, licensing, methodology, and pricing source by source.

### 2026-09-03 — Phase 2: polling-source discovery

A broad inventory was built across research projects, primary pollsters, commissioning media, issue-poll projects, aggregators, and commercial APIs. The inventory deliberately separates structured reusable datasets from poll-release webpages and discovery-only aggregators.

Strong structured/free candidates discovered:

1. **Irish Polling Indicator (UCD-hosted research project)** — https://pollingindicator.com/ and https://github.com/Irish-Polling-Indicator/ipi-data
   - Raw national voting-intention polls plus daily aggregated estimates.
   - Public documentation states raw polls cover roughly 1982/1983 onward and estimates 1987 onward; development data update after new polls.
   - CSV/XLSX/Stata/R formats are exposed publicly.
   - Raw poll CSV visibly includes poll date, fieldwork start/end/midpoint, pollster, sample size and party results.
   - This is the leading candidate for a recurring national historical polling series.

2. **Irish Demographic Polling Datasets (UCD-linked research project)** — https://github.com/Irish-Dem-Polling/datasets
   - Aggregated vote intention, government satisfaction, and party-leader satisfaction.
   - Covers Red C and Behaviour & Attitudes source reports and includes demographic/geographic subsamples.
   - Public CSV/Stata/R files; dashboard also supports subset downloads.
   - Repository documentation reports 100+ polls; the main project site states coverage through 2025, while an older repository README snapshot still says 2011–2023. Exact current coverage will be verified from files rather than inferred from the stale text.

Primary pollsters/publication channels discovered:

3. **RED C Research / Business Post** — https://redcresearch.com/ and Business Post poll reports.
   - Recurring national voting intention; public poll reports also contain political issues, satisfaction and demographic tables depending on wave.
   - Current reports indicate online polling through RED C Live; historical methodology changed over time.

4. **Ireland Thinks / Sunday Independent / The Evidence** — https://www.irelandthinks.ie/ and https://analysis.irelandthinks.ie/
   - Recurring national voting intention plus election, referendum, presidential, European/local and issue polling.
   - Public analysis contains demographic breakdowns for some projects and explicit methodology discussion.

5. **Ipsos B&A / The Irish Times** — https://www.ipsosbanda.ie/news-polls/ and Irish Times poll pages.
   - National voting intention, government/leader satisfaction, issue polling, referendum polling and occasional constituency/byelection and European-election polling.
   - Ipsos B&A is the successor/combination of Ipsos Ireland/MRBI and Behaviour & Attitudes, so historical series require methodology/version awareness.

6. **Opinions / The Sunday Times** — https://opinions.ie/
   - National voting-intention polling is confirmed in the 2024 election cycle; RTÉ's poll-of-polls documentation identifies its 2024 political surveys as online.
   - A clean public historical data archive was not found in discovery and needs further verification.

Additional issue/referendum/all-island polling sources discovered:

7. **Amárach Research** — https://amarach.com/
   - Public/commissioned issue polling; government and civil-society clients publish some results.
   - Relevant for referendum, constitutional, EU and public-policy topics rather than a single continuous party-support series.

8. **European Movement Ireland — Ireland and the EU Poll** — https://www.europeanmovement.ie/em-ireland-eu-poll/
   - Annual series since 2013, currently conducted by Amárach.
   - Covers attitudes to the EU across the Republic of Ireland and Northern Ireland and exposes downloadable findings.

9. **ARINS / Irish Times constitutional-future surveys** — project/publication sources associated with UCD/Irish Times.
   - Useful for United-Ireland/constitutional-attitude comparisons and repeated issue questions.
   - Needs source-by-source licensing/download verification.

Secondary aggregators/discovery candidates discovered:

10. **IrelandElection.com** — https://irelandelection.com/opinion_polls_party.php
    - Searchable HTML table of national polls including pollster, commissioner, sample, margin and party values; current data visible through 2026.
    - Potentially useful for cross-checking/discovery, but reuse/scraping rights and provenance need verification before any production use.

11. **Ireland Votes** — https://www.irelandvotes.com/polling/ireland/dail
    - Displays a polling series back to 2007 plus an aggregate and government-approval tracker.
    - Its terms explicitly prohibit scraping/bulk downloading/automated extraction without permission, so it is likely discovery/reference only unless permission is obtained.

12. **IrishPolitics.ie** — https://www.irishpolitics.ie/tracker
    - Current poll tracker with visible poll history and a CSV control in the public interface.
    - Secondary source using Red C, Ireland Thinks and Ipsos B&A. Licensing/provenance must be checked before reuse.

13. **Wikipedia Irish election polling pages** — e.g. https://en.wikipedia.org/wiki/Opinion_polling_for_the_2024_Irish_general_election
    - Useful discovery/index source for national, referendum and constituency polls and source links.
    - Not preferred as the canonical production source where primary/curated research data are available.

Commercial / paid access candidates discovered:

14. **Europe Elects polling database** — https://europeelects.eu/data-access-request/
    - Ireland included. Academic/non-commercial CSV is advertised at €15 per country, but commercial entities must contact Europe Elects first.
    - Therefore the €15 price is not applicable to EirePolitic commercial/public use without separate permission; pricing for commercial use is contact-only.

15. **PolitPro API** — https://politpro.eu/en/api
    - Raw individual voting-intention polls and polling trends for Ireland are within its supported-country coverage.
    - JSON API intended for media/publishers/think tanks and can be used editorially/in own products, subject to restrictions on competing polling platforms.
    - Pricing is not public on the reviewed API page; contact/inquiry required.

16. **PoliticalAPI polling endpoint** — https://politicalapi.com/political-polls-api
    - REST API with Ireland coverage and normalized poll metadata.
    - It states its polling data are normalized from publicly available Wikipedia polling tables, so it is a convenience layer rather than a primary source.
    - Published subscription tiers exist and will be assessed in Phase 3.

Discovery-only/non-Irish dead end:

17. **PollBase / PollBase Pro** — UK/Great Britain focused, not a realistic Republic of Ireland polling source. It should not be pursued for this investigation.

Production changes: none.

## Sources checked

### Polling research/data projects

- Irish Polling Indicator: https://pollingindicator.com/
- Irish Polling Indicator method: https://pollingindicator.com/method
- Irish Polling Indicator development data: https://github.com/Irish-Polling-Indicator/ipi-data
- Irish Demographic Polling Datasets: https://github.com/Irish-Dem-Polling/datasets

### Pollsters and commissioners

- RED C Research: https://redcresearch.com/
- RED C omnibus: https://redcresearch.com/our-omnibus/
- Ireland Thinks: https://www.irelandthinks.ie/
- Ireland Thinks analysis/public data site: https://analysis.irelandthinks.ie/
- Ipsos B&A polls: https://www.ipsosbanda.ie/news-polls/
- Ipsos B&A omnibus: https://www.ipsosbanda.ie/research-approaches/omnibus/
- Opinions omnibus: https://opinions.ie/omnibus/
- Amárach omnibus: https://amarach.com/amarach-omnibus-survey.html
- European Movement Ireland EU Poll: https://www.europeanmovement.ie/em-ireland-eu-poll/

### Secondary/paid aggregators

- IrelandElection.com polls: https://irelandelection.com/opinion_polls_party.php
- Ireland Votes Dáil polling: https://www.irelandvotes.com/polling/ireland/dail
- Ireland Votes terms: https://www.irelandvotes.com/about/terms
- IrishPolitics.ie tracker: https://www.irishpolitics.ie/tracker
- Europe Elects Ireland: https://europeelects.eu/ireland/
- Europe Elects data access: https://europeelects.eu/data-access-request/
- PolitPro API: https://politpro.eu/en/api
- PoliticalAPI polls API: https://politicalapi.com/political-polls-api
- Wikipedia 2024 polling index: https://en.wikipedia.org/wiki/Opinion_polling_for_the_2024_Irish_general_election

## Rejected sources

- **PollBase / PollBase Pro** — British/Great Britain polling database, not Republic of Ireland coverage. Rejected as out of scope.
- **Ireland Votes as an automated ingestion source** — not rejected as a reference site, but its published terms prohibit scraping, bulk downloading and automated extraction without permission. Treat as discovery/reference only unless permission is obtained.

## Ranked recommendations

Pending completion of access/licensing/methodology verification and non-polling research.

## Living next-step plan

1. **Completed:** discover a comprehensive inventory of Irish polling sources.
2. **Current:** verify primary-source access, licensing, methodology, historical coverage, and cost for each promising polling source.
3. Test programmatic feasibility using read-only checks where useful.
4. Assess historical-series comparability and recurring-graphic value.
5. Investigate high-value non-polling Irish political datasets.
6. Rank candidates and select the top five for a future ingestion investigation.
