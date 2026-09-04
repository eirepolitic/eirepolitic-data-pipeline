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

## Sources checked

Substantive source research has not started yet. Entries will be added phase by phase with URLs, evidence, access tests, pricing, licensing, methodology, and conclusions.

## Rejected sources

None yet.

## Ranked recommendations

Pending completion of the investigation.

## Living next-step plan

1. Discover a comprehensive inventory of Irish polling sources.
2. Verify primary-source access, licensing, methodology, historical coverage, and cost for each promising source.
3. Test programmatic feasibility using read-only checks where useful.
4. Assess historical-series comparability and recurring-graphic value.
5. Investigate high-value non-polling Irish political datasets.
6. Rank candidates and select the top five for a future ingestion investigation.
