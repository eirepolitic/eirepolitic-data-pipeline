# Oral versus Written parliamentary questions

Status: **Research complete; no new production dataset recommended**  
Date: **2 September 2026**  
Parent research: [Parliamentary questions investigation](parliamentary_questions_investigation.md)

This note records the comparison of Oral and Written Parliamentary Question channels using the current certified production foundations.

## Scope

Production batch:

`structure-oral-exchange-participants-20260902-1`

Sources used:

- `silver_questions`;
- `daily_question_dimensions`;
- `oral_question_sections`.

No classifier or model call was used.

Question counts below are distinct submitted Oireachtas `question_id` records unless explicitly described as exchanges.

## Overall channel mix

Across the current 2025–2026 history:

- total question records: **121,355**;
- Written: **117,695 (96.98%)**;
- Oral: **3,660 (3.02%)**;
- Written-to-Oral ratio: about **32.2:1**.

### By year

| Year | Oral | Written | Oral share |
| --- | ---: | ---: | ---: |
| 2025 | **2,071** | **64,939** | **3.09%** |
| 2026 | **1,589** | **52,756** | **2.92%** |

The basic channel balance is therefore stable across the two years: Written questions overwhelmingly dominate submitted-question volume.

## Oral exchanges and recipient attribution

The current history contains **2,127 Oral-question exchanges**.

A key structural result from this comparison is:

- **2,127 of 2,127 exchanges have exactly one recorded question recipient**;
- **0 multi-recipient Oral exchanges**;
- **0 Oral exchanges without a recorded recipient**.

This means the exchange-level transcript measures already materialized in `oral_question_sections` can be attributed safely to the recorded recipient without duplicating one exchange across multiple departments.

This is stronger than expected and makes recipient-level Oral exchange analysis straightforward.

## Oral questions are more concentrated by recipient

Recipient concentration is noticeably higher for Oral questions than Written questions.

Herfindahl concentration index over recipient question counts:

- Oral: **0.1596**;
- Written: **0.0981**.

This is primarily driven by the special parliamentary role of questions to the Taoiseach.

## Taoiseach is structurally different from line departments

Questions to the Taoiseach are an exceptional channel:

- Oral questions: **1,354**;
- Written questions: **1,137**;
- Oral share: **54.36%**;
- Written-to-Oral ratio: **0.84:1**;
- single-recipient exchanges: **107**;
- median questions per exchange: **12.65**;
- grouped-exchange share: **95.3%**;
- median transcript interventions: **26**;
- median transcript words: **3,396**.

The Taoiseach receives approximately **37% of all Oral question records** in the current history.

This makes Taoiseach questions structurally incomparable with most line-department question channels unless the distinction is shown explicitly.

## Large service departments are overwhelmingly Written

Several of the highest-volume question recipients use the Written channel almost exclusively by volume.

| Recipient | Oral | Written | Oral share | Written per Oral |
| --- | ---: | ---: | ---: | ---: |
| Health | **130** | **26,495** | **0.49%** | **203.8** |
| Education | **131** | **13,064** | **0.99%** | **99.7** |
| Transport | **127** | **9,762** | **1.28%** | **76.9** |
| Justice | **122** | **9,253** | **1.30%** | **75.8** |
| Children | **116** | **8,489** | **1.35%** | **73.2** |
| Housing | **147** | **8,921** | **1.62%** | **60.7** |

This does **not** mean these departments receive less scrutiny. It shows that their recorded Parliamentary Question activity is overwhelmingly conducted through Written questions rather than Oral questions.

## Recipients with relatively higher Oral use

Excluding the exceptional Taoiseach channel, the recipients with the highest Oral share among recipients with at least 50 total questions include:

| Recipient | Oral share | Written per Oral |
| --- | ---: | ---: |
| Defence | **6.70%** | **13.9** |
| Culture | **5.65%** | **16.7** |
| Rural | **5.32%** | **17.8** |
| Foreign | **4.87%** | **19.5** |
| Public Expenditure | **4.43%** | **21.6** |
| Tourism | **4.41%** | **21.7** |
| Further and Higher Education | **3.78%** | **25.4** |
| Enterprise | **3.44%** | **28.1** |

Even here, Written questions remain the majority channel.

## Oral exchange structure varies by recipient

Because every Oral exchange has one recipient, the existing exchange metrics can be compared safely by department.

Most line-department Oral exchanges have a median of roughly:

- **6 transcript interventions**;
- around **1,100–1,300 words**;
- roughly **1 Oral question per exchange**.

The Taoiseach channel is the major outlier because grouped questions are the norm.

Other recipients with notable grouped-exchange shares include:

- Foreign: about **26.9%** grouped;
- Tourism: about **26.1%**;
- Defence: about **21.9%**;
- Culture: about **13.3%**;
- Finance: about **12.2%**.

Recipients such as Children and Social Protection were effectively single-question exchanges in this snapshot.

## Member channel use

There are **157 member identities** with recorded question submissions in the current foundation.

- **126** used both Oral and Written channels;
- **31** used Written questions only;
- **0** used Oral questions without also using Written questions.

Thus, Oral questioning appears to be an additional channel used by a subset of members rather than a replacement for Written questions.

### High-volume members with relatively high Oral share

Among members with at least 20 total questions, examples with the highest Oral share include:

- Mary Lou McDonald — **19.1% Oral**;
- Rose Conway-Walsh — **18.4%**;
- Tony McCormack — **15.5%**;
- Donnchadh Ó Laoghaire — **12.8%**;
- Ruairí Ó Murchú — **12.7%**;
- Ruth Coppinger — **10.7%**;
- Paul Murphy — **8.65%**.

These are **channel-use profiles only**. They should not be interpreted as rankings of activity quality, effectiveness or scrutiny performance.

### Written-only examples

Some high-volume members have zero recorded Oral questions in this history while submitting large numbers of Written questions.

Examples include members with hundreds of Written questions and no Oral question records.

This is further evidence that the two channels represent different parliamentary usage patterns and should not be collapsed into one undifferentiated question count when describing behaviour.

## Party channel profiles

`daily_question_dimensions` already supplies event-date party attribution, so party-level channel mix can be calculated without a new dataset.

Current party aggregates include:

| Party/group | Oral | Written | Oral share |
| --- | ---: | ---: | ---: |
| Sinn Féin | **1,101** | **29,647** | **3.58%** |
| Fianna Fáil | **929** | **25,322** | **3.54%** |
| Fine Gael | **436** | **18,542** | **2.30%** |
| People Before Profit–Solidarity | **311** | **3,495** | **8.17%** |
| Social Democrats | **288** | **9,475** | **2.95%** |
| Labour | **234** | **8,855** | **2.57%** |
| Independent | **167** | **6,964** | **2.34%** |
| Aontú | **115** | **4,142** | **2.70%** |
| Green Party | **52** | **1,738** | **2.91%** |
| Independent Ireland | **15** | **9,163** | **0.16%** |

### Party interpretation caution

These are raw question-record channel shares, **not per-TD rates**.

They are affected by:

- party size;
- which individual TDs submit large Written batches;
- member tenure during the period;
- access and parliamentary scheduling for Oral questions.

Do not interpret a higher Oral share as stronger scrutiny or greater parliamentary effectiveness.

A per-TD party comparison would require a separately certified exposure/eligibility denominator and is not added here.

## Constituency channel profiles

The same event-date foundation supports constituency channel profiles.

Examples with relatively high Oral share include:

- Meath East — **6.61% Oral**;
- Louth — **6.22%**;
- Mayo — **5.81%**;
- Dublin South-West — **5.32%**;
- Cork East — **4.59%**.

Examples with much lower Oral share include:

- Longford–Westmeath — **0.26% Oral**;
- Cork South-West — **0.64%**;
- Wexford — **0.95%**.

Again, these are aggregated submitted-question records for TDs attributed to each constituency on the event date. They are not per-representative rates and must not be interpreted as constituency representation quality.

## Important data-contract observation

`daily_question_dimensions` stores `question_type` dimension values as lowercase:

- `oral`;
- `written`.

The first exploratory query assumed title case and therefore produced an empty party/constituency profile before the values were normalized.

This was an analysis-script assumption only. The production dataset itself was correct.

Future consumers should normalize or use the exact documented dimension values rather than assume presentation capitalization.

## Existing foundations are sufficient

No new materialized Oral-vs-Written comparison dataset is needed.

The recurring calculations are already supported by:

- `daily_question_dimensions` for member/party/constituency/national channel counts;
- source question recipient fields for question-level recipient mix;
- `oral_question_sections` for recipient-level Oral exchange size and grouped structure;
- `oral_question_exchange_participants` for detailed observed participation if needed.

A new comparison table would largely duplicate existing additive components and introduce another materialization to maintain.

## Safe reusable measures

Useful descriptive measures that can be calculated from existing foundations include:

- Oral question count;
- Written question count;
- Oral share of submitted questions;
- Written-to-Oral question ratio;
- Oral exchange count;
- Oral questions per exchange;
- grouped-exchange share;
- Oral exchange intervention count;
- Oral exchange word volume;
- recipient channel mix;
- member/party/constituency channel mix.

### Interpretation rules

Do not label these measures as:

- scrutiny effectiveness;
- answer quality;
- opposition performance;
- ministerial accountability quality;
- representation quality.

Safe language is:

- channel use;
- submitted-question mix;
- recorded Oral exchange activity;
- recipient mix;
- transcript volume;
- grouped-question structure.

## Evidence

Read-only production investigations:

- main Oral-vs-Written comparison: **33680594824** — SUCCESS;
- corrected event-date member/party/constituency channel profiles: **33680698374** — SUCCESS.

No production data changed.

No classifier was called.

## Living next-steps plan

### 1. Continue deterministic speech-context research

This is now the immediate research task.

The question structures are sufficiently mature and the existing foundations already support Oral-vs-Written analysis without further schema work.

Investigate whether additional transcript context categories can be certified from source metadata without AI, especially:

- Leaders' Questions;
- Bills / legislation debates;
- motions;
- statements;
- parliamentary business / procedural sections;
- other.

For each candidate category:

- identify the exact source field/rule;
- measure coverage;
- test overlap between categories;
- establish precedence when more than one rule applies;
- inspect false positives manually;
- retain `other` when source evidence is insufficient.

Do not infer context from speech text unless a deterministic source relationship is proven.

### 2. Determine whether `speech_question_context` should become broader `speech_context`

Only after the deterministic categories are certified.

Potential architecture:

- keep `speech_question_context` for compatibility if needed;
- introduce a broader context foundation only if it has multiple well-certified categories;
- ensure every speech receives exactly one top-level context under explicit precedence rules;
- preserve source section identifiers and evidence method.

Do not rename or replace the current production dataset until the broader taxonomy has been validated on historical production data.

### 3. Return to legislation investigation

Bills/legislation context work should connect directly to the previously deferred legislation investigation rather than inventing a speech-only label in isolation.

Investigate:

- bill identifiers and debate-section relationships;
- stages/readings;
- sponsoring member/minister where source-supported;
- speech linkage;
- divisions linked to legislation;
- whether legislation provides useful cross-metric structure.

### 4. Improve voting insight after legislation context

Once legislation/motion context is better understood, revisit division analysis so voting behaviour can be described with more substantive context than raw party unity alone.

### 5. Question issue classification remains deferred

The Oral-vs-Written comparison produced useful channel insights entirely from deterministic data.

Do not classify the full question history unless a concrete use case later requires topic analysis that cannot be answered by recipients, Oral section headings, legislation/context relationships, or existing issue-labelled speech data.
