# Instagram post candidates

Status: **Research complete; shortlist for editorial review**  
Date: **3 September 2026**

This note identifies ten defensible Instagram post candidates from the current EirePolitic production metrics and research record. It is an editorial shortlist, **not** the final five.

No production schema, architecture, data pointer, classifier, or model call was changed or used for this work.

## Data sources examined

Research records:

- `parliamentary_questions_investigation.md`
- `parliamentary_questions_question_taking.md`
- `parliamentary_questions_exchange_metrics.md`
- `parliamentary_questions_exchange_metrics_implementation.md`
- `parliamentary_questions_section_headings.md`
- `parliamentary_questions_oral_vs_written.md`
- `broader_speech_context.md`
- `legislation_investigation.md`
- `legislation_bridge_implementation_plan.md`

Production/configuration reviewed:

- `configs/political_metrics/catalogue/questions.yml`
- `configs/political_metrics/catalogue/core_speeches.yml`
- `configs/political_metrics/catalogue/votes.yml`
- `configs/political_metrics/materialization.yml`
- `docs/political_metrics_foundation.md`

The current certified question/exchange snapshot used by the source research covers 2025–2026 and includes 121,355 submitted Parliamentary Questions, 2,127 Oral-question exchanges, 18,485 exchange transcript interventions and 2,763,441 exchange words.

## Metrics investigated

- Oral vs Written submitted-question mix;
- recipient channel mix and concentration;
- Taoiseach question structure;
- single vs grouped Oral exchanges;
- exchange interventions and transcript word volume;
- ministerial/respondent, ordinary-member and chair contributions;
- submitting-member participation and non-submitting TD participation;
- member, party and constituency channel profiles;
- exact Leaders' Questions context;
- section-heading structure;
- certified/potential Bill-to-section, Bill-to-speech and Bill-to-division relationships;
- Bill stages and sponsors;
- existing division/voting components and denominator requirements.

## Ranked shortlist of ten

### 1. Almost every Parliamentary Question is Written

- **Working title:** 97% of Parliamentary Questions are Written
- **Hook:** Oral questions are the visible chamber format, but they are only a small fraction of submitted Parliamentary Questions.
- **Visual:** 100-dot or stacked bar: Written vs Oral, with a small two-year comparison underneath.
- **Metric/relationship:** distinct submitted `question_id` by certified `question_type`.
- **Period:** current 2025–2026 history.
- **Denominator:** all 121,355 submitted question records.
- **Key result:** Written **117,695 (96.98%)**; Oral **3,660 (3.02%)**; about **32.2 Written questions per Oral question**. 2025 Oral share 3.09%; 2026 2.92%.
- **Why interesting:** immediately understandable and corrects the natural tendency to equate chamber visibility with overall PQ volume.
- **Caveats:** counts submitted questions, not scrutiny quality, answer quality or parliamentary effectiveness.
- **Production-ready:** **Yes.** Existing certified foundations are sufficient.
- **Confidence:** **strong**.
- **Final-five competitiveness:** **Very high.**

### 2. Questions to the Taoiseach work very differently

- **Working title:** The Taoiseach is the big exception to the Written-question rule
- **Hook:** While most departments receive overwhelmingly Written questions, questions to the Taoiseach are actually majority Oral.
- **Visual:** Taoiseach vs six large line departments, showing Oral share of submitted questions.
- **Metric/relationship:** recipient-level Oral/Written counts from certified question records.
- **Period:** current 2025–2026 history.
- **Denominator:** all submitted questions to each displayed recipient.
- **Key result:** Taoiseach: **1,354 Oral, 1,137 Written, 54.36% Oral**. The Taoiseach accounts for about **37% of all Oral question records**. By contrast Health is **0.49% Oral**, Education **0.99%**, Transport **1.28%**, Justice **1.30%**, Children **1.35%**, Housing **1.62%**.
- **Why interesting:** strong visual contrast and a real structural feature of the Dáil rather than a partisan ranking.
- **Caveats:** the Taoiseach channel has a special parliamentary role and should not be treated as directly comparable to line departments on scrutiny/effectiveness.
- **Production-ready:** **Yes.**
- **Confidence:** **strong**.
- **Final-five competitiveness:** **Very high.**

### 3. Grouped Oral questions are much bigger exchanges

- **Working title:** When Oral questions are grouped, the debate roughly doubles in size
- **Hook:** Grouping several questions into one exchange changes the shape of the chamber discussion substantially.
- **Visual:** paired bars for single-question vs grouped exchanges: median interventions and median words.
- **Metric/relationship:** `grouped_exchange`, `related_speech_count`, `related_speech_word_count` in certified Oral exchange metrics.
- **Period:** current 2025–2026 Oral-exchange history.
- **Denominator:** 2,127 certified Oral-question exchanges: **1,865 single-question**, **262 grouped**.
- **Key result:** single-question median **6 interventions / 1,147 words**; grouped median **12 interventions / 2,275.5 words**. Grouped exchanges are **12.3%** of exchanges.
- **Why interesting:** visually simple and explains a procedural feature most users will not know.
- **Caveats:** transcript size is not debate quality, importance, effectiveness or responsiveness.
- **Production-ready:** **Yes.** Exchange metrics are implemented and audited.
- **Confidence:** **strong**.
- **Final-five competitiveness:** **Very high.**

### 4. Ministers speak fewer times, but account for most Oral-exchange words

- **Working title:** Fewer interventions, more words: who fills Oral-question transcripts?
- **Hook:** Ordinary TDs make slightly more interventions, but ministerial/respondent speakers account for about three-fifths of the words.
- **Visual:** two 100% bars: share of interventions vs share of words by role.
- **Metric/relationship:** certified participant-role intervention and word components.
- **Period:** current 2025–2026 Oral-exchange history.
- **Denominator:** **18,485 interventions** and **2,763,441 words** inside 2,127 Oral-question exchanges.
- **Key result:** ordinary members: **9,262 interventions (50.1%)**, **1,061,771 words (38.4%)**; ministerial/respondent: **7,967 interventions (43.1%)**, **1,684,996 words (61.0%)**; chair/procedural: **1,251 interventions (6.8%)**, **16,659 words (0.6%)**.
- **Why interesting:** a clean contrast between how often people intervene and how much transcript they generate.
- **Caveats:** do not describe word share as dominance, answer quality or responsiveness.
- **Production-ready:** **Yes.**
- **Confidence:** **strong**.
- **Final-five competitiveness:** **High.**

### 5. The Taoiseach's Oral questions are usually grouped

- **Working title:** 95% of Taoiseach Oral-question exchanges are grouped
- **Hook:** Questions to the Taoiseach are not just more Oral; they are also organised very differently from typical departmental Oral exchanges.
- **Visual:** Taoiseach vs selected line departments: grouped-exchange share and median questions per exchange.
- **Metric/relationship:** recipient + `grouped_exchange` + question count per certified Oral exchange.
- **Period:** current 2025–2026 Oral-exchange history.
- **Denominator:** Oral exchanges for each displayed recorded recipient.
- **Key result:** Taoiseach grouped-exchange share **95.3%**, with median **12.65 questions per exchange**, median **26 interventions** and **3,396 words**. Most line-department Oral exchanges have roughly one question, six interventions and about 1,100–1,300 words.
- **Why interesting:** turns an abstract institutional difference into a concrete visual explainer.
- **Caveats:** recipient channels are structurally different; avoid implying better/worse scrutiny.
- **Production-ready:** **Yes.** Every one of the 2,127 Oral exchanges has exactly one recorded recipient, preventing double attribution.
- **Confidence:** **strong**.
- **Final-five competitiveness:** **High**, though it overlaps thematically with candidate 2.

### 6. Other TDs often join an Oral exchange they did not submit

- **Working title:** Oral questions can draw in TDs beyond the original submitter
- **Hook:** A sizeable share of Oral exchanges include an ordinary TD who was not one of the recorded question submitters.
- **Visual:** single vs grouped exchange bars for the share containing at least one ordinary non-submitting TD.
- **Metric/relationship:** certified observed exchange participation vs recorded submitter identities.
- **Period:** current 2025–2026 Oral-exchange history.
- **Denominator:** single-question and grouped exchanges respectively.
- **Key result:** about **21.4%** of single-question exchanges and **43.9%** of grouped exchanges contain at least one ordinary TD who was not a recorded submitter.
- **Why interesting:** shows that an Oral-question exchange can be broader than a one-question/one-TD interaction.
- **Caveats:** observed participation is not the same as formal question-taking or substitution attribution. Do not infer who "took" a question from participation alone.
- **Production-ready:** **Yes.** Participant foundation is deterministic and separate from question-taker inference.
- **Confidence:** **strong**.
- **Final-five competitiveness:** **High.**

### 7. Most TDs who ask questions use both channels — but Oral-only use does not appear

- **Working title:** Oral questions are an extra channel, not a replacement for Written questions
- **Hook:** Every member who used the Oral channel in this history also used Written questions.
- **Visual:** three-category member count: both channels / Written only / Oral only.
- **Metric/relationship:** distinct submitting member identities by observed question channel use.
- **Period:** current 2025–2026 history.
- **Denominator:** **157 member identities** with at least one recorded submitted question.
- **Key result:** **126** used both Oral and Written; **31** used Written only; **0** used Oral only.
- **Why interesting:** simple behavioural structure that avoids raw volume rankings.
- **Caveats:** the time window is limited to current production history; member tenure and eligibility differ. Do not generalise to all historical Dáileanna.
- **Production-ready:** **Yes.**
- **Confidence:** **strong**.
- **Final-five competitiveness:** **Medium-high.**

### 8. Health receives more than 200 Written questions for every Oral question

- **Working title:** Some departments are almost entirely questioned in writing
- **Hook:** For large service departments, the recorded PQ channel is overwhelmingly Written.
- **Visual:** horizontal bars of Written questions per Oral question for selected high-volume recipients.
- **Metric/relationship:** recipient-level Written/Oral count ratio.
- **Period:** current 2025–2026 history.
- **Denominator:** submitted questions to each displayed recipient.
- **Key result:** Health **203.8 Written per Oral**; Education **99.7**; Transport **76.9**; Justice **75.8**; Children **73.2**; Housing **60.7**.
- **Why interesting:** more intuitive than percentages and useful for explaining why chamber-visible questioning is not the full picture.
- **Caveats:** does not mean these departments receive less scrutiny; it is a channel-mix measure only. Recipient naming/portfolio structure should be labelled exactly as represented in the source period.
- **Production-ready:** **Yes.**
- **Confidence:** **strong**.
- **Final-five competitiveness:** **Medium-high**, but overlaps candidate 2.

### 9. Chair interventions are visible in the exchange count but tiny in word volume

- **Working title:** The chair appears often, but says very little of the transcript
- **Hook:** Procedural chair interventions are nearly 7% of Oral-exchange interventions but only about 0.6% of the words.
- **Visual:** intervention share vs word share for chair/procedural contributions.
- **Metric/relationship:** certified chair intervention and word components.
- **Period:** current 2025–2026 Oral-exchange history.
- **Denominator:** all 18,485 Oral-exchange interventions and 2,763,441 words.
- **Key result:** chair/procedural contributions: **1,251 interventions (6.77%)**, **16,659 words (0.60%)**.
- **Why interesting:** a neat structural explainer showing why intervention counts and word counts answer different questions.
- **Caveats:** procedural role only; not a measure of influence or speaking performance.
- **Production-ready:** **Yes.**
- **Confidence:** **strong**.
- **Final-five competitiveness:** **Medium.** Better as an explainer/carousel panel than a headline post if stronger ideas are available.

### 10. Leaders' Questions is a large, cleanly identifiable parliamentary context

- **Working title:** What does a Leaders' Questions session look like in the transcript data?
- **Hook:** Leaders' Questions can be isolated deterministically from exact Oireachtas section headings, giving a clean foundation for a public explainer.
- **Visual:** introductory structural card using section count and speech/intervention coverage, followed by a simple explanation of the exact proceeding label.
- **Metric/relationship:** exact certified source-heading allowlist for genuine Leaders' Questions sections.
- **Period:** current 2025–2026 speech history.
- **Denominator:** speeches/sections whose source heading exactly matches one of the two certified Leaders' Questions headings.
- **Key result:** **10,821 speeches across 159 sections**; main heading 10,702 speeches, resumed heading 119. A broad substring rule would incorrectly include a separate standing-orders motion, so exact headings matter.
- **Why interesting:** recognizable public format and a clean bridge from raw transcript data to a structural Dáil explainer.
- **Caveats:** there is not yet a broad production `speech_context` foundation. The exact Leaders' Questions rule is research-certified, but publishing richer comparisons (party/member speaking shares, session length trends, etc.) should first use a dedicated reproducible extraction or production implementation.
- **Production-ready:** **Not as a reusable production metric yet; deterministic evidence is strong.**
- **Confidence:** **usable with caveat**.
- **Final-five competitiveness:** **Medium**, unless a focused Leaders' Questions analysis is completed before production.

## Ideas considered and rejected for this shortlist

### Raw party Oral-share ranking

Current data can calculate party Oral/Written shares, but raw shares are affected by party size, member tenure, high-volume individual submitters and access/scheduling. There is no separately certified per-TD exposure/eligibility denominator. A ranking could be visually appealing but invites a misleading "who scrutinises most" interpretation. **Rejected for now.**

### Constituency Oral-share ranking

The event-date foundation supports constituency aggregation, but these are not per-representative rates and should not be treated as representation quality. The denominator and tenure context are too easy to lose in an Instagram ranking. **Rejected for now.**

### Member Oral-share leaderboard

Member channel profiles are deterministic, but a leaderboard risks equating channel choice with activity quality/effectiveness and is sensitive to tenure and scheduling. **Rejected as a top-ten post.**

### Question-taker/substitution leaderboard

Explicit evidence certifies only a bounded subset of question-taking relationships; many cases remain unknown and grouped exchanges require extra caution. Observed participation must not be converted into taker attribution. **Rejected.**

### Section-heading topic ranking

Headings are useful source labels but are not a clean issue taxonomy; headings can span portfolios and their relationship to recipients can shift. **Rejected as a "what issues dominate" post.**

### Recipient concentration index

Oral recipient concentration (HHI 0.1596) is higher than Written (0.0981), but HHI is not intuitive for a general Instagram audience and much of the difference is already explained more clearly by the Taoiseach exception. **Rejected in favour of candidates 2 and 5.**

## Promising ideas that are not yet safe

### Bill-linked speeches

The legislation investigation found a conservative deterministic subset covering **168 Bills, 371 debate sections and 7,352 speeches**. This is strong evidence, but the certified Bill-section bridge is not yet production-deployed and current transcript coverage excludes much Seanad/committee/history. Do not publish "most debated Bills" or whole-Oireachtas coverage yet.

### Bill-linked divisions

The same conservative subset links **168 divisions across 94 Bill-linked sections**. This is promising for posts about which Bills reached recorded divisions or how stages relate to votes, but should wait until the certified section bridge is implemented and denominator/stage context is production-safe.

### Bill sponsors

All 406 Bills have sponsor rows, but member sponsors and office/ministerial sponsors are different source entities. Only source-URI member links should map directly to people; office sponsors require separate date-aware attribution. Avoid sponsor rankings until the interpretation layer is explicit.

### Bill stages/readings

The stage table has **1,395 rows across 406 Bills**, but stage outcomes are often blank and `Cream List` requires interpretation before exposing stage names as a public taxonomy. Promising for an explainer after cleanup/certification.

### Voting behaviour / party unity

Voting components exist, but a public behavioural ranking should only be produced where the eligible-vote denominator, absence/abstention handling, party attribution date and substantive division context are all explicit. The strongest next step is to connect voting to certified Bill/motion context rather than publish raw unity rankings.

### Broader motions/statements/procedural context

Deterministic source signals exist, but broad regex families are heterogeneous or incomplete. Exact heading allowlists and scope definitions are needed before public comparisons.

## Editorial ranking rationale

The top six are favoured because they combine certified production evidence, clear denominators, general-interest procedural insight and simple visuals. Candidates 7–9 are safe but slightly less novel or overlap stronger themes. Candidate 10 is promising and recognizable but should not outrank production-ready exchange/PQ findings until its reusable metric path is implemented.

No candidate should be framed as measuring political effectiveness, scrutiny quality, representation quality, answer quality or performance.

## Living next-step plan

1. Review these ten editorially with the user; do **not** select the final five before that review.
2. For ideas retained after review, remove thematic duplication — especially among candidates 2, 5 and 8 — so the final five remain meaningfully different.
3. Prefer candidates 1–9 for immediate production because their denominators and attribution are already certified.
4. If Leaders' Questions survives editorial review, run a focused deterministic extraction/validation for the exact intended visual before production.
5. Keep Bill-linked, sponsor, stage and voting posts in the promising queue until the certified Bill-section relationship and required public denominators are deployed/audited.
6. Do not add classifiers merely to create variety; continue using existing deterministic relationships unless a specific editorial question cannot otherwise be answered.
