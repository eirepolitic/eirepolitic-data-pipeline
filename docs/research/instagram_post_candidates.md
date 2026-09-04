# Instagram post candidates

Status: **Research complete; rebuilt across the full EirePolitic dataset**  
Date: **3 September 2026**

This replaces the earlier Parliamentary-Questions-heavy shortlist. The scope is now all usable EirePolitic political data: speeches, issue labels, divisions/member votes, legislation, deterministic parliamentary context and Parliamentary Questions.

No production schema, architecture or source data was changed for this research. Temporary read-only diagnostics were run only on the isolated branch `analysis/instagram-candidates-all-data-20260903` and are not part of the documentation PR.

## Data sources examined

Research and implementation records:

- `broader_speech_context.md`
- `legislation_investigation.md`
- `legislation_bridge_implementation.md`
- `parliamentary_questions_oral_vs_written.md`
- `parliamentary_questions_exchange_metrics.md`
- `parliamentary_questions_question_taking.md`
- `oireachtas_speech_issue_classifier_v2.md`

Current production/derived data inspected directly:

- `silver_speeches`
- `silver_divisions`
- `silver_member_votes`
- `silver_bills`
- `silver_bill_stages`
- `silver_bill_sponsors`
- `gold_member_activity_yearly`
- `gold_constituency_activity_yearly`
- `gold_current_members`
- 2025 classified debate speeches compatibility output
- production `bill_debate_sections`
- certified Parliamentary Question and Oral-exchange metrics

The read-only diagnostic runs used deterministic existing data only. No classifier/API calls were made.

## Ranked shortlist of ten

### 1. What the Dáil talked about most in 2025

- **Working title:** Housing, health and education dominated policy speech topics in 2025
- **Hook:** More than a third of policy-labelled speeches fell into just three issue categories.
- **Visual:** ranked horizontal bars for the top 8–10 issues, with the top three highlighted.
- **Metric/relationship:** count of 2025 speeches by primary `PoliticalIssues` label, excluding `NONE`/blank.
- **Period:** 2025.
- **Denominator:** **23,936 policy-labelled speeches**.
- **Key result:** Housing and Community Development **3,385 (14.1%)**; Health **3,208 (13.4%)**; Education **2,440 (10.2%)**. Together they account for **37.7%** of policy-labelled speeches. International Affairs follows at 1,699 (7.1%).
- **Why interesting:** immediately understandable, high public relevance, and naturally visual.
- **Caveats:** classifier assigns one main issue per speech; this measures recorded speaking attention, not policy importance, time spent, agreement, effectiveness or public concern. Issue metrics remain classification-dependent.
- **Production-ready:** **Usable from existing classified output; public copy must carry classifier context.**
- **Confidence:** **usable with caveat**.
- **Final-five competitiveness:** **Very high.**

### 2. How long is a recorded Dáil speech?

- **Working title:** The median recorded Dáil speech in the current snapshot is 156 words
- **Hook:** A large share of transcript interventions are very short, while a smaller tail runs much longer.
- **Visual:** distribution bands: under 50 words / 50–156 / 157–500 / over 500, plus median callout.
- **Metric/relationship:** deterministic `word_count` across canonical `silver_speeches`.
- **Period:** current 2026 production speech snapshot, covering **10 debate days**.
- **Denominator:** **3,430 recorded speeches/interventions**.
- **Key result:** median **156 words**; mean 257.7; **1,108 (32.3%)** are under 50 words; **495 (14.4%)** exceed 500 words; total transcript volume **883,872 words**.
- **Why interesting:** a simple structural explainer about what a parliamentary transcript actually looks like.
- **Caveats:** `speech` here is a transcript intervention, not necessarily a prepared standalone speech. Current production snapshot is only 10 debate days, so do not describe this as a timeless historical norm.
- **Production-ready:** **Yes for the current snapshot.**
- **Confidence:** **strong**.
- **Final-five competitiveness:** **Very high.**

### 3. Close Dáil divisions are uncommon in the current 2026 snapshot

- **Working title:** Only 3 of 56 recorded Dáil divisions were decided by 10 votes or fewer
- **Hook:** The closest recorded division in the current production window still had a nine-vote margin.
- **Visual:** histogram or dot plot of Tá–Níl margins, with <=10 highlighted.
- **Metric/relationship:** absolute difference between recorded Tá and Níl member-vote counts per division.
- **Period:** current 2026 division snapshot.
- **Denominator:** **56 divisions** with member-vote records.
- **Key result:** minimum margin **9**; median **14.5**; **0/56** within five votes; **3/56 (5.4%)** within ten votes.
- **Why interesting:** clear, intuitive and likely surprising without making a partisan claim.
- **Caveats:** margin is based on recorded Tá/Níl votes, not the full membership of the chamber; this snapshot is not a historical claim about all Dáil voting.
- **Production-ready:** **Yes for the current snapshot.**
- **Confidence:** **strong**.
- **Final-five competitiveness:** **Very high.**

### 4. Abstentions are rare among recorded division votes

- **Working title:** Recorded Dáil division votes are overwhelmingly Tá or Níl
- **Hook:** Of more than 8,000 recorded member-vote entries, only a few dozen are recorded as abstentions.
- **Visual:** 100% stacked bar: Tá / Níl / Staon.
- **Metric/relationship:** deterministic member-vote label counts.
- **Period:** current 2026 division snapshot.
- **Denominator:** **8,252 recorded member-vote rows** across 56 divisions.
- **Key result:** Tá **4,206 (51.0%)**; Níl **4,010 (48.6%)**; abstain/Staon **36 (0.44%)**.
- **Why interesting:** very clean visual and a useful explainer of what recorded division data contains.
- **Caveats:** an absent/non-voting TD is not the same thing as a recorded abstention; denominator is recorded vote entries, not all eligible members across all divisions.
- **Production-ready:** **Yes.**
- **Confidence:** **strong**.
- **Final-five competitiveness:** **High.**

### 5. How far have the Bills in the current production set progressed?

- **Working title:** From First Stage to Fifth: where 45 current tracked Bills have reached
- **Hook:** Almost all have reached Second Stage, while substantially fewer have reached Committee/Report/Fifth Stage so far.
- **Visual:** stage funnel using distinct Bills reaching each named stage.
- **Metric/relationship:** distinct `bill_id` by observed `stage_name`.
- **Period:** current production Bill snapshot: 45 Bills (2 from 2024, 7 from 2025, 36 from 2026).
- **Denominator:** **45 Public Bills** in the current production set.
- **Key result:** First Stage **45**; Second Stage **42 (93.3%)**; Committee Stage **29 (64.4%)**; Report Stage **27 (60.0%)**; Fifth Stage **27 (60.0%)**; Enacted stage recorded for **3**. Current status table shows **39 Bills are still Current**, so the latter figures are not final completion rates.
- **Why interesting:** strong civic explainer about the legislative path and easy to turn into a funnel graphic.
- **Caveats:** stage rows can occur in different Houses and the snapshot contains many still-active Bills. Do not frame later-stage counts as failure/dropout rates. `Cream List` is excluded from the public funnel because its semantics need separate explanation.
- **Production-ready:** **Yes with explicit snapshot wording.**
- **Confidence:** **strong**.
- **Final-five competitiveness:** **Very high.**

### 6. We can now link thousands of speeches directly to Bills without guessing

- **Working title:** 7,352 speeches are deterministically linked to 168 Bills in the certified production subset
- **Hook:** The new Bill-section bridge shows exactly which transcript sections belong to a Bill, without assigning an entire debate to legislation.
- **Visual:** simple relationship graphic: 168 Bills → 371 debate sections → 7,352 speeches + 168 divisions.
- **Metric/relationship:** production `bill_debate_sections` joined by exact `debate_section_id` to speeches/divisions.
- **Period:** current production source batch certified on 3 September 2026.
- **Denominator:** the conservative certified Bill-section subset: **168 Bills / 371 sections**.
- **Key result:** **7,352 speeches** and **168 divisions** link through **371 certified Bill debate sections** across **168 Bills**.
- **Why interesting:** a strong “how legislation actually appears in the chamber record” explainer and demonstrates a newly safe relationship in the pipeline.
- **Caveats:** this is deliberately a certified subset, not whole-Oireachtas legislation coverage; unresolved Seanad, committee and older historical coverage is excluded rather than guessed.
- **Production-ready:** **Yes.** The bridge was deployed and post-audited in production on 3 September 2026.
- **Confidence:** **strong**.
- **Final-five competitiveness:** **High.**

### 7. The transcript footprint of Leaders' Questions

- **Working title:** Leaders' Questions: 159 identifiable sections and 10,821 transcript interventions
- **Hook:** Leaders' Questions can be isolated cleanly from exact official section headings rather than broad text matching.
- **Visual:** big-number explainer with section count, transcript-intervention count and exact-heading methodology.
- **Metric/relationship:** exact certified Leaders' Questions source-heading allowlist.
- **Period:** broader 2025–2026 speech research history.
- **Denominator:** sections/speeches whose source heading exactly matches the two certified genuine Leaders' Questions headings.
- **Key result:** **10,821 speeches/interventions across 159 sections**; 10,702 under the main heading and 119 under the resumed heading.
- **Why interesting:** highly recognisable parliamentary format with a clean deterministic basis.
- **Caveats:** broader reusable `speech_context` is not yet productionised. A final production post should rerun the exact intended visual against the current certified source scope.
- **Production-ready:** **Deterministic rule is certified; reusable context metric is not fully productionised.**
- **Confidence:** **usable with caveat**.
- **Final-five competitiveness:** **High.**

### 8. Almost every Parliamentary Question is Written

- **Working title:** 97% of Parliamentary Questions are Written
- **Hook:** The chamber-visible Oral format is only a small fraction of submitted Parliamentary Questions.
- **Visual:** 100-dot or stacked bar: Written vs Oral.
- **Metric/relationship:** distinct submitted question records by certified question type.
- **Period:** current 2025–2026 PQ history used by the certified question research.
- **Denominator:** **121,355 submitted questions**.
- **Key result:** Written **117,695 (96.98%)**; Oral **3,660 (3.02%)**; about 32 Written questions per Oral question.
- **Why interesting:** extremely simple and corrects a likely public misconception about where PQ activity occurs.
- **Caveats:** volume/channel only; not scrutiny quality, effectiveness or answer quality.
- **Production-ready:** **Yes.**
- **Confidence:** **strong**.
- **Final-five competitiveness:** **High.**

### 9. Grouping Oral questions roughly doubles the exchange size

- **Working title:** Grouped Oral questions produce much larger chamber exchanges
- **Hook:** When several questions are grouped together, the typical exchange roughly doubles in both interventions and transcript words.
- **Visual:** two paired bars: single vs grouped median interventions and median words.
- **Metric/relationship:** certified Oral exchange `grouped_exchange`, intervention count and word count.
- **Period:** current 2025–2026 Oral-exchange history.
- **Denominator:** **2,127 Oral-question exchanges**: 1,865 single-question, 262 grouped.
- **Key result:** single median **6 interventions / 1,147 words**; grouped median **12 interventions / 2,275.5 words**.
- **Why interesting:** good “how the Dáil works” procedural explainer and visually immediate.
- **Caveats:** exchange size is not quality, importance, effectiveness or responsiveness.
- **Production-ready:** **Yes.**
- **Confidence:** **strong**.
- **Final-five competitiveness:** **High.**

### 10. Where the current tracked Bills began

- **Working title:** Most Bills in the current production snapshot originated in the Dáil
- **Hook:** The current tracked set contains Bills originating in both Houses, but the Dáil accounts for the large majority.
- **Visual:** 38 vs 7 split: Dáil Éireann / Seanad Éireann.
- **Metric/relationship:** distinct Bills by `origin_house_name`.
- **Period:** current production Bill snapshot.
- **Denominator:** **45 Public Bills**.
- **Key result:** **38 (84.4%)** originated in Dáil Éireann; **7 (15.6%)** in Seanad Éireann.
- **Why interesting:** simple civic-structure fact and a useful companion to the stage-funnel post.
- **Caveats:** this is the current production subset, not a historical rate for all Irish legislation. Do not imply that Seanad-origin Bills are less important.
- **Production-ready:** **Yes.**
- **Confidence:** **strong**.
- **Final-five competitiveness:** **Medium.**

## Ideas considered and rejected from the top ten

### Raw “most speeches by TD” leaderboard

The current metrics can rank speech/intervention counts, but office-holders, chair roles, speaking opportunities and differing coverage make a simple leaderboard easy to misread as performance or effectiveness. **Rejected.**

### Raw constituency speech ranking

The current constituency activity mart can rank speech counts, but representative counts and current-member coverage make raw totals unsuitable for a public “most active constituency” claim without a stronger exposure denominator. **Rejected.**

### Current member vote-participation leaderboard

The legacy 2025 member-profile output currently contains zero vote-participation values, while the newer 2026 gold member activity table has usable vote counts. That inconsistency is itself a reason not to publish a member attendance/participation ranking from the existing consumer layer. **Rejected.**

### Party voting-unity/performance ranking

Potentially interesting, but eligibility, absence, abstention, party-at-vote attribution and substantive vote context must all be explicit. Raw unity percentages are too easy to turn into partisan performance claims. **Rejected for now.**

### Bill sponsor leaderboard

There are 84 sponsor rows across 45 current Bills; 8 Bills have multiple sponsor rows and one has 21. However member sponsors and ministerial-office sponsors are different source entities. A leaderboard would require a more explicit interpretation layer. **Rejected from the top ten.**

### “Most debated Bill” ranking

The deployed Bill-section bridge makes this technically feasible for the certified subset, but whole-Oireachtas coverage is incomplete. A ranking can be reconsidered once the public denominator is explicitly limited to the certified coverage window. **Deferred.**

### Question-only department/member/party rankings

Technically available but intentionally deprioritised now that the content brief covers the full pipeline. The strongest two PQ structural explainers remain candidates 8 and 9; the rest do not outrank the broader speech/vote/legislation findings.

## Promising ideas that are not yet safe

### Party issue emphasis

Draft metrics already define party issue share and comparison with all TDs / average party. These could become excellent posts, but they depend on classifier quality gates and event-date party attribution and should be commissioned explicitly before public ranking.

### Bill-linked voting behaviour

The production Bill-section bridge now safely links **168 divisions** to Bill context. The next valuable step is to combine that with stage/proceeding context and safe vote denominators before making claims such as “closest Bill votes” or party behaviour on legislation.

### Most debated Bills

The production bridge links **7,352 speeches** to 168 Bills through 371 sections. A properly scoped “most discussed Bills in the certified dataset” ranking is now technically possible, but should first be computed with clear coverage wording and duplicate-safe section/speech aggregation.

### Cross-year speech trends

Current canonical speech production inspected here contains a short 2026 snapshot (10 debate days), while the issue-classified compatibility history is much broader. Cross-year comparisons should wait for one harmonised certified historical speech surface.

### Multi-sponsor Bills

Eight of 45 current Bills have more than one sponsor row and the maximum is 21. This is visually promising, but the member-vs-office sponsor distinction needs to be made explicit before publishing specific sponsor comparisons.

## Living next-step plan

1. Review these ten editorially with the user; do **not** choose the final five yet.
2. Prefer the strongest cross-domain mix rather than defaulting back to Parliamentary Questions.
3. For candidate 1, carry the issue-classifier caveat directly into the post copy and methodology slide.
4. For candidates 2–5 and 10, freeze the production snapshot date/coverage in the source metadata so numbers cannot silently drift during rendering.
5. For candidate 6, use only exact production `bill_debate_sections` joins; never whole-debate Bill attribution.
6. If candidate 7 survives, rerun the exact Leaders' Questions extraction against the intended publication scope before rendering.
7. Keep raw member/party performance-style rankings out unless exposure/eligibility denominators are explicitly certified.
8. Do not merge the temporary investigative workflow branch; the final research record belongs only in this documentation branch/PR.
