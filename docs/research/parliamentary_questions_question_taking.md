# Parliamentary question-taking certification

Status: **Current follow-up investigation**  
Date: **2 September 2026**  
Parent research record: [Parliamentary questions investigation](parliamentary_questions_investigation.md)

This note records the follow-up investigation into whether the TD who actually takes an Oral Parliamentary Question in the chamber can be derived deterministically from transcript order, explicit substitution/proxy language, and question-to-speech text matching.

## Confirmed findings

The production snapshot contains:

- **2,127 oral-question exchanges**;
- **3,660 Oral question records**.

### Submitter normally appears in the exchange

In **1,906 exchanges (89.6%)**, at least one recorded submitter appears under their own member code in the related transcript.

At question-record level, **3,410 of 3,660 Oral questions (93.2%)** belong to exchanges where a submitter is present in the transcript.

### Transcript ordering is highly reliable, but not perfect

After excluding ministers and chair/procedural speakers using both dated office history and transcript role labels, the **first non-officeholder member speaker** was one of the recorded question submitters in:

- **1,890 of 1,901 eligible submitter-present exchanges**;
- **99.42%**.

This is strong evidence that transcript order is useful supporting evidence for identifying the person taking a question.

However, the remaining **11 exceptions** show that ordering alone is not safe enough to materialize `taken_by_member_code` universally.

### Explicit substitution/proxy-taking is demonstrable

The refined deterministic rules now recognize both English and Irish formulations, including examples equivalent to:

- "I am taking this question on behalf of Deputy ...";
- "I am covering for Deputy ...";
- "I ask this question on behalf of ...";
- chair announcements that a named TD is taking another TD's question;
- Irish chair language such as a TD `ag glacadh Ceist ... faoi choinne an Teachta ...`;
- Irish wording where a question is `á tógáil ag an Teachta ...`.

The refined audit found:

- **59 exchanges (2.77%)** with explicit substitution/proxy evidence and an identifiable ordinary-member candidate;
- **65 Oral question records (1.78%)** inside those exchanges.

Because grouped exchanges can contain several submitted questions, an exchange-level substitute candidate must not automatically be assigned to every question in that section.

### Safe question-level relationships

Using a deliberately strict rule:

> single-question exchange + explicit substitution/proxy language + identifiable first non-ministerial/non-chair member speaker

we can safely derive **57 question-level substitute relationships**.

Examples include:

- Naoise Ó Cearúil's question taken by Martin Daly after an Irish chair announcement;
- Pádraig O'Sullivan's question taken by John Connolly;
- Cathal Crowe's question taken by John Connolly;
- Naoise Ó Cearúil's question taken by Aindrias Moynihan;
- Brendan Smith's question taken by Shane Moynihan;
- Ciarán Ahern's question taken by Conor Sheehan;
- Mark Wall's question taken by Eoghan Kenny;
- Roderic O'Gorman's question taken by Paul Murphy;
- Ryan O'Meara's question taken by Brendan Smith;
- Darren O'Rourke's question taken by Mark Ward.

These are examples of a real parliamentary mechanism rather than anomalous data.

## Remaining unresolved structure

After the refined rules:

### Exchange-level status

| Status | Exchanges | Share |
| --- | ---: | ---: |
| submitter present | 1,906 | 89.61% |
| unresolved with ordinary-member candidate | 156 | 7.33% |
| explicit substitute candidate | 59 | 2.77% |
| unresolved with no ordinary-member candidate | 5 | 0.24% |
| procedural/interrupted | 1 | 0.05% |

### Question-record status

| Status | Oral question records | Share |
| --- | ---: | ---: |
| submitter present | 3,410 | 93.17% |
| unresolved with ordinary-member candidate | 168 | 4.59% |
| explicit substitute candidate | 65 | 1.78% |
| procedural/interrupted | 11 | 0.30% |
| unresolved with no ordinary-member candidate | 6 | 0.16% |

The unresolved-with-candidate group should **not** be treated as erroneous. Manual examples strongly suggest that many are also substitutions where the transcript does not contain one of the currently certified phrases.

Examples include cases where a different TD clearly conducts the entire exchange while the official submitted question belongs to another TD.

## Role-detection correction made during this investigation

An earlier pass incorrectly treated some acting chairs as ordinary members because dated office history did not always identify the temporary chair role.

The refined method also checks transcript speaker labels for:

- Ceann Comhairle;
- Leas-Cheann Comhairle;
- Cathaoirleach;
- Cathaoirleach Gníomhach;
- Acting Chairman / Acting Chairperson.

This improved the validation of the transcript-order rule from approximately **98.3% to 99.4%**.

This is another reason source transcript role labels should supplement office history when classifying procedural speakers.

## Investigation of the 11 transcript-order exceptions

All 11 exceptions were inspected individually, including a raw-XML check of the first speech in each affected debate section.

### Raw XML result

The parsed first-speaker attribution matched the Oireachtas XML in all 11 cases examined.

This rules out the suspected general speaker/text alignment bug in `silver_speeches` for these cases.

Two apparently suspicious grouped-answer introductions were correctly attributed in the raw XML to the Taoiseach. The earlier compact exception digest had shown the **first ordinary-member candidate** separately from the **first speech text**, which made those rows look mismatched. The underlying production speech records were correct.

### What the 11 exceptions actually represent

The exceptions break down structurally as:

- **8 grouped-question exchanges** where an ordinary TD who was not one of the recorded submitters spoke before the first recorded submitter;
- **2 single-question exchanges with explicit substitute-taking**, where the substitute spoke first even though the original submitter later appeared in the same exchange;
- **1 single-question exchange with a likely substitute/alternate taker but no currently certified explicit proxy phrase**.

Representative cases include:

- `Middle East`, 13 February 2025: Pearse Doherty spoke before either of the two recorded submitters; Pa Daly appeared shortly afterwards.
- `Military Neutrality`, 26 February 2025: Richard Boyd Barrett spoke before the recorded submitters in a four-question grouped exchange.
- `Cabinet Committees`, 19 March 2025: the Taoiseach opened the grouped answer, then Paul Lawless spoke before the first recorded submitter.
- `Financial Instruments`, 3 April 2025: Paul Murphy spoke before the recorded submitters in a grouped exchange.
- `Cabinet Committees`, 24 June 2025: the Taoiseach opened the grouped answer, then Marie Sherlock spoke before the first recorded submitter.
- `Fishing Industry`, 3 July 2025: Conor D. McGuinness explicitly stated that he was asking the question on behalf of Pádraig Mac Lochlainn; Mac Lochlainn later spoke in the same exchange.
- `Just Transition`, 18 December 2025: the Leas-Cheann Comhairle announced the grouped questions, and Réada Cronin participated well before the first recorded submitter.
- `Job Creation`, 13 January 2026: Erin McGreehan opened a single-question exchange whose official submitted question belonged to Aisling Dempsey; the current rules do not contain explicit enough substitution evidence to certify the relationship.
- `General Practitioner Services`, 5 March 2026: Martin Daly spoke before the two recorded submitters in a grouped exchange.
- `Disease Management`, 5 March 2026: Martin Daly explicitly said he was taking the question on behalf of Deputy Byrne; the original submitter later appeared.
- `Trade Agreements`, 21 May 2026: Malcolm Byrne spoke before the recorded submitters in a six-question grouped exchange.

### Consequence for the ordering rule

The 99.42% figure remains useful as a **validation statistic**, but the first ordinary-member speaker cannot be treated as the question taker by default.

There are two reasons:

1. grouped oral-question exchanges can contain other TD participants before any recorded submitter speaks;
2. an explicit substitute can take the question first even when the original submitter later participates.

Therefore:

> `submitter_present` is not equivalent to `submitter_took_question`, and `first ordinary speaker` is not equivalent to `taken_by_member_code`.

Transcript order should remain supporting evidence only unless combined with stronger question-level evidence.

## Question-text to opening-speech matching

A second deterministic signal was tested on **single-question oral exchanges only**.

The idea was to compare the official submitted `question_text` with the first non-ministerial/non-chair member speech and determine whether a sufficiently strong text match could certify who actually took the question.

### Scope

The audit found **1,856 single-question exchanges** with an identifiable ordinary-member opening speaker:

- **1,649** where the opening ordinary-member speaker was the official submitter;
- **207** where the opening ordinary-member speaker was not the official submitter.

A follow-up benchmark separated the non-submitter openings into:

- **58 explicit substitute cases** under the expanded proxy-language rules;
- **149 unresolved non-submitter-first cases**.

### Main result: text similarity does not distinguish self from substitute

The official question wording is often repeated or closely paraphrased by whoever takes the question in the chamber.

That means a substitute can match the official question **more closely** than the original submitter does in many self-taken cases.

Median normalized question-token coverage:

- self-first cases: **35.0%**;
- explicit substitutes: **52.6%**;
- unresolved non-submitter-first cases: **35.3%**.

Median token-set similarity:

- self-first cases: **63.1**;
- explicit substitutes: **74.9**;
- unresolved non-submitter-first cases: **63.2**.

This is the opposite of what would be needed for a safe rule that distinguishes submitters from substitutes.

### Threshold tests do not provide a safe separator

For example, requiring at least **60% question-token coverage and 80 token-set similarity** passed:

- about **10.1%** of self-first cases;
- about **34.5%** of explicit substitute cases;
- about **14.1%** of unresolved non-submitter-first cases.

At the stricter **75% coverage / 90 token-set** threshold, it still passed:

- about **2.1%** of self-first cases;
- about **17.2%** of explicit substitutes;
- about **2.7%** of unresolved non-submitter-first cases.

These results show that high similarity can support the proposition that a speaker is **voicing the formal question**, but cannot establish whether they are the official submitter or a substitute.

### Low text similarity also does not rule out a real taker

Many known self-taken and explicit substitute cases have low similarity because the speaker:

- begins with acknowledgements or procedural remarks;
- paraphrases rather than repeats the official wording;
- expands immediately into political/contextual argument;
- speaks in Irish while the question record is in English;
- starts with a short intervention before later stating the substance;
- receives or responds to a procedural interjection before the substantive wording appears.

Examples of explicit substitutes with low opening-text overlap included cases where the first speech merely said that the TD was taking the question on behalf of another member, with the substance appearing later.

### High-match unresolved cases are useful review candidates

The unresolved non-submitter-first set contains many cases where the opening TD speech closely reproduces the official question despite lacking a currently certified proxy phrase.

Examples include cases where:

- Joe Neville closely voiced a Barry Ward question;
- John Connolly closely voiced a Tony McCormack question;
- Louis O'Hara closely voiced a Pa Daly question;
- Malcolm Byrne closely voiced a Naoise Ó Cearúil question;
- Thomas Gould closely voiced a Mark Ward question;
- Sorca Clarke closely voiced a David Cullinane question;
- Cathy Bennett closely voiced another David Cullinane question.

These are strong candidates for further deterministic substitution-pattern research, but the text match itself is not enough to publish `taken_by_member_code` as fact.

### Decision on text matching

Question-to-opening-speech similarity should be treated as:

- **review prioritization evidence**;
- a potential feature in a deterministic evidence bundle;
- a way to identify likely cases where a non-submitting TD voiced the official question.

It should **not** be treated as:

- proof that the speaker was the official submitter;
- proof that the speaker was an authorized substitute;
- a standalone production `taken_by_member_code` rule.

The proposed `substitute_text_match` status is therefore **not certified** and should not be added to production at this stage.

## Current methodological decision

A production `question_taking_relationships` foundation is **promising but not yet approved**.

The currently certified safe rule is:

- Do not infer `taken_by_member_code` from submitter presence alone.
- Do not infer `taken_by_member_code` from first ordinary-member transcript position alone.
- Do not infer `taken_by_member_code` from question-text similarity alone.
- For a **single-question exchange**, if explicit substitution/proxy language is present and the first non-ministerial/non-chair member speaker is identifiable, that member may be recorded as a deterministic `taken_by_member_code` candidate.
- For grouped exchanges, do not assign one exchange-level participant to every grouped question.
- For unresolved cases, retain `unknown`; absence of the submitter or high textual similarity alone is not evidence of certified substitution.

## Evidence

Important runs for this follow-up:

- initial deterministic question-taking audit: **33578667504**;
- compact initial audit digest: **33578749629**;
- refined chair-role + Irish-language substitution audit: **33578804943**;
- refined compact digest: **33578865429**;
- 11-case transcript-order exception extraction: **33579196132**;
- compact exception digest: **33579257409**;
- raw XML speaker-attribution check: **33579429125**;
- single-question text-match audit: **33579960048**;
- explicit-substitute/unresolved text-match benchmark: **33580045161**.

No production pipeline or production data was changed during this investigation.

## Revised next-steps plan

### 1. Use high-match unresolved cases to discover additional deterministic substitution wording

This is now the immediate next task.

The text-match experiment should not become an attribution rule, but it gives us an efficient way to prioritize the unresolved cases most likely to contain real substitution/proxy-taking.

Investigate the highest-match unresolved non-submitter-first cases for:

- additional English formulations such as "my question is..." when another TD is named elsewhere in the section;
- wording such as "I am covering the questions for..." not captured by current patterns;
- chair announcements before the member's first substantive speech;
- Irish substitution announcements outside the current regex set;
- references to the named submitter elsewhere in the first few interventions;
- whether the substitute's identity can be extracted from explicit chair text rather than inferred from order.

Goal: expand **explicit-evidence coverage**, not create a similarity-based inference rule.

### 2. Separate unresolved single-question and grouped-question cases

The remaining unresolved population should be split structurally.

For single-question exchanges:

- explicit substitution wording may allow safe question-level attribution;
- high text similarity can prioritize review;
- unknown remains unknown without explicit evidence.

For grouped exchanges:

- question-level taker attribution may be inherently ambiguous;
- focus on exchange-level participation unless the transcript explicitly maps individual questions to speakers.

### 3. Reassess whether a production foundation is worth building

After the additional deterministic wording pass, decide whether the certified coverage is high enough to justify `question_taking_relationships`.

A possible foundation would include:

- `question_id`;
- `debate_section_id`;
- `submitted_by_member_code`;
- `taken_by_member_code` when certified;
- `is_substitute`;
- `relationship_status`;
- `evidence_method`;
- `evidence_speech_id` or section evidence reference;
- provenance/version fields.

Recommended status values remain conservative:

- `substitute_explicit`;
- `unknown`;
- `procedural_or_interrupted`.

A separate `self_confirmed` status should only be introduced if a robust rule for actual self-taking, rather than mere submitter presence, is certified.

Do **not** add `substitute_text_match` as a factual status based on the current evidence.

### 4. Then return to exchange participant metrics

Once question-taking attribution is settled or deliberately bounded, continue the parent plan by certifying reusable exchange measures such as:

- participating submitter count/share;
- ordinary non-submitting TD participants;
- respondent/minister word share;
- chair intervention count;
- exchange word volume;
- grouped vs single-question exchange.

### 5. Question issue classification remains deferred

Nothing in this follow-up changes the earlier decision: do not classify the roughly 121k question history until deterministic structure has been exhausted and a concrete use case justifies the cost.
