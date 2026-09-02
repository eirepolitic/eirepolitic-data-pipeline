# Parliamentary question-taking certification

Status: **Current follow-up investigation**  
Date: **2 September 2026**  
Parent research record: [Parliamentary questions investigation](parliamentary_questions_investigation.md)

This note records the follow-up investigation into whether the TD who actually takes an Oral Parliamentary Question in the chamber can be derived deterministically from transcript order and explicit substitution/proxy language.

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

This is strong evidence that transcript order is useful for identifying the person taking the question.

However, the remaining **11 exceptions** mean ordering alone is not yet safe enough to materialize `taken_by_member_code` universally.

### Explicit substitution/proxy-taking is demonstrable

The refined deterministic rules now recognize both English and Irish formulations, including examples equivalent to:

- "I am taking this question on behalf of Deputy ...";
- "I am covering for Deputy ...";
- "I ask this question on behalf of ...";
- chair announcements that a named TD is taking another TD's question;
- Irish chair language such as a TD `ag glacadh Ceist ... faoi choinne an Teachta ...`;
- Irish wording where a question `á tógáil ag an Teachta ...`.

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

## Current methodological decision

A production `question_taking_relationships` foundation is **promising but not yet approved**.

The currently certified safe rule is:

- If the recorded submitter is present, retain the official submitter relationship; do not create a substitute relationship solely from ordering.
- For a **single-question exchange**, if explicit substitution/proxy language is present and the first non-ministerial/non-chair member speaker is identifiable, that member may be recorded as a deterministic `taken_by_member_code` candidate.
- For grouped exchanges, do not assign one exchange-level candidate to every grouped question.
- For unresolved cases, retain `unknown`; absence of the submitter alone is not evidence of substitution.

## Evidence

Important runs for this follow-up:

- initial deterministic question-taking audit: **33578667504**;
- compact initial audit digest: **33578749629**;
- refined chair-role + Irish-language substitution audit: **33578804943**;
- refined compact digest: **33578865429**.

The refined audit corrected temporary-chair identification and expanded explicit substitution language. No production pipeline or production data was changed during this investigation.

## Revised next-steps plan

### 1. Investigate the 11 transcript-order exceptions

This is now the immediate next task.

For each of the 11 submitter-present exchanges where the first non-officeholder member speaker is not one of the recorded submitters:

- inspect transcript order;
- check whether the first speaker is another participating TD before the actual question taker;
- check section boundary/order issues;
- check temporary office/chair-role gaps;
- check grouped-question procedure;
- determine whether the ordering rule can be refined deterministically.

Target: determine whether the 99.42% rule can become effectively certified, or whether ordering must remain supporting evidence only.

### 2. Review the 156 unresolved-with-candidate exchanges

Prioritize deterministic patterns rather than manual one-by-one attribution.

Investigate:

- additional English proxy wording;
- additional Irish formulations;
- chair announcements not captured by current regexes;
- apology/absence formulations that imply another TD is taking the question;
- whether named substitute extraction can be performed from chair text;
- whether the first ordinary-member speaker is consistently the substitute in these cases.

Unknown must remain unknown where explicit evidence cannot be established.

### 3. Decide production foundation scope

If the first two tasks support it, design `question_taking_relationships` with fields such as:

- `question_id`;
- `debate_section_id`;
- `submitted_by_member_code`;
- `taken_by_member_code`;
- `is_substitute`;
- `relationship_status`;
- `evidence_method`;
- `evidence_speech_id` or section evidence reference;
- provenance/version fields.

Recommended status values would distinguish at least:

- `self_confirmed`;
- `substitute_explicit`;
- `unknown`;
- `procedural_or_interrupted`.

Do not expose an inferred substitute as fact without certified evidence.

### 4. Then return to exchange participant metrics

Once question-taking attribution is settled, continue the parent plan by certifying reusable exchange measures such as:

- participating submitter count/share;
- ordinary non-submitting TD participants;
- respondent/minister word share;
- chair intervention count;
- exchange word volume;
- grouped vs single-question exchange.

### 5. Question issue classification remains deferred

Nothing in this follow-up changes the earlier decision: do not classify the roughly 121k question history until deterministic structure has been exhausted and a concrete use case justifies the cost.
