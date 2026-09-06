# Close-division research

Status: **Framework established; first three cases completed; two strongest cases deepened into post briefs**  
Date: **5 September 2026**

This directory is the durable research record for close Dáil divisions that may support public-facing EirePolitic content.

A close division is not automatically a meaningful political story. The research process therefore treats each qualifying division as a case file and separates the formal vote, surrounding debate, voting coalition, external reporting and editorial interpretation.

## Initial detection rule

For current research, flag a division when the absolute difference between recorded Tá and Níl votes is **10 votes or fewer**.

This is a discovery threshold, not a claim that every such vote was politically close in the same way. Procedural business, amendments, substantive Bill votes and motions must be distinguished.

## Required workflow for every case

1. **Detect deterministically**
   - division ID/date/chamber;
   - exact Tá/Níl/Staon totals;
   - absolute Tá–Níl margin;
   - recorded outcome;
   - linked debate/section and Bill where available.

2. **Reconstruct the exact proposition**
   - capture the precise `Question put` / `Amendment put` wording;
   - classify the vote as Bill stage, Bill amendment, motion amendment, substantive motion or procedural/Order-of-Business vote;
   - never infer the proposition from the section title alone.

3. **Build the debate packet**
   - read the immediately preceding debate and relevant earlier discussion;
   - identify proposer, Government response and principal arguments on each side;
   - distinguish arguments about substance from arguments about procedure/timing.

4. **Analyse the voting coalition**
   - retain exact member-level recorded votes;
   - resolve party at the event date where the membership history is complete;
   - keep recorded abstention separate from absence/non-voting;
   - never treat an unresolved party join as an Independent or political category;
   - do not infer motive from party alignment alone.

5. **Add external context**
   - prefer official Oireachtas/Bill material first;
   - then reputable contemporary reporting and directly relevant stakeholder material;
   - record what the outside source adds that the parliamentary transcript does not.

6. **Maintain a claim ledger**
   - `confirmed`: explicit in vote record, Bill text or transcript;
   - `externally_reported`: attributable to a named contemporary source;
   - `supported_interpretation`: synthesis supported by multiple sources;
   - `unresolved`: plausible but unsafe to state publicly.

7. **Editorial assessment**
   - substantive vs procedural;
   - public-interest level;
   - visual clarity;
   - sensitivity/caveats;
   - whether the evidence is strong enough for a post.

## First certification cases

- [9 June 2026 — Order of Business / EU Migration and Asylum Pact debate request](2026-06-09_order_of_business.md)
- [10 June 2026 — counselling-record disclosure amendment](2026-06-10_counselling_records.md)
- [8 July 2026 — Planning for Constitutional Change Bill 2026](2026-07-08_constitutional_change.md)
- [Instagram post briefs for the two strongest cases](post_briefs.md)

## Deep-dive result

The **Planning for Constitutional Change Bill** and **counselling-record disclosure amendment** both remain strong standalone content candidates after a deeper transcript and external-source pass.

The constitutional-change case is the easier first production candidate because the proposition is simple to visualise and the principal caveat is political framing: the 69–79 division was on a statutory planning process/timetable, not a direct vote on Irish unity.

The counselling-record case is equally substantive but more sensitive. The strongest framing is that both sides accepted the need for stronger privacy safeguards and disagreed over the legal threshold for exceptional disclosure and fair-trial protections.

## Party-at-vote data-quality finding

The member-vote surface contains exact member votes for both target divisions, but the raw `party_name_at_vote` field is blank. A research-only event-date join against `silver_member_parties` leaves a material unresolved set:

- constitutional-change vote: party resolved for **84/148**, unresolved for **64/148**;
- counselling-record amendment: party resolved for **86/146**, unresolved for **60/146**.

The resolved rows support the broad Government/opposition pattern visible in the debates and reporting, but precise EirePolitic party totals are **not yet certified for publication**. `Unknown` means attribution coverage is incomplete, not that those TDs lacked a party.

## Important correction from initial exploration

The 9 June 2026 close division should **not** be described as a vote on the Occupied Territories Bill. The exact amendment proposed by Matt Carthy sought to add statements and questions on the **EU Migration and Asylum Pact and the international protection system** to that week's Dáil schedule. Other scheduling disputes, including the Occupied Territories Bill, were raised in the same Order-of-Business exchange, but they were not the proposition being put in this division.

## Future automation boundary

Safe to automate later:

- detection of margin <= 10;
- metadata and exact division totals;
- linked debate/section/Bill lookup;
- extraction of nearby transcript text;
- generation of an unreviewed research packet.

Keep human/research review before publication for:

- identifying the true political dispute;
- deciding whether a procedural vote is editorially meaningful;
- explaining why the coalition formed;
- selecting and evaluating external reporting;
- assigning public-facing framing.

## Living next-step plan

1. Editorially review the two post briefs before any graphics are made.
2. Produce the **Planning for Constitutional Change Bill** post first if one close-vote case is selected for immediate production.
3. If the counselling-record case is selected, require an additional sensitivity/legal-copy check before final caption approval.
4. Do not add party vote charts until event-date party attribution coverage is complete enough to certify exact totals.
5. For future close votes, freeze the division ID, proposition, vote totals, source links and data snapshot in the post source record.
6. Keep the temporary analysis branch `analysis/close-divisions-deep-dive-20260905` unmerged; it contains research-only workflow changes.
