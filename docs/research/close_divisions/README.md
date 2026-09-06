# Close-division research

Status: **Research framework established; first three cases completed**  
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
   - group recorded votes by event-date party/member attribution where reliable;
   - keep recorded abstention separate from absence/non-voting;
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

1. Use these three cases as the reference set for future close-division research.
2. Add deterministic party/member vote breakdowns to each case once the event-date attribution extraction is run in a reusable form.
3. For future production work, add a research-only close-division detector rather than changing existing voting denominators.
4. Revisit the threshold after a larger history exists; <=10 is a discovery rule, not a permanent editorial definition.
5. If a case becomes an Instagram post, freeze the exact division ID, proposition text, source links and snapshot date in the post's source record.
