# 8 July 2026 — Planning for Constitutional Change Bill 2026

## Formal vote

- **Type:** substantive Private Members' Bill, Second Stage (resumed).
- **Division:** `vote_184`.
- **Result:** Tá 69; Níl 79; Staon 0.
- **Margin:** 10.
- **Outcome:** Bill defeated at Second Stage.

## What was being voted on

The Bill proposed a statutory planning process for possible constitutional change / Irish reunification. Its central mechanisms were:

- requiring the Taoiseach to prepare and publish a Green Paper on reunification within a defined timeframe;
- broad consultation, including political parties, civil society, experts and unionist/Protestant communities;
- an all-island Citizens' Assembly following the Green Paper;
- periodic reporting to the Oireachtas.

This was a direct Second Stage vote on whether the Bill should proceed. It was **not** a referendum on Irish unity itself.

## Core political dispute

Sinn Féin presented the Bill as a way to move Irish-unity planning from aspiration into an organised Government process. Mary Lou McDonald argued that constitutional change should be prepared for in advance and that the Green Paper and Citizens' Assembly would force practical questions into a structured public process.

The Government did not frame its opposition as opposition to Irish unity in principle. Taoiseach Micheál Martin argued that the statutory timetable and mechanism were not credible and that a Citizens' Assembly was not the appropriate way to conduct the preparatory work. Government messaging instead emphasised the Good Friday Agreement, reconciliation, Shared Island work and building consent across communities.

Contemporary reporting therefore describes a substantial difference over **timing, mechanism and who should lead formal preparation**, rather than a simple binary split between people who favour or oppose a united Ireland.

## Key speakers and positions

### Mary Lou McDonald — Sinn Féin

McDonald introduced the Bill as a formal planning mechanism. Her case was that nearly three decades after the Good Friday Agreement, Government should begin detailed preparation rather than wait until a referendum is imminent. The Bill would require a Green Paper and then an all-island Citizens' Assembly.

### Micheál Martin — Taoiseach, Fianna Fáil

Martin rejected the proposal as the wrong approach. Contemporary reporting records him describing the timetable as not credible and the Citizens' Assembly mechanism as unsuitable for the scale of the work. His argument was that building consent, reconciliation and practical cooperation should precede a statutory unity-planning timetable.

### Fine Gael / Government position

Fine Gael also opposed the Bill as part of the Government position. External reporting characterised the Government and Sinn Féin as operating on substantially different timetables for unity planning rather than disagreeing on whether constitutional change is a legitimate future possibility.

## Voting coalition

### Exact data

The member-vote table contains **148 exact recorded member votes** for this division: 69 Tá and 79 Níl. These member-level votes are safe to use.

### Party-at-vote limitation discovered during deep dive

The raw Oireachtas member-vote rows have blank `party_name_at_vote` values for this division. A research-only event-date join against `silver_member_parties` resolved party labels for **84 of 148 rows** and left **64 unresolved** because the membership-history surface is incomplete for those members.

Among the rows that resolve, the directional coalition is clear:

- mapped Fianna Fáil and Fine Gael voters are on the **Níl** side;
- mapped Sinn Féin, Social Democrats, Labour, Independent Ireland, People Before Profit-Solidarity and Aontú voters are on the **Tá** side;
- mapped Independent voters split across both sides.

The Irish Times subsequently reported that the Bill failed to win support from Fianna Fáil and Fine Gael and described the overall split as Government versus Sinn Féin/opposition on the timetable and mechanism. However, because EirePolitic's event-date party join still leaves 64 rows unresolved, **precise party totals should not yet be published from our derived party layer**.

The unresolved rows include recognisable party members, so `Unknown` is a data-coverage category, not a political category.

## External context

### Sinn Féin case for the Bill

Sinn Féin's public presentation said the Bill would put formal planning for reunification at the centre of Government by requiring a Green Paper and an all-island Citizens' Assembly.

### Government case against the Bill

The Irish Times reported Martin arguing that the Bill would amount to an ineffective gesture, that the proposed deadline was not credible and that a Citizens' Assembly was not the correct mechanism. Government statements instead emphasised the Good Friday Agreement, Shared Island work and reconciliation.

### Contemporary political interpretation

The Irish Times' later Inside Politics discussion described Sinn Féin and Government as operating on completely different timetables for unity. That is a more accurate summary of the political disagreement than saying 69 TDs were “for unity” and 79 “against unity”.

## Why the vote was close

### Confirmed

- The Bill lost by only ten votes, 69–79.
- It was a substantive Second Stage vote, not a procedural scheduling division.
- Sinn Féin sponsored the Bill.
- The proposition was whether Government should be legally required to begin a defined Green Paper/Citizens' Assembly process on a statutory timetable.
- The recorded member coalition closely resembles Government versus opposition, with some independent variation.

### Supported interpretation

The narrow margin reflects a meaningful parliamentary split over **whether formal state-led preparation for reunification should begin now under a statutory timetable**. The principal Government objection centred on the proposed process and timing rather than rejection of constitutional change as a legitimate future question.

## Editorial assessment

- **Public interest:** very high.
- **Substantive significance:** very high.
- **Visual clarity:** very high.
- **Sensitivity/caveat:** high political salience; wording must describe the actual proposition.
- **Post potential:** **very high**.

Best framing: **“A Bill requiring formal planning for constitutional change was defeated 69–79. What exactly did it propose, and why did Government oppose it?”**

### Do not say

- “69 TDs voted for a united Ireland and 79 voted against.”
- “The Dáil rejected Irish unity.”
- “The Government opposes Irish unity” based on this division alone.
- precise party totals until event-date party attribution coverage is certified.

## Sources

- Houses of the Oireachtas, Planning for Constitutional Change Bill 2026 / debate record.
- ContactYourTD vote record: https://www.contactyourtd.ie/debates/votes/782/planning-for-constitutional-change-bill-2026-second-stage-resumed-private-members
- The Irish Times, 7 July 2026: https://www.irishtimes.com/politics/2026/07/07/government-should-get-behind-border-poll-bill-says-sinn-fein/
- The Irish Times, 10 July 2026, Inside Politics: https://www.irishtimes.com/podcasts/inside-politics/sinn-fein-and-government-on-completely-different-timetables-for-irish-unity/
- Sinn Féin, Mary Lou McDonald speech, 7 July 2026: https://sinnfein.ie/news/reunification-a-new-deal-for-the-people-of-ireland-mary-lou-mcdonald/

## Research provenance

The exact vote totals/member rows come from the EirePolitic production `silver_divisions` / `silver_member_votes` surfaces. The incomplete party-at-vote diagnostic was run read-only on isolated branch `analysis/close-divisions-deep-dive-20260905`; that branch is not to be merged into production.
