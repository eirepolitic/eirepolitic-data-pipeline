# Instagram speech-metrics carousel follow-up

Status: **Promising candidate; member coverage needs one certification step before publication**  
Date: **6 September 2026**

This refines candidate 2 in `instagram_post_candidates.md` from a generic speech-length explainer into a multi-slide carousel about who speaks most, longest and most often in the current Dáil transcript history.

## Proposed post

**Working title:** Who talks most in the Dáil?

Potential slides:

1. most transcript interventions;
2. most total words spoken;
3. longest average intervention, with a minimum-speech threshold;
4. longest single recorded intervention;
5. most debate days with at least one intervention;
6. least transcript interventions, but only after a fair tenure/coverage denominator is certified.

Do not use issue labels in this post.

## Deterministic probe

The full-session compatibility speech history currently spans **18 December 2024 to 26 February 2026**, covering **116 debate days** and **47,275 transcript rows**.

The compatibility table contains speaker names but not member IDs. A temporary read-only analysis therefore normalized speaker names and matched them to `gold_current_members`.

That match produced **25,822 transcript interventions across 96 matched TD names**.

Illustrative results from that matched subset:

- **Most transcript interventions:** Micheál Martin — **4,424**; Simon Harris — **2,074**; Pearse Doherty — **1,018**.
- **Most total words:** Micheál Martin — **440,599**; Simon Harris — **392,995**; Dara Calleary — **153,097**.
- **Longest average intervention, minimum 20 interventions:** Ciarán Ahern — **562 words**; Cormac Devlin — **517**; Emer Higgins — **411**.
- **Longest single matched intervention:** Jennifer Carroll MacNeill — **4,517 words**, on 26 February 2025 in `Future of Healthcare for Longer, Healthier Lives: Statements`.
- **Most speaking days:** Michael Collins — **102 of 116 covered debate days**; Ruth Coppinger — **98**; Pearse Doherty and Paul Gogarty — **95** each.

These are useful diagnostics, not yet publication-certified leaderboards.

## Main caveat discovered

`gold_current_members` currently contains only **98 rows**, and the name match covers 96 of them. That is not a sufficient reference surface for a definitive whole-Dáil "most/least talkative TD" leaderboard.

The low end is especially unsafe: a member can appear to have few speeches because of incomplete reference coverage, tenure, office changes, transcript-name variation or time outside the covered period.

Therefore:

- do **not** publish "least talkative TD" yet;
- do **not** describe the current top-end diagnostic as a complete all-TD ranking yet;
- do not treat intervention counts or word totals as effectiveness, performance or quality;
- preserve the distinction between a transcript intervention and a prepared speech.

## What is safe now

The overall carousel concept is strong and should remain in the shortlist.

The metrics themselves are deterministic and visually useful. The remaining work is identity/eligibility certification, not classifier work.

## Living next-step plan

1. Build or identify the complete current-Dáil member reference for the full covered period, using event-date membership where possible.
2. Match every transcript speaker to member IDs and quantify unmatched/ambiguous names.
3. Recalculate the six proposed carousel metrics on the certified TD population.
4. For average speech length, retain a minimum-intervention threshold so tiny samples cannot top the ranking.
5. For the low-end ranking, require full-period eligibility or normalize by eligible debate days; otherwise omit the "least talkative" slide.
6. Once those checks pass, replace candidate 2 in the main Instagram shortlist with this carousel concept and freeze the publication snapshot.

No production data or architecture was changed. Temporary diagnostics remain isolated on the unmerged analysis branch.
