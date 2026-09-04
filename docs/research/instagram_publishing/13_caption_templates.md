# Step 13 — Caption Templates and Explicit Final Caption Storage

Status: **complete**

Research date: 2026-09-04

Scope: design a reusable caption-template/default system that keeps recurring Eirepolitic post formats consistent while preserving conversational editing through High Director. The exact final caption must always be stored and approved before publication.

No caption-generation implementation, template files, publication code, account connection, or live post was created.

---

## Short conclusion

Eirepolitic should use **versioned caption templates as drafting defaults, not as publication-time instructions**.

Recommended flow:

```text
post type / prior publication
      ↓
versioned caption template + defaults
      ↓
High Director conversationally adapts wording
      ↓
exact proposed caption
      ↓
human approval
      ↓
exact final caption stored in PublicationRequest
      ↓
publisher sends stored text unchanged
```

The existing repo already follows the most important safety principle: `instagram_build_copy_pack.py` generates an explicit caption string and writes it to files/manifest data. That should be preserved.

What should change later is **where reusable defaults live**. Today the Member Profile caption format and five default hashtags are embedded directly in Python. Recurring post types should instead reference a versioned template/configuration file so wording conventions can evolve without changing publishing execution code.

---

# 1. Existing repo behaviour

`process/instagram_build_copy_pack.py` currently contains:

- a hard-coded `DEFAULT_HASHTAGS` list;
- a deterministic `build_caption()` function;
- a deterministic `build_alt_text()` function;
- explicit `.caption.txt` files;
- explicit caption text stored in the copy manifest/CSV.

Current default hashtags are:

```text
#EirePolitic
#IrishPolitics
#DailEireann
#Oireachtas
#DataPolitics
```

The current Member Profile caption structure includes:

- identity/party/constituency;
- top debate issue;
- vote participation;
- speech activity/rank;
- source/review line;
- hashtags.

This is a useful deterministic prototype.

### What should be preserved

Keep:

```text
caption is explicit data
```

rather than:

```text
caption is regenerated later from instructions
```

The publication worker should never execute `build_caption()` or invoke an LLM at publication time.

---

# 2. What a caption template is

A caption template should represent **editorial defaults and structure** for a recurring post type.

It may define:

- default opening sentence;
- section/order conventions;
- standard source attribution;
- standard disclaimer;
- default hashtags;
- optional hashtags;
- standard account mentions;
- link-language conventions;
- period formatting;
- optional call-to-action wording;
- optional notes for High Director.

It should not contain:

- Meta access tokens;
- schedule state;
- provider IDs;
- temporary container IDs;
- final publication approval;
- runtime instructions.

---

# 3. Proposed versioned template structure

Conceptual example only:

```yaml
template_id: party_speech_breakdown
version: 1

platform_defaults:
  instagram:
    opening:
      default: "What did the parties focus on in the Dáil this month?"

    attribution:
      default: "Source: Houses of the Oireachtas data analysed by EirePolitic."

    disclaimer:
      enabled: true
      text: "Figures reflect the published dataset and classification methodology."

    hashtags:
      default:
        - "#EirePolitic"
        - "#IrishPolitics"
        - "#DailEireann"
      optional:
        - "#Oireachtas"
        - "#DataPolitics"

    mentions:
      default: []

    period:
      format: "%B %Y"

    editorial_notes:
      - "Lead with the most notable finding if one is clearly supported by the data."
      - "Do not imply causation that is not established by the source data."
```

This is a drafting/default definition, not the final post.

---

# 4. Template files should be version-controlled

Recommended future location:

```text
instagram/caption_templates/
```

Example:

```text
instagram/caption_templates/
  party_speech_breakdown.yml
  member_profile.yml
```

Each file should contain an explicit version number.

### Why versioning matters

If the standard attribution changes in October, High Director should still be able to answer:

```text
"Use the same caption structure as the August post."
```

without silently applying a new November disclaimer/template.

The historical `PublicationRequest` should record:

```yaml
template_ref:
  template_id: party_speech_breakdown
  template_version: 3
```

but the **stored final caption remains authoritative**.

---

# 5. Template versus final caption

The template is provenance.

The final caption is publication content.

Recommended record:

```yaml
caption:
  text: |
    This is the complete exact caption that will be sent to Instagram.

  template_ref:
    template_id: party_speech_breakdown
    template_version: 3
```

The publisher sends only:

```text
caption.text
```

It does not reopen the template and reconstruct the caption.

This prevents a scheduled post from changing because a template file changed after approval.

---

# 6. Conversational modification

Desired request:

```text
"Use the normal Party Speech Breakdown caption but mention that August had unusually low Dáil activity."
```

High Director should:

1. load the relevant template version/defaults;
2. load the approved August data/context;
3. produce a proposed complete caption;
4. ensure the additional statement is supported by the available data;
5. show/edit the proposed text conversationally;
6. store the final exact caption in the publication request;
7. require publication approval according to Step 11.

The phrase:

```text
"use the normal caption"
```

should resolve to a deterministic template reference, not rely solely on conversational memory.

---

# 7. Reusing a previous caption structure

Desired request:

```text
"Use the same caption structure as last month's Party Speech Breakdown."
```

Recommended lookup order:

```text
1. resolve post type/project
2. identify previous published publication
3. load its PublicationRequest
4. read its template_ref
5. use prior final caption as structural reference where helpful
6. generate current proposed caption from current approved data
```

Do not copy old facts, periods, mentions or hashtags blindly.

The previous publication is a **style/structure reference**, not the source of current factual content.

---

# 8. Default hashtags

The repo currently hard-codes five hashtags globally for the Member Profile copy builder.

That works for a prototype but is too coarse for a broader publishing system.

Recommended model:

```yaml
hashtags:
  standard:
    - "#EirePolitic"
    - "#IrishPolitics"

  post_type_default:
    - "#DailEireann"

  optional:
    - "#Oireachtas"
    - "#DataPolitics"
```

High Director can then conversationally decide:

```text
"Use the usual hashtags but leave out #DataPolitics."
```

The resulting exact caption still contains the final chosen hashtag text before approval.

### Do not maintain two contradictory sources of truth

If hashtags are stored structurally for audit/validation, they should match the exact caption text.

The publisher should not append a separate live hashtag list at execution time.

---

# 9. Mentions

Templates may define standard optional account mentions, but mentions should never be inserted automatically without being visible in the final caption.

Recommended example:

```yaml
mentions:
  defaults: []
  suggested:
    - username: houses_of_the_oireachtas_example
      reason: source attribution where editorially appropriate
```

A suggested mention is not the same as an approved mention.

High Director should only include it in the final caption after the publication request reflects that decision.

---

# 10. Attribution and disclaimers

Recurring Eirepolitic formats should keep standard attribution/disclaimer language in templates rather than duplicating it across Python functions.

Benefits:

- consistent wording;
- easier editorial review;
- explicit history/versioning;
- easier updates without changing execution code;
- High Director can explain which standard language was used.

Example:

```yaml
attribution:
  required: true
  text: "Source: Houses of the Oireachtas data analysed by EirePolitic."
```

If required text is removed during conversational editing, validation can flag it before approval.

---

# 11. Required versus optional template components

Templates should distinguish:

```text
required
optional
suggested/default
```

Example:

```yaml
components:
  attribution:
    required: true

  disclaimer:
    required: true

  hashtags:
    required: false

  call_to_action:
    required: false
```

This prevents High Director from accidentally removing a mandatory source/disclaimer while allowing natural variation in optional wording.

---

# 12. Template variables

If template variables are used, keep them narrow and explicit.

Example:

```yaml
variables:
  period_label:
    source: publication.period
  project_display_name:
    source: project metadata
```

Avoid building a complex mini programming language for captions.

High Director is the conversational composition layer; the template only supplies safe defaults/structure.

---

# 13. Deterministic validation after conversational drafting

Before the caption can enter final publication approval, deterministic checks should verify things such as:

```text
caption is non-empty
required attribution exists
required disclaimer exists
structured mentions match caption text
structured hashtags match caption text
caption references the correct period/project where applicable
no unresolved template placeholders remain
no accidental "Review before publishing" draft-only text remains
```

The last point is relevant to the current prototype: `instagram_build_copy_pack.py` includes:

```text
Source: Oireachtas data pipeline. Review before publishing.
```

That is appropriate draft/review language but may not be appropriate in a final public caption.

The final template system should distinguish internal review notices from public attribution text.

---

# 14. Current Meta API consideration

Meta's current Content Publishing API accepts caption text as a media/container parameter for supported feed/Reel publication flows.

Source:

- Meta current Instagram Content Publishing documentation: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-ab559ffb-8e2c-4b0a-b43a-5737b6d2f672

The current official high-level guide reviewed here does not expose a single caption-limit rule that should be hard-coded into this research design for every post type/API route.

Therefore the eventual platform adapter should validate caption constraints against the current selected Graph API version rather than embedding an unverified historical Instagram limit into the template system.

Templates should focus on editorial structure; platform adapters enforce current technical limits.

---

# 15. Caption status lifecycle

Recommended conceptual lifecycle:

```text
template defaults
      ↓
draft caption
      ↓
conversational edits
      ↓
final proposed caption
      ↓
PublicationRequest version
      ↓
human approval
      ↓
immutable exact caption
      ↓
publisher
```

Once approved, the caption does not change in place.

If edited:

```text
publication version N
      ↓
caption changed
      ↓
publication version N+1
      ↓
pending_publication_approval
```

This follows Step 11's approval model.

---

# 16. Do not regenerate at publication time

The publisher worker must not:

- call an LLM;
- load `latest` template defaults;
- re-run an editorial caption builder;
- append hashtags from current config;
- rewrite spelling/style;
- replace mentions;
- modify line breaks.

It should perform only technical validation and then send the exact approved string.

Conceptually:

```python
caption_to_send = approved_publication_request.caption.text
```

not:

```python
caption_to_send = generate_caption(template, current_data)
```

---

# 17. High Director behaviour

High Director should be able to understand instructions such as:

```text
"Use the normal Member Profile caption."
"Use last month's structure but shorter."
"Keep the attribution but remove the optional hashtags."
"Mention @example in the second paragraph."
"Don't use hashtags on this one."
```

For every one of these, the output of the conversation is still:

```text
one complete explicit caption string
```

before final approval.

---

# 18. Proposed template registry

Avoid requiring High Director to guess which template applies.

A small registry could map:

```yaml
projects:
  member_profile:
    caption_template: member_profile

  party_speech_breakdown:
    caption_template: party_speech_breakdown
```

The current repo does not appear to contain a `party_speech_breakdown` implementation yet, so that template would be introduced alongside/after that post type exists rather than pre-populating speculative production copy now.

---

# 19. Template changes and backwards compatibility

When editorial defaults change:

```text
template v3
   ↓ edit
create template v4
```

Do not mutate v3 in a way that makes old publication provenance misleading.

Historical publication records should continue to say:

```text
used template v3
```

while retaining their complete exact caption independently.

---

# 20. First comment is not part of the caption template

A first comment may have its own reusable default later, but it should remain a separate publication field because the API executes it as a separate action.

Do not hide first-comment text inside the caption template.

Possible future structure:

```yaml
first_comment_defaults:
  enabled: false
  template: null
```

The final exact first comment, if used, must be stored and approved separately.

---

# 21. Alt text is not caption text

Likewise, alt text remains attached to the media asset/publication media entry.

A caption template may define editorial guidance for alt text, but should not generate a single generic alt-text block for the entire carousel.

The existing repo already generates explicit alt text separately, which is the correct conceptual split.

---

# 22. Suggested future repo structure

Research recommendation only — do not create during this phase:

```text
instagram/
  caption_templates/
    member_profile.yml
    party_speech_breakdown.yml
    schema/
      caption_template_schema_v1.json
```

A small loader/validator could replace hard-coded caption defaults in the current Python builder later.

No publishing runtime should depend directly on these template files after a publication has been approved; it uses the stored `PublicationRequest.caption.text`.

---

# 23. Step 13 verdict

Recommended caption architecture:

```text
version-controlled template/defaults
        ↓
High Director conversational drafting
        ↓
explicit complete final caption
        ↓
PublicationRequest
        ↓
human approval fingerprint
        ↓
publisher sends unchanged caption
```

Key rules:

1. Preserve the repo's current deterministic explicit-caption principle.
2. Move reusable editorial defaults out of Python and into versioned template data later.
3. Templates guide drafting; they are not execution-time generators.
4. Store both `template_ref` and the complete final caption.
5. Required attribution/disclaimer components should be deterministically validated.
6. Hashtags and mentions may have defaults, but final values must appear explicitly in the approved caption.
7. Previous publications may be used as structure references, but old facts must not be copied blindly.
8. Editing an approved caption creates a new publication version and requires new approval.
9. First comments and alt text remain separate fields.
10. The publication worker never regenerates or edits approved caption text.

---

## Repository sources

- `process/instagram_build_copy_pack.py`
- `instagram/campaigns/member_profile_batch_v1/campaign_brief.md`

## External source

- Meta current Instagram Content Publishing documentation: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-ab559ffb-8e2c-4b0a-b43a-5737b6d2f672

---

## Confidence / unresolved items

**High confidence:**

- current repo hard-codes Member Profile caption structure/default hashtags in Python;
- current repo already stores explicit caption text, which should be preserved;
- versioned editorial templates are preferable to hard-coded defaults for recurring formats;
- final approved caption must remain independent from future template changes;
- publication-time LLM/template regeneration is unsafe and unnecessary.

**Still to determine later:**

- exact final template schema;
- editorial wording for each production post type;
- current Graph API caption constraints for the exact API version/post type during implementation;
- whether templates live as YAML or JSON after implementation planning.

**Next research step:**

Step 14 will define timezone handling using `Europe/Dublin`, UTC execution instants, daylight-saving transitions and schedule-confirmation rules.
