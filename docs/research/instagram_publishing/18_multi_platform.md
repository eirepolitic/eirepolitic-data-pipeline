# Step 18 — Multi-Platform Extension Approach

Status: **complete**

Research date: 2026-09-04

Scope: define how the publication control layer can support Instagram first while remaining extensible to Facebook, LinkedIn, X, Bluesky or other future networks without making Instagram/Meta-specific concepts the core data model.

No multi-platform implementation, provider adapter, social account connection, or live publication was created.

---

## Short conclusion

The publication architecture should distinguish three separate concepts:

```text
CONTENT INTENT
   ↓
SOCIAL PLATFORM
   ↓
DELIVERY PROVIDER
```

Example:

```text
content: Party Speech Breakdown
platform: instagram
provider: meta_direct
```

or later:

```text
content: Party Speech Breakdown
platform: instagram
provider: buffer
```

and independently:

```text
content: Party Speech Breakdown
platform: linkedin
provider: linkedin_direct
```

The **platform** describes where the public post appears.

The **provider** describes which API/service Eirepolitic uses to deliver it.

Do not conflate those two.

For one piece of content going to several networks, create a **distribution group** containing one independently versioned/approved `PublicationRequest` per platform. Each platform publication can then have its own caption, asset derivative, tags, schedule, execution state and provider result.

This preserves conversational simplicity while avoiding the false assumption that every social network supports the same post fields.

---

# 1. Existing repo finding

A targeted repository search found no current generic `platform` or `provider` publishing abstraction.

The existing Instagram code is generation/review oriented and Instagram-specific.

This means a future publication control layer can introduce the abstraction cleanly without needing to maintain compatibility with an existing social-publishing interface.

---

# 2. Platform and provider must be separate

Bad model:

```yaml
provider: instagram
```

This becomes ambiguous if Instagram can be delivered through either Meta directly or Buffer.

Recommended model:

```yaml
target:
  platform: instagram
  account_ref: eirepolitic_instagram

delivery:
  provider: meta_direct
```

Alternative:

```yaml
target:
  platform: instagram
  account_ref: eirepolitic_instagram

delivery:
  provider: buffer
```

The human-facing destination is unchanged.

Only the execution adapter changes.

---

# 3. Common core publication fields

Keep truly common intent at the core:

```yaml
publication_id: pub_...
publication_version: 3

content:
  project_id: party_speech_breakdown
  period: "2026-08"
  asset_package_id: asset_...

target:
  platform: instagram
  account_ref: eirepolitic_instagram

caption:
  text: "Exact platform-specific final text"

media:
  - asset_id: slide_01
    ordinal: 1
    alt_text: "..."

schedule:
  scheduled_local: "2026-09-08T19:30:00"
  timezone: Europe/Dublin
  scheduled_at_utc: "2026-09-08T18:30:00Z"
```

These concepts are reusable across networks.

---

# 4. Platform-specific fields belong in extensions

Instagram-specific fields should remain under:

```yaml
platform_options:
  instagram:
    post_type: carousel
    media_tags: ...
    collaborators: ...
    location: ...
    first_comment: ...
```

A future LinkedIn request might instead contain:

```yaml
platform_options:
  linkedin:
    visibility: ...
    organization_ref: ...
    article_options: ...
```

A future X request may need different reply/thread/media semantics.

Do not add every platform's possible field to one giant flat schema.

---

# 5. Do not pretend feature parity exists

A multi-platform architecture should **not** normalize features into misleading generic concepts when platform semantics differ.

Examples:

```text
Instagram collaborator
≠ LinkedIn co-author concept

Instagram first comment
≠ X reply/thread semantics

Instagram image user tag
≠ Facebook/LinkedIn mention semantics

Instagram location_id
≠ arbitrary free-text location on another network
```

Only normalize concepts that really have common meaning.

Everything else remains platform-specific.

---

# 6. Capability registry

Each platform/provider adapter should expose deterministic capabilities.

Conceptual example:

```yaml
capabilities:
  platform: instagram
  provider: meta_direct

  post_types:
    image: true
    carousel: true
    reel: true
    story: true

  features:
    caption: true
    alt_text: true
    media_tags: true
    collaborators: true
    location: true
    first_comment: true
```

Buffer might currently differ:

```yaml
capabilities:
  platform: instagram
  provider: buffer

  features:
    media_tags: true
    collaborators: false
    first_comment: false   # disabled while current provider bug remains
```

High Director reads this registry before promising a feature.

The LLM must not infer provider capabilities from memory at execution time.

---

# 7. Capability validation occurs before approval

Example request:

```text
"Post this to Instagram with @example as collaborator and also to LinkedIn."
```

The control layer should:

1. create separate Instagram and LinkedIn draft requests;
2. validate each against its selected platform/provider capability registry;
3. identify unsupported fields;
4. tell the human what differs;
5. create exact platform-specific final requests;
6. approve only executable configurations.

Do not approve one generic manifest and silently drop unsupported fields inside adapters later.

---

# 8. Distribution group

When one content item is intended for several networks, introduce a logical grouping record such as:

```yaml
distribution_id: dist_01J...
project_id: party_speech_breakdown
period: "2026-08"

publications:
  - pub_instagram_...
  - pub_linkedin_...
  - pub_bluesky_...
```

The distribution group is organizational.

Each child publication remains independently authoritative.

---

# 9. Why separate child publications

One universal multi-network publication record becomes problematic because:

- captions have different lengths/style;
- hashtags differ;
- mentions differ;
- asset constraints differ;
- carousels/media limits differ;
- features differ;
- schedules may differ;
- one platform may fail while another succeeds;
- cancellation may apply to one platform only;
- provider authentication differs.

Therefore:

```text
one distribution intent
   ↓
multiple platform PublicationRequests
```

is safer than:

```text
one mutable publication row with platform=[instagram,linkedin,x]
```

---

# 10. Approval of a multi-platform bundle

High Director may make the UX simple by presenting one combined confirmation.

Example:

```text
Ready to schedule Party Speech Breakdown — August 2026:

Instagram @eirepolitic
- 8-slide carousel
- Instagram caption: ...
- tag @example on slide 3
- Tue 19:30 Dublin

LinkedIn EirePolitic
- 8 images / platform-valid package
- LinkedIn caption: ...
- Tue 19:35 Dublin

Confirm both exact publications?
```

One human response can create two approval records if the user clearly approves the entire displayed bundle.

Each child still gets:

```text
its own publication_id
its own publication_version
its own fingerprint
its own schedule
its own execution/result state
```

---

# 11. Partial success

Multi-platform publication must allow:

```text
Instagram = published
LinkedIn = failed
```

without flattening the entire distribution to simply:

```text
failed
```

A distribution-level status can summarize:

```text
scheduled
partially_published
published_all
needs_attention
cancelled_all
```

but child publication states remain authoritative.

---

# 12. Scheduling model

Each child publication receives its own schedule.

Even when several platforms should publish at the same time, create independent scheduled executions:

```text
pub_instagram → EventBridge schedule A
pub_linkedin  → EventBridge schedule B
```

This prevents one platform's API latency/failure from blocking another.

High Director can still interpret:

```text
"Move all of Friday's versions to 8pm"
```

as a grouped change requiring confirmation of the affected child publications.

---

# 13. Provider adapter interface

A future publisher interface should remain small and deterministic.

Conceptual shape:

```python
class PublicationProvider:
    def validate(self, publication): ...
    def prepare(self, publication, attempt): ...
    def publish(self, publication, attempt): ...
    def reconcile(self, publication, attempt): ...
    def cancel_scheduled(self, publication): ...
    def capabilities(self): ...
```

Depending on whether scheduling is Eirepolitic-owned or vendor-owned, schedule methods may belong in a separate scheduler/provider interface.

Do not force every provider to implement operations its platform does not support.

---

# 14. Recommended adapter layering

Prefer two conceptual boundaries:

```text
Platform adapter
  → validates platform semantics/assets/fields

Delivery provider adapter
  → talks to Meta/Buffer/LinkedIn/etc.
```

For a small v1 these can be implemented in one module/class, for example:

```text
MetaInstagramPublisher
```

but the data model should preserve the distinction.

This avoids over-engineering v1 while keeping future migration possible.

---

# 15. Example adapters

Potential future modules:

```text
publishers/
  base.py
  instagram_meta.py
  instagram_buffer.py
  linkedin_direct.py
  bluesky_direct.py
```

or equivalent.

Do not implement these during the research phase.

The point is that adding a network should not require modifying High Director's core publication state machine.

---

# 16. Provider-neutral execution result

Normalize common result facts:

```yaml
provider_result:
  provider: meta_direct
  platform: instagram
  external_media_id: "..."
  permalink: "..."
  published_at: "..."
```

Keep raw/provider-specific details underneath:

```yaml
provider_details:
  instagram:
    media_type: CAROUSEL_ALBUM
    container_id: "..."
```

High Director can answer common questions using normalized result fields while diagnostics retain provider detail.

---

# 17. Normalized error categories

Adapters should translate provider errors into the Step 16 categories where possible:

```text
transient_safe_retry
transient_reconcile_first
permanent_input
permanent_auth
rate_limited
provider_unknown
success_response_lost
operator_action_required
```

The raw provider error/code is retained safely for diagnostics.

This gives High Director consistent failure language across platforms.

---

# 18. Asset derivation by platform

Step 12 established that the current PNG render should be finalized into an Instagram-compatible JPEG delivery package.

Multi-platform extension should allow platform-specific delivery derivatives from one reviewed source package.

Conceptually:

```text
reviewed source content package
        ↓
Instagram derivative package
        ↓
LinkedIn derivative package
        ↓
other platform derivative package
```

Each derivative records its own:

- MIME type;
- dimensions;
- file size;
- SHA-256;
- platform policy/version.

Do not lower the quality/source asset globally just because the first platform has a JPEG-specific requirement.

---

# 19. Caption derivation by platform

Likewise, use the source editorial intent/template as input to create **separate explicit final captions**.

Example:

```text
source editorial message
   ↓
Instagram caption draft
   ↓ human approval
Instagram final caption

source editorial message
   ↓
LinkedIn caption draft
   ↓ human approval
LinkedIn final caption
```

The publisher never transforms one platform's approved caption into another platform format at execution time.

---

# 20. Account registry

Use stable internal account references.

Example:

```yaml
accounts:
  eirepolitic_instagram:
    platform: instagram
    provider: meta_direct
    provider_account_ref: ...
    default_timezone: Europe/Dublin

  eirepolitic_linkedin:
    platform: linkedin
    provider: linkedin_direct
    provider_account_ref: ...
    default_timezone: Europe/Dublin
```

High Director talks primarily in human account names/handles.

The deterministic system resolves them to `account_ref`.

---

# 21. Provider migration

A key benefit of separating platform from provider is future migration.

Example:

```text
Instagram delivery today:
provider = meta_direct

future operational decision:
provider = buffer
```

Historical publications retain the provider that actually executed them.

New publications use the newly configured provider.

No `publication_id` semantics or High Director conversational model need to change.

---

# 22. Do not design to Buffer's schema

Buffer supports several networks through one API, which is useful, but the Eirepolitic core schema should **not** mirror Buffer's post object.

Why:

- direct Meta may remain the preferred Instagram route;
- Buffer can lag native platform features;
- vendor APIs/pricing can change;
- provider migration would become painful;
- Eirepolitic needs richer approval/audit semantics than a scheduler vendor object.

Buffer should remain one possible delivery adapter, not the system's domain model.

---

# 23. Do not design to Meta's schema either

Similarly, fields such as:

```text
creation_id
container_id
instagram_business_account
```

must remain inside the Meta execution adapter/attempt records.

They should not become required fields for every publication.

The core publication model describes approved intent, not Graph API mechanics.

---

# 24. High Director conversational examples

## Add a platform

```text
User:
"Post the August breakdown to Instagram and LinkedIn next Tuesday."

High Director:
resolve source content
→ create two platform drafts
→ apply platform-specific defaults
→ identify differences/unresolved choices
→ show both exact proposals
→ obtain approval
→ schedule each independently
```

## Cancel one platform

```text
User:
"Keep Instagram but cancel LinkedIn."

High Director:
resolve distribution children
→ confirm LinkedIn cancellation
→ cancel only LinkedIn child
→ Instagram schedule remains unchanged
```

## Move both

```text
User:
"Move both versions to 8pm."

High Director:
resolve both schedule records
→ show both affected publications
→ obtain grouped schedule-change confirmation
→ update each schedule
```

---

# 25. Cross-platform query model

The normalized ledger allows High Director to answer:

```text
"What's scheduled this week across all platforms?"
```

by filtering platform publications independently.

Example result:

```text
Tue 19:30 — Instagram — Party Speech Breakdown
Tue 19:35 — LinkedIn — Party Speech Breakdown
Fri 18:00 — Instagram — Member Profile
```

Likewise:

```text
"What failed yesterday?"
```

can search normalized states across providers.

---

# 26. Idempotency remains per platform publication

Do not use the distribution-group ID as the sole idempotency key.

Use:

```text
publication_id + version
```

for each platform child.

This allows one network to retry/reconcile without affecting another already-successful platform publication.

---

# 27. Approval remains per exact platform output

Because captions/assets/features can differ, the approval fingerprint is per child publication.

A grouped human confirmation may approve several fingerprints at once, but there should still be independent immutable approval records underneath.

This is important when one child later changes.

Example:

```text
Instagram caption edited
```

should invalidate only the Instagram child approval, not the unchanged LinkedIn approval.

---

# 28. Monitoring remains normalized

Step 17's application metrics can use low-cardinality dimensions:

```text
platform
provider
environment
```

Examples:

```text
PublicationPublished platform=instagram provider=meta_direct
PublicationFailed platform=linkedin provider=linkedin_direct
```

Provider-specific logs remain available for diagnosis.

---

# 29. Secrets remain provider-specific

Step 15's secret model scales naturally:

```text
/eirepolitic/prod/instagram/meta-page-access-token
/eirepolitic/prod/linkedin/access-token
/eirepolitic/prod/buffer/api-token
```

The publication record stores only a provider/account reference.

High Director never receives any of these token values.

---

# 30. Recommended v1 scope

Do **not** build multi-platform publishing before Instagram is proven.

Implement the first version so the boundaries exist, but only provide one production adapter initially:

```text
Instagram + selected provider
```

Once the first platform proves:

- approval model;
- scheduler;
- ledger;
- idempotency;
- monitoring;
- secrets;

then add a second platform as a new adapter/asset/caption policy rather than redesigning the system.

This is the lowest-risk path.

---

# 31. Avoid premature universal schema design

Do not attempt to predict every future social network field now.

Use:

```text
small common core
+
versioned platform_options extension
+
capability registry
+
provider adapters
```

When a new network is introduced, add only the fields it actually requires.

This prevents a complex schema full of unused nullable fields.

---

# 32. Recommended conceptual model

```text
Source Content / Reviewed Asset Package
                ↓
         DistributionGroup
          /       |       \
         /        |        \
Instagram Pub   LinkedIn Pub   Future Pub
     ↓               ↓             ↓
Instagram       LinkedIn      Future platform
capability      capability    capability
validator       validator     validator
     ↓               ↓             ↓
provider adapter  provider adapter  provider adapter
     ↓               ↓             ↓
external network APIs
```

Each child publication has its own:

```text
request
approval
schedule
execution attempts
published result
```

---

# 33. Step 18 verdict

Recommended multi-platform architecture:

```text
High Director
     ↓
Distribution intent (optional grouping)
     ↓
independent platform PublicationRequests
     ↓
platform capability validation
     ↓
per-platform approval fingerprints
     ↓
independent schedules
     ↓
provider adapters
     ↓
external networks
```

Key rules:

1. Separate `platform` from `delivery provider`.
2. Keep a small common publication core and platform-specific extensions.
3. Never silently normalize or discard platform-specific features.
4. Validate provider/platform capability before approval.
5. One multi-platform distribution creates independent child publications.
6. Each child has its own caption, assets, approval, schedule, idempotency and result.
7. Grouped confirmation is allowed, but approvals remain independently fingerprinted.
8. Partial success is a normal multi-platform state.
9. Keep Buffer/Meta schemas behind adapters rather than making either the core domain model.
10. Build Instagram first; add the second platform only after the core publication system is proven.

---

## Repository findings

Targeted repository searches found no current generic publishing `platform` or `provider` abstraction, so this would be new control-plane design rather than a migration of existing publishing code.

## Dependencies from earlier research

This design builds directly on:

- Step 8 — Buffer as optional delivery provider;
- Step 10 — platform-neutral publication records with `platform_options`;
- Step 11 — independent immutable approval fingerprints;
- Step 12 — platform-specific asset derivatives;
- Step 13 — platform-specific explicit final captions;
- Step 15 — provider-specific credential references;
- Step 16 — per-publication idempotency;
- Step 17 — normalized monitoring and status queries.

---

## Confidence / unresolved items

**High confidence:**

- platform and delivery provider should be distinct concepts;
- multi-platform distribution should use independent child publication records;
- platform-specific capabilities should be validated before approval;
- provider APIs should remain behind adapters;
- adding a second platform later should not require changing the core approval/scheduling model.

**Intentionally deferred:**

- which second platform Eirepolitic should support first;
- direct API versus Buffer for future non-Instagram platforms;
- exact schema for LinkedIn/X/Bluesky-specific fields;
- exact adapter class/module layout;
- whether grouped publication approvals use a parent approval event in addition to child approval records.

**Next research step:**

Step 19 will synthesize all findings into the final option comparison, expected cost/complexity/risk/lock-in analysis, recommended architecture, phased implementation proposal and explicit decision gates.
