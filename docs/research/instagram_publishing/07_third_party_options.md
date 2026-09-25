# Step 7 — Third-Party Social Scheduling / Publishing Platforms

Status: **complete**

Research date: 2026-09-03

Scope: compare current third-party platforms that could receive a deterministic publication instruction from Eirepolitic and perform final Instagram scheduling/delivery.

Platforms reviewed:

- Buffer;
- Metricool;
- Sprout Social;
- Hootsuite;
- Later.

No account was created, connected or upgraded and nothing was scheduled or published.

---

## Short conclusion

**Buffer is the strongest third-party API candidate for Eirepolitic at current scale.**

It now has a current public GraphQL API that can:

- create and edit posts;
- schedule exact publication times;
- automatically publish;
- retrieve scheduled/sent/error posts;
- delete posts;
- target Instagram, Facebook, LinkedIn, X, Bluesky and other networks;
- create Instagram post/reel/story types;
- supply image alt text;
- supply Instagram image user tags;
- supply Instagram geolocation;
- configure first comments;
- integrate with custom code/agents.

API access is available even on Buffer Free; Essentials is currently $5/channel/month billed annually.

However, Buffer is **not feature-equivalent to direct Meta** today:

- current Instagram API input does not expose collaborators;
- its new API is still evolving rapidly;
- Buffer's own current roadmap reports a bug where Instagram `firstComment` can be silently ignored when posts are created through the API;
- Buffer requires media URLs to remain publicly reachable until the scheduled publication time and explicitly advises against expiring S3 presigned URLs for scheduled posts;
- a public webhook/post-status callback was not identified in the current API documentation reviewed, so Eirepolitic would likely need status polling/reconciliation;
- Eirepolitic would depend on both Buffer and Meta for final delivery.

Metricool is technically viable but its programmatic API is on the significantly more expensive **Advanced** tier (currently from €54/month) and its public API documentation is less developer-focused.

Sprout Social is not a suitable final delivery API at present because its Publishing API currently supports creating posts only in **Draft** status, despite its high API-tier cost.

Hootsuite is capable as a social-management product and has a current developer/API platform, but its current publicly discoverable API documentation does not provide a sufficiently clear self-service publishing contract for this project to select it without vendor confirmation. Pricing starts at approximately $99/month.

Later has strong Instagram scheduling features in its product but no generally available customer publishing API suitable for Eirepolitic's own deterministic backend was identified in the current documentation reviewed.

---

# 1. Buffer

## API availability

### Status: strong / public / self-service

Buffer launched its new public API in 2026. It is a GraphQL API at:

```text
https://api.buffer.com
```

Buffer explicitly describes use cases including:

- custom scripts;
- automation tools;
- agents;
- applications;
- creating and scheduling posts from an external system.

API access is available on every current plan, including Free.

Sources:

- https://buffer.com/api
- https://developers.buffer.com/
- https://developers.buffer.com/guides/introduction.html
- https://buffer.com/resources/buffer-api-is-here/

## Programmatic scheduling

### Supported

The API supports an exact scheduled timestamp using:

```text
mode: customScheduled
dueAt: <ISO 8601 UTC timestamp>
schedulingType: automatic
```

Buffer then owns the timed delivery.

It exposes a lifecycle including:

- scheduled;
- sent;
- error.

Sources:

- https://developers.buffer.com/examples/create-scheduled-post.html
- https://developers.buffer.com/guides/posts-and-scheduling.html

### Conversational fit

High Director could create the Eirepolitic publication record, obtain final human approval, and then create a corresponding Buffer job.

That creates a clean hybrid possibility:

```text
High Director
  ↓
Eirepolitic publication request/approval/ledger
  ↓
Buffer API
  ↓
Buffer scheduler
  ↓
Instagram
```

This will be examined formally in Step 8.

---

## Instagram formats

Buffer's current API supports Instagram as a service and allows the Instagram post type to be specified as:

- post;
- story;
- reel.

Its product supports automatic publishing for professional Instagram accounts subject to Meta's limitations.

Sources:

- https://developers.buffer.com/guides/posts-and-scheduling.html
- https://developers.buffer.com/types/InstagramPostMetadataInput.html
- https://support.buffer.com/en-us/articles/what-is-buffers-api-GtIYIQilz5

### Carousels

The API accepts an ordered `assets` array, including multiple image/media assets. It is therefore suitable for Eirepolitic's carousel model subject to Buffer/Instagram per-post limits.

The ordered-asset concept aligns well with Eirepolitic's future publication manifest.

---

## Caption control

### Supported

`CreatePostInput.text` provides the exact post text.

This means Eirepolitic can continue storing the final caption explicitly and pass that exact caption to Buffer.

No LLM regeneration would be required at execution time.

---

## Image tagging

### Supported

Buffer's current API has an explicit Instagram example using:

```text
image.metadata.userTags
```

with:

- Instagram handle;
- normalized `x` position;
- normalized `y` position.

Buffer's example also requires image alt text when supplying image metadata.

Source:

- https://developers.buffer.com/examples/create-instagram-post-with-user-tags.html

This is a strong fit for Eirepolitic's requirement to conversationally say which account should be tagged on which image.

---

## Alt text

### Supported

Buffer's image metadata exposes:

```text
altText
```

This maps cleanly to the existing Eirepolitic deterministic alt-text outputs.

Source:

- https://developers.buffer.com/reference.html

---

## Location

### Supported in current API schema

Current `InstagramPostMetadataInput` exposes:

```text
geolocation
```

with location ID/text fields.

Source:

- https://developers.buffer.com/types/InstagramPostMetadataInput.html

As with direct Meta, location IDs should be validated before final approval rather than guessed.

---

## Collaborators

### Not exposed in the current Buffer Instagram input schema reviewed

The current `InstagramPostMetadataInput` contains:

- firstComment;
- geolocation;
- isAiGenerated;
- link;
- shouldShareToFeed;
- stickerFields;
- type.

It does **not** expose a collaborators field.

The asset metadata exposes user tags, but a user tag is not the same as an Instagram collaborator/co-author.

Source:

- https://developers.buffer.com/types/InstagramPostMetadataInput.html

### Conclusion

If collaborator posts are an important Eirepolitic capability, Buffer currently has a gap relative to Meta's richer direct Graph API surface.

Do not represent media tags as collaborator invitations.

---

## First comment

### Schema support exists, but there is a current known API problem

Buffer added:

```text
InstagramPostMetadataInput.firstComment
```

in February 2026.

However, Buffer's current public API roadmap, last updated 28 August 2026, contains an active report titled:

```text
Instagram firstComment field not persisted when creating posts via API
```

The report states the value can currently be silently ignored and returned as null for Instagram API-created posts.

Sources:

- https://developers.buffer.com/changelog.html
- https://developers.buffer.com/roadmap.html

### Conclusion

Treat Buffer Instagram first-comment automation as **currently unreliable until Buffer confirms the roadmap issue is resolved**.

This is a concrete example of the additional failure layer introduced by a scheduling vendor: an API field can exist while the vendor implementation still fails to forward it correctly.

---

## Approval workflows

Buffer's `CreatePostInput` exposes:

```text
needsApproval
```

which submits a post for approval when the target channel's posting policy requires approval.

It also supports saving as a draft.

Source:

- https://developers.buffer.com/types/CreatePostInput.html

### Eirepolitic use

Even if Buffer approval were used, Eirepolitic should still keep its **own human approval record**.

Buffer approval should be treated as an optional secondary operational policy rather than replacing High Director's approval fingerprint and audit history.

---

## Edit/delete/reschedule

Buffer's current API supports:

- post creation;
- editing;
- retrieval;
- deletion;
- scheduling modes;
- scheduled timestamps.

This makes changing or cancelling a Buffer-held scheduled job straightforward through an API adapter.

Sources:

- https://developers.buffer.com/guides/introduction.html
- https://developers.buffer.com/reference.html
- https://developers.buffer.com/changelog.html

---

## Rate limits

Current limits per API client:

| Plan | 15 min | 24 hr | 30 days |
|---|---:|---:|---:|
| Free | 100 | 250 | 3,000 |
| Essentials | 100 | 250 | 7,500 |
| Team | 100 | 500 | 15,000 |

These are vastly higher than Eirepolitic's expected initial usage.

Source:

- https://developers.buffer.com/guides/api-limits.html

---

## Pricing

Current pricing:

- Free: $0, up to 3 channels, 10 scheduled posts/channel, API included;
- Essentials: $5/month per channel when billed annually, unlimited scheduling subject to fair-use cap, 3 API keys;
- Team: $10/month per channel when billed annually, additional collaboration/approval capability.

Sources:

- https://buffer.com/pricing
- https://support.buffer.com/en-us/articles/buffer-pricing-and-features-6pJrOPuzIt

### Eirepolitic cost

For one Instagram channel, Buffer is inexpensive enough that cost is not a strong argument against it.

---

## Media hosting requirement

This is a material architectural difference from direct Meta.

Buffer's current API has **no media upload endpoint** for ordinary post creation. The application supplies a public media URL.

For a scheduled post, Buffer says the URL must remain reachable until the scheduled publication time and explicitly advises avoiding expiring/signed URLs such as S3 presigned links because they may expire before Buffer retrieves the media.

Source:

- https://developers.buffer.com/guides/hosting-media.html

### Eirepolitic implication

If Buffer performs future scheduling days in advance, we would need either:

- stable public-but-hard-to-guess asset URLs;
- a controlled CDN/public media-delivery path;
- or another stable hosting mechanism.

This is less attractive than the direct-Meta design, where Eirepolitic can retain private S3 assets and generate retrieval URLs only when its own worker executes.

A stable public URL is not necessarily insecure, but it changes the preferred asset-access model and must be weighed in Step 8/12.

---

## Webhooks / event callbacks

No generally documented post-status webhook mechanism was identified in the current Buffer public API documentation reviewed in this step.

Buffer does expose post retrieval/status, so Eirepolitic can reconcile states such as scheduled/sent/error by querying Buffer.

### Conclusion

Do not assume push webhooks exist.

A Buffer hybrid architecture should be designed to work with polling/reconciliation unless Buffer documents a suitable webhook before implementation.

---

## Reliability / operational risk

Buffer is an established social publishing platform and says it is an official API partner with Meta and other networks.

However, its **new public GraphQL API is very new in 2026 and evolving quickly**. The current changelog shows frequent additions/changes, and the first-comment issue demonstrates that production integration should still be capability-tested.

### Buffer assessment

| Dimension | Assessment |
|---|---|
| Programmatic publishing | **Strong** |
| Exact scheduling | **Strong** |
| Cost | **Very low** |
| Image tagging | **Strong** |
| Alt text | **Strong** |
| Location | **Supported** |
| Collaborators | **Gap in current input schema** |
| First comment | **Schema exists, current Instagram API bug reported** |
| Approval workflow | **Available** |
| Multi-platform | **Excellent** |
| API maturity | **New/evolving** |
| Vendor lock-in | **Medium** |
| High Director fit | **Very good** |

---

# 2. Metricool

## API availability

### Supported, but plan-restricted

Metricool has a scheduler API suitable for use from a custom backend.

Its current documentation shows a call conceptually like:

```text
POST /v2/scheduler/posts
```

with fields including:

- publication date;
- timezone;
- text;
- providers/networks;
- `autoPublish`;
- `draft`;
- media;
- network-specific data.

Source:

- https://help.metricool.com/wli-scheduler-endpoint-example-on-a-custom-backend-proxy-frko7

This means Eirepolitic **could programmatically schedule from its own pipeline**.

---

## Subscription restriction

Metricool's current pricing places its API in the **Advanced** tier.

Current starting price:

```text
€54/month
```

for Advanced, before VAT, at the current annual pricing shown.

The Advanced tier also includes:

- team/client management;
- roles;
- approval system;
- API access.

Source:

- https://metricool.com/pricing/

This is materially more expensive than Buffer for a single Eirepolitic account.

---

## Instagram capability

Metricool currently auto-publishes professional-account:

- posts;
- Reels;
- Trial Reels;
- Stories.

Some Instagram-native features still require manual notification publishing because Meta's API does not support them.

Source:

- https://help.metricool.com/schedule-and-post-on-instagram-6b6q5

Metricool also supports first-comment scheduling in its product for Instagram Posts and Reels.

Source:

- https://help.metricool.com/how-to-schedule-a-first-comment-on-your-posts-pwm25

It exposes Instagram image tagging in its product for supported formats.

Source:

- https://help.metricool.com/mention-and-tag-other-accounts-9jqzi

Its current documentation also recommends the Facebook-connected Instagram Graph route for the fullest feature set and reports product-tagging/collaboration analytics differences between connection types.

Source:

- https://help.metricool.com/access-to-instagram-from-metricool-connection-types-and-differences-kcsrb

---

## API completeness risk

Metricool's API is real, but its public developer experience is less transparent/comprehensive than Buffer's current schema-driven API docs.

Some advanced feature capability is documented primarily at the product/help level rather than with a clear API field-by-field contract.

Therefore a Metricool choice would require an API proof specifically for:

- user tags;
- collaborators;
- first comment;
- location;
- alt text;
- cancellation/rescheduling;
- result/status retrieval;
- webhook availability.

Do not assume every capability available in the Metricool UI is exposed by the external API.

### Metricool assessment

| Dimension | Assessment |
|---|---|
| Programmatic scheduling | **Yes** |
| Cost | **Medium/high for this project** |
| Instagram product features | **Strong** |
| Exact API metadata coverage | **Requires proof** |
| Approval workflows | **Strong product feature** |
| Multi-platform | **Strong** |
| Public API clarity | **Moderate** |
| Vendor lock-in | **Medium-high** |
| High Director fit | **Good** |

---

# 3. Sprout Social

## API availability

Sprout has a documented API and places Sprout API access on its **Advanced** plan.

Current Advanced price:

```text
$399 per seat/month billed annually
```

Source:

- https://sproutsocial.com/pricing/

---

## Critical Publishing API limitation

Sprout's current Publishing Post API documentation says:

- Publishing Posts are intended for future social publication;
- but **Create Publishing Post currently supports only Draft status**.

It also says Instagram Mobile Publisher and Story posts cannot be directly created through that Publishing API path.

Source:

- https://api.sproutsocial.com/docs/

### Consequence

This does not meet Eirepolitic's requirement:

```text
approved instruction
   ↓
automatically publishes at scheduled time
```

If our API can only create a Sprout draft that requires another manual/product-side action, Sprout is not providing the deterministic final-delivery abstraction we need.

### Sprout assessment

| Dimension | Assessment |
|---|---|
| API exists | Yes |
| API-created final scheduled delivery | **Not suitable currently — draft-only create** |
| Cost | **Very high** |
| Enterprise approvals | Strong |
| Vendor lock-in | High |
| High Director fit | **Poor for autonomous final delivery** |

### Verdict

**Exclude from shortlist for this project.**

---

# 4. Hootsuite

## Product capability

Hootsuite is a mature social-management platform with:

- scheduling;
- Instagram publishing;
- approvals;
- multi-account management;
- a current REST API/developer portal;
- MCP/AI integration initiatives.

Current pricing starts approximately at:

- Standard: $99/month;
- Professional: $199/month;
- Advanced: $399/month.

Sources:

- https://www.hootsuite.com/plans
- https://www.hootsuite.com/docs
- https://apidocs.hootsuite.com/docs/api/index.html

---

## API availability finding

Hootsuite clearly has a current API platform and developer portal.

However, during this research pass, the current publicly discoverable API reference did **not expose a sufficiently clear self-service publishing endpoint/schema and subscription entitlement model** to establish with confidence that Eirepolitic can create fully scheduled Instagram posts through its own backend under an ordinary current plan.

Hootsuite historically offered an open Publishing API, and its present platform continues to describe API/developer integrations, but historical availability is not sufficient for this architecture decision.

### Recommendation

Do not select Hootsuite unless Hootsuite directly confirms, for the intended plan/account:

1. current outbound Publishing API access;
2. Instagram post/carousel/Reel support;
3. user tags/collaborators/location/alt text fields;
4. exact schedule/cancel/update operations;
5. API rate limits;
6. webhook/status mechanisms;
7. plan/API entitlement.

At Eirepolitic's scale, paying $99+/month merely to discover/maintain this additional abstraction is difficult to justify when Buffer has a much clearer current API.

### Hootsuite assessment

| Dimension | Assessment |
|---|---|
| Product scheduling | Strong |
| Approval product | Strong |
| API platform | Exists |
| Clear current publishing API contract for this use | **Insufficiently verified** |
| Cost | High |
| Vendor lock-in | High |
| High Director fit | Unclear until publishing API is confirmed |

### Verdict

**Not shortlisted without vendor confirmation.**

---

# 5. Later

## Product capability

Later currently supports professional Instagram scheduling for formats including:

- single image;
- carousel;
- Reels;
- Stories;
- collaborative posts in its product.

Some features require notification/manual publishing because of Instagram API restrictions.

Sources:

- https://help.later.com/hc/en-us/articles/360060842914-Supported-Social-Platforms-Post-Types
- https://help.later.com/hc/en-us/articles/36919457087639-Instagram-Publishing-Restrictions-in-Later

Current pricing begins at:

```text
$18.75/month billed yearly
```

for Starter.

Source:

- https://later.com/pricing/

---

## Programmatic API finding

No generally available public API was identified in Later's current public documentation that would let Eirepolitic's backend create and control Later scheduled publishing jobs in the same way Buffer or Metricool can.

Later's current help/developer-facing material focuses on using the Later application itself and connecting social profiles to Later.

This does not mean Later has no internal/partner API. It means **no suitable public customer publishing contract was found that should be used as the foundation of Eirepolitic's architecture**.

### Consequence

Later is a potentially good manual social scheduler, but it does not currently solve the central requirement:

```text
High Director
   ↓
our deterministic pipeline
   ↓
third-party scheduling API
```

without a documented API interface.

### Later assessment

| Dimension | Assessment |
|---|---|
| Instagram scheduling product | Strong |
| Instagram collaboration product features | Strong |
| Public customer publishing API found | **No** |
| Programmatic own-pipeline fit | **Poor/unknown** |
| Cost | Low-medium |
| Vendor lock-in | High if workflows live in its UI |
| High Director fit | **Poor without API** |

### Verdict

**Exclude from API shortlist unless Later provides a suitable private/partner API contract.**

---

# 6. Comparison table

| Platform | Own-pipeline API publishing | Scheduled final delivery | Approx starting cost relevant to API use | Instagram metadata fit | Approval capability | Main concern | Shortlist? |
|---|---|---|---:|---|---|---|---|
| **Buffer** | **Yes** | **Yes** | **$0 / ~$5 per channel** | Good; image tags, alt text, geolocation; no collaborator input found | API/product support | New API; first-comment bug; stable public media URL needed | **Yes** |
| **Metricool** | **Yes** | **Yes** | **~€54/month** | Product strong; exact API fields require proof | Strong | Cost + less transparent API surface | Maybe |
| Sprout Social | API exists | **API create is draft-only** | **$399/seat/month** | Enterprise product | Strong | Does not provide required final API delivery | No |
| Hootsuite | API platform exists | Not sufficiently verified | **~$99+/month** | Product strong | Strong | Current public publishing API contract/entitlement unclear | No pending proof |
| Later | No suitable public publishing API identified | Product UI yes | **$18.75+/month** | Product strong | Growth+ product workflows | Cannot reliably integrate own deterministic backend | No |

---

# 7. Advantages of a third-party scheduler in general

Compared with direct Meta, a suitable vendor can handle:

- social-account authorization UI;
- connection/token lifecycle;
- network-specific scheduler workers;
- timed delivery;
- some retries;
- some platform constraint normalization;
- multiple social networks through one API;
- a backup operator UI/calendar.

This can reduce Eirepolitic's implementation burden.

Buffer in particular can potentially eliminate the need for Eirepolitic to build its own time-triggered Instagram worker.

---

# 8. Disadvantages of a third-party scheduler in general

A vendor adds another stateful system:

```text
Eirepolitic ledger
     ↓
Vendor scheduled post
     ↓
Meta
```

Potential problems:

- our ledger and vendor state can disagree;
- vendor APIs can lag native Instagram features;
- additional outages can block publication;
- feature bugs can exist independently of Meta;
- vendor subscription/pricing/API policies can change;
- media may need to remain externally accessible for longer;
- cancellation/rescheduling must update both our record and vendor state;
- exact delivery errors may be normalized/hidden by the vendor;
- future migration requires replacing a delivery adapter and reconciling historical vendor IDs.

The Buffer first-comment issue is a concrete current example of this additional abstraction risk.

---

# 9. Vendor lock-in design rule

If a third-party service is ever used, High Director should **not** create vendor-specific content as the source of truth.

Keep:

```text
Eirepolitic PublicationRequest
Eirepolitic PublicationApproval
Eirepolitic PublicationLedger
```

and only add external execution state such as:

```text
provider: buffer
provider_post_id: ...
provider_status: scheduled
```

This means Eirepolitic can later replace:

```text
BufferPublisher
```

with:

```text
MetaPublisher
```

without rebuilding the conversational/approval model.

---

# 10. Step 7 verdict

The only third-party option strong enough to carry forward as a serious alternative is:

## **Buffer**

Metricool remains a technically viable but more expensive secondary candidate.

The current candidate set for the architecture comparison is therefore:

```text
Option A
Direct Meta + Eirepolitic-owned scheduling

Option B
Buffer API + Buffer-owned final scheduling/delivery
```

Step 8 will examine the hybrid model properly, including whether Buffer actually removes enough work to justify:

- vendor dependency;
- public/stable media hosting requirements;
- missing collaborator support;
- reconciliation complexity.

---

## Sources

### Buffer

- https://buffer.com/api
- https://buffer.com/pricing
- https://developers.buffer.com/
- https://developers.buffer.com/guides/introduction.html
- https://developers.buffer.com/guides/posts-and-scheduling.html
- https://developers.buffer.com/examples/create-scheduled-post.html
- https://developers.buffer.com/types/CreatePostInput.html
- https://developers.buffer.com/types/InstagramPostMetadataInput.html
- https://developers.buffer.com/examples/create-instagram-post-with-user-tags.html
- https://developers.buffer.com/guides/api-limits.html
- https://developers.buffer.com/guides/hosting-media.html
- https://developers.buffer.com/changelog.html
- https://developers.buffer.com/roadmap.html

### Metricool

- https://metricool.com/pricing/
- https://help.metricool.com/wli-scheduler-endpoint-example-on-a-custom-backend-proxy-frko7
- https://help.metricool.com/schedule-and-post-on-instagram-6b6q5
- https://help.metricool.com/how-to-schedule-a-first-comment-on-your-posts-pwm25
- https://help.metricool.com/access-to-instagram-from-metricool-connection-types-and-differences-kcsrb
- https://help.metricool.com/mention-and-tag-other-accounts-9jqzi

### Sprout Social

- https://api.sproutsocial.com/docs/
- https://sproutsocial.com/pricing/

### Hootsuite

- https://www.hootsuite.com/docs
- https://apidocs.hootsuite.com/docs/api/index.html
- https://www.hootsuite.com/plans

### Later

- https://later.com/pricing/
- https://help.later.com/hc/en-us/articles/360060842914-Supported-Social-Platforms-Post-Types
- https://help.later.com/hc/en-us/articles/36919457087639-Instagram-Publishing-Restrictions-in-Later
- https://help.later.com/hc/en-us/articles/1500001601602-Instagram-and-Facebook-Page-Requirements-in-Later

---

## Confidence / unresolved items

**High confidence:**

- Buffer currently exposes a real public API capable of programmatic scheduling/final delivery;
- Buffer API access is available on all plans;
- Buffer supports Instagram user tags/alt text/geolocation in its current API;
- Buffer's current Instagram input schema does not expose collaborators;
- Buffer currently reports an Instagram first-comment persistence issue in its API roadmap;
- Buffer scheduled API media URLs must remain reachable until publication;
- Metricool exposes a programmatic scheduler endpoint and API access is Advanced-tier;
- Sprout's current Publishing API create operation is draft-only;
- no suitable current public Later customer publishing API was identified.

**Requires proof before choosing a vendor:**

- Buffer carousel/user-tag edge cases and collaborator roadmap;
- Buffer webhooks/event callbacks, if added later;
- Metricool's exact API fields for all desired Instagram metadata;
- Hootsuite's current outbound publishing API availability/entitlement for an ordinary Eirepolitic account.

**Next research step:**

Step 8 will compare the hybrid model — Eirepolitic owns intent/approval/ledger, while Buffer (or another vendor) owns final scheduled delivery.
