# Step 2 — Existing Instagram Account Conversion

Status: **complete**

Research date: 2026-09-03

Scope: determine whether the existing Eirepolitic Instagram personal account can be converted to a professional account without replacing the account or losing its existing audience/history, and document material trade-offs. No account changes were made.

## Short conclusion

**Yes — the existing Instagram account can be switched in place from Personal to a Professional account (Creator or Business). A replacement account is not required.**

The available current evidence indicates that the existing profile, username/handle, posts, followers and messages remain attached to the same account when its account type is switched.

The main practical risk is **privacy**, not loss of followers/posts:

- professional accounts cannot be private;
- if the current personal account is private, switching it to professional makes it public;
- Meta's current professional-account guidance states that pending follow requests are automatically accepted when the account becomes public.

Therefore, if Eirepolitic is currently private, the pending-follow-request queue should be reviewed before any future conversion.

No conversion should be performed during this research phase.

---

## 1. Is conversion in place or is a new account required?

It is an **account-type switch on the existing Instagram account**.

Meta describes professional accounts as either **Business** or **Creator** and exposes a `Switch to professional account` flow. Meta also states that a professional account can later be switched back to Personal or changed between professional account types.

This means the intended path is:

```text
Existing personal Instagram account
        ↓
Switch account type
        ↓
Existing account becomes Creator or Business
```

not:

```text
Existing personal account
        ↓
Create replacement account
        ↓
Attempt to migrate audience/content
```

### Account-preservation confidence

Meta's public Help Center page is the canonical source (`About professional accounts on Instagram`), but its direct page was intermittently login/rate-limit blocked during this research session. A current June 2026 court record reproduces the current Meta Help Center text, including Meta's statement that users may switch back to Personal or change professional account type at any time.

Multiple current 2026 integration/provider guides independently confirm that the same account's followers, posts, messages and username are unaffected by the account-type switch.

**Research conclusion:** high confidence that switching account type does not create a new account or discard the existing audience/content.

---

## 2. Existing followers, posts and handle

The current conversion flow changes the account type rather than replacing the account.

Current 2026 provider documentation explicitly states that switching a Personal account to Professional does not affect existing:

- followers;
- posts/content;
- messages;
- username/handle.

This is consistent with Meta's own language describing the operation as switching the account/profile type and allowing the user to switch back later.

### Recommendation

**Do not create a new Eirepolitic Instagram account simply to gain API access.**

If professional API access is eventually approved, convert the existing account in place unless a later Meta-specific constraint appears that makes doing so unsafe.

---

## 3. Privacy is the biggest conversion risk

Meta's current professional-account guidance states:

- professional accounts cannot be set to private;
- all pending follow requests are automatically accepted when the account becomes public.

Therefore:

### If the current Eirepolitic account is already public

There is no meaningful privacy-state change from conversion itself.

### If the current Eirepolitic account is private

Before any future conversion:

1. inspect pending follow requests;
2. remove/decline any requests that should not become followers;
3. explicitly accept that the account will become public;
4. only then switch account type.

This is the one conversion effect identified here that could change the audience unexpectedly.

---

## 4. Can the account switch back later?

Yes.

Meta's current professional-account guidance says the account can:

- switch back to Personal; or
- change professional account type (Creator ↔ Business).

Switching back removes access to professional tools such as Insights/API-oriented functionality.

Current third-party documentation that references Instagram's current behaviour also reports that historical professional Insights may only remain recoverable for a limited period after switching back. This is not important to the publishing architecture itself, but it means repeated casual switching between account types should not be treated as consequence-free operational practice.

**Architecture implication:** once publishing automation is deployed, the account type should be treated as an operational dependency. If the account is changed back to Personal, API publishing will stop working.

---

## 5. Personal accounts cannot use the publishing API

Meta's current official Instagram API documentation is clear:

- the API works with Instagram **Professional** accounts — Business and Creator;
- the Facebook Login API cannot access consumer/personal Instagram accounts;
- the Instagram Login API is likewise for professional accounts.

Therefore the existing account can remain Personal during research and design, but **real API publishing will eventually require converting that same account to Creator or Business**.

This conversion should be a later explicit approval gate.

---

## 6. Creator versus Business — do not decide yet

Both Creator and Business are professional account types and are eligible for Instagram API publishing in Meta's current documentation.

There are trade-offs that need later research before choosing between them:

- Meta currently documents API Stories publishing as Business-only on the Facebook Login route;
- certain Business accounts and certain types of posts can have restricted access to Instagram's licensed music library;
- tagging/API-login differences may influence which Meta authentication route we need;
- Business may fit an organisation/brand identity better, while Creator may have fewer content/music trade-offs depending on actual usage.

Because the primary requirement here is preservation of the existing account, **Step 2 does not recommend Creator or Business yet**. The later API-capability and authentication steps will determine which is the least disruptive option that still supports the required automation.

---

## 7. Facebook Page linkage

A Facebook Page is **not required merely to switch the Instagram account from Personal to Professional**.

However, Meta currently has two API login models:

- Instagram API with Instagram Login — does not require a linked Facebook Page, but Meta says this setup cannot access tagging;
- Instagram API with Facebook Login — requires a Facebook Page linked to the Professional Instagram account and supports the broader Facebook/Instagram permission model.

Because Eirepolitic wants conversational control over tagging, Page linkage may ultimately be required for the preferred API path.

That is an API/authentication design question, not an account-conversion requirement, and will be researched in Steps 3–5.

---

## 8. Music/library trade-off

Meta's current official Help Center states that its licensed music library is intended for personal, non-commercial use and that **certain Business accounts and certain types of posts do not have access to the licensed music library**.

Meta offers its Sound Collection as an alternative for commercial-safe music where licensed music is unavailable.

This matters if Eirepolitic currently relies on Instagram's licensed music library for Reels or Stories.

Before choosing Business over Creator, later research should check whether Eirepolitic's actual publishing formats rely on music that would be affected.

For static image/carousel publishing, this is unlikely to be a major concern.

---

## 9. Conversion risks identified

| Risk | Result | Severity for Eirepolitic |
|---|---|---|
| Loss of existing followers | No evidence that switching account type removes them | Low |
| Loss of existing posts/media | No evidence that switching account type removes them | Low |
| Username/handle change | Not required by account-type switch | Low |
| New account required | No | Low |
| Account becomes public | Yes if currently private | **Important** |
| Pending follow requests auto-accepted | Yes when a private account becomes public | **Important if currently private** |
| Professional/API features disappear if switched back to Personal | Yes | Medium once automation exists |
| Licensed music availability changes | Possible, especially for certain Business accounts/posts | Medium if Reels/Stories use licensed music |
| Facebook Page required just to convert | No | Low |
| Facebook Page may be required for desired API/tagging route | Yes, potentially | To be decided in later steps |

---

## 10. Decision for the architecture research

Continue the research assuming:

```text
Current Eirepolitic personal account
       ↓ future explicit approval only
Convert SAME account to appropriate Professional type
       ↓
Retain existing profile/content/audience
       ↓
Connect API publishing architecture
```

Do **not** assume a second/replacement Instagram account will be required.

Do **not** switch the live account during research.

The next step is to document the current Meta Instagram publishing API capability matrix before choosing Creator versus Business or an authentication route.

---

## Sources

### Primary / Meta

- Meta Instagram API documentation (official Meta Postman workspace): https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api
  - Current documentation states that Instagram Professionals — Businesses and Creators — are supported and that consumer/personal accounts are not available through the Facebook Login API.
- Meta Instagram API with Facebook Login: https://www.postman.com/meta/instagram/folder/u4g5a2a/instagram-api-with-facebook-login
- Instagram Help Center — About professional accounts on Instagram (canonical Meta help URL): https://www.facebook.com/help/instagram/138925576505882
- Instagram Help Center — Access to the licensed music library on Instagram: https://www.facebook.com/help/instagram/402084904469945

### Current corroboration used where Meta Help Center content was login/rate-limit blocked

- Delhi High Court order dated 29 May 2026 reproducing Meta's current `About professional accounts on Instagram` Help Center text: https://indiankanoon.org/doc/199762348/
- Creator Ads help, updated June 2026, explicitly confirming that switching does not affect followers/posts/existing content: https://creatorads.zendesk.com/hc/en-us/articles/46020285365780-Why-do-you-require-an-Instagram-Professional-account
- SnapWidget help, updated June 2026, confirming followers/posts/messages remain on the account during Creator conversion: https://help.snapwidget.com/en/articles/9955847-how-to-switch-to-a-creator-account-on-instagram-2026-guide

## Confidence / unresolved items

**High confidence:**

- existing account can be converted in place;
- Personal → Creator/Business is supported;
- existing posts/followers are retained;
- professional accounts must be public;
- switching back is supported;
- API publishing requires a professional account.

**Still to resolve in later steps:**

- whether Creator or Business best matches all desired Eirepolitic publishing features;
- exact Facebook Page requirement for the final API route;
- exact tagging/collaborator/location differences;
- whether any Eirepolitic-specific use of music makes Business materially worse than Creator.
