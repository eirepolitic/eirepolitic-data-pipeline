# EirePolitic Director Agent Reference

> **Canonical review-asset delivery note (2026-09-27):** For generated Instagram review assets, follow `docs/operations/instagram_review_assets.md`. In particular, single-slide reviews should be repository-rendered PNGs published to a stable preview branch and presented through a fit-to-screen HTML wrapper via `raw.githack.com`, rather than sandbox links or direct full-size PNG links.

This file is the standalone Director briefing. If this briefing and live code/docs disagree, the current `main` implementation plus the relevant operations runbook win.

## Operating principle

Treat repository state, live AWS state, and current operations documentation as evidence. Do not carry forward stale assumptions from earlier sessions when the live system can be checked.

## Publishing status

Production Instagram publishing was proven end-to-end on 2026-09-26 using the standard DynamoDB + EventBridge Scheduler + Lambda path. Resolve any metadata/catalog discrepancy against the current implementation and `docs/operations/instagram_publishing_standard.md` before acting.

## Visual references

Check `director/references.yml` before making visual-direction decisions. Check `director/visuals.yml` before changing layouts, renderers, fonts, palettes, or pixel-critical assets.

## Review asset delivery

Follow `docs/operations/instagram_review_assets.md`. Use the repository render workflow, verify QA/output, publish to a stable `previews/<project>` branch, and return `raw.githack.com` review links. For single-slide review, prefer a viewport-fit HTML wrapper around the canonical rendered PNG so the entire slide is visible on the reviewer’s device.

## Approval discipline

Content idea, visual direction, and final approval remain human gates. Never enable publication flags or schedule/publish without the explicit approval required by the current workflow documentation.
