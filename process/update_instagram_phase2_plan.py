from pathlib import Path

# One-time canonical milestone updater. The workflow removes this file after success.
path = Path("instagram/CONTENT_FACTORY_PLAN.md")
text = path.read_text(encoding="utf-8")
marker = "### Phase 2 — complete-slide test generator\n\nBuild:"
replacement = """### Phase 2 — complete-slide test generator

**Status: constituency pilot implemented and technically validated (2026-07-19).**

Validated pilot:

- project: `instagram/projects/constituency_issue_profile_v1/project.yml`
- grain: constituency
- slides: cover and classified issue profile
- minimum, maximum, and real-example complete-slide renders
- per-scenario and per-slide contact sheets
- local fixture and live S3 execution
- production-pointer resolution and join-coverage manifests
- assistant-visible preview publication
- final live validation run: `29703335986`
- preview root: `instagram-preview-output/preview/factory/projects/constituency_issue_profile_v1/`

This validates the Phase 2 architecture for one pilot. A generic multi-project scenario builder remains future work. Human factual and visual approval is still required before batch generation.

Build:"""
if marker not in text:
    raise SystemExit("Phase 2 marker not found")
path.write_text(text.replace(marker, replacement, 1), encoding="utf-8")
