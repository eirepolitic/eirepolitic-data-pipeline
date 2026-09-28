# Instagram review asset delivery standard

*Added 2026-09-27 after live review testing with Warren.*

## Purpose

Generated Instagram review assets must be easy to open directly from chat on desktop and mobile. Do not make the reviewer download files, navigate GitHub, or fight the browser's native full-size image viewer.

## Canonical review flow

For every meaningful slide revision:

1. Render through the repository workflow that will actually generate the production/review asset.
2. Confirm the workflow completed successfully and declarative QA passed.
3. Verify the published preview branch contains the expected PNG(s), contact sheet(s), and review page.
4. Reuse a stable `previews/<project-slug>` branch across revisions unless parallel alternatives genuinely need separate branches.
5. Return `raw.githack.com` browser links only after the render/publish workflow has completed and the preview branch has been verified.

## Single-slide review

Do **not** use the direct PNG URL as the primary review link. Some browsers open a 1080×1350 PNG at native size and do not provide enough zoom-out range to see the whole slide.

Instead, publish a small HTML wrapper beside the rendered PNG, for example:

`slide-001-01-headline-fit.html`

The wrapper must display the real rendered PNG and constrain it to the viewport. A minimal working pattern is:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=5,user-scalable=yes">
<style>
  html, body { margin: 0; padding: 0; width: 100%; min-height: 100%; background: #111; }
  body { display: flex; align-items: center; justify-content: center; min-height: 100vh; overflow: auto; }
  .frame { box-sizing: border-box; width: 100vw; height: 100vh; padding: 8px; display: flex; align-items: center; justify-content: center; }
  img { display: block; max-width: calc(100vw - 16px); max-height: calc(100vh - 16px); width: auto; height: auto; object-fit: contain; }
</style>
</head>
<body>
  <div class="frame">
    <img src="slide-001-01-headline.png" alt="Slide review">
  </div>
</body>
</html>
```

Return the wrapper URL using the pattern:

`https://raw.githack.com/eirepolitic/eirepolitic-data-pipeline/previews/<project-slug>/<slide>-fit.html`

The PNG remains the canonical rendered asset; the HTML file is only a review surface.

## Multi-slide review

For a carousel or sequence review, publish and return:

- `index.html` containing the rendered slides in intended order;
- a contact sheet where useful;
- the individual PNGs;
- fit-to-screen single-slide wrappers for slides being reviewed individually.

The generic factory's `index.html`, contact sheets, and `slides.zip` remain useful, but individual review should use the fit wrapper above when browser-native image display is inconvenient.

## Visual assets

When a project is intended to follow the standard EirePolitic house style, use the pinned canonical corner ornaments rather than approximate replacements:

- `instagram/templates/assets/corner_tl.png`
- `instagram/templates/assets/corner_tr.png`
- `instagram/templates/assets/corner_bl.png`
- `instagram/templates/assets/corner_br.png`

These are pinned pixel-critical assets. Current canonical placement in `title_text_media_v1.json` is 155×155 at the four canvas corners. Verify the pinned ref/hashes in `director/visuals.yml` and the factory workflow before changing them.

## What not to return as the primary review surface

- `/mnt/data/...` paths;
- ChatGPT sandbox download links;
- GitHub repository-navigation links;
- direct raw PNG links when the reviewer needs the whole slide fit to screen;
- mock SVGs or locally generated substitutes when the repository render path is available.
