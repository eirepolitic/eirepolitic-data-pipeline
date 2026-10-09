#!/usr/bin/env python3
from __future__ import annotations

import io
import json
from datetime import date
from pathlib import Path

import boto3
import pandas as pd
from PIL import Image, ImageDraw

import process.tmp_render_all_second_stage_posts as renderer
from extract.oireachtas.batch import resolve_production_key

ENRICH = Path('artifacts/second-stage-enrichment/copy.json')
if not ENRICH.is_file():
    raise SystemExit(f'missing enrichment dataset: {ENRICH}')

data = json.loads(ENRICH.read_text(encoding='utf-8'))
rows = data.get('rows', [])
if len(rows) != 108:
    raise SystemExit(f'expected 108 enrichment rows, got {len(rows)}')
by_id = {r['bill_id']: r for r in rows}

_original_generic = renderer.generic_copy
_original_load_rows = renderer.load_rows


def enriched_generic(row):
    item = by_id.get(str(row.get('bill_id', '')))
    if not item:
        return _original_generic(row)
    vals = tuple(str(item.get(k, '')).strip() for k in ('what', 'practical', 'context', 'issues'))
    return vals if all(vals) else _original_generic(row)


def _read_latest_csv(key: str) -> pd.DataFrame:
    s3 = boto3.client('s3', region_name='ca-central-1')
    real = resolve_production_key(s3, bucket=renderer.BUCKET, production_key=key)
    obj = s3.get_object(Bucket=renderer.BUCKET, Key=real)
    return pd.read_csv(io.BytesIO(obj['Body'].read()), dtype=str, keep_default_na=False)


def enriched_load_rows():
    bill_rows, production_key = _original_load_rows()
    bridge = _read_latest_csv(renderer.KEYS['bridge'])
    speeches = _read_latest_csv(renderer.KEYS['speeches'])

    latest_debate = {}
    if {'bill_id', 'debate_section_id'}.issubset(bridge.columns) and {'debate_section_id', 'debate_date'}.issubset(speeches.columns):
        joined = bridge[['bill_id', 'debate_section_id']].drop_duplicates().merge(
            speeches[['debate_section_id', 'debate_date']].drop_duplicates(),
            on='debate_section_id', how='inner'
        )
        joined['_date'] = pd.to_datetime(joined['debate_date'], errors='coerce')
        joined = joined.dropna(subset=['_date'])
        if not joined.empty:
            latest_debate = joined.groupby('bill_id')['_date'].max().dt.strftime('%Y-%m-%d').to_dict()

    for row in bill_rows:
        row['latest_debate_date'] = latest_debate.get(str(row.get('bill_id', '')), '')
    return bill_rows, production_key


def _parse(value):
    if not value:
        return None
    ts = pd.to_datetime(str(value), errors='coerce')
    if pd.isna(ts):
        return None
    return ts.date()


def _fmt_date(value) -> str:
    d = _parse(value)
    return d.strftime('%-d %B %Y') if d else ''


def _duration(start, end) -> str:
    a = _parse(start)
    b = _parse(end)
    if not a or not b or b < a:
        return ''
    days = (b - a).days
    if days < 14:
        return f'{days} day' + ('' if days == 1 else 's')
    if days < 60:
        weeks = max(2, days // 7)
        return f'{weeks} weeks'
    months = (b.year - a.year) * 12 + (b.month - a.month)
    if b.day < a.day:
        months = max(0, months - 1)
    if months < 12:
        return f'{max(1, months)} month' + ('' if months == 1 else 's')
    years, rem = divmod(months, 12)
    if rem == 0:
        return f'{years} year' + ('' if years == 1 else 's')
    return f'{years} year' + ('' if years == 1 else 's') + f', {rem} month' + ('' if rem == 1 else 's')


def _house_plain(value: str) -> str:
    low = str(value or '').lower()
    if 'dáil' in low or 'dail' in low:
        return 'Dáil'
    if 'seanad' in low:
        return 'Seanad'
    return 'House'


def status_parts(row):
    snapshot = row.get('snapshot_date') or date.today().isoformat()
    stage_date = row.get('current_stage_date', '')
    debate_date = row.get('latest_debate_date', '')
    division_date = row.get('latest_division_date', '')
    last_event = row.get('last_event_date', '')
    house = _house_plain(row.get('current_stage_house_name', ''))
    outcome = str(row.get('current_stage_outcome', '') or '').strip().lower()

    duration = _duration(stage_date, snapshot) or 'an unknown length of time'
    headline = f'AT SECOND STAGE FOR {duration.upper()}'
    entered = f'Reached Second Stage: {_fmt_date(stage_date)}' if _fmt_date(stage_date) else 'Reached Second Stage: date not available'

    candidates = []
    for kind, value in [('stage', stage_date), ('debate', debate_date), ('vote', division_date), ('update', last_event)]:
        d = _parse(value)
        if d:
            candidates.append((d, kind, value))
    candidates.sort(key=lambda x: (x[0], {'stage': 0, 'debate': 2, 'vote': 1, 'update': 3}[x[1]]))
    last_date, kind, raw = candidates[-1] if candidates else (None, '', '')

    # Prefer a known debate over a same-day generic update; it is clearer to readers.
    dd = _parse(debate_date)
    if dd and last_date and dd == last_date:
        kind, raw = 'debate', debate_date
    vd = _parse(division_date)
    if vd and last_date and vd == last_date and not (dd and dd == last_date):
        kind, raw = 'vote', division_date

    pretty = _fmt_date(raw)
    if kind == 'debate':
        activity = f'Last activity: The Bill was discussed in the {house} on {pretty}.'
    elif kind == 'vote':
        activity = f'Last activity: A recorded vote linked to the Bill took place on {pretty}.'
    elif kind == 'update':
        activity = f'Last activity: The latest recorded update for the Bill was on {pretty}.'
    elif kind == 'stage':
        activity = f'Last activity: The Bill reached Second Stage on {pretty}.'
    else:
        activity = 'Last activity: No later activity is recorded.'

    if 'adjourn' in outcome:
        follow = 'The debate was paused before it finished, and no later progress is recorded.'
    elif 'postpon' in outcome or 'defer' in outcome:
        follow = 'Further consideration was postponed, and no later progress is recorded.'
    else:
        since = _duration(raw, snapshot) if raw else ''
        if kind == 'debate' and since:
            follow = f'It has remained at Second Stage for {since} since that discussion.'
        elif kind in {'update', 'vote'} and since:
            follow = f'No move to the next stage is recorded in the {since} since then.'
        elif kind == 'stage':
            follow = 'There is no recorded Second Stage debate yet, and it has not moved to the next stage.'
        else:
            follow = 'It has not moved to the next stage.'

    if not any(k in outcome for k in ('adjourn', 'postpon', 'defer')):
        reason = 'The official record does not say why it has not progressed further.'
    else:
        reason = ''
    return headline, entered, activity, follow, reason


def enriched_bill_slide(out, idx, row):
    im = renderer.base_slide()
    d = ImageDraw.Draw(im)
    title = row['title']
    tf = renderer.fit(d, title, 900, 95, 34, 22, 3, True)
    lines = renderer.wrap_text_px(d, title, tf, 900)
    cur = 54
    for ln in lines:
        d.text((renderer.W // 2, cur), ln, font=tf, fill=renderer.TEXT, anchor='ma')
        bb = d.textbbox((renderer.W // 2, cur), ln, font=tf, anchor='ma')
        cur = bb[3] + 4
    renderer.rule(d, max(145, cur + 8))
    ry = max(175, cur + 38)
    sponsor = str(row.get('primary_sponsor_name', '')).strip() or str(row.get('primary_sponsor_role_name', '')).strip() or 'Sponsor not named in snapshot'
    meta = f"Bill No. {row.get('bill_no','')} of {row.get('bill_year','')} · {row.get('origin_house_name','')} · {sponsor}"
    mf = renderer.fit(d, meta, 900, 45, 17, 13, 2)
    renderer.draw_block(d, meta, (90, ry, 990, ry + 48), mf, fill=renderer.MUTED, center=True)

    texts = renderer.SPECIAL.get(title) or enriched_generic(row)
    labels = ['WHAT THE BILL COVERS', 'PRACTICAL EFFECT', 'SPONSOR / CONTEXT', 'SECOND STAGE RECORD']
    top = ry + 62
    bh = 282
    gap = 22
    boxes = [(58, top, 518, top + bh), (562, top, 1022, top + bh), (58, top + bh + gap, 518, top + 2 * bh + gap), (562, top + bh + gap, 1022, top + 2 * bh + gap)]
    shared = 14
    for size in range(24, 13, -1):
        ff = renderer.font(size)
        if all(renderer.lines_h(d, renderer.wrap_text_px(d, txt, ff, 408), ff, 4) <= 190 for txt in texts):
            shared = size
            break
    bf = renderer.font(shared)
    for box, label, txt in zip(boxes, labels, texts):
        renderer.panel(d, box)
        d.text((box[0] + 22, box[1] + 20), label, font=renderer.font(16, True), fill=renderer.ACCENT, anchor='la')
        end = renderer.draw_block(d, txt, (box[0] + 22, box[1] + 58, box[2] - 22, box[3] - 18), bf)
        if end > box[3] - 12:
            raise RuntimeError(f'overflow {title} {label} at shared font {shared}')

    sy = boxes[2][3] + 28
    renderer.panel(d, (58, sy, 1022, 1210), outline=renderer.ACCENT, fill=renderer.BG, w=3)
    d.text((renderer.W // 2, sy + 22), 'SECOND STAGE STATUS', font=renderer.font(18, True), fill=renderer.ACCENT, anchor='ma')
    headline, entered, activity, follow, reason = status_parts(row)
    d.text((renderer.W // 2, sy + 57), headline, font=renderer.font(25, True), fill=renderer.TEXT, anchor='ma')
    status_text = '\n'.join([x for x in (entered, activity, follow, reason) if x])
    sf = renderer.fit(d, status_text.replace('\n', ' '), 850, 125, 20, 16, 7)
    y = sy + 91
    for line in status_text.split('\n'):
        wrapped = renderer.wrap_text_px(d, line, sf, 850)
        for sub in wrapped:
            d.text((renderer.W // 2, y), sub, font=sf, fill=renderer.TEXT, anchor='ma')
            bb = d.textbbox((renderer.W // 2, y), sub, font=sf, anchor='ma')
            y = bb[3] + 3
        y += 3
    if y > 1195:
        raise RuntimeError(f'status overflow {title}: y={y}')

    renderer.footer(d, f'Second Stage · Bill {idx}', source=f"Source: Houses of the Oireachtas · {row.get('bill_id','')}")
    im.save(out / f'{idx:02d}-bill.png')


renderer.generic_copy = enriched_generic
renderer.load_rows = enriched_load_rows
renderer.bill_slide = enriched_bill_slide
renderer.main()
