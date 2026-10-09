#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import process.tmp_render_all_second_stage_posts as renderer

ENRICH=Path('artifacts/second-stage-enrichment/copy.json')
if not ENRICH.is_file():
    raise SystemExit(f'missing enrichment dataset: {ENRICH}')
data=json.loads(ENRICH.read_text(encoding='utf-8'))
rows=data.get('rows',[])
if len(rows)!=108:
    raise SystemExit(f'expected 108 enrichment rows, got {len(rows)}')
by_id={r['bill_id']:r for r in rows}

_original_generic=renderer.generic_copy

def enriched_generic(row):
    item=by_id.get(str(row.get('bill_id','')))
    if not item:
        return _original_generic(row)
    vals=tuple(str(item.get(k,'')).strip() for k in ('what','practical','context','issues'))
    if not all(vals):
        return _original_generic(row)
    return vals

renderer.generic_copy=enriched_generic
renderer.main()
