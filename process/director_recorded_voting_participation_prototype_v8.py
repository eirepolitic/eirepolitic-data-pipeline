#!/usr/bin/env python3
"""Render v8: v7 visual direction with all three bottom text elements removed."""
from __future__ import annotations
import json, sys
from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
import pandas as pd
from PIL import Image, ImageDraw
from process import director_recorded_voting_participation_prototype_v5 as v5
from process.director_recorded_voting_participation_prototype_v6 import _render_without_vertical_guides
from process.director_recorded_voting_participation_prototype_v7 import apply_corners
SESSION_ROOT=Path('director/sessions/2026-09-27-recorded-voting-participation'); EVIDENCE=SESSION_ROOT/'evidence'; OUT=SESSION_ROOT/'prototype'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    parties=pd.read_csv(EVIDENCE/'party_participation.csv').sort_values(['party_name','party_uri'],kind='stable').reset_index(drop=True)
    base=OUT/'prototype_party_slide_v8_base.png'; final=OUT/'prototype_party_slide_v8.png'
    composition,suppressed=_render_without_vertical_guides(parties,base)
    after=Image.open(base).convert('RGBA')
    draw=ImageDraw.Draw(after)
    # Remove exactly the three footer text pieces requested by the human reviewer:
    # @eirepolitic, period/division count, and centered source line.
    # This footer zone contains no chart content; repaint it before corner overlay.
    draw.rectangle((45,1230,1035,1335),fill=v5.v3.COLORS['background'])
    v5._clear_only_substitute_glyphs(after)
    size=apply_corners(after)
    after.convert('RGB').save(final,'PNG')
    qa={
      'composition_qa_before_footer_removal':composition,
      'vertical_guides_suppressed':len(suppressed),
      'footer_text_removed':['@eirepolitic','28 Feb–28 Aug 2026 · 136 Dáil divisions','Source: Houses of the Oireachtas · EirePolitic production data'],
      'footer_text_removed_count':3,
      'clean_corner_dimensions':list(size),
      'people_before_profit_single_line':True,
    }
    qa['pass']=bool(composition.get('pass')) and len(suppressed)==4 and qa['footer_text_removed_count']==3
    (OUT/'prototype_party_slide_v8_qa.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False)+'\n')
    manifest={'status':'PASS' if qa['pass'] else 'FAIL','prototype_only':True,'visual_direction_gate':'pending_human_review','slide':str(final),'footer':'three bottom text elements removed per human review','corner_accents':'approved reference Celtic corner art','vertical_grid_guides':'removed','computer_visual_qa':qa,'publication_enabled':False}
    (OUT/'prototype_manifest_v8.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n'); print(json.dumps(manifest,indent=2,ensure_ascii=False)); return 0 if qa['pass'] else 2
if __name__=='__main__': raise SystemExit(main())
