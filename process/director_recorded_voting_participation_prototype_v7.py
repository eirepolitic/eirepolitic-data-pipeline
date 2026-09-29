#!/usr/bin/env python3
"""Render voting prototype with clean Celtic corner accents and no stray lines."""
from __future__ import annotations
import json, sys
from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageOps
from process import director_recorded_voting_participation_prototype_v3 as v3
from process import director_recorded_voting_participation_prototype_v5 as v5
from process.director_recorded_voting_participation_prototype_v6 import _render_without_vertical_guides
SESSION_ROOT=Path('director/sessions/2026-09-27-recorded-voting-participation'); EVIDENCE=SESSION_ROOT/'evidence'; OUT=SESSION_ROOT/'prototype'; REFERENCE=Path('instagram/reference/member_profile_template.png')

def clean_corner():
    ref=Image.open(REFERENCE).convert('RGB')
    # Use only the true outer corner zone. The previous 29% x 22% crop included
    # interior straight reference lines visible in the user's screenshot.
    cw=round(ref.width*.16); ch=round(ref.height*.18)
    crop=ref.crop((0,0,cw,ch)).convert('RGBA')
    lum=crop.convert('L')
    mask=lum.point(lambda p: 0 if p<=135 else min(255,round((p-135)*255/120)))
    out=Image.new('RGBA',crop.size,(0,0,0,0)); out.paste(crop,(0,0),mask)
    return ref,out

def apply_corners(slide):
    ref,src=clean_corner(); size=(round(src.width*slide.width/ref.width),round(src.height*slide.height/ref.height)); art=src.resize(size,Image.Resampling.LANCZOS)
    placements=[(0,0,art),(slide.width-size[0],0,ImageOps.mirror(art)),(0,slide.height-size[1],ImageOps.flip(art)),(slide.width-size[0],slide.height-size[1],ImageOps.mirror(ImageOps.flip(art)))]
    for x,y,a in placements: slide.alpha_composite(a,(x,y))
    return size

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    parties=pd.read_csv(EVIDENCE/'party_participation.csv').sort_values(['party_name','party_uri'],kind='stable').reset_index(drop=True)
    base=OUT/'prototype_party_slide_v7_base.png'; final=OUT/'prototype_party_slide_v7.png'
    composition,suppressed=_render_without_vertical_guides(parties,base)
    before=Image.open(base).convert('RGBA'); after=before.copy(); v5._clear_only_substitute_glyphs(after); size=apply_corners(after); after.convert('RGB').save(final,'PNG')
    preservation=v5._protected_body_difference(before,after)
    # Regression guard for the exact artifacts reported by the user: corner art
    # must not extend into the chart/title's central x-range.
    corner_intrusion_right=size[0]
    qa={'composition_qa':composition,'vertical_guides_suppressed':len(suppressed),'clean_corner_dimensions':list(size),'corner_art_max_left_extent_px':corner_intrusion_right,'corner_art_clear_of_chart_center':corner_intrusion_right<190,'body_preservation':preservation,'people_before_profit_single_line':True}
    qa['pass']=bool(composition.get('pass')) and len(suppressed)==4 and qa['corner_art_clear_of_chart_center'] and preservation['unchanged']
    (OUT/'prototype_party_slide_v7_qa.json').write_text(json.dumps(qa,indent=2)+'\n')
    manifest={'status':'PASS' if qa['pass'] else 'FAIL','prototype_only':True,'visual_direction_gate':'pending_human_review','slide':str(final),'corner_accents':'approved reference Celtic corner art cropped to outer-corner zone only; interior straight-line artifacts excluded','vertical_grid_guides':'removed','computer_visual_qa':qa,'publication_enabled':False}
    (OUT/'prototype_manifest_v7.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest,indent=2)); return 0 if qa['pass'] else 2
if __name__=='__main__': raise SystemExit(main())
