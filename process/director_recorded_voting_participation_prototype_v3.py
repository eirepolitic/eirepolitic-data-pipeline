#!/usr/bin/env python3
"""Render corrected representative party slide for Director visual review."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from instagram.renderer.constants import FONT_CANDIDATES

SESSION_ROOT = Path("director/sessions/2026-09-27-recorded-voting-participation")
EVIDENCE = SESSION_ROOT / "evidence"
OUT = SESSION_ROOT / "prototype"
W, H = 1080, 1350
COLORS = {
    "background": "#0f2f24", "panel": "#173d30", "panel_alt": "#214a3b",
    "text": "#f4ead7", "muted": "#cbbf9f", "accent": "#d8b45f", "grid": "#6c8978",
}

def font(kind: str, size: int) -> ImageFont.ImageFont:
    key = "bold" if kind == "bold" else "regular"
    for candidate in FONT_CANDIDATES[key]:
        if Path(candidate).exists(): return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()

def bbox(draw, xy, text, ft, anchor="la", spacing=4): return tuple(int(v) for v in draw.multiline_textbbox(xy,text,font=ft,anchor=anchor,spacing=spacing))
def intersects(a,b,pad=0): return not (a[2]+pad<=b[0] or b[2]+pad<=a[0] or a[3]+pad<=b[1] or b[3]+pad<=a[1])
def within(box,outer,pad=0): return box[0]>=outer[0]+pad and box[1]>=outer[1]+pad and box[2]<=outer[2]-pad and box[3]<=outer[3]-pad
def fit_single(draw,text,kind,max_size,min_size,max_width):
    for size in range(max_size,min_size-1,-1):
        ft=font(kind,size)
        if draw.textbbox((0,0),text,font=ft)[2]<=max_width:return ft
    return font(kind,min_size)
def fit_wrapped(draw,text,kind,max_size,min_size,max_width,max_lines=2):
    words=text.split()
    for size in range(max_size,min_size-1,-1):
        ft=font(kind,size);lines=[];current=""
        for word in words:
            probe=word if not current else f"{current} {word}"
            if draw.textbbox((0,0),probe,font=ft)[2]<=max_width:current=probe
            else:
                if current:lines.append(current)
                current=word
        if current:lines.append(current)
        if len(lines)<=max_lines:return ft,"\n".join(lines)
    return font(kind,min_size),text
def register(elements,element_id,kind,box,row=None):elements.append({"id":element_id,"kind":kind,"bbox":list(box),"row":row})
def ornament_tile(ft,color):
    probe=Image.new("RGBA",(160,160),(0,0,0,0));d=ImageDraw.Draw(probe);b=d.textbbox((0,0),"❦",font=ft);pad=4
    tile=Image.new("RGBA",(b[2]-b[0]+pad*2,b[3]-b[1]+pad*2),(0,0,0,0));td=ImageDraw.Draw(tile);td.text((pad-b[0],pad-b[1]),"❦",font=ft,fill=color);return tile
def paste_ornament(image,tile,position):
    transformed=tile
    if position in {"tr","br"}:transformed=ImageOps.mirror(transformed)
    if position in {"bl","br"}:transformed=ImageOps.flip(transformed)
    x=10 if position in {"tl","bl"} else W-10-transformed.width;y=8 if position in {"tl","tr"} else H-8-transformed.height
    image.paste(transformed,(x,y),transformed);return (x,y,x+transformed.width,y+transformed.height)
def centered(draw,text,y,ft,fill):
    b=bbox(draw,(W//2,y),text,ft,anchor="ma");draw.text((W//2,y),text,font=ft,fill=fill,anchor="ma");return b

def render(parties: pd.DataFrame, output: Path):
    image=Image.new("RGB",(W,H),COLORS["background"]);draw=ImageDraw.Draw(image);elements=[]
    tile=ornament_tile(font("regular",86),COLORS["accent"])
    for pos in ("tl","tr","bl","br"):register(elements,f"ornament_{pos}","ornament",paste_ornament(image,tile,pos))
    title="Recorded Voting Participation — Parties";title_ft=fit_single(draw,title,"bold",44,38,900);register(elements,"title","text",centered(draw,title,78,title_ft,COLORS["text"]))
    subtitle="Share of eligible Dáil division opportunities with a recorded Tá, Níl or formal abstention";register(elements,"subtitle","text",centered(draw,subtitle,145,font("regular",22),COLORS["muted"]))
    qualifier="Alphabetical order · recorded participation is not an attendance measure";register(elements,"qualifier","text",centered(draw,qualifier,181,font("regular",18),COLORS["muted"]))
    rule=(70,222,1010,230);draw.rectangle(rule,fill=COLORS["accent"]);register(elements,"top_rule","rule",rule)
    panel=(58,258,1022,1192);draw.rounded_rectangle(panel,radius=28,fill=COLORS["panel"],outline=COLORS["panel_alt"],width=2);register(elements,"panel","container",panel)
    header_ft=font("bold",15);draw.text((86,282),"PARTY / GROUP",font=header_ft,fill=COLORS["muted"]);draw.text((982,282),"RATE  ·  RECORDED / ELIGIBLE",font=header_ft,fill=COLORS["muted"],anchor="ra")
    register(elements,"header_left","text",bbox(draw,(86,282),"PARTY / GROUP",header_ft));register(elements,"header_right","text",bbox(draw,(982,282),"RATE  ·  RECORDED / ELIGIBLE",header_ft,anchor="ra"))
    chart_top,chart_bottom=320,1144;row_h=(chart_bottom-chart_top)/len(parties);label_x,label_w,bar_x,bar_w,value_x=86,320,420,370,982
    for tick in (25,50,75,100):
        gx=int(bar_x+bar_w*tick/100);draw.line((gx,chart_top+4,gx,chart_bottom-5),fill=COLORS["grid"],width=1);draw.text((gx,chart_bottom+8),f"{tick}%",font=font("regular",12),fill=COLORS["muted"],anchor="ma")
    for idx,row in enumerate(parties.itertuples(index=False)):
        y0=int(chart_top+idx*row_h);y1=int(chart_top+(idx+1)*row_h);cy=(y0+y1)//2
        if idx:draw.line((80,y0,994,y0),fill=COLORS["panel_alt"],width=1)
        raw_name=str(row.party_name);display_name="People Before Profit" if "People Before Profit" in raw_name else raw_name
        if display_name=="People Before Profit":name_ft=fit_single(draw,display_name,"bold",20,15,label_w);name_text=display_name
        else:name_ft,name_text=fit_wrapped(draw,display_name,"bold",20,16,label_w)
        name_box=bbox(draw,(label_x,cy),name_text,name_ft,anchor="lm",spacing=2);draw.multiline_text((label_x,cy),name_text,font=name_ft,fill=COLORS["text"],anchor="lm",spacing=2);register(elements,f"party_{idx}_name","text",name_box,row=idx)
        pct=float(row.recorded_participation_pct);bar_h=34;full_bar=(bar_x,cy-bar_h//2,bar_x+bar_w,cy+bar_h//2);value_bar=(bar_x,cy-bar_h//2,int(bar_x+bar_w*max(0,min(100,pct))/100),cy+bar_h//2)
        draw.rounded_rectangle(full_bar,radius=8,fill=COLORS["panel_alt"]);draw.rounded_rectangle(value_bar,radius=8,fill=COLORS["accent"]);register(elements,f"party_{idx}_bar","bar",value_bar,row=idx)
        pct_text=f"{pct:.1f}%";pct_ft=font("bold",22);pct_box=bbox(draw,(value_x,cy-5),pct_text,pct_ft,anchor="rs");draw.text((value_x,cy-5),pct_text,font=pct_ft,fill=COLORS["text"],anchor="rs");register(elements,f"party_{idx}_pct","text",pct_box,row=idx)
        denom_text=f"{int(row.recorded_participation_opportunities):,} / {int(row.eligible_division_opportunities):,}";denom_ft=font("regular",14);denom_box=bbox(draw,(value_x,cy+11),denom_text,denom_ft,anchor="ra");draw.text((value_x,cy+11),denom_text,font=denom_ft,fill=COLORS["muted"],anchor="ra");register(elements,f"party_{idx}_denom","text",denom_box,row=idx)
    footer_ft=font("regular",16);draw.text((66,1278),"@eirepolitic",font=footer_ft,fill=COLORS["muted"],anchor="la");draw.text((1014,1278),"28 Feb–28 Aug 2026 · 136 Dáil divisions",font=footer_ft,fill=COLORS["muted"],anchor="ra")
    register(elements,"footer_left","text",bbox(draw,(66,1278),"@eirepolitic",footer_ft));register(elements,"footer_right","text",bbox(draw,(1014,1278),"28 Feb–28 Aug 2026 · 136 Dáil divisions",footer_ft,anchor="ra"))
    source="Source: Houses of the Oireachtas · EirePolitic production data";source_ft=font("regular",12);draw.text((W//2,1310),source,font=source_ft,fill=COLORS["muted"],anchor="ma");register(elements,"source","text",bbox(draw,(W//2,1310),source,source_ft,anchor="ma"))
    collisions=[];texts=[e for e in elements if e["kind"]=="text"]
    for i,a in enumerate(texts):
        for b in texts[i+1:]:
            if a.get("row") is not None and a.get("row")==b.get("row") and {a["id"].split("_")[-1],b["id"].split("_")[-1]}=={"pct","denom"}:continue
            if intersects(tuple(a["bbox"]),tuple(b["bbox"]),2):collisions.append({"a":a["id"],"b":b["id"]})
    for e in texts:
        if e.get("row") is None:continue
        bar=next(x for x in elements if x["id"]==f"party_{e['row']}_bar")
        if intersects(tuple(e["bbox"]),tuple(bar["bbox"]),5):collisions.append({"a":e["id"],"b":bar["id"]})
    out_of_bounds=[e["id"] for e in elements if e["kind"] in {"text","ornament"} and not within(tuple(e["bbox"]),(0,0,W,H),4)]
    panel_texts=[e for e in texts if e["id"].startswith("party_") or e["id"].startswith("header_")];panel_out=[e["id"] for e in panel_texts if not within(tuple(e["bbox"]),panel,16)]
    row_spacing=[]
    for idx in range(len(parties)-1):
        ra=[e for e in texts if e.get("row")==idx];rb=[e for e in texts if e.get("row")==idx+1];gap=min(e["bbox"][1] for e in rb)-max(e["bbox"][3] for e in ra)
        if gap<7:row_spacing.append({"rows":[idx,idx+1],"gap_px":gap})
    qa={"text_collision_count":len(collisions),"text_collisions":collisions,"out_of_bounds_count":len(out_of_bounds),"out_of_bounds":out_of_bounds,"panel_text_out_of_bounds_count":len(panel_out),"panel_text_out_of_bounds":panel_out,"row_spacing_issue_count":len(row_spacing),"row_spacing_issues":row_spacing,"ornament_count":4,"ornament_transform_reference":{"tl":"normal","tr":"scaleX(-1)","bl":"scaleY(-1)","br":"scale(-1,-1)"},"people_before_profit_display_label":"People Before Profit","people_before_profit_single_line":True}
    qa["pass"]=not collisions and not out_of_bounds and not panel_out and not row_spacing;output.parent.mkdir(parents=True,exist_ok=True);image.save(output,"PNG");return qa

def main():
    OUT.mkdir(parents=True,exist_ok=True);parties=pd.read_csv(EVIDENCE/"party_participation.csv").sort_values(["party_name","party_uri"],kind="stable").reset_index(drop=True);slide=OUT/"prototype_party_slide_v3.png";qa=render(parties,slide);(OUT/"prototype_party_slide_v3_qa.json").write_text(json.dumps(qa,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");return 0 if qa["pass"] else 2
if __name__=="__main__":raise SystemExit(main())
