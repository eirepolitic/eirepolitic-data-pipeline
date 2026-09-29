#!/usr/bin/env python3
"""Build full review carousel from validated recorded-voting evidence.
Publication is intentionally disabled; output is for Director/human review only.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
import pandas as pd
from PIL import Image,ImageDraw,ImageFont,ImageOps
from instagram.renderer.constants import FONT_CANDIDATES
from process.director_recorded_voting_participation_prototype_v7 import clean_corner
W,H=1080,1350; C={'bg':'#0f2f24','panel':'#173d30','alt':'#214a3b','text':'#f4ead7','muted':'#cbbf9f','gold':'#d8b45f'}
SESSION=Path('director/sessions/2026-09-27-recorded-voting-participation'); EV=SESSION/'evidence'; OUT=SESSION/'carousel_review'
def ft(size,bold=False):
 key='bold' if bold else 'regular'
 for p in FONT_CANDIDATES[key]:
  if Path(p).exists(): return ImageFont.truetype(p,size)
 return ImageFont.load_default()
def corners(im):
 ref,src=clean_corner(); size=(round(src.width*W/ref.width),round(src.height*H/ref.height));a=src.resize(size,Image.Resampling.LANCZOS)
 for x,y,z in [(0,0,a),(W-size[0],0,ImageOps.mirror(a)),(0,H-size[1],ImageOps.flip(a)),(W-size[0],H-size[1],ImageOps.mirror(ImageOps.flip(a)))]:im.alpha_composite(z,(x,y))
def base(title,subtitle=None):
 im=Image.new('RGBA',(W,H),C['bg']);d=ImageDraw.Draw(im);corners(im);tf=ft(43,True)
 while d.textbbox((0,0),title,font=tf)[2]>850 and tf.size>31:tf=ft(tf.size-1,True)
 d.text((W//2,80),title,font=tf,fill=C['text'],anchor='ma')
 if subtitle:d.text((W//2,145),subtitle,font=ft(20),fill=C['muted'],anchor='ma')
 d.rectangle((70,215,1010,223),fill=C['gold']);return im,d
def wrap(d,text,font,maxw):
 words=text.split();lines=[];cur=''
 for w in words:
  p=w if not cur else cur+' '+w
  if d.textbbox((0,0),p,font=font)[2]<=maxw:cur=p
  else:lines.append(cur);cur=w
 if cur:lines.append(cur)
 return '\n'.join(lines)
def save(im,n):p=OUT/f'{n:02d}.png';im.convert('RGB').save(p);return str(p)
def table_slide(title,subtitle,df,name_col,rows_per=18):
 slides=[]
 for page,start in enumerate(range(0,len(df),rows_per),1):
  chunk=df.iloc[start:start+rows_per];suffix=f' · {page}/{math.ceil(len(df)/rows_per)}' if len(df)>rows_per else ''
  im,d=base(title+suffix,subtitle);panel=(58,255,1022,1215);d.rounded_rectangle(panel,28,fill=C['panel'],outline=C['alt'],width=2)
  d.text((86,280),name_col.replace('_',' ').upper(),font=ft(14,True),fill=C['muted']);d.text((982,280),'RATE  ·  RECORDED / ELIGIBLE',font=ft(14,True),fill=C['muted'],anchor='ra')
  top,bottom=315,1170;rh=(bottom-top)/len(chunk)
  for j,row in enumerate(chunk.itertuples(index=False)):
   cy=int(top+(j+.5)*rh);y0=int(top+j*rh)
   if j:d.line((80,y0,994,y0),fill=C['alt'],width=1)
   name=str(getattr(row,name_col)); name='People Before Profit' if 'People Before Profit' in name else name
   nf=ft(17,True)
   while d.textbbox((0,0),name,font=nf)[2]>315 and nf.size>12:nf=ft(nf.size-1,True)
   d.text((86,cy),name,font=nf,fill=C['text'],anchor='lm')
   pct=float(row.recorded_participation_pct);bx,bw=420,370;bh=24;d.rounded_rectangle((bx,cy-bh//2,bx+bw,cy+bh//2),7,fill=C['alt']);d.rounded_rectangle((bx,cy-bh//2,int(bx+bw*pct/100),cy+bh//2),7,fill=C['gold'])
   d.text((982,cy-4),f'{pct:.1f}%',font=ft(18,True),fill=C['text'],anchor='rs');d.text((982,cy+11),f'{int(row.recorded_participation_opportunities):,} / {int(row.eligible_division_opportunities):,}',font=ft(12),fill=C['muted'],anchor='ra')
  slides.append(im)
 return slides
def main():
 OUT.mkdir(parents=True,exist_ok=True);party=pd.read_csv(EV/'party_participation.csv').sort_values('party_name');const=pd.read_csv(EV/'constituency_participation.csv').sort_values('constituency_name');td=pd.read_csv(EV/'td_participation.csv').sort_values('member_name')
 imgs=[]
 im,d=base('Recorded Voting Participation','28 Feb–28 Aug 2026 · Dáil Éireann');d.text((W//2,525),'How often did TDs have a recorded vote\nor recorded abstention when eligible\nto participate in a Dáil division?',font=ft(38,True),fill=C['text'],anchor='ma',align='center',spacing=14);d.text((W//2,770),'136 divisions · 23,408 eligible opportunities',font=ft(22),fill=C['muted'],anchor='ma');imgs.append(im)
 im,d=base('What This Metric Measures');d.rounded_rectangle((100,300,980,1035),30,fill=C['panel']);blocks=[('Recorded participation','Eligible opportunities with a recorded Tá, Níl or formal abstention.'),('Eligible divisions','TD membership dates determine whether a division enters that TD’s denominator.'),('What “no recorded vote” means','Only that no qualifying vote or abstention was recorded for that eligible opportunity. It does not establish physical absence.'),('Overall','19,956 recorded participation opportunities out of 23,408 eligible opportunities (85.3%).')];y=355
 for h,t in blocks:d.text((145,y),h,font=ft(24,True),fill=C['gold']);body=wrap(d,t,ft(21),760);d.multiline_text((145,y+38),body,font=ft(21),fill=C['text'],spacing=7);y+=165
 imgs.append(im)
 imgs+=table_slide('Recorded Voting Participation — Parties','Alphabetical · recorded participation is not an attendance measure',party,'party_name',18)
 imgs+=table_slide('Recorded Voting Participation — Constituencies','Alphabetical · totals are summed opportunities, not averaged TD percentages',const,'constituency_name',15)
 imgs+=table_slide('Recorded Voting Participation — TDs','Alphabetical · recorded participation is not an attendance measure',td,'member_name',18)
 im,d=base('Methodology & Limitations');d.rounded_rectangle((80,270,1000,1110),28,fill=C['panel']);items=['Period: 28 Feb–28 Aug 2026 inclusive; 136 production-supported Dáil divisions.','Grain: one eligible TD × division opportunity. Numerator = recorded Tá, Níl or formal abstention; denominator = eligible division opportunities.','Membership dates define eligibility. Party and constituency are attributed using event-date histories.','The identified presiding member is excluded from an ordinary eligible opportunity unless that member has a recorded vote, preserving a casting-vote case.','73 formal abstention records occurred across 3 divisions.','No canonical pairing/statutory-leave field exists in the promoted production batch, so the denominator is not adjusted for those circumstances.','Recorded voting participation does not by itself measure a TD’s attendance, workload, effectiveness, or overall job performance.'];y=320
 for x in items:body=wrap(d,'• '+x,ft(19),820);d.multiline_text((125,y),body,font=ft(19),fill=C['text'],spacing=6);y+=d.multiline_textbbox((125,y),body,font=ft(19),spacing=6)[3]-y+28
 imgs.append(im)
 paths=[save(im,i+1) for i,im in enumerate(imgs)];manifest={'status':'PASS','review_only':True,'publication_enabled':False,'visual_direction':'approved_v8','slide_count':len(paths),'slides':paths,'period':{'start':'2026-02-28','end':'2026-08-28'},'division_count':136,'eligible_opportunities':23408,'recorded_opportunities':19956,'ordering':'alphabetical for parties, constituencies and TDs','required_statement':'Recorded voting participation does not by itself measure a TD’s attendance, workload, effectiveness, or overall job performance.'};(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n');print(json.dumps(manifest,indent=2,ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
