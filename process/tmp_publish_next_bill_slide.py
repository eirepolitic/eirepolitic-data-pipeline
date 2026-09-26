#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT=Path('artifacts/bill-next-slide'); OUT.mkdir(parents=True,exist_ok=True)
W,H=1080,1350
BG='#0f2f24'; TEXT='#f4ead7'; MUTED='#c8bda8'; ACCENT='#d8b45f'; FOR=ACCENT; AGAINST=TEXT; NO_VOTE='#65756d'
PARTIES=[
('Fianna Fáil',48,0,41,7),('Sinn Féin',39,37,0,2),('Fine Gael',38,0,32,6),('Independent',15,3,6,6),
('Social Democrats',12,11,0,1),('Labour',11,9,0,2),('Independent Ireland',4,2,0,2),('PBP–S',3,2,0,1),
('Aontú',2,2,0,0),('Green',1,1,0,0),('100% Redress',1,0,0,1)]

def font(size,bold=False):
    paths=[('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),('/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf')]
    for p in paths:
        if Path(p).exists(): return ImageFont.truetype(p,size=size)
    return ImageFont.load_default()

def size(d,t,f):
    b=d.textbbox((0,0),str(t),font=f); return b[2]-b[0],b[3]-b[1]

def centered(d,y,t,f,fill):
    tw,_=size(d,t,f); d.text(((W-tw)/2,y),t,font=f,fill=fill)

def bar(d,x,y,w,h,yes,no,nr):
    total=max(1,yes+no+nr); vals=[(yes,FOR),(no,AGAINST),(nr,NO_VOTE)]; vals=[v for v in vals if v[0]>0]; cur=x
    for i,(val,color) in enumerate(vals):
        sw=(x+w-cur) if i==len(vals)-1 else round(w*val/total); d.rectangle([cur,y,cur+sw,y+h],fill=color); cur+=sw
    d.rectangle([x,y,x+w,y+h],outline=MUTED,width=2)

im=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(im)
# Approved B3 geometry.
centered(d,62,'Israeli Settlements Bill · Services Amendment',font(40,True),TEXT)
d.rectangle([110,130,970,135],fill=ACCENT)
centered(d,160,'Dáil amendment No. 16 · 7 July 2026',font(25,True),ACCENT)
bar(d,70,208,940,56,67,79,28)
legend=[('Tá',FOR),('Níl',AGAINST),('No recorded vote',NO_VOTE)]; lf=font(19,True); widths=[18+12+size(d,l,lf)[0] for l,_ in legend]; gap=42; cur=(W-(sum(widths)+gap*2))/2
for (label,color),iw in zip(legend,widths):
    d.rectangle([cur,290,cur+18,308],fill=color,outline=MUTED); d.text((cur+30,286),label,font=lf,fill=TEXT); cur+=iw+gap
centered(d,328,'174 eligible TDs · 67 Tá · 79 Níl · 0 abstentions · 28 no recorded vote',font(20,True),TEXT)
centered(d,382,'PARTY BREAKDOWN',font(31,True),ACCENT)
name_x=64; bar_x=348; bar_w=470; row_top=438; row_h=73; bar_h=36; nf=font(24,True); numf=font(20,True); hf=font(16,True); centers=[866,922,998]
for (label,color),cx in zip([('Tá',ACCENT),('Níl',TEXT),('No vote',MUTED)],centers):
    tw,_=size(d,label,hf); d.text((cx-tw/2,row_top-32),label,font=hf,fill=color)
for i,(name,eligible,yes,no,nr) in enumerate(PARTIES[:8]):
    y=row_top+i*row_h; d.text((name_x,y+2),name,font=nf,fill=TEXT); bar(d,bar_x,y,bar_w,bar_h,yes,no,nr)
    for (n,color),cx in zip([(str(yes),ACCENT),(str(no),TEXT),(str(nr),MUTED)],centers):
        tw,_=size(d,n,numf); d.text((cx-tw/2,y+5),n,font=numf,fill=color)
centered(d,1064,'Smaller groups',font(27,True),ACCENT)
centered(d,1102,'Aontú 2 / 0 / 0 · Green 1 / 0 / 0 · 100% Redress 0 / 0 / 1',font(22),MUTED)
d.rectangle([70,1235,1010,1238],fill=ACCENT)
centered(d,1248,'Row format: Tá · Níl · No vote',font(16,True),MUTED)
centered(d,1274,'Vote shown is amendment No. 16 on services — not the final passage vote.',font(15),MUTED)
im.save(OUT/'slide-02-services-amendment.png')

sheet=Image.new('RGB',(1200,1500),'white'); sd=ImageDraw.Draw(sheet); sd.text((40,30),'Bill Tracker · Next slide',font=font(28,True),fill='black'); prev=im.copy(); prev.thumbnail((1080,1350)); sheet.paste(prev,(60,90)); sheet.save(OUT/'contact-sheet.png')
print(OUT/'slide-02-services-amendment.png')
