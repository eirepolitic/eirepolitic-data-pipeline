from __future__ import annotations
import math
from pathlib import Path
from typing import Iterable
from PIL import Image, ImageDraw, ImageFont
W,H=1080,1350
BG='#0f2f24'; TEXT='#f4ead7'; ACCENT='#d8b45f'; MUTED='#c8bda8'
CORNER_DIR=Path('instagram/templates/assets')
def font(size:int,bold:bool=False):
    paths=['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']
    for p in paths:
        if Path(p).exists(): return ImageFont.truetype(p,size=size)
    return ImageFont.load_default()
def base_slide():
    missing=[n for n in ('corner_tl.png','corner_tr.png','corner_bl.png','corner_br.png') if not (CORNER_DIR/n).is_file()]
    if missing: raise FileNotFoundError(f'Approved Instagram corner assets are missing: {missing}')
    im=Image.new('RGB',(W,H),BG)
    for n,pos in [('corner_tl.png',(0,0)),('corner_tr.png',(925,0)),('corner_bl.png',(0,1195)),('corner_br.png',(925,1195))]:
        c=Image.open(CORNER_DIR/n).convert('RGBA').resize((155,155),Image.Resampling.LANCZOS); im.paste(c,pos,c)
    return im
def _width(d,text,f):
    b=d.textbbox((0,0),text,font=f,anchor='la'); return b[2]-b[0]
def wrap_text_px(d,text,f,max_width:int):
    words=str(text or '').split()
    if not words: return []
    lines=[]; cur=words[0]
    for w in words[1:]:
        cand=f'{cur} {w}'
        if _width(d,cand,f)<=max_width: cur=cand
        else: lines.append(cur); cur=w
    lines.append(cur); return lines
def contact_sheet(items:Iterable[tuple[str,Path]],out_path:Path,*,columns:int=4):
    items=list(items); tw,th,lh,g=250,312,34,18; rows=math.ceil(len(items)/columns)
    canvas=Image.new('RGB',(columns*(tw+g)+g,rows*(th+lh+g)+g),BG); d=ImageDraw.Draw(canvas); lf=font(18,True)
    for idx,(label,path) in enumerate(items):
        row,col=divmod(idx,columns); x=g+col*(tw+g); y=g+row*(th+lh+g); im=Image.open(path).convert('RGB'); im.thumbnail((tw,th),Image.Resampling.LANCZOS); canvas.paste(im,(x+(tw-im.width)//2,y)); d.text((x+tw//2,y+th+20),label,font=lf,fill=TEXT,anchor='mm')
    out_path.parent.mkdir(parents=True,exist_ok=True); canvas.save(out_path,quality=92)
