#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path('artifacts/bill-gas-reserve-pair')
OUT.mkdir(parents=True, exist_ok=True)
ASSET_DIR = Path('artifacts/approved-factory-assets')

W, H = 1080, 1350
BG = '#0f2f24'; TEXT = '#f4ead7'; MUTED = '#c8bda8'; ACCENT = '#d8b45f'; FOR = ACCENT; AGAINST = TEXT; NO_VOTE = '#65756d'; PANEL = '#173f31'
party_rows = [
    ('Fianna Fáil',48,42,0,6),('Sinn Féin',39,0,31,8),('Fine Gael',38,35,0,3),('Independent',15,10,2,3),
    ('Social Democrats',12,0,12,0),('Labour',11,0,7,4),('Independent Ireland',4,3,0,1),('PBP–S',3,0,3,0),
    ('Aontú',2,0,0,2),('Green',1,0,1,0),('100% Redress',1,0,1,0)]

POST1_BILLS = [
    'Development (Strategic Gas Reserve) Bill 2026',
    'Israeli Settlements in the Occupied Palestinian Territory (Prohibition of Importation of Goods) Bill 2026',
    'Criminal Law, Civil Law and Defence (Miscellaneous Provisions) Bill 2026',
]

def font(size,bold=False):
    candidates=['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']
    for p in candidates:
        if Path(p).exists(): return ImageFont.truetype(p,size=size)
    return ImageFont.load_default()

def measure(d,t,f):
    b=d.textbbox((0,0),str(t),font=f); return b[2]-b[0],b[3]-b[1]

def centered(d,y,t,f,fill):
    tw,_=measure(d,t,f); d.text(((W-tw)/2,y),t,font=f,fill=fill)

def wrap(d,text,f,width):
    words=str(text).split(); lines=[]; cur=''
    for word in words:
        trial=word if not cur else cur+' '+word
        if measure(d,trial,f)[0] <= width: cur=trial
        else:
            if cur: lines.append(cur)
            cur=word
    if cur: lines.append(cur)
    return lines

def draw_wrapped(d,x,y,text,f,fill,width,gap=7,max_lines=None):
    lines=wrap(d,text,f,width)
    if max_lines is not None and len(lines)>max_lines: raise RuntimeError(f'copy overflow ({len(lines)} lines): {text}')
    for line in lines:
        d.text((x,y),line,font=f,fill=fill); _,h=measure(d,line,f); y += h+gap
    return y

def draw_centered_wrapped(d,y,text,f,fill,width,gap=7,max_lines=None):
    lines=wrap(d,text,f,width)
    if max_lines is not None and len(lines)>max_lines: raise RuntimeError(f'copy overflow ({len(lines)} lines): {text}')
    for line in lines:
        tw,h=measure(d,line,f); d.text(((W-tw)/2,y),line,font=f,fill=fill); y += h+gap
    return y

def add_ornaments(im):
    specs = [('corner_tl.png',(0,0)),('corner_tr.png',(W,0)),('corner_bl.png',(0,H)),('corner_br.png',(W,H))]
    for filename,(ax,ay) in specs:
        path=ASSET_DIR/filename
        if not path.exists():
            continue
        corner=Image.open(path).convert('RGBA')
        corner.thumbnail((185,185),Image.Resampling.LANCZOS)
        x=0 if ax==0 else W-corner.width
        y=0 if ay==0 else H-corner.height
        im.alpha_composite(corner,(x,y))

def base_slide(with_ornaments=True):
    im=Image.new('RGBA',(W,H),BG)
    if with_ornaments: add_ornaments(im)
    return im

def footer(d, text='EirePolitic · Bills of the Current Session'):
    d.rectangle([80,1260,1000,1263],fill=ACCENT)
    centered(d,1277,text,font(15,True),MUTED)

def panel(d,x,y,w,h,heading,body,body_size=25):
    d.rounded_rectangle([x,y,x+w,y+h],radius=20,fill=PANEL,outline='#31594a',width=2)
    d.text((x+24,y+20),heading,font=font(21,True),fill=ACCENT)
    end_y=draw_wrapped(d,x+24,y+58,body,font(body_size),TEXT,w-48,gap=3,max_lines=8)
    if end_y > y+h-18: raise RuntimeError(f'panel overflow: {heading}')

def glossary_card(d,x,y,w,h,term,definition,term_size=25,body_size=22):
    d.rounded_rectangle([x,y,x+w,y+h],radius=20,fill=PANEL,outline='#31594a',width=2)
    d.text((x+25,y+22),term,font=font(term_size,True),fill=ACCENT)
    end=draw_wrapped(d,x+25,y+65,definition,font(body_size),TEXT,w-50,gap=6,max_lines=5)
    if end > y+h-18: raise RuntimeError(f'glossary overflow: {term}')

def build_cover():
    im=base_slide(); d=ImageDraw.Draw(im)
    centered(d,120,'BILLS OF THE',font(30,True),ACCENT)
    centered(d,165,'CURRENT SESSION',font(56,True),TEXT)
    d.rectangle([185,242,895,248],fill=ACCENT)
    centered(d,282,'ENACTED · PART 1',font(28,True),ACCENT)
    draw_centered_wrapped(d,345,'Three recently enacted Bills, explained in plain English — what each does, the arguments around it, and what the recorded vote shown actually decided.',font(23),TEXT,820,gap=9,max_lines=5)
    centered(d,520,'IN THIS PART',font(24,True),ACCENT)
    y=575
    for i,title in enumerate(POST1_BILLS,1):
        d.rounded_rectangle([95,y,985,y+180],radius=20,fill=PANEL,outline='#31594a',width=2)
        d.ellipse([125,y+46,181,y+102],fill=ACCENT)
        num=str(i); nw,nh=measure(d,num,font(24,True)); d.text((153-nw/2,y+74-nh/2),num,font=font(24,True),fill=BG)
        end=draw_wrapped(d,210,y+38,title,font(25,True),TEXT,720,gap=7,max_lines=4)
        if end>y+155: raise RuntimeError(f'cover bill title overflow: {title}')
        y += 202
    footer(d,'EirePolitic · Enacted · Part 1')
    im.convert('RGB').save(OUT/'00-title-enacted-part-1.png')

def bar(d,x,y,w,h,yes,no,nr):
    total=max(1,yes+no+nr); segs=[(yes,FOR),(no,AGAINST),(nr,NO_VOTE)]; segs=[s for s in segs if s[0]>0]; cur=x
    for i,(value,color) in enumerate(segs):
        sw=(x+w-cur) if i==len(segs)-1 else round(w*value/total); d.rectangle([cur,y,cur+sw,y+h],fill=color); cur += sw
    d.rectangle([x,y,x+w,y+h],outline=MUTED,width=2)

def build_explainer():
    im=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(im)
    centered(d,54,'Development (Strategic Gas Reserve) Bill 2026',font(36,True),TEXT)
    d.rectangle([110,118,970,123],fill=ACCENT)
    centered(d,146,'WHAT IT DOES & WHAT THE DÁIL VOTE MEANT',font(22,True),ACCENT)
    centered(d,184,'Introduced by the Government · Minister for Climate, Energy and the Environment',font(17),MUTED)
    panel(d,60,232,460,270,'WHAT THE BILL DOES','Sets up a special legal route for approving a strategic gas reserve at Cahiracon, Co. Clare. It replaces the normal planning route for this project, but environmental assessments still apply.')
    panel(d,560,232,460,270,'PRACTICAL EFFECT','The Minister could decide the project directly under a faster process. The reserve is intended for emergencies if Ireland\'s normal gas supplies are seriously disrupted.')
    panel(d,60,526,460,286,'WHY SOME TDs BACKED IT','Supporters said Ireland relies heavily on imported gas and needs a back-up supply if imports are seriously disrupted. They described it as an energy-security measure while Ireland moves toward renewables.')
    panel(d,560,526,460,286,'WHY SOME TDs OPPOSED IT','Critics said the project could prolong reliance on fossil fuels. They also objected to the special planning route, faster timetable and limited time for scrutiny.')
    d.rounded_rectangle([60,850,1020,1206],radius=22,fill='#102b22',outline=ACCENT,width=3)
    centered(d,878,'WHAT THE 30 JUNE DÁIL VOTE MEANT',font(24,True),ACCENT)
    explainer_end=draw_centered_wrapped(d,928,'TDs — members of the Dáil — were voting on one combined question that completed the Bill\'s remaining Dáil steps and passed it. A Tá meant pass the Bill and send it on. A Níl meant reject that passage motion.',font(23),TEXT,850,gap=7,max_lines=6)
    result_y=explainer_end+8
    centered(d,result_y,'RESULT',font(19,True),MUTED); centered(d,result_y+32,'90 Tá · 57 Níl — carried',font(27,True),TEXT)
    effect_end=draw_centered_wrapped(d,result_y+78,'The Bill passed the Dáil and moved to the Seanad, Ireland\'s second parliamentary chamber.',font(22,True),MUTED,850,gap=6,max_lines=3)
    if effect_end>1186: raise RuntimeError(f'bottom vote box overflow: final_y={effect_end}')
    d.rectangle([70,1235,1010,1238],fill=ACCENT); centered(d,1250,'Sources: Houses of the Oireachtas bill text, explanatory memorandum and Dáil debate record',font(15),MUTED)
    im.save(OUT/'01-gas-reserve-explainer.png')

def build_vote():
    im=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(im)
    centered(d,62,'Strategic Gas Reserve · Party Split',font(45,True),TEXT); d.rectangle([110,130,970,135],fill=ACCENT)
    centered(d,160,'Dáil passage vote · 30 June 2026',font(25,True),ACCENT); bar(d,70,208,940,56,90,57,27)
    legend=[('Tá',FOR),('Níl',AGAINST),('No recorded vote',NO_VOTE)]; lf=font(19,True); widths=[18+12+measure(d,l,lf)[0] for l,_ in legend]; gap=42; cur=(W-(sum(widths)+gap*2))/2
    for (label,color),iw in zip(legend,widths):
        d.rectangle([cur,290,cur+18,308],fill=color,outline=MUTED); d.text((cur+30,286),label,font=lf,fill=TEXT); cur += iw+gap
    centered(d,328,'174 eligible TDs · 90 Tá · 57 Níl · 0 abstentions · 27 no recorded vote',font(20,True),TEXT); centered(d,382,'PARTY BREAKDOWN',font(31,True),ACCENT)
    name_x,bar_x,bar_w,row_top,row_h,bar_h=64,348,470,438,73,36; centers=[866,922,998]
    for (label,color),cx in zip([('Tá',ACCENT),('Níl',TEXT),('No vote',MUTED)],centers):
        tw,_=measure(d,label,font(16,True)); d.text((cx-tw/2,row_top-32),label,font=font(16,True),fill=color)
    for i,(name,eligible,yes,no,nr) in enumerate(party_rows[:8]):
        y=row_top+i*row_h; d.text((name_x,y+2),name,font=font(24,True),fill=TEXT); bar(d,bar_x,y,bar_w,bar_h,yes,no,nr)
        for (n,color),cx in zip([(str(yes),ACCENT),(str(no),TEXT),(str(nr),MUTED)],centers):
            tw,_=measure(d,n,font(20,True)); d.text((cx-tw/2,y+5),n,font=font(20,True),fill=color)
    centered(d,1064,'Smaller groups',font(27,True),ACCENT); centered(d,1102,'Aontú 0 / 0 / 2 · Green 0 / 1 / 0 · 100% Redress 0 / 1 / 0',font(22),MUTED)
    d.rectangle([70,1235,1010,1238],fill=ACCENT); centered(d,1248,'Row format: Tá · Níl · No vote',font(16,True),MUTED); centered(d,1274,'No recorded vote is shown separately and does not automatically mean absent.',font(15),MUTED)
    im.save(OUT/'02-gas-reserve-vote.png')

def build_glossary_terms():
    im=base_slide(); d=ImageDraw.Draw(im)
    centered(d,95,'GLOSSARY',font(48,True),TEXT)
    centered(d,157,'HOW BILLS MOVE THROUGH PARLIAMENT',font(22,True),ACCENT)
    d.rectangle([150,205,930,210],fill=ACCENT)
    cards=[
        ('BILL','A proposed law. It must go through parliamentary stages before it can become law.'),
        ('DÁIL ÉIREANN','Ireland’s directly elected house of parliament. Its elected members are called TDs.'),
        ('SEANAD ÉIREANN','Ireland’s second parliamentary chamber. Its members are Senators.'),
        ('STAGE','A formal step in considering a Bill. Different stages cover its principles, detailed text, amendments and final approval.'),
        ('ENACTED','The Bill has completed the required parliamentary process and has become law.'),
    ]
    y=265
    heights=[155,155,155,176,155]
    for (term,definition),h in zip(cards,heights):
        glossary_card(d,110,y,860,h,term,definition,term_size=24,body_size=22)
        y+=h+18
    footer(d,'EirePolitic · Glossary · Parliamentary terms')
    im.convert('RGB').save(OUT/'07-glossary-parliamentary-terms.png')

def build_glossary_votes():
    im=base_slide(); d=ImageDraw.Draw(im)
    centered(d,95,'GLOSSARY',font(48,True),TEXT)
    centered(d,157,'HOW TO READ THE VOTE SLIDES',font(22,True),ACCENT)
    d.rectangle([150,205,930,210],fill=ACCENT)
    cards=[
        ('TÁ / NÍL','Tá means Yes. Níl means No. What “Yes” or “No” means depends on the exact question being voted on.'),
        ('DIVISION','A recorded vote where individual members’ votes are listed.'),
        ('AMENDMENT','A proposed change. A Tá on an amendment supports that change; it does not automatically mean support for the whole Bill.'),
        ('NO RECORDED VOTE','No vote is recorded for that TD in the division data. It does not automatically mean the TD was absent.'),
    ]
    y=270
    heights=[175,150,200,185]
    for (term,definition),h in zip(cards,heights):
        glossary_card(d,110,y,860,h,term,definition,term_size=24,body_size=22)
        y+=h+20
    d.rounded_rectangle([110,1055,970,1208],radius=20,fill='#102b22',outline=ACCENT,width=3)
    centered(d,1080,'OUR VOTE-LABEL RULE',font(22,True),ACCENT)
    draw_centered_wrapped(d,1122,'We describe the exact proposition shown. Speaking about a Bill does not equal supporting it, and a linked vote may be about an amendment or procedure rather than the Bill itself.',font(19),TEXT,790,gap=6,max_lines=4)
    footer(d,'EirePolitic · Glossary · Vote terms & safeguards')
    im.convert('RGB').save(OUT/'08-glossary-vote-terms.png')

def build_contact_sheet():
    names=['00-title-enacted-part-1.png','01-gas-reserve-explainer.png','02-gas-reserve-vote.png','07-glossary-parliamentary-terms.png','08-glossary-vote-terms.png']
    slides=[Image.open(OUT/n).convert('RGB') for n in names]
    for slide in slides: assert slide.size==(1080,1350)
    thumb_w,thumb_h=430,538; margin=40; gap=30
    sheet=Image.new('RGB',(3*thumb_w+2*gap+2*margin,2*thumb_h+160),'white'); d=ImageDraw.Draw(sheet)
    d.text((40,24),'Bill Tracker · Post 1 framing + approved Gas Reserve pair',font=font(28,True),fill='black')
    for i,slide in enumerate(slides):
        slide.thumbnail((thumb_w,thumb_h)); x=margin+(i%3)*(thumb_w+gap); y=90+(i//3)*(thumb_h+40); sheet.paste(slide,(x,y))
    sheet.save(OUT/'contact-sheet.png')

if __name__=='__main__':
    build_cover(); build_explainer(); build_vote(); build_glossary_terms(); build_glossary_votes(); build_contact_sheet()
    print(OUT)
