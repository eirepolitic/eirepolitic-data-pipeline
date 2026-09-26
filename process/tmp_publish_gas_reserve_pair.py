#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path('artifacts/bill-gas-reserve-pair')
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1080, 1350
BG = '#0f2f24'; TEXT = '#f4ead7'; MUTED = '#c8bda8'; ACCENT = '#d8b45f'; FOR = ACCENT; AGAINST = TEXT; NO_VOTE = '#65756d'; PANEL = '#173f31'
party_rows = [
    ('Fianna Fáil',48,42,0,6),('Sinn Féin',39,0,31,8),('Fine Gael',38,35,0,3),('Independent',15,10,2,3),
    ('Social Democrats',12,0,12,0),('Labour',11,0,7,4),('Independent Ireland',4,3,0,1),('PBP–S',3,0,3,0),
    ('Aontú',2,0,0,2),('Green',1,0,1,0),('100% Redress',1,0,1,0)]

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

def panel(d,x,y,w,h,heading,body,body_size=20):
    d.rounded_rectangle([x,y,x+w,y+h],radius=20,fill=PANEL,outline='#31594a',width=2)
    d.text((x+24,y+20),heading,font=font(21,True),fill=ACCENT)
    end_y=draw_wrapped(d,x+24,y+58,body,font(body_size),TEXT,w-48,gap=7,max_lines=7)
    if end_y > y+h-18:
        raise RuntimeError(f'panel overflow: {heading}')

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
    centered(d,184,'Government Bill · Minister for Climate, Energy and the Environment',font(18),MUTED)

    panel(d,60,232,460,270,'WHAT THE BILL DOES',
          'Sets up a special legal route for approving a strategic gas reserve at Cahiracon, Co. Clare. It replaces the normal planning route for this project, but environmental assessments still apply.')
    panel(d,560,232,460,270,'PRACTICAL EFFECT',
          'The Minister could decide the project directly under a faster process. The reserve is intended for emergencies if Ireland\'s normal gas supplies are seriously disrupted.')
    panel(d,60,526,460,286,'WHY SOME TDs BACKED IT',
          'Supporters said Ireland relies heavily on imported gas and needs a back-up supply if imports are seriously disrupted. They described it as an energy-security measure while Ireland moves toward renewables.')
    panel(d,560,526,460,286,'WHY SOME TDs OPPOSED IT',
          'Critics said the project could prolong reliance on fossil fuels. They also objected to the special planning route, faster timetable and limited time for scrutiny.')

    d.rounded_rectangle([60,850,1020,1206],radius=22,fill='#102b22',outline=ACCENT,width=3)
    centered(d,878,'WHAT THE 30 JUNE DÁIL VOTE MEANT',font(24,True),ACCENT)
    explainer_end = draw_centered_wrapped(
        d,928,
        'TDs — members of the Dáil — were voting on one combined question that completed the Bill\'s remaining Dáil steps and passed it. A Tá meant pass the Bill and send it on. A Níl meant reject that passage motion.',
        font(19),TEXT,850,gap=8,max_lines=5)
    result_y = explainer_end + 10
    centered(d,result_y,'RESULT',font(19,True),MUTED)
    centered(d,result_y+32,'90 Tá · 57 Níl — carried',font(27,True),TEXT)
    effect_end = draw_centered_wrapped(
        d,result_y+78,
        'The Bill passed the Dáil and moved to the Seanad, Ireland\'s second parliamentary chamber.',
        font(18,True),MUTED,850,gap=6,max_lines=3)
    if effect_end > 1186:
        raise RuntimeError(f'bottom vote box overflow: final_y={effect_end}')

    d.rectangle([70,1235,1010,1238],fill=ACCENT)
    centered(d,1250,'Sources: Houses of the Oireachtas bill text, explanatory memorandum and Dáil debate record',font(15),MUTED)
    assert im.size == (1080,1350)
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
    assert im.size == (1080,1350)
    im.save(OUT/'02-gas-reserve-vote.png')

def build_contact_sheet():
    slides=[Image.open(OUT/'01-gas-reserve-explainer.png').convert('RGB'),Image.open(OUT/'02-gas-reserve-vote.png').convert('RGB')]
    for slide in slides:
        assert slide.size == (1080,1350)
    sheet=Image.new('RGB',(1600,1120),'white'); d=ImageDraw.Draw(sheet); d.text((40,24),'Bill Tracker · Strategic Gas Reserve · two-slide pattern',font=font(28,True),fill='black')
    for slide,x in zip(slides,[40,820]): slide.thumbnail((720,900)); sheet.paste(slide,(x,90))
    sheet.save(OUT/'contact-sheet.png')

if __name__=='__main__':
    build_explainer(); build_vote(); build_contact_sheet(); print(OUT)
