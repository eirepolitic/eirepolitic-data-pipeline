#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw
from instagram.factory.render_primitives import base_slide, font, wrap_text_px, contact_sheet, BG, TEXT, ACCENT, MUTED

OUT=Path('artifacts/bill-tracker-second-stage-review'); OUT.mkdir(parents=True,exist_ok=True)
W,H=1080,1350; PANEL='#174638'; OUTLINE='#346857'; NO_VOTE='#65756d'

def rule(d,y,l=82,r=998,w=5): d.rectangle((l,y,r,y+w),fill=ACCENT)
def panel(d,box,outline=OUTLINE,fill=PANEL,w=2): d.rounded_rectangle(box,radius=18,fill=fill,outline=outline,width=w)
def footer(d,text='EirePolitic · Bills of the Current Session'):
    rule(d,1260,58,1022,4); d.text((W//2,1286),text,font=font(14,True),fill=MUTED,anchor='ma')
def wrapped(d,text,x,y,f,width,fill=TEXT,gap=5,center=False):
    lines=wrap_text_px(d,text,f,width); cur=y
    for line in lines:
        ax=W//2 if center else x; anchor='ma' if center else 'la'
        d.text((ax,cur),line,font=f,fill=fill,anchor=anchor)
        b=d.textbbox((ax,cur),line,font=f,anchor=anchor); cur=b[3]+gap
    return cur
def fit(d,text,width,start=24,min_size=17,max_lines=7,bold=False):
    for s in range(start,min_size-1,-1):
        f=font(s,bold); lines=wrap_text_px(d,text,f,width)
        if len(lines)<=max_lines:return f
    raise RuntimeError(text)
def title(d,text,top=44,maxw=910,start=34,min_size=25,max_lines=2):
    f=fit(d,text,maxw,start,min_size,max_lines,True); lines=wrap_text_px(d,text,f,maxw); cur=top
    for line in lines:
        d.text((W//2,cur),line,font=f,fill=TEXT,anchor='ma'); b=d.textbbox((W//2,cur),line,font=f,anchor='ma'); cur=b[3]+4
    return cur

def shared_font(d,texts,width,height,start=24,min_size=18,max_lines=7):
    for s in range(start,min_size-1,-1):
        f=font(s); ok=True
        for t in texts:
            lines=wrap_text_px(d,t,f,width); h=0
            for ln in lines:
                b=d.textbbox((0,0),ln,font=f); h += (b[3]-b[1])+4
            if len(lines)>max_lines or h>height: ok=False; break
        if ok:return f
    raise RuntimeError('shared text does not fit')

def cover():
    im=base_slide(); d=ImageDraw.Draw(im)
    d.text((W//2,126),'BILLS OF THE',font=font(28,True),fill=ACCENT,anchor='ma')
    d.text((W//2,174),'CURRENT SESSION',font=font(44,True),fill=TEXT,anchor='ma')
    rule(d,240,138,942,5)
    d.text((W//2,282),'SECOND STAGE · FORMAT REVIEW',font=font(26,True),fill=ACCENT,anchor='ma')
    wrapped(d,'At Second Stage, the House debates the general principles of a Bill. Some Bills are awaiting debate; others reach a decision and may have a recorded vote.',0,345,font(22),790,center=True)
    d.text((W//2,520),'IN THIS REVIEW',font=font(23,True),fill=ACCENT,anchor='ma')
    bills=['Anti-Shrinkflation Bill 2026','Broadcasting (Amendment) Bill 2026','Electoral (Postal Voting) (Carers) Bill 2026']
    y=570
    for i,b in enumerate(bills,1):
        panel(d,(76,y,1004,y+176)); d.ellipse((106,y+58,164,y+116),fill=ACCENT); d.text((135,y+87),str(i),font=font(23,True),fill=BG,anchor='mm')
        f=fit(d,b,760,24,19,3,True); lines=wrap_text_px(d,b,f,760); lineh=sum(d.textbbox((0,0),ln,font=f)[3] for ln in lines)+4*(len(lines)-1); cur=y+(176-lineh)//2
        for ln in lines:
            d.text((570,cur),ln,font=f,fill=TEXT,anchor='ma'); cur=d.textbbox((570,cur),ln,font=f,anchor='ma')[3]+4
        y+=198
    footer(d,'EirePolitic · Second Stage · Format review'); im.save(OUT/'00-cover.png')

def explainer(filename,title_text,meta,what,effect,case,issues,status_label,status_body):
    im=base_slide(); d=ImageDraw.Draw(im); tb=title(d,title_text); ry=max(116,tb+18); rule(d,ry)
    d.text((W//2,ry+34),'WHAT IT DOES & WHAT SECOND STAGE MEANS',font=font(22,True),fill=ACCENT,anchor='ma')
    d.text((W//2,ry+70),meta,font=font(16),fill=MUTED,anchor='ma')
    top=ry+112; bw=456; bh=256; gap=24; boxes=[(60,top,516,top+bh),(564,top,1020,top+bh),(60,top+bh+gap,516,top+2*bh+gap),(564,top+bh+gap,1020,top+2*bh+gap)]
    labels=['WHAT THE BILL PROPOSES','PRACTICAL EFFECT','CASE MADE FOR THE BILL','ISSUES FOR SECOND STAGE']; texts=[what,effect,case,issues]
    bf=shared_font(d,texts,402,175,24,18,7)
    for box,label,txt in zip(boxes,labels,texts):
        panel(d,box); d.text((box[0]+24,box[1]+20),label,font=font(17,True),fill=ACCENT,anchor='la'); end=wrapped(d,txt,box[0]+24,box[1]+61,bf,box[2]-box[0]-48)
        if end>box[3]-16: raise RuntimeError(f'overflow {filename} {label}')
    sy=boxes[2][3]+34; panel(d,(60,sy,1020,1206),outline=ACCENT,fill=BG,w=3); d.text((W//2,sy+29),status_label,font=font(21,True),fill=ACCENT,anchor='ma')
    sf=fit(d,status_body,850,20,17,5); wrapped(d,status_body,0,sy+76,sf,850,center=True)
    footer(d); im.save(OUT/filename)

def decision():
    im=base_slide(); d=ImageDraw.Draw(im)
    title(d,'Broadcasting Bill · Second Stage Decision',top=54,maxw=940,start=37,min_size=28,max_lines=1); rule(d,132)
    d.text((W//2,168),'DÁIL ÉIREANN · 10 JUNE 2026',font=font(23,True),fill=ACCENT,anchor='ma')
    d.text((W//2,214),'EXACT QUESTION',font=font(17,True),fill=MUTED,anchor='ma')
    d.text((W//2,251),'“That the Bill be now read a Second Time.”',font=font(24,True),fill=TEXT,anchor='ma')
    # overall stacked bar
    x,y,w,h=70,322,940,66; total=175; ta,nil,nr=102,44,29; a=round(w*ta/total); b=round(w*nil/total)
    d.rectangle((x,y,x+a,y+h),fill=ACCENT); d.rectangle((x+a,y,x+a+b,y+h),fill=TEXT); d.rectangle((x+a+b,y,x+w,y+h),fill=NO_VOTE); d.rectangle((x,y,x+w,y+h),outline=MUTED,width=2)
    d.text((W//2,420),'175 eligible TDs · 102 Tá · 44 Níl · 0 abstentions · 29 no recorded vote',font=font(19,True),fill=TEXT,anchor='ma')
    d.text((W//2,485),'RESULT',font=font(19,True),fill=MUTED,anchor='ma'); d.text((W//2,530),'102 Tá · 44 Níl — carried',font=font(31,True),fill=TEXT,anchor='ma')
    panel(d,(84,600,996,812)); d.text((W//2,635),'WHAT A TÁ / NÍL MEANT HERE',font=font(20,True),fill=ACCENT,anchor='ma')
    wrapped(d,'Tá supported the Second Reading motion and allowed the Bill to proceed. Níl opposed that motion. This was a vote on the Bill’s general principles at Second Stage, not on individual amendments.',0,686,font(21),820,center=True)
    panel(d,(84,848,996,1122),outline=ACCENT,fill=BG,w=3); d.text((W//2,885),'WHAT HAPPENED NEXT',font=font(20,True),fill=ACCENT,anchor='ma')
    wrapped(d,'The Second Reading motion carried and the Bill was referred onward for detailed scrutiny. Later Committee Stage work is a separate part of the process and should not be folded into this Second Stage vote.',0,934,font(21),820,center=True)
    d.text((W//2,1172),'Party-by-party bars are omitted from this review until the vote-date affiliation join is revalidated.',font=font(14),fill=MUTED,anchor='ma')
    footer(d,'EirePolitic · Second Stage · Recorded Dáil decision'); im.save(OUT/'03-broadcasting-decision.png')

def process_glossary():
    im=base_slide(); d=ImageDraw.Draw(im); d.text((W//2,75),'GLOSSARY',font=font(40,True),fill=TEXT,anchor='ma'); d.text((W//2,136),'HOW A BILL MOVES THROUGH PARLIAMENT',font=font(22,True),fill=ACCENT,anchor='ma'); rule(d,176,112,968,4)
    labels=['FIRST','SECOND','COMMITTEE','REPORT','FINAL','OTHER HOUSE','PRESIDENT','ENACTED']; x0=35; y=220; bw=105; bh=104; gap=15
    for i,lbl in enumerate(labels):
        x=x0+i*(bw+gap); active=lbl=='SECOND'; panel(d,(x,y,x+bw,y+bh),outline=ACCENT if active else OUTLINE,fill=ACCENT if active else PANEL); d.text((x+bw//2,y+24),str(i+1),font=font(14,True),fill=BG if active else ACCENT,anchor='mm'); f=fit(d,lbl,bw-12,12,10,2,True); wrapped(d,lbl,0,y+56,f,bw-12,fill=BG if active else ACCENT,center=False); # overwrite centered below
        # centered label
        lines=wrap_text_px(d,lbl,f,bw-12); cy=y+64
        for ln in lines: d.text((x+bw//2,cy),ln,font=f,fill=BG if active else ACCENT,anchor='mm'); cy+=16
        if i<len(labels)-1:
            ax=x+bw+4; d.polygon([(ax,y+52),(ax+9,y+45),(ax+9,y+59)],fill=ACCENT)
    d.text((W//2,356),'THIS POST: SECOND STAGE',font=font(18,True),fill=ACCENT,anchor='ma')
    wrapped(d,'Second Stage is where the House debates the Bill’s general principles. If agreed, detailed section-by-section scrutiny normally follows at Committee Stage.',0,398,font(18),860,center=True)
    d.text((W//2,505),'THE HOUSE MATTERS TOO',font=font(21,True),fill=ACCENT,anchor='ma')
    cards=[('FIRST HOUSE','A Bill can be at Second Stage in the House where it began.'),('SECOND HOUSE','After completing the first House, a Bill normally goes through stages in the other House too.'),('CURRENT POSITION','Always read the stage together with Dáil or Seanad: “Second Stage” alone does not tell you how far through the full journey the Bill is.')]
    boxes=[(70,555,505,745),(575,555,1010,745),(170,790,910,1035)]
    for (hd,bd),box in zip(cards,boxes): panel(d,box); d.text(((box[0]+box[2])//2,box[1]+36),hd,font=font(19,True),fill=ACCENT,anchor='ma'); wrapped(d,bd,0,box[1]+82,font(20),box[2]-box[0]-50,center=True)
    footer(d,'EirePolitic · Glossary · Parliamentary process'); im.save(OUT/'05-process-glossary.png')

def second_stage_glossary():
    im=base_slide(); d=ImageDraw.Draw(im); d.text((W//2,82),'SECOND STAGE',font=font(41,True),fill=TEXT,anchor='ma'); d.text((W//2,143),'WHAT IS THE HOUSE DECIDING?',font=font(23,True),fill=ACCENT,anchor='ma'); rule(d,190,112,968,4)
    cards=[
      ('GENERAL PRINCIPLES','The main debate is about the broad purpose and approach of the Bill, rather than line-by-line amendment.'),
      ('SECOND READING','The ordinary motion asks whether the Bill should be read a second time. If that question is agreed, the Bill can move forward.'),
      ('NOT COMMITTEE STAGE','Detailed examination of sections and amendments normally comes later at Committee Stage.'),
      ('NO RECORDED DIVISION','A decision can stand without a member-by-member division. If there is no recorded division, we do not invent a party tally.'),
      ('EXACT PROPOSITION','If a division occurs, always read the exact question. A vote may be on Second Reading itself, or on a procedural/amending proposition connected with it.')]
    y=240; heights=[165,175,165,185,205]
    bf=shared_font(d,[x[1] for x in cards],820,105,22,18,4)
    for (hd,bd),hh in zip(cards,heights):
        panel(d,(82,y,998,y+hh)); d.text((106,y+24),hd,font=font(20,True),fill=ACCENT,anchor='la'); end=wrapped(d,bd,106,y+64,bf,820); y+=hh+18
    footer(d,'EirePolitic · Glossary · Understanding Second Stage'); im.save(OUT/'06-second-stage-explainer.png')

def build_sheet():
    files=['00-cover.png','01-anti-shrinkflation.png','02-broadcasting-explainer.png','03-broadcasting-decision.png','04-carers.png','05-process-glossary.png','06-second-stage-explainer.png']
    for f in files:
        if Image.open(OUT/f).size!=(1080,1350): raise RuntimeError(f'bad size {f}')
    contact_sheet([(str(i+1),OUT/f) for i,f in enumerate(files)],OUT/'contact-sheet.png',columns=4)

if __name__=='__main__':
    cover()
    explainer('01-anti-shrinkflation.png','Anti-Shrinkflation Bill 2026','Private Member’s Bill · Holly Cairns · Dáil Éireann',
      'Would require clearer retail labelling when a product’s quantity is reduced in a way that raises its unit price, so consumers can spot hidden price increases.',
      'Large retailers would have to flag qualifying quantity reductions for a set period. The proposal focuses on price transparency rather than banning smaller packs or setting prices.',
      'The sponsor argues shoppers should be told clearly when they are paying effectively more for less, particularly during a period of cost-of-living pressure.',
      'At Second Stage the House can test the proposal’s general approach: which retailers/products should be covered, how notice rules work, exemptions, enforcement and proportionality.',
      'WHERE IT IS NOW','The Bill is in the Second Stage part of the process, but the available certified record does not show a substantive Second Stage decision or recorded division. No party vote chart is therefore shown.')
    explainer('02-broadcasting-explainer.png','Broadcasting (Amendment) Bill 2026','Government Bill · Dáil Éireann',
      'Would reform governance, transparency, funding and oversight arrangements for RTÉ and TG4, expand Coimisiún na Meán functions and implement parts of the European Media Freedom Act.',
      'The Bill would change how public service media governance, auditing, performance assessment and some public-service-content funding arrangements operate.',
      'Government presented the Bill as implementing recommendations from the Future of Media Commission and the independent RTÉ governance review, alongside EU media-law requirements.',
      'Second Stage debate raised issues including governance, long-term public-service-media funding, Irish-language provision, independent production, geo-blocking and implementation detail.',
      'SECOND STAGE OUTCOME','The Dáil later voted on the explicit Second Reading motion. Because that proposition has a recorded division, the next slide shows the certified overall result and explains exactly what the vote meant.')
    decision()
    explainer('04-carers.png','Electoral (Postal Voting) (Carers) Bill 2026','Private Member’s Bill · Mark Wall · Dáil Éireann',
      'Would extend eligibility for the postal-voter register to certain people who provide care for others, by amending the Electoral Act 1992.',
      'Qualifying carers who cannot readily attend a polling station because of caring responsibilities could gain a postal-voting route if the Bill eventually becomes law.',
      'The sponsor presented the proposal as a way to reduce barriers to electoral participation faced by family carers whose responsibilities can make in-person voting difficult.',
      'The substantive Second Stage debate would be the point to test the proposal’s general principles, eligibility rules, safeguards and how a carers postal-voting category should operate.',
      'WHERE IT IS NOW','The Bill has been introduced and is positioned for Second Stage, but the available certified data does not show a substantive Second Stage debate or recorded division yet. The absence of a vote is therefore part of the story, not missing content.')
    process_glossary(); second_stage_glossary(); build_sheet(); print(OUT)
