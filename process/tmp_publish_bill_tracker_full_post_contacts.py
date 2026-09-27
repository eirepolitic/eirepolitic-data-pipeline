#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

OUT=Path('artifacts/bill-tracker-full-posts-review'); OUT.mkdir(parents=True,exist_ok=True)
W,H=1080,1350
BG,TEXT,MUTED,ACCENT='#0f2f24','#f4ead7','#c8bda8','#d8b45f'
PANEL,PANEL2,BORDER='#173f31','#102b22','#31594a'
FOR,AGAINST,NO_VOTE=ACCENT,TEXT,'#65756d'

P1_TITLES=[
 'Development (Strategic Gas Reserve) Bill 2026',
 'Israeli Settlements in the Occupied Palestinian Territory (Prohibition of Importation of Goods) Bill 2026',
 'Criminal Law, Civil Law and Defence (Miscellaneous Provisions) Bill 2026']
P2_TITLES=[
 'Housing and Residential Tenancies (Miscellaneous Provisions) Bill 2026',
 'Health (Provision of Contraception Prescribing Service in Retail Pharmacy Businesses) Bill 2026',
 'Regulation of Artificial Intelligence Bill 2026']

PARTY={
'gas':[("Fianna Fáil",48,42,0,6),("Sinn Féin",39,0,31,8),("Fine Gael",38,35,0,3),("Independent",15,10,2,3),("Social Democrats",12,0,12,0),("Labour",11,0,7,4),("Independent Ireland",4,3,0,1),("PBP–S",3,0,3,0),("Aontú",2,0,0,2),("Green",1,0,1,0),("100% Redress",1,0,1,0)],
'israeli':[("Fianna Fáil",48,0,41,7),("Sinn Féin",39,37,0,2),("Fine Gael",38,0,32,6),("Independent",15,3,6,6),("Social Democrats",12,11,0,1),("Labour",11,9,0,2),("Independent Ireland",4,2,0,2),("PBP–S",3,2,0,1),("Aontú",2,2,0,0),("Green",1,1,0,0),("100% Redress",1,0,0,1)],
'criminal':[("Fianna Fáil",48,42,0,6),("Sinn Féin",39,0,34,5),("Fine Gael",38,28,0,10),("Independent",15,9,2,4),("Social Democrats",12,12,0,0),("Labour",11,11,0,0),("Independent Ireland",4,0,3,1),("PBP–S",3,0,3,0),("Aontú",2,0,1,1),("Green",1,0,1,0),("100% Redress",1,0,1,0)],
'housing':[("Fianna Fáil",48,42,0,6),("Sinn Féin",39,0,38,1),("Fine Gael",38,32,0,6),("Independent",15,6,4,5),("Social Democrats",12,0,10,2),("Labour",11,0,9,2),("Independent Ireland",4,0,2,2),("PBP–S",3,0,1,2),("Aontú",2,0,2,0),("Green",1,0,1,0),("100% Redress",1,0,0,1)],
'ai':[("Fianna Fáil",48,41,0,7),("Sinn Féin",39,0,31,8),("Fine Gael",38,34,0,4),("Independent",15,6,2,7),("Social Democrats",12,0,12,0),("Labour",11,0,7,4),("Independent Ireland",4,0,2,2),("PBP–S",3,0,2,1),("Aontú",2,0,0,2),("Green",1,0,1,0),("100% Redress",1,0,1,0)],
}

TOTALS={
 'gas':(174,90,57,27), 'israeli':(174,67,79,28), 'criminal':(174,102,45,27), 'housing':(174,80,67,27), 'ai':(174,81,58,35)
}

def font(size,bold=False):
    cs=['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']
    for c in cs:
        if Path(c).exists(): return ImageFont.truetype(c,size=size)
    return ImageFont.load_default()

def measure(d,t,f):
    b=d.textbbox((0,0),str(t),font=f); return b[2]-b[0],b[3]-b[1]

def wrap(d,text,f,width):
    lines=[]; cur=''
    for word in str(text).split():
        trial=word if not cur else cur+' '+word
        if measure(d,trial,f)[0] <= width: cur=trial
        else:
            if cur: lines.append(cur)
            cur=word
    if cur: lines.append(cur)
    return lines

def centered(d,y,text,f,fill=TEXT):
    w,_=measure(d,text,f); d.text(((W-w)/2,y),text,font=f,fill=fill)

def draw_wrapped(d,x,y,text,f,fill,width,gap=5,max_lines=None):
    lines=wrap(d,text,f,width)
    if max_lines and len(lines)>max_lines: raise RuntimeError(f'overflow: {text}')
    for line in lines:
        d.text((x,y),line,font=f,fill=fill); _,h=measure(d,line,f); y+=h+gap
    return y

def draw_centered_wrapped(d,y,text,f,fill,width,gap=5,max_lines=None):
    lines=wrap(d,text,f,width)
    if max_lines and len(lines)>max_lines: raise RuntimeError(f'overflow: {text}')
    for line in lines:
        w,h=measure(d,line,f); d.text(((W-w)/2,y),line,font=f,fill=fill); y+=h+gap
    return y

def draw_centered_block(d,box,text,f,fill,gap=7,max_lines=4):
    x1,y1,x2,y2=box; lines=wrap(d,text,f,x2-x1)
    if len(lines)>max_lines: raise RuntimeError(f'block overflow: {text}')
    d.multiline_text(((x1+x2)/2,(y1+y2)/2),'\n'.join(lines),font=f,fill=fill,anchor='mm',align='center',spacing=gap)

def slide(): return Image.new('RGB',(W,H),BG)

def footer(d,txt):
    d.rectangle([80,1260,1000,1263],fill=ACCENT); centered(d,1276,txt,font(15,True),MUTED)

def panel(d,x,y,w,h,title,body,head_size=21,body_size=23,max_lines=8):
    d.rounded_rectangle([x,y,x+w,y+h],radius=20,fill=PANEL,outline=BORDER,width=2)
    d.text((x+24,y+18),title,font=font(head_size,True),fill=ACCENT)
    end=draw_wrapped(d,x+24,y+56,body,font(body_size),TEXT,w-48,gap=4,max_lines=max_lines)
    if end>y+h-18: raise RuntimeError(f'panel overflow {title}')

def top_title(d,title):
    # Long Bill names are deliberately wrapped into 2–3 centred lines.
    size=32
    while size>=24:
        f=font(size,True); lines=wrap(d,title,f,940)
        if len(lines)<=3:
            block='\n'.join(lines)
            bb=d.multiline_textbbox((0,0),block,font=f,spacing=2,align='center')
            if bb[3]-bb[1] <= 82: break
        size-=1
    d.multiline_text((W/2,68),'\n'.join(lines),font=f,fill=TEXT,anchor='mm',align='center',spacing=2)
    d.rectangle([110,118,970,123],fill=ACCENT)

def make_cover(path,post_num,bills,desc):
    im=slide(); d=ImageDraw.Draw(im)
    centered(d,120,'BILLS OF THE',font(30,True),ACCENT); centered(d,165,'CURRENT SESSION',font(56,True),TEXT)
    d.rectangle([185,242,895,248],fill=ACCENT); centered(d,282,f'ENACTED · POST {post_num}',font(28,True),ACCENT)
    draw_centered_wrapped(d,345,desc,font(23),TEXT,820,gap=8,max_lines=5); centered(d,520,'IN THIS POST',font(24,True),ACCENT)
    y=575
    for i,title in enumerate(bills,1):
        d.rounded_rectangle([95,y,985,y+180],radius=20,fill=PANEL,outline=BORDER,width=2)
        d.ellipse([125,y+62,181,y+118],fill=ACCENT); d.text((153,y+90),str(i),font=font(24,True),fill=BG,anchor='mm')
        draw_centered_block(d,(210,y+18,965,y+162),title,font(25,True),TEXT,gap=7,max_lines=4)
        y+=202
    footer(d,f'EirePolitic · Enacted · Post {post_num}'); im.save(path)

def make_explainer(path,title,subtitle,a,b,c,e,bottom_title,bottom_body,result=None,effect=None):
    im=slide(); d=ImageDraw.Draw(im); top_title(d,title)
    centered(d,146,'WHAT IT DOES & WHY IT WAS DEBATED',font(22,True),ACCENT); centered(d,184,subtitle,font(17),MUTED)
    panel(d,60,232,460,270,*a,body_size=23); panel(d,560,232,460,270,*b,body_size=23)
    panel(d,60,526,460,286,*c,body_size=22); panel(d,560,526,460,286,*e,body_size=22)
    d.rounded_rectangle([60,850,1020,1206],radius=22,fill=PANEL2,outline=ACCENT,width=3)
    centered(d,878,bottom_title,font(24,True),ACCENT)
    y=draw_centered_wrapped(d,928,bottom_body,font(22),TEXT,860,gap=7,max_lines=7)
    if result:
        y += 24  # requested extra breathing room before RESULT / carried line
        centered(d,y,'RESULT',font(18,True),MUTED); centered(d,y+34,result,font(26,True),TEXT); y+=70
    if effect: draw_centered_wrapped(d,y+14,effect,font(21,True),MUTED,850,gap=6,max_lines=4)
    footer(d,'EirePolitic · Draft review copy'); im.save(path)

def bar(d,x,y,w,h,yes,no,nr):
    total=max(1,yes+no+nr); cur=x; segs=[(yes,FOR),(no,AGAINST),(nr,NO_VOTE)]; segs=[s for s in segs if s[0]>0]
    for i,(v,c) in enumerate(segs):
        sw=(x+w-cur) if i==len(segs)-1 else round(w*v/total); d.rectangle([cur,y,cur+sw,y+h],fill=c); cur+=sw
    d.rectangle([x,y,x+w,y+h],outline=MUTED,width=2)

def make_vote(path,key,title,stage,context_note):
    eligible,yes,no,nr=TOTALS[key]; rows=PARTY[key]
    im=slide(); d=ImageDraw.Draw(im)
    centered(d,62,title,font(42,True),TEXT); d.rectangle([110,130,970,135],fill=ACCENT)
    centered(d,160,stage,font(24,True),ACCENT); bar(d,70,208,940,56,yes,no,nr)
    legend=[('Tá',FOR),('Níl',AGAINST),('No recorded vote',NO_VOTE)]; lf=font(19,True); widths=[18+12+measure(d,l,lf)[0] for l,_ in legend]; gap=42; cur=(W-(sum(widths)+gap*2))/2
    for (lab,col),iw in zip(legend,widths):
        d.rectangle([cur,290,cur+18,308],fill=col,outline=MUTED); d.text((cur+30,286),lab,font=lf,fill=TEXT); cur+=iw+gap
    centered(d,328,f'{eligible} eligible TDs · {yes} Tá · {no} Níl · 0 abstentions · {nr} no recorded vote',font(20,True),TEXT)
    centered(d,382,'PARTY BREAKDOWN',font(31,True),ACCENT)
    name_x,bar_x,bar_w,row_top,row_h,bar_h=64,348,470,438,73,36; centers=[866,922,998]
    for (lab,col),cx in zip([('Tá',ACCENT),('Níl',TEXT),('No vote',MUTED)],centers):
        tw,_=measure(d,lab,font(16,True)); d.text((cx-tw/2,row_top-32),lab,font=font(16,True),fill=col)
    for i,(name,elig,yv,nv,nrv) in enumerate(rows[:8]):
        y=row_top+i*row_h; d.text((name_x,y+2),name,font=font(24,True),fill=TEXT); bar(d,bar_x,y,bar_w,bar_h,yv,nv,nrv)
        for n,c,cx in [(yv,ACCENT,centers[0]),(nv,TEXT,centers[1]),(nrv,MUTED,centers[2])]:
            s=str(n); tw,_=measure(d,s,font(20,True)); d.text((cx-tw/2,y+5),s,font=font(20,True),fill=c)
    smaller=' · '.join([f'{n} {y}/{nn}/{nr0}' for n,_,y,nn,nr0 in rows[8:]])
    centered(d,1064,'Smaller groups',font(27,True),ACCENT); centered(d,1102,smaller,font(21),MUTED)
    draw_centered_wrapped(d,1150,context_note,font(16),MUTED,900,gap=4,max_lines=3)
    d.rectangle([70,1235,1010,1238],fill=ACCENT); centered(d,1248,'Row format: Tá · Níl · No vote',font(16,True),MUTED); centered(d,1274,'No recorded vote does not automatically mean absent.',font(15),MUTED)
    im.save(path)

def make_no_division(path):
    im=slide(); d=ImageDraw.Draw(im)
    centered(d,74,'Pharmacy Contraception · Vote Record',font(42,True),TEXT); d.rectangle([110,146,970,151],fill=ACCENT)
    centered(d,184,'NO RECORDED DIVISION FOUND',font(27,True),ACCENT)
    d.rounded_rectangle([100,260,980,780],radius=24,fill=PANEL,outline=BORDER,width=2)
    centered(d,305,'WHAT THIS MEANS',font(25,True),ACCENT)
    draw_centered_wrapped(d,365,'The Bill completed its parliamentary stages and was enacted, but the Oireachtas votes API contains no recorded member-by-member division linked to this Bill.',font(24),TEXT,790,gap=9,max_lines=6)
    centered(d,560,'SO THERE IS NO PARTY SPLIT TO SHOW',font(23,True),ACCENT)
    draw_centered_wrapped(d,615,'That is different from saying nobody voted. A stage can be agreed without a recorded division, so individual Tá / Níl votes are not available for a party breakdown here.',font(23),TEXT,790,gap=8,max_lines=6)
    d.rounded_rectangle([130,865,950,1085],radius=20,fill=PANEL2,outline=ACCENT,width=3)
    centered(d,905,'EDITORIAL RULE',font(22,True),ACCENT)
    draw_centered_wrapped(d,955,'We do not invent or infer a party vote when the official record does not contain a recorded division.',font(22,True),TEXT,720,gap=7,max_lines=4)
    footer(d,'EirePolitic · No recorded division in Oireachtas vote data'); im.save(path)

def make_glossary_terms(path):
    im=slide(); d=ImageDraw.Draw(im)
    centered(d,72,'GLOSSARY',font(48,True),TEXT); centered(d,134,'HOW A BILL MOVES THROUGH PARLIAMENT',font(22,True),ACCENT); d.rectangle([150,180,930,185],fill=ACCENT)
    # Compact process strip: under half a page.
    steps=['1\nFIRST','2\nSECOND','3\nCOMMITTEE','4\nREPORT','5\nFINAL','6\nOTHER HOUSE','7\nPRESIDENT']
    x0=58; y0=225; bw=126; gap=18
    for i,s in enumerate(steps):
        x=x0+i*(bw+gap); d.rounded_rectangle([x,y0,x+bw,y0+105],radius=14,fill=PANEL,outline=BORDER,width=2)
        d.multiline_text((x+bw/2,y0+52),s,font=font(15,True),fill=ACCENT,anchor='mm',align='center',spacing=3)
        if i<len(steps)-1: d.polygon([(x+bw+5,y0+52),(x+bw+13,y0+46),(x+bw+13,y0+58)],fill=ACCENT)
    draw_centered_wrapped(d,350,'A Government Bill normally moves through stages in one House, then repeats the process in the other House. Once both Houses agree the text, it goes to the President to be signed into law.',font(20),TEXT,900,gap=6,max_lines=4)
    centered(d,455,'COMMON TERMS',font(22,True),ACCENT)
    defs=[('BILL','A proposed law.'),('DÁIL ÉIREANN','Ireland’s directly elected house; members are TDs.'),('SEANAD ÉIREANN','Ireland’s second parliamentary chamber; members are Senators.'),('STAGE','A formal step in considering, amending or approving a Bill.'),('ENACTED','The Bill has completed the required process and become law.')]
    boxes=[(80,505,520,650),(560,505,1000,650),(80,675,520,820),(560,675,1000,820),(300,845,780,990)]
    for (term,body),(x1,y1,x2,y2) in zip(defs,boxes):
        d.rounded_rectangle([x1,y1,x2,y2],radius=18,fill=PANEL,outline=BORDER,width=2); centered_x=(x1+x2)/2
        tw,_=measure(d,term,font(20,True)); d.text((centered_x-tw/2,y1+24),term,font=font(20,True),fill=ACCENT)
        lines=wrap(d,body,font(18),x2-x1-40); d.multiline_text((centered_x,y1+78),'\n'.join(lines),font=font(18),fill=TEXT,anchor='mm',align='center',spacing=4)
    footer(d,'EirePolitic · Glossary · Parliamentary process & terms'); im.save(path)

def make_glossary_votes(path):
    im=slide(); d=ImageDraw.Draw(im)
    centered(d,95,'GLOSSARY',font(48,True),TEXT); centered(d,157,'HOW TO READ THE VOTE SLIDES',font(22,True),ACCENT); d.rectangle([150,205,930,210],fill=ACCENT)
    cards=[('TÁ / NÍL','Tá means Yes. Níl means No. What “Yes” or “No” means depends on the exact proposition being voted on.'),('DIVISION','A recorded vote where individual members’ votes are listed.'),('AMENDMENT','A proposed change. A Tá on an amendment supports that change. It does not automatically mean support for the whole Bill.'),('NO RECORDED VOTE','No vote is recorded for that TD in the division data. It does not automatically mean the TD was absent.')]
    y=270
    for (t,b),h in zip(cards,[175,150,200,185]): panel(d,110,y,860,h,t,b,head_size=24,body_size=22,max_lines=6); y+=h+20
    d.rounded_rectangle([110,1055,970,1208],radius=20,fill=PANEL2,outline=ACCENT,width=3); centered(d,1080,'OUR VOTE-LABEL RULE',font(22,True),ACCENT)
    draw_centered_wrapped(d,1122,'We describe the exact proposition shown. Speaking about a Bill does not equal supporting it, and a linked vote may be about an amendment or procedure rather than the Bill itself.',font(19),TEXT,790,gap=6,max_lines=4)
    footer(d,'EirePolitic · Glossary · Vote terms & safeguards'); im.save(path)

def contact(paths,out,title):
    ims=[Image.open(p).convert('RGB') for p in paths]; tw,th=420,525; margin,gap=40,28; cols=3
    sheet=Image.new('RGB',(cols*tw+(cols-1)*gap+2*margin,3*th+2*55+135),'white'); d=ImageDraw.Draw(sheet)
    d.text((40,24),title,font=font(30,True),fill='black'); d.text((40,62),'Full final-review draft · 9 slides',font=font(19),fill='#444')
    for i,im in enumerate(ims):
        thumb=im.copy(); thumb.thumbnail((tw,th)); x=margin+(i%cols)*(tw+gap); y=115+(i//cols)*(th+55)
        d.text((x,y-28),str(i+1),font=font(19,True),fill='#333'); sheet.paste(thumb,(x,y))
    sheet.save(out)

# POST 1
make_cover(OUT/'post1_00_title.png',1,P1_TITLES,'These are the Bills that have been passed so far this session. These are the first three of six that we’re going to look at.')
make_explainer(OUT/'post1_01_gas_explainer.png',P1_TITLES[0],'Introduced by the Government · Minister for Climate, Energy and the Environment',('WHAT THE BILL DOES','Sets up a special legal route for approving a strategic gas reserve at Cahiracon, Co. Clare. It replaces the normal planning route for this project, but environmental assessments still apply.'),('PRACTICAL EFFECT','The Minister could decide the project directly under a faster process. The reserve is intended for emergencies if Ireland’s normal gas supplies are seriously disrupted.'),('WHY SOME TDs BACKED IT','Supporters said Ireland relies heavily on imported gas and needs a back-up supply if imports are seriously disrupted. They described it as an energy-security measure while Ireland moves toward renewables.'),('WHY SOME TDs OPPOSED IT','Critics said the project could prolong reliance on fossil fuels. They also objected to the special planning route, faster timetable and limited time for scrutiny.'),'WHAT THE 30 JUNE DÁIL VOTE MEANT','TDs were voting on one combined question that completed the Bill’s remaining Dáil steps and passed it. A Tá meant pass the Bill and send it on. A Níl meant reject that passage motion.','90 Tá · 57 Níl — carried','The Bill passed the Dáil and moved to the Seanad, Ireland’s second parliamentary chamber.')
make_vote(OUT/'post1_02_gas_vote.png','gas','Strategic Gas Reserve · Party Split','Dáil passage vote · 30 June 2026','This was the combined final Dáil question completing the Bill’s remaining Dáil stages.')
make_explainer(OUT/'post1_03_israeli_explainer.png',P1_TITLES[1],'Introduced by the Government · Minister for Foreign Affairs and Trade',('WHAT THE BILL DOES','Makes it unlawful to import goods into Ireland that originate in Israeli settlements in the occupied Palestinian territory. Those imports become enforceable under customs law.'),('PRACTICAL EFFECT','Businesses could not lawfully import those goods into Ireland. The enacted Bill is about goods only and does not extend the ban to services.'),('WHY SOME TDs BACKED IT','Supporters said Ireland should not trade in goods from settlements they view as illegal and that the State should reflect international-law obligations in domestic law.'),('WHY SOME TDs RAISED CONCERNS','A major debate was whether the Bill should also cover services. Others raised questions about trade law, enforceability and whether Ireland acting alone could face legal complications.'),'WHAT THE KEY DÁIL AMENDMENT VOTE MEANT','The recorded vote shown next was on adding certain settlement-related services to the Bill. It was not a final yes-or-no vote on the whole Bill.','67 Tá · 79 Níl — amendment lost','The proposal to add services did not pass, so the Bill continued in goods-only form.')
make_vote(OUT/'post1_04_israeli_vote.png','israeli','Settlement Goods Ban · Party Split','Services amendment · 7 July 2026','This division was on the services amendment, not final passage of the Bill. Tá = add the services provision; Níl = reject that amendment.')
make_explainer(OUT/'post1_05_criminal_explainer.png',P1_TITLES[2],'Introduced by the Government · Minister for Justice, Home Affairs and Migration',('WHAT THE BILL DOES','Bundles a wide range of legal changes into one Bill across criminal law, civil law, court procedure and Defence, including practical changes to evidence, administration and powers.'),('PRACTICAL EFFECT','Instead of changing these areas through many separate Bills, the legislation updates several parts of the justice and defence system together in one package.'),('WHY SOME TDs BACKED IT','Supporters argued that a large number of practical legal fixes were needed and that progressing them together would address gaps more quickly.'),('WHY SOME TDs RAISED CONCERNS','Critics said the Bill was too broad and moved too quickly. They questioned whether significant Defence and justice provisions got enough detailed scrutiny.'),'WHAT THE 10 JUNE DÁIL VOTE MEANT','The recorded division shown next was the concluding question during Report and Final Stages. The Dáil’s agreed business arrangements provided for those stages to finish by one question from the Chair.','102 Tá · 45 Níl — carried','The Bill completed its Dáil consideration and was sent to the Seanad.')
make_vote(OUT/'post1_06_criminal_vote.png','criminal','Criminal, Civil & Defence · Party Split','Dáil Report and Final Stages · 10 June 2026','This was the recorded concluding question during Report and Final Stages; the motion carried.')
make_glossary_terms(OUT/'post1_07_glossary_terms.png'); make_glossary_votes(OUT/'post1_08_glossary_votes.png')
p1=[OUT/f'post1_{i:02d}_{n}.png' for i,n in [(0,'title'),(1,'gas_explainer'),(2,'gas_vote'),(3,'israeli_explainer'),(4,'israeli_vote'),(5,'criminal_explainer'),(6,'criminal_vote'),(7,'glossary_terms'),(8,'glossary_votes')]]; contact(p1,OUT/'post1_contact_sheet.png','Bill Tracker · Post 1')

# POST 2
make_cover(OUT/'post2_00_title.png',2,P2_TITLES,'These are the Bills that have been passed so far this session. These are the next three of six that we’re going to look at.')
make_explainer(OUT/'post2_01_housing_explainer.png',P2_TITLES[0],'Introduced by the Government · Minister for Housing, Local Government and Heritage',('WHAT THE BILL DOES','Puts residency requirements for social-housing eligibility into legislation, creates a statutory appeals process for eligibility decisions, and also changes parts of residential-tenancy law.'),('PRACTICAL EFFECT','Local authorities get a clearer legal framework for deciding social-housing eligibility, and applicants get a formal route to challenge those decisions.'),('WHY SOME TDs BACKED IT','Supporters said the Bill would create clearer and more consistent rules for social-housing eligibility across different local authorities.'),('WHY SOME TDs RAISED CONCERNS','Critics said the interaction between housing, immigration and EU law was complex. They also warned that significant changes affecting vulnerable people needed closer scrutiny.'),'WHAT THE 8 JULY DÁIL VOTE MEANT','The recorded division shown next was the concluding question during Report and Final Stages. It was carried by 80 votes to 67.','80 Tá · 67 Níl — carried','The Bill completed its Dáil consideration and continued through the Seanad before enactment.')
make_vote(OUT/'post2_02_housing_vote.png','housing','Housing & Tenancies · Party Split','Dáil Report and Final Stages · 8 July 2026','This was the recorded concluding question during Report and Final Stages; the motion carried.')
make_explainer(OUT/'post2_03_health_explainer.png',P2_TITLES[1],'Introduced by the Government · Minister for Health',('WHAT THE BILL DOES','Creates the legal basis for a community-pharmacy contraception service, including pharmacist repeat prescribing for specified contraception after an initial GP prescription.'),('PRACTICAL EFFECT','Eligible people could be able to renew certain contraception through trained pharmacists instead of always returning to a GP for every repeat prescription.'),('WHY SOME TDs BACKED IT','Supporters said the Bill could make access quicker, easier and more convenient while expanding the role community pharmacies can play in routine care.'),('WHY SOME TDs RAISED CONCERNS','Questions focused mainly on implementation: pharmacist training, clinical guidance, record-sharing, patient safety and how the service would work in practice.'),'WHY THERE IS NO PARTY BREAKDOWN','The Bill completed its parliamentary stages and became law, but the official Oireachtas vote data contains no recorded member-by-member division linked to this Bill.','No recorded division','Because there is no recorded division, there is no reliable Tá / Níl party split to display.')
make_no_division(OUT/'post2_04_health_vote.png')
make_explainer(OUT/'post2_05_ai_explainer.png',P2_TITLES[2],'Introduced by the Government · Minister for Enterprise, Tourism and Employment',('WHAT THE BILL DOES','Builds Ireland’s domestic enforcement system for the EU AI Act, including an AI Office of Ireland and powers for regulators to supervise, investigate and enforce the EU rules.'),('PRACTICAL EFFECT','AI providers and users in Ireland face national oversight and enforcement mechanisms under the EU framework, with a central office coordinating implementation.'),('WHY SOME TDs BACKED IT','Supporters said Ireland needed the national infrastructure required to make the EU AI Act work in practice before key EU obligations took effect.'),('WHY SOME TDs RAISED CONCERNS','Critics questioned the AI Office’s independence and resourcing, how regulators would coordinate, and whether protections around rights, privacy, children and work were strong enough.'),'WHAT THE 30 JUNE DÁIL VOTE MEANT','The recorded division shown next was the concluding question during Committee and Remaining Stages. It was carried by 81 votes to 58.','81 Tá · 58 Níl — carried','The Bill completed its Dáil consideration and continued through the Seanad before enactment.')
make_vote(OUT/'post2_06_ai_vote.png','ai','AI Regulation · Party Split','Dáil Committee and Remaining Stages · 30 June 2026','This was the recorded concluding question during Committee and Remaining Stages; the motion carried.')
make_glossary_terms(OUT/'post2_07_glossary_terms.png'); make_glossary_votes(OUT/'post2_08_glossary_votes.png')
p2=[OUT/f'post2_{i:02d}_{n}.png' for i,n in [(0,'title'),(1,'housing_explainer'),(2,'housing_vote'),(3,'health_explainer'),(4,'health_vote'),(5,'ai_explainer'),(6,'ai_vote'),(7,'glossary_terms'),(8,'glossary_votes')]]; contact(p2,OUT/'post2_contact_sheet.png','Bill Tracker · Post 2')

print(OUT)
