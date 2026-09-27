#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

OUT = Path('artifacts/bill-tracker-full-posts-review')
OUT.mkdir(parents=True, exist_ok=True)
W,H=1080,1350
BG,TEXT,MUTED,ACCENT='#0f2f24','#f4ead7','#c8bda8','#d8b45f'
PANEL,PANEL2,BORDER='#173f31','#102b22','#31594a'
FOR,AGAINST,NO_VOTE=ACCENT,TEXT,'#65756d'

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

def slide(): return Image.new('RGB',(W,H),BG)

def footer(d,txt):
    d.rectangle([80,1260,1000,1263],fill=ACCENT); centered(d,1276,txt,font(15,True),MUTED)

def panel(d,x,y,w,h,title,body,head_size=21,body_size=23,max_lines=8):
    d.rounded_rectangle([x,y,x+w,y+h],radius=20,fill=PANEL,outline=BORDER,width=2)
    d.text((x+24,y+18),title,font=font(head_size,True),fill=ACCENT)
    end=draw_wrapped(d,x+24,y+56,body,font(body_size),TEXT,w-48,gap=4,max_lines=max_lines)
    if end>y+h-18: raise RuntimeError(f'panel overflow: {title}')

def draw_bar(d,x,y,w,h,yes,no,nr):
    total=max(1,yes+no+nr); cur=x
    vals=[(yes,FOR),(no,AGAINST),(nr,NO_VOTE)]
    nz=[v for v in vals if v[0]>0]
    for i,(val,col) in enumerate(nz):
        sw=(x+w-cur) if i==len(nz)-1 else round(w*val/total)
        d.rectangle([cur,y,cur+sw,y+h],fill=col); cur+=sw
    d.rectangle([x,y,x+w,y+h],outline=MUTED,width=2)

def make_cover(path,post_num,bills,desc):
    im=slide(); d=ImageDraw.Draw(im)
    centered(d,120,'BILLS OF THE',font(30,True),ACCENT); centered(d,165,'CURRENT SESSION',font(56,True),TEXT)
    d.rectangle([185,242,895,248],fill=ACCENT); centered(d,282,f'ENACTED · POST {post_num}',font(28,True),ACCENT)
    draw_centered_wrapped(d,345,desc,font(23),TEXT,820,gap=8,max_lines=5); centered(d,520,'IN THIS POST',font(24,True),ACCENT)
    y=575
    for i,title in enumerate(bills,1):
        d.rounded_rectangle([95,y,985,y+180],radius=20,fill=PANEL,outline=BORDER,width=2)
        d.ellipse([125,y+46,181,y+102],fill=ACCENT); d.text((153,y+74),str(i),font=font(24,True),fill=BG,anchor='mm')
        lines=wrap(d,title,font(25,True),720)
        d.multiline_text((590,y+90),'\n'.join(lines),font=font(25,True),fill=TEXT,anchor='mm',align='center',spacing=7)
        y+=202
    footer(d,f'EirePolitic · Enacted · Post {post_num}'); im.save(path)

def make_explainer(path,title,subtitle,a,b,c,e,bottom_title,bottom_body,result=None,effect=None):
    im=slide(); d=ImageDraw.Draw(im)
    centered(d,54,title,font(34,True),TEXT); d.rectangle([110,118,970,123],fill=ACCENT)
    centered(d,146,'WHAT IT DOES & WHY IT WAS DEBATED',font(22,True),ACCENT); centered(d,184,subtitle,font(17),MUTED)
    panel(d,60,232,460,270,*a,body_size=23); panel(d,560,232,460,270,*b,body_size=23)
    panel(d,60,526,460,286,*c,body_size=22); panel(d,560,526,460,286,*e,body_size=22)
    d.rounded_rectangle([60,850,1020,1206],radius=22,fill=PANEL2,outline=ACCENT,width=3)
    centered(d,878,bottom_title,font(24,True),ACCENT)
    y=draw_centered_wrapped(d,928,bottom_body,font(22),TEXT,860,gap=6,max_lines=7)
    if result:
        centered(d,y+6,'RESULT',font(18,True),MUTED); centered(d,y+36,result,font(26,True),TEXT); y+=74
    if effect: draw_centered_wrapped(d,y+16,effect,font(21,True),MUTED,850,gap=6,max_lines=4)
    footer(d,'EirePolitic · Draft review copy'); im.save(path)

def make_context(path,title,stage,q,meaning,outcome,counts=None,foot='EirePolitic · Draft review copy'):
    im=slide(); d=ImageDraw.Draw(im)
    centered(d,62,title,font(40,True),TEXT); d.rectangle([110,130,970,135],fill=ACCENT); centered(d,160,stage,font(23,True),ACCENT)
    if counts:
        yes,no,nr,elig,label=counts; draw_bar(d,90,210,900,58,yes,no,nr)
        cur=180
        for lab,col in [('Tá',FOR),('Níl',AGAINST),('No recorded vote',NO_VOTE)]:
            d.rectangle([cur,292,cur+18,310],fill=col,outline=MUTED); d.text((cur+30,288),lab,font=font(18,True),fill=TEXT); cur+=230
        centered(d,328,label,font(20,True),TEXT)
    else:
        d.rounded_rectangle([120,220,960,350],radius=22,fill=PANEL2,outline=ACCENT,width=3); centered(d,250,'RECORDED-VOTE STATUS',font(24,True),ACCENT)
        draw_centered_wrapped(d,290,'For this review draft, this slide gives the vote context in words rather than a party-split chart.',font(20),TEXT,760,gap=6,max_lines=4)
    panel(d,70,390,940,190,'WHAT WAS BEING DECIDED?',q,head_size=22,body_size=24,max_lines=6)
    panel(d,70,610,940,170,'WHAT DID A TÁ / NÍL MEAN?',meaning,head_size=22,body_size=23,max_lines=5)
    panel(d,70,810,940,170,'WHAT HAPPENED?',outcome,head_size=22,body_size=23,max_lines=5)
    footer(d,foot); im.save(path)

def make_glossary_terms(path):
    im=slide(); d=ImageDraw.Draw(im)
    centered(d,95,'GLOSSARY',font(48,True),TEXT); centered(d,157,'HOW BILLS MOVE THROUGH PARLIAMENT',font(22,True),ACCENT); d.rectangle([150,205,930,210],fill=ACCENT)
    cards=[('BILL','A proposed law. It must go through parliamentary stages before it can become law.'),('DÁIL ÉIREANN','Ireland’s directly elected house of parliament. Its elected members are called TDs.'),('SEANAD ÉIREANN','Ireland’s second parliamentary chamber. Its members are Senators.'),('STAGE','A formal step in considering a Bill. Different stages cover its principles, detailed text, amendments and final approval.'),('ENACTED','The Bill has completed the required parliamentary process and has become law.')]
    y=265
    for (t,b),h in zip(cards,[155,155,155,176,155]): panel(d,110,y,860,h,t,b,head_size=24,body_size=22,max_lines=5); y+=h+18
    footer(d,'EirePolitic · Glossary · Parliamentary terms'); im.save(path)

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
    ims=[Image.open(p).convert('RGB') for p in paths]; tw,th=420,525; margin,gap=40,28; cols=3; rows=3
    sheet=Image.new('RGB',(cols*tw+(cols-1)*gap+2*margin,rows*th+(rows-1)*55+135),'white'); d=ImageDraw.Draw(sheet)
    d.text((40,24),title,font=font(30,True),fill='black'); d.text((40,62),'Full text-review draft · 9 slides',font=font(19),fill='#444')
    for i,im in enumerate(ims):
        thumb=im.copy(); thumb.thumbnail((tw,th)); x=margin+(i%cols)*(tw+gap); y=115+(i//cols)*(th+55)
        d.text((x,y-28),f'{i+1}',font=font(19,True),fill='#333'); sheet.paste(thumb,(x,y))
    sheet.save(out)

p1_titles=['Development (Strategic Gas Reserve) Bill 2026','Israeli Settlements in the Occupied Palestinian Territory (Prohibition of Importation of Goods) Bill 2026','Criminal Law, Civil Law and Defence (Miscellaneous Provisions) Bill 2026']
p2_titles=['Housing and Residential Tenancies (Miscellaneous Provisions) Bill 2026','Health (Provision of Contraception Prescribing Service in Retail Pharmacy Businesses) Bill 2026','Regulation of Artificial Intelligence Bill 2026']

make_cover(OUT/'post1_00_title.png',1,p1_titles,'These are the Bills that have been passed so far this session. These are the first three of six that we’re going to look at.')
make_explainer(OUT/'post1_01_gas_explainer.png','Development (Strategic Gas Reserve) Bill 2026','Introduced by the Government · Minister for Climate, Energy and the Environment',('WHAT THE BILL DOES','Sets up a special legal route for approving a strategic gas reserve at Cahiracon, Co. Clare. It replaces the normal planning route for this project, but environmental assessments still apply.'),('PRACTICAL EFFECT','The Minister could decide the project directly under a faster process. The reserve is intended for emergencies if Ireland’s normal gas supplies are seriously disrupted.'),('WHY SOME TDs BACKED IT','Supporters said Ireland relies heavily on imported gas and needs a back-up supply if imports are seriously disrupted. They described it as an energy-security measure while Ireland moves toward renewables.'),('WHY SOME TDs OPPOSED IT','Critics said the project could prolong reliance on fossil fuels. They also objected to the special planning route, faster timetable and limited time for scrutiny.'),'WHAT THE 30 JUNE DÁIL VOTE MEANT','TDs — members of the Dáil — were voting on one combined question that completed the Bill’s remaining Dáil steps and passed it. A Tá meant pass the Bill and send it on. A Níl meant reject that passage motion.','90 Tá · 57 Níl — carried','The Bill passed the Dáil and moved to the Seanad, Ireland’s second parliamentary chamber.')
make_context(OUT/'post1_02_gas_vote.png','Strategic Gas Reserve · Vote Slide','Dáil passage vote · 30 June 2026','The question bundled the Bill’s remaining Dáil steps. It was effectively the passage motion at the end of Dáil consideration.','A Tá meant pass the Bill in the Dáil and send it to the Seanad. A Níl meant reject that motion.','The motion carried. The Bill passed the Dáil and then completed Seanad consideration before enactment.',(90,57,27,174,'174 eligible TDs · 90 Tá · 57 Níl · 27 no recorded vote'),'EirePolitic · Review draft of vote context')
make_explainer(OUT/'post1_03_israeli_explainer.png','Israeli Settlements in the Occupied Palestinian Territory (Prohibition of Importation of Goods) Bill 2026','Introduced by the Government · Minister for Foreign Affairs and Trade',('WHAT THE BILL DOES','Makes it unlawful to import goods into Ireland that originate in Israeli settlements in the occupied Palestinian territory. Those imports become enforceable under customs law.'),('PRACTICAL EFFECT','Businesses could not lawfully import those goods into Ireland. The enacted Bill is about goods only and does not extend the ban to services.'),('WHY SOME TDs BACKED IT','Supporters said Ireland should not trade in goods from settlements they view as illegal and that the State should reflect international-law obligations in domestic law.'),('WHY SOME TDs RAISED CONCERNS','A major debate was whether the Bill should also cover services. Others raised questions about trade law, enforceability and whether Ireland acting alone could face legal complications.'),'WHAT THE KEY DÁIL AMENDMENT VOTE MEANT','For this draft, the vote shown is the amendment on adding services. It is not the same thing as a final yes-or-no vote on the whole Bill.','67 Tá · 79 Níl — amendment lost','The proposal to add services did not pass, so the Bill continued in goods-only form.')
make_context(OUT/'post1_04_israeli_vote.png','Settlement Goods Ban · Amendment Vote','Amendment No. 16 on services · Dáil Report Stage · 7 July 2026','TDs were deciding whether the Bill should go beyond goods and also include certain settlement-related services.','A Tá meant add services to the Bill. A Níl meant leave the Bill focused on goods.','The amendment was defeated, so services were not added. The enacted Bill remained limited to goods.',(67,79,28,174,'174 eligible TDs · 67 Tá · 79 Níl · 28 no recorded vote'),'EirePolitic · Review draft of vote context')
make_explainer(OUT/'post1_05_criminal_explainer.png','Criminal Law, Civil Law and Defence (Miscellaneous Provisions) Bill 2026','Introduced by the Government · Minister for Justice, Home Affairs and Migration',('WHAT THE BILL DOES','Bundles a wide range of legal changes into one Bill across criminal law, civil law, court procedure and Defence, including practical changes to evidence, administration and powers.'),('PRACTICAL EFFECT','Instead of changing these areas through many separate Bills, the legislation updates several parts of the justice and defence system together in one package.'),('WHY SOME TDs BACKED IT','Supporters argued that a large number of practical legal fixes were needed and that progressing them together would address gaps more quickly.'),('WHY SOME TDs RAISED CONCERNS','Critics said the Bill was too broad and moved too quickly. They questioned whether significant Defence and justice provisions got enough detailed scrutiny.'),'HOW THE VOTE CONTEXT IS BEING HANDLED IN THIS DRAFT','The Bill record links several recorded votes during scrutiny. To avoid mislabelling a procedural or amendment vote as support for the whole Bill, the paired slide explains the context in neutral terms pending final proposition sign-off.')
make_context(OUT/'post1_06_criminal_vote.png','Criminal, Civil & Defence Changes · Vote Context','Linked committee-stage division(s) · proposition wording under final review','This Bill has multiple linked recorded votes in the current record. At least some appear to concern amendments or procedural questions during Seanad scrutiny rather than a simple final-passage motion.','Until the exact proposition is signed off, we should avoid implying that a Tá automatically meant support for the whole Bill, or that a Níl automatically meant opposition to the whole Bill.','For text review, this slide is intentionally neutral. Once the exact proposition is locked, it can be converted into the final party-split format if needed.',None,'EirePolitic · Context wording intentionally neutral pending proposition sign-off')
make_glossary_terms(OUT/'post1_07_glossary_terms.png'); make_glossary_votes(OUT/'post1_08_glossary_votes.png')
p1=[OUT/f'post1_{i:02d}_{n}.png' for i,n in [(0,'title'),(1,'gas_explainer'),(2,'gas_vote'),(3,'israeli_explainer'),(4,'israeli_vote'),(5,'criminal_explainer'),(6,'criminal_vote'),(7,'glossary_terms'),(8,'glossary_votes')]]
contact(p1,OUT/'post1_contact_sheet.png','Bill Tracker · Post 1')

make_cover(OUT/'post2_00_title.png',2,p2_titles,'These are the Bills that have been passed so far this session. These are the next three of six that we’re going to look at.')
make_explainer(OUT/'post2_01_housing_explainer.png','Housing and Residential Tenancies (Miscellaneous Provisions) Bill 2026','Introduced by the Government · Minister for Housing, Local Government and Heritage',('WHAT THE BILL DOES','Puts residency requirements for social-housing eligibility into legislation, creates a statutory appeals process for eligibility decisions, and also changes parts of residential-tenancy law.'),('PRACTICAL EFFECT','Local authorities get a clearer legal framework for deciding social-housing eligibility, and applicants get a formal route to challenge those decisions.'),('WHY SOME TDs BACKED IT','Supporters said the Bill would create clearer and more consistent rules for social-housing eligibility across different local authorities.'),('WHY SOME TDs RAISED CONCERNS','Critics said the interaction between housing, immigration and EU law was complex. They also warned that significant changes affecting vulnerable people needed closer scrutiny.'),'HOW THE PAIRED VOTE SLIDE IS FRAMED IN THIS DRAFT','The Bill record links several recorded votes. To avoid over-claiming what one committee-stage division meant, the paired slide uses neutral wording until the final proposition is signed off.')
make_context(OUT/'post2_02_housing_vote.png','Housing & Tenancies · Vote Context','Linked committee-stage division(s) · proposition wording under final review','The linked divisions appear to come from scrutiny and amendments rather than a straightforward whole-Bill passage vote.','That means a Tá may have supported a particular amendment or procedural motion, not necessarily every part of the Bill as enacted.','For review purposes, this slide keeps the explanation neutral. Once the exact proposition is locked, it can be converted into the final vote graphic if appropriate.',None,'EirePolitic · Context wording intentionally neutral pending proposition sign-off')
make_explainer(OUT/'post2_03_health_explainer.png','Health (Provision of Contraception Prescribing Service in Retail Pharmacy Businesses) Bill 2026','Introduced by the Government · Minister for Health',('WHAT THE BILL DOES','Creates the legal basis for a community-pharmacy contraception service, including pharmacist repeat prescribing for specified contraception after an initial GP prescription.'),('PRACTICAL EFFECT','Eligible people could be able to renew certain contraception through trained pharmacists instead of always returning to a GP for every repeat prescription.'),('WHY SOME TDs BACKED IT','Supporters said the Bill could make access quicker, easier and more convenient while expanding the role community pharmacies can play in routine care.'),('WHY SOME TDs RAISED CONCERNS','Questions focused mainly on implementation: pharmacist training, clinical guidance, record-sharing, patient safety and how the service would work in practice.'),'WHY THIS BILL’S SECOND SLIDE IS DIFFERENT','The current bill-tracker record notes no certified linked division for this Bill. So the paired slide explains the recorded-vote status instead of showing a party split.')
make_context(OUT/'post2_04_health_vote.png','Pharmacy Contraception · Recorded-Vote Status','No certified linked division in the current bill record','This Bill completed its stages and was enacted, but the current bill-tracker record does not link a certified recorded division to it.','Because there is no linked division selected, there is no reliable party-split chart to show on this draft slide.','The practical upshot is that the post can still explain what changed, but the vote slide should either stay descriptive or be replaced with a stage/status slide.',None,'EirePolitic · No linked division currently selected for this Bill')
make_explainer(OUT/'post2_05_ai_explainer.png','Regulation of Artificial Intelligence Bill 2026','Introduced by the Government · Minister for Enterprise, Tourism and Employment',('WHAT THE BILL DOES','Builds Ireland’s domestic enforcement system for the EU AI Act, including an AI Office of Ireland and powers for regulators to supervise, investigate and enforce the EU rules.'),('PRACTICAL EFFECT','AI providers and users in Ireland would face national oversight and enforcement mechanisms under the EU framework, with a central office coordinating implementation.'),('WHY SOME TDs BACKED IT','Supporters said Ireland needed the national infrastructure required to make the EU AI Act work in practice before key EU obligations took effect.'),('WHY SOME TDs RAISED CONCERNS','Critics questioned the AI Office’s independence and resourcing, how regulators would coordinate, and whether protections around rights, privacy, children and work were strong enough.'),'HOW THE VOTE CONTEXT IS TREATED HERE','The Bill record links recorded votes during scrutiny. Until the exact proposition is signed off, the paired slide explains the context carefully rather than presenting a definitive party-split claim.')
make_context(OUT/'post2_06_ai_vote.png','AI Regulation · Vote Context','Linked scrutiny division(s) · proposition wording under final review','This Bill had linked recorded votes during its scrutiny stages. Those votes may relate to amendments or procedural decisions rather than a simple up-or-down final-passage motion.','So a Tá cannot automatically be described as support for the entire Bill unless the exact proposition clearly means that.','For this review draft, the slide stays neutral and explains the caution. It can be finalised into the standard vote layout once the proposition wording is confirmed.',None,'EirePolitic · Context wording intentionally neutral pending proposition sign-off')
make_glossary_terms(OUT/'post2_07_glossary_terms.png'); make_glossary_votes(OUT/'post2_08_glossary_votes.png')
p2=[OUT/f'post2_{i:02d}_{n}.png' for i,n in [(0,'title'),(1,'housing_explainer'),(2,'housing_vote'),(3,'health_explainer'),(4,'health_vote'),(5,'ai_explainer'),(6,'ai_vote'),(7,'glossary_terms'),(8,'glossary_votes')]]
contact(p2,OUT/'post2_contact_sheet.png','Bill Tracker · Post 2')
print(OUT)
