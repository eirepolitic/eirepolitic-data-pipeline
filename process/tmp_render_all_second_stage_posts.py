#!/usr/bin/env python3
from __future__ import annotations
import io, json, os, re, sys
from pathlib import Path
import boto3, pandas as pd
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from extract.oireachtas.batch import resolve_production_key
from political_metrics.bill_content_snapshot import build_bill_content_snapshot
from instagram.factory.render_primitives import base_slide, font, wrap_text_px, contact_sheet, BG, TEXT, ACCENT, MUTED

OUT=Path('artifacts/bill-tracker-second-stage-all'); OUT.mkdir(parents=True,exist_ok=True)
W,H=1080,1350; PANEL='#174638'; OUTLINE='#346857'
BUCKET=os.getenv('S3_BUCKET','eirepolitic-data')
KEYS={'bills':'processed/oireachtas_unified/latest/csv/silver_bills.csv','stages':'processed/oireachtas_unified/latest/csv/silver_bill_stages.csv','sponsors':'processed/oireachtas_unified/latest/csv/silver_bill_sponsors.csv','bridge':'processed/oireachtas_unified/latest/metrics/event/bill_debate_sections/csv/bill_debate_sections.csv','speeches':'processed/oireachtas_unified/latest/csv/silver_speeches.csv','divisions':'processed/oireachtas_unified/latest/csv/silver_divisions.csv','member_votes':'processed/oireachtas_unified/latest/csv/silver_member_votes.csv'}

SPECIAL={
'Anti-Shrinkflation Bill 2026':('Would require clearer retail labelling when a product’s quantity is reduced in a way that raises its unit price, so consumers can spot hidden price increases.','Large retailers would have to flag qualifying quantity reductions for a set period. The proposal focuses on price transparency rather than banning smaller packs or setting prices.','The sponsor argues shoppers should be told clearly when they are paying effectively more for less, particularly during a period of cost-of-living pressure.','At Second Stage the House can test the proposal’s general approach: which retailers or products should be covered, how notice rules work, exemptions, enforcement and proportionality.'),
'Broadcasting (Amendment) Bill 2026':('Would reform governance, transparency, funding and oversight arrangements for RTÉ and TG4, expand Coimisiún na Meán functions and implement parts of the European Media Freedom Act.','The Bill would change how public service media governance, auditing, performance assessment and some public-service-content funding arrangements operate.','Government presented the Bill as implementing recommendations from the Future of Media Commission and the independent RTÉ governance review, alongside EU media-law requirements.','Second Stage debate raised issues including governance, long-term public-service-media funding, Irish-language provision, independent production, geo-blocking and implementation detail.'),
'Electoral (Postal Voting) (Carers) Bill 2026':('Would extend eligibility for the postal-voter register to certain people who provide care for others, by amending the Electoral Act 1992.','Qualifying carers who cannot readily attend a polling station because of caring responsibilities could gain a postal-voting route if the Bill eventually becomes law.','The sponsor presented the proposal as a way to reduce barriers to electoral participation faced by family carers whose responsibilities can make in-person voting difficult.','The substantive Second Stage debate can test the proposal’s general principles, eligibility rules, safeguards and how a carers postal-voting category should operate.'),
'Defence (Amendment) (No. 2) Bill 2026':('Would prohibit U.S. military aircraft, and civilian aircraft carrying munitions of war, from landing in the State, subject to limited emergency search-and-rescue exceptions.','Routine landings in Ireland by aircraft covered by the prohibition would no longer be permitted if the Bill became law.','The sponsor presented the proposal as a way to end U.S. military use of Shannon and to align airport practice with his view of Irish neutrality.','A Second Stage debate can test the Bill’s broad approach, definitions and exceptions, its relationship with neutrality and foreign policy, enforcement and aviation implications.'),
'Adult Safeguarding Strategy Bill 2026':('Would require the Minister for Health to prepare recurring statutory strategies for protecting adults at risk of harm, with implementation, review and reporting requirements.','Adult safeguarding would be placed on a regular statutory planning cycle, with public accountability for actions, outstanding work and review.','The sponsors presented the Bill as a way to strengthen coordination and accountability for protecting adults at risk across health, care and other relevant settings.','A Second Stage debate can examine who should be covered, which bodies should have duties, oversight, implementation, resourcing and interaction with existing safeguarding policy.'),
'Prevention of Energy Wastage Bill 2026':('Would create a statutory basis for using renewable electricity that would otherwise be curtailed or constrained, linking that unused energy to climate, just-transition and energy-poverty objectives.','The proposal would enable a scheme for eligible electricity customers to benefit from otherwise-unused renewable power, with a focus on affordability and vulnerable households.','The sponsor argues renewable electricity should not be wasted while households face energy costs, and that surplus clean power should contribute to climate and energy-poverty goals.','A Second Stage debate can examine electricity-market and grid rules, customer eligibility, cost allocation, system operation and how the scheme would work in practice.'),
}
FIRST_EIGHT=['Anti-Shrinkflation Bill 2026','Broadcasting (Amendment) Bill 2026','Electoral (Postal Voting) (Carers) Bill 2026','Electoral (Amendment) (Voting Age) Bill 2026','Defence (Amendment) (No. 2) Bill 2026','Adult Safeguarding Strategy Bill 2026','Prevention of Energy Wastage Bill 2026','Protection of Children (Online Age Verification) Bill 2026']

def read_s3(s3,k):
    rk=resolve_production_key(s3,bucket=BUCKET,production_key=k); obj=s3.get_object(Bucket=BUCKET,Key=rk)
    return pd.read_csv(io.BytesIO(obj['Body'].read()),dtype=str,keep_default_na=False),rk

def load_rows():
    s3=boto3.client('s3',region_name=os.getenv('AWS_REGION','ca-central-1')); f={}; resolved={}
    for n,k in KEYS.items(): f[n],resolved[n]=read_s3(s3,k)
    snap=build_bill_content_snapshot(bills=f['bills'],stages=f['stages'],sponsors=f['sponsors'],bill_debate_sections=f['bridge'],speeches=f['speeches'],divisions=f['divisions'],member_votes=f['member_votes'],batch_size=6)
    x=snap[snap['series_bucket'].eq('second_stage')].copy(); x['_last']=pd.to_datetime(x['last_event_date'],errors='coerce')
    def keep(r):
        h=str(r.get('current_stage_house_name','')).lower(); dt=r['_last']
        if pd.isna(dt): return False
        return (h in {'34th dáil','34th dail'} and dt>=pd.Timestamp('2024-12-18')) or (h=='27th seanad' and dt>=pd.Timestamp('2025-02-12'))
    x=x[x.apply(keep,axis=1)].copy().sort_values(['_last','bill_year','bill_no'],ascending=[False,False,False])
    by={r['title']:r for _,r in x.iterrows()}; ordered=[]
    for t in FIRST_EIGHT:
        if t in by: ordered.append(by.pop(t))
    ordered += [r for _,r in x.iterrows() if r['title'] in by]
    if len(ordered)!=108: raise RuntimeError(f'expected 108 current-term Second Stage bills, got {len(ordered)}')
    return ordered,resolved['bills']

def rule(d,y,l=82,r=998,w=5): d.rectangle((l,y,r,y+w),fill=ACCENT)
def panel(d,box,outline=OUTLINE,fill=PANEL,w=2): d.rounded_rectangle(box,radius=18,fill=fill,outline=outline,width=w)
def footer(d,text,source='Source: Houses of the Oireachtas'):
    rule(d,1254,58,1022,4); d.text((64,1278),source,font=font(12),fill=MUTED,anchor='la'); d.text((1016,1278),text,font=font(12,True),fill=MUTED,anchor='ra')
def lines_h(d,lines,f,g=5): return sum(d.textbbox((0,0),ln,font=f)[3]-d.textbbox((0,0),ln,font=f)[1] for ln in lines)+g*max(0,len(lines)-1)
def fit(d,text,width,height,start=28,min_size=15,max_lines=8,bold=False):
    for s in range(start,min_size-1,-1):
        f=font(s,bold); lines=wrap_text_px(d,text,f,width)
        if len(lines)<=max_lines and lines_h(d,lines,f,4)<=height:return f
    return font(min_size,bold)
def draw_block(d,text,box,f,fill=TEXT,center=False,gap=4):
    x0,y0,x1,y1=box; lines=wrap_text_px(d,text,f,x1-x0); cur=y0
    for ln in lines:
        if center: d.text(((x0+x1)//2,cur),ln,font=f,fill=fill,anchor='ma'); bb=d.textbbox(((x0+x1)//2,cur),ln,font=f,anchor='ma')
        else: d.text((x0,cur),ln,font=f,fill=fill,anchor='la'); bb=d.textbbox((x0,cur),ln,font=f,anchor='la')
        cur=bb[3]+gap
    return cur

def subject_from_title(title):
    s=re.sub(r'\s+Bill\s+\d{4}$','',title).strip(); s=re.sub(r'\s*\((?:Amendment|Miscellaneous Provisions|No\. ?\d+)\)\s*',' ',s,flags=re.I); return re.sub(r'\s+',' ',s).strip()
def generic_copy(r):
    title=r['title']; subj=subject_from_title(title)
    if 'Amendment of the Constitution' in title or 'Amendment of Constitution' in title: what=f'Proposes a constitutional amendment concerning {subj.lower()}. The exact wording and legal effect are determined by the Bill text and, if passed by both Houses, would require approval at referendum.'
    elif 'Amendment' in title: what=f'Would amend existing legislation in the area of {subj.lower()}. The Bill is currently before the House at Second Stage, where its overall approach is considered.'
    else: what=f'Would create or change legislation concerning {subj.lower()}. At Second Stage, the House considers whether the Bill’s general approach should proceed to detailed scrutiny.'
    practical='If enacted, the Bill would change the legal framework in the area named above. This slide stays at that verified scope level unless the underlying Bill text has been separately researched and certified.'
    sponsor=str(r.get('primary_sponsor_name','')).strip() or str(r.get('primary_sponsor_role_name','')).strip() or 'Sponsor recorded by the Oireachtas'
    context=f'{sponsor} introduced or sponsors the Bill. EirePolitic treats the sponsor’s position separately from the views of other TDs or Senators and does not infer wider support from sponsorship or speeches.'
    sections=int(r.get('certified_section_count',0)); speeches=int(r.get('certified_speech_count',0))
    if speeches: issues=f'The pipeline links {sections} certified debate section{"s" if sections!=1 else ""} and {speeches} certified intervention{"s" if speeches!=1 else ""} to this Bill. Those records provide context, but speaking does not by itself establish support or opposition.'
    else: issues='No certified substantive debate interventions are linked in the current production snapshot. That is not evidence that the Bill lacks importance; it means there is no debate record here to summarise safely.'
    return what,practical,context,issues

def status_copy(r):
    house=r.get('current_stage_house_name',''); d=r.get('current_stage_date',''); speeches=int(r.get('certified_speech_count',0))
    if speeches: return f'The Bill is recorded at Second Stage in the {house} (stage date {d}). The production snapshot contains certified debate material, but it does not show the Bill having moved beyond Second Stage.'
    return f'The Bill is recorded at Second Stage in the {house} (stage date {d}). No substantive Second Stage debate is certified in the current snapshot, so the next meaningful step is consideration of its general principles.'

def cover(out,post_no,bills):
    im=base_slide(); d=ImageDraw.Draw(im); d.text((W//2,126),'BILLS OF THE',font=font(28,True),fill=ACCENT,anchor='ma'); d.text((W//2,174),'CURRENT SESSION',font=font(44,True),fill=TEXT,anchor='ma'); rule(d,240,138,942,5); d.text((W//2,282),f'SECOND STAGE · POST {post_no}',font=font(26,True),fill=ACCENT,anchor='ma')
    intro='Four Bills currently at Second Stage in the 34th Dáil or 27th Seanad. Second Stage is where the House considers a Bill’s general principles before detailed Committee Stage scrutiny.'; f=fit(d,intro,800,135,22,18,6); draw_block(d,intro,(140,340,940,475),f,center=True); d.text((W//2,510),'IN THIS POST',font=font(23,True),fill=ACCENT,anchor='ma'); y=555
    for i,r in enumerate(bills,1):
        panel(d,(72,y,1008,y+150)); d.ellipse((102,y+46,158,y+102),fill=ACCENT); d.text((130,y+74),str(i),font=font(22,True),fill=BG,anchor='mm'); t=r['title']; tf=fit(d,t,760,105,24,17,3,True); lines=wrap_text_px(d,t,tf,760); total=lines_h(d,lines,tf,4); cur=y+(150-total)//2
        for ln in lines: d.text((580,cur),ln,font=tf,fill=TEXT,anchor='ma'); bb=d.textbbox((580,cur),ln,font=tf,anchor='ma'); cur=bb[3]+4
        y+=166
    footer(d,f'Second Stage · Post {post_no}'); im.save(out/'00-cover.png')

def bill_slide(out,idx,r):
    im=base_slide(); d=ImageDraw.Draw(im); title=r['title']; tf=fit(d,title,900,95,34,22,3,True); lines=wrap_text_px(d,title,tf,900); cur=54
    for ln in lines: d.text((W//2,cur),ln,font=tf,fill=TEXT,anchor='ma'); bb=d.textbbox((W//2,cur),ln,font=tf,anchor='ma'); cur=bb[3]+4
    rule(d,max(145,cur+8)); ry=max(175,cur+38); sponsor=str(r.get('primary_sponsor_name','')).strip() or str(r.get('primary_sponsor_role_name','')).strip() or 'Sponsor not named in snapshot'; meta=f"Bill No. {r.get('bill_no','')} of {r.get('bill_year','')} · {r.get('origin_house_name','')} · {sponsor}"; mf=fit(d,meta,900,45,17,13,2); draw_block(d,meta,(90,ry,990,ry+48),mf,fill=MUTED,center=True)
    texts=SPECIAL.get(title) or generic_copy(r); labels=['WHAT THE BILL COVERS','PRACTICAL EFFECT','SPONSOR / CONTEXT','SECOND STAGE RECORD']; top=ry+62; bh=282; gap=22; boxes=[(58,top,518,top+bh),(562,top,1022,top+bh),(58,top+bh+gap,518,top+2*bh+gap),(562,top+bh+gap,1022,top+2*bh+gap)]
    shared=14
    for s in range(24,13,-1):
        ff=font(s); ok=True
        for txt in texts:
            if lines_h(d,wrap_text_px(d,txt,ff,408),ff,4)>190: ok=False; break
        if ok: shared=s; break
    bf=font(shared)
    for box,label,txt in zip(boxes,labels,texts):
        panel(d,box); d.text((box[0]+22,box[1]+20),label,font=font(16,True),fill=ACCENT,anchor='la'); end=draw_block(d,txt,(box[0]+22,box[1]+58,box[2]-22,box[3]-18),bf)
        if end>box[3]-12: raise RuntimeError(f'overflow {title} {label} at shared font {shared}')
    sy=boxes[2][3]+28; panel(d,(58,sy,1022,1210),outline=ACCENT,fill=BG,w=3); d.text((W//2,sy+26),'WHERE IT IS NOW',font=font(20,True),fill=ACCENT,anchor='ma'); stat=status_copy(r); sf=fit(d,stat,850,130,29,18,6); lines=wrap_text_px(d,stat,sf,850); total=lines_h(d,lines,sf,5); start=sy+62+(1210-(sy+62)-total)//2; draw_block(d,stat,(115,start,965,1190),sf,center=True,gap=5); footer(d,f'Second Stage · Bill {idx}',source=f"Source: Houses of the Oireachtas · {r.get('bill_id','')}"); im.save(out/f'{idx:02d}-bill.png')

def process_glossary(out):
    im=base_slide(); d=ImageDraw.Draw(im); d.text((W//2,74),'GLOSSARY',font=font(40,True),fill=TEXT,anchor='ma'); d.text((W//2,133),'HOW A BILL MOVES THROUGH PARLIAMENT',font=font(22,True),fill=ACCENT,anchor='ma'); rule(d,176,112,968,4); labels=['FIRST','SECOND','COMMITTEE','REPORT','FINAL','OTHER HOUSE','PRESIDENT','ENACTED']; x0,y,bw,bh,gap=35,220,105,104,15
    for i,l in enumerate(labels):
        x=x0+i*(bw+gap); active=l=='SECOND'; panel(d,(x,y,x+bw,y+bh),outline=ACCENT if active else OUTLINE,fill=ACCENT if active else PANEL); d.text((x+bw//2,y+24),str(i+1),font=font(14,True),fill=BG if active else ACCENT,anchor='mm'); lf=font(10,True); lines=wrap_text_px(d,l,lf,bw-12); cy=y+61
        for ln in lines: d.text((x+bw//2,cy),ln,font=lf,fill=BG if active else ACCENT,anchor='mm'); cy+=15
        if i<len(labels)-1: ax=x+bw+4; d.polygon([(ax,y+52),(ax+9,y+45),(ax+9,y+59)],fill=ACCENT)
    d.text((W//2,356),'THIS POST: SECOND STAGE',font=font(18,True),fill=ACCENT,anchor='ma'); expl='Second Stage is where the House debates the Bill’s general principles. If agreed, detailed section-by-section scrutiny normally follows at Committee Stage.'; ef=fit(d,expl,860,100,20,17,5); draw_block(d,expl,(110,398,970,500),ef,center=True); cards=[('FIRST HOUSE','A Bill can be at Second Stage in the House where it began.'),('SECOND HOUSE','After completing the first House, a Bill normally goes through stages in the other House too.'),('CURRENT POSITION','Read the stage together with Dáil or Seanad: “Second Stage” alone does not tell you how far through the full legislative journey the Bill is.')]; boxes=[(70,555,505,745),(575,555,1010,745),(170,790,910,1035)]
    for (hd,bd),box in zip(cards,boxes): panel(d,box); cx=(box[0]+box[2])//2; d.text((cx,box[1]+35),hd,font=font(19,True),fill=ACCENT,anchor='ma'); bf=fit(d,bd,box[2]-box[0]-55,box[3]-box[1]-85,22,17,6); lines=wrap_text_px(d,bd,bf,box[2]-box[0]-55); total=lines_h(d,lines,bf,4); start=box[1]+72+(box[3]-(box[1]+72)-total)//2; draw_block(d,bd,(box[0]+28,start,box[2]-28,box[3]-18),bf,center=True)
    footer(d,'Second Stage · Process glossary'); im.save(out/'05-process-glossary.png')

def stage_glossary(out):
    im=base_slide(); d=ImageDraw.Draw(im); d.text((W//2,82),'SECOND STAGE',font=font(41,True),fill=TEXT,anchor='ma'); d.text((W//2,143),'WHAT IS THE HOUSE DECIDING?',font=font(23,True),fill=ACCENT,anchor='ma'); rule(d,190,112,968,4); cards=[('GENERAL PRINCIPLES','The main debate is about the broad purpose and approach of the Bill, rather than line-by-line amendment.'),('SECOND READING','The ordinary motion asks whether the Bill should be read a second time. If agreed, the Bill can move forward.'),('NOT COMMITTEE STAGE','Detailed examination of sections and amendments normally comes later at Committee Stage.'),('NO RECORDED DIVISION','A decision can stand without a member-by-member division. If there is no recorded division, EirePolitic does not invent a party tally.'),('EXACT PROPOSITION','If a division occurs, always read the exact question. A vote may be on Second Reading itself or on a procedural or amending proposition connected with it.')]; y=240; heights=[165,175,165,185,205]
    for (hd,bd),hh in zip(cards,heights): panel(d,(82,y,998,y+hh)); d.text((106,y+24),hd,font=font(20,True),fill=ACCENT,anchor='la'); bf=fit(d,bd,820,hh-72,22,17,5); draw_block(d,bd,(106,y+64,926,y+hh-15),bf); y+=hh+18
    footer(d,'Second Stage · Understanding the stage'); im.save(out/'06-second-stage-explainer.png')

def main():
    rows,prod=load_rows(); posts=[rows[i:i+4] for i in range(0,108,4)]; manifest={'production_key':prod,'bill_count':108,'post_count':27,'posts':[]}
    for n,bills in enumerate(posts,1):
        out=OUT/f'post{n:02d}'; out.mkdir(parents=True,exist_ok=True); cover(out,n,bills)
        for i,r in enumerate(bills,1): bill_slide(out,i,r)
        process_glossary(out); stage_glossary(out); files=['00-cover.png']+[f'{i:02d}-bill.png' for i in range(1,5)]+['05-process-glossary.png','06-second-stage-explainer.png']
        for f in files:
            if Image.open(out/f).size!=(1080,1350): raise RuntimeError(f'bad size {out/f}')
        contact_sheet([(str(i+1),out/f) for i,f in enumerate(files)],out/'contact-sheet.png',columns=4); manifest['posts'].append({'post':n,'bills':[r['title'] for r in bills],'files':files})
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({'posts':27,'bills':108,'production_key':prod},indent=2))
if __name__=='__main__': main()
