#!/usr/bin/env python3
from __future__ import annotations
import io, json, os, re, sys, time
from pathlib import Path
import boto3, pandas as pd, requests
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from extract.oireachtas.batch import resolve_production_key
from political_metrics.bill_content_snapshot import build_bill_content_snapshot

BUCKET=os.getenv('S3_BUCKET','eirepolitic-data')
OUT=Path('artifacts/second-stage-enrichment'); OUT.mkdir(parents=True,exist_ok=True)
KEYS={'bills':'processed/oireachtas_unified/latest/csv/silver_bills.csv','stages':'processed/oireachtas_unified/latest/csv/silver_bill_stages.csv','sponsors':'processed/oireachtas_unified/latest/csv/silver_bill_sponsors.csv','bridge':'processed/oireachtas_unified/latest/metrics/event/bill_debate_sections/csv/bill_debate_sections.csv','speeches':'processed/oireachtas_unified/latest/csv/silver_speeches.csv','divisions':'processed/oireachtas_unified/latest/csv/silver_divisions.csv','member_votes':'processed/oireachtas_unified/latest/csv/silver_member_votes.csv','docs':'processed/oireachtas_unified/latest/csv/silver_bill_related_docs.csv','versions':'processed/oireachtas_unified/latest/csv/silver_bill_versions.csv'}
s3=boto3.client('s3',region_name=os.getenv('AWS_REGION','ca-central-1'))

MANUAL={
'Adult Safeguarding Strategy Bill 2026':{
 'what':'Would require the Minister for Health to prepare and adopt a national adult safeguarding strategy within six months, followed by a new three-year strategy every three years.',
 'practical':'The strategies would set safeguarding objectives and measures for adults at risk, including duties that can apply to public authorities, implementation measures and recurring public accountability.',
 'context':'The Bill is intended to place adult safeguarding on a statutory planning cycle rather than relying only on policy. It defines an adult at risk and requires structured objectives, measures and reporting.',
 'issues':'Second Stage can examine who should be covered, which public bodies should have duties, oversight and reporting arrangements, implementation and resourcing.'},
'Protection of Voice and Image Bill 2025':{
 'what':'Would create specific offences for knowingly misusing a person’s name, photograph, voice or likeness in order to cause harm.',
 'practical':'The proposal targets conduct such as harmful deepfakes and other misuse of a person’s identity or likeness. It would also require the Minister for Justice to review the law after one year of operation.',
 'context':'The explanatory memorandum says the Bill responds to the growing use of deepfakes and misuse of personal data, particularly where an individual’s voice or image is used without permission.',
 'issues':'Second Stage can test the scope of the offences, how harm and consent are defined, enforcement, interaction with existing criminal law, and safeguards for legitimate uses.'},
'Housing Loans Regulations (Fresh Start - Buyout) (Amendment) Bill 2026':{
 'what':'Would extend local-authority housing-loan eligibility to a person remaining in the family home after divorce or legal separation who needs finance to buy out a former partner’s share.',
 'practical':'Someone who retains the family home but cannot obtain the necessary mortgage finance could apply to a local authority for a housing loan specifically to complete that buyout.',
 'context':'The Bill would amend the Housing Loans Regulations 2021 so this group is expressly included within the eligibility rules for local-authority housing loans.',
 'issues':'Second Stage can consider eligibility criteria, affordability and credit assessment, treatment of existing ownership interests, and how the change would operate within the local-authority loan scheme.'},
'Repeal of Exempted Development Regulations Bill 2026':{
 'what':'Would repeal two planning exemptions introduced in 2022 and 2023 for buildings used as temporary accommodation for people seeking international protection or receiving temporary protection.',
 'practical':'Existing accommodation relying on those exemptions could need to be regularised through the planning system. A transitional provision would allow the responsible State agency to seek retention permission.',
 'context':'The Bill specifically targets S.I. 605/2022 and S.I. 376/2023, which created and extended exempted-development treatment for certain temporary accommodation.',
 'issues':'Second Stage can examine planning oversight, transitional arrangements for existing accommodation, effects on State accommodation capacity, and local planning requirements.'},
'Harassment, Harmful Communications and Related Offences (Amendment) Bill 2026':{
 'what':'Would amend the Harmful Communications and Related Offences Act 2020 by prohibiting the creation of non-consensual intimate images and increasing penalties for existing offences.',
 'practical':'It would add a new prohibition on creating non-consensual images, raise penalties under sections 3 and 4 of the 2020 Act, and extend the time allowed to begin summary proceedings.',
 'context':'The Bill is aimed at strengthening the existing legal framework for harmful communications and image-based abuse, including conduct that may occur before an image is distributed.',
 'issues':'Second Stage can examine the definition of non-consensual creation, evidential requirements, proportionality of penalties, limitation periods and overlap with existing offences.'},
'Education (Leave for Injuries) Bill 2025':{
 'what':'Would set minimum standards for any injury-leave scheme maintained by the Minister for Education for teachers and special needs assistants who suffer certain injuries.',
 'practical':'The scheme would have to cover matters including medical expenses, certified paid leave, support services, possible early retirement for long-term incapacity and equal treatment of teachers and SNAs.',
 'context':'The Bill seeks to put baseline protections around injury leave where a teacher or SNA is hurt or becomes ill in circumstances connected with their work.',
 'issues':'Second Stage can examine which injuries qualify, medical certification, duration and cost of paid leave, employer and school duties, support services and equality between staff groups.'},
'Maternity Protection (Child Bereavement) (Amendment) Bill 2026':{
 'what':'Would allow a relevant employee to postpone all or part of maternity leave following the death of her child, subject to notification and medical-certification requirements.',
 'practical':'The postponed maternity leave could be resumed later as one continuous period, with the Bill allowing postponement for up to 52 weeks from the date it begins under specified conditions.',
 'context':'The proposal amends the Maternity Protection Act 1994 so a bereaved mother would not necessarily have to use the remainder of maternity leave immediately after the death of her child.',
 'issues':'Second Stage can examine eligibility, notice and medical-certificate rules, the maximum postponement period, employment protections and how resumed leave would operate.'},
'National Minimum Wage (Inclusion of Young Persons, Apprentices and Interns) Bill 2025':{
 'what':'Would extend the full national minimum hourly rate to employed young people, bring apprentices within the minimum-wage legislation and cover certain interns and work-experience workers.',
 'practical':'Age-based reduced rates would be removed for employed young people. Apprentices would be brought into the Acts, while some interns doing more than 30 hours in four weeks would be treated as employees for minimum-wage purposes.',
 'context':'The Bill seeks to widen who is protected by the National Minimum Wage Acts, while retaining exceptions for genuine charitable work and prescribed education or training placements.',
 'issues':'Second Stage can examine the treatment of apprentices and trainees, education-placement exceptions, employer costs, enforcement and the effect of removing age-based minimum-wage rates.'},
'Local Government (Support for Elected Members) Bill 2024':{
 'what':'Would require a scheme providing administrative support to every elected member of a local authority to help them carry out their duties as public representatives.',
 'practical':'The Minister would have to establish the support scheme by regulation. The Bill defines administrative support by reference to the secretarial facilities used for Oireachtas members.',
 'context':'The proposal would amend the Local Government Act 2001 so administrative support for councillors becomes a statutory requirement rather than a discretionary arrangement.',
 'issues':'Second Stage can consider the level and form of support, costs, staffing and procurement arrangements, consistency across councils and ministerial regulation of the scheme.'},
'Electricity (Supply) (Amendment) (No. 2) Bill 2025':{
 'what':'Would reduce legal restrictions on ESB works affecting water levels on the Shannon lakes, including Lough Derg, Lough Ree and Lough Allen.',
 'practical':'It would remove specified historic water-level limits, allow provision for controlling Lough Ree levels and expressly permit works such as dredging new channels and deepening existing channels.',
 'context':'The Bill would amend the Electricity (Supply) (Amendment) (No. 2) Act 1934 to give the ESB greater flexibility in managing lake levels and related works.',
 'issues':'Second Stage can examine flood management, environmental and ecological impacts, navigation and land impacts, the scope of ESB powers and interaction with modern environmental law.'},
'Emergency Inspection of Dublin Zoo Bill 2025':{
 'what':'Would require the Minister for Housing, Local Government and Heritage to report on appointing an emergency independent inspector for Dublin Zoo.',
 'practical':'The Minister would have to consider and report on an independent emergency inspection rather than leaving the issue solely within existing administrative arrangements.',
 'context':'The explanatory memorandum states that the Bill’s purpose is specifically to require a ministerial report on the appointment of an emergency independent inspector for Dublin Zoo.',
 'issues':'Second Stage can consider the trigger and scope of an emergency inspection, inspector independence and powers, reporting, animal-welfare oversight and interaction with existing regulators.'},
'Domestic Violence (Free Travel Scheme) Bill 2025':{
 'what':'Would require the Minister for Social Protection to establish a free-travel scheme for victims who are fleeing, or have recently fled, domestic violence.',
 'practical':'A qualifying pass would normally last three months, with a possible three-month extension, and would provide the same transport access as the existing Free Travel Scheme while not revealing why the pass was issued.',
 'context':'Assessments would be carried out by approved bodies after consultation with Cuan, and the Minister would have to report annually to the Oireachtas on use and effectiveness of the scheme.',
 'issues':'Second Stage can examine eligibility and verification, privacy and safety, duration of support, participating transport services, administration and cost.'},
'Protection of Retail Workers Bill 2025':{
 'what':'Would create specific public-order offences for assaulting, threatening or abusing a retail worker while that person is engaged in retail work.',
 'practical':'A person convicted summarily could face up to 12 months’ imprisonment and/or a fine, with the offence covering both individual incidents and a course of threatening or abusive conduct.',
 'context':'The Bill would amend the Criminal Justice (Public Order) Act 1994 and define retail workers broadly enough to include employees, owners, agency staff and people delivering goods from retail premises.',
 'issues':'Second Stage can consider the need for a retail-specific offence, definitions, penalties, proof that the accused knew the victim was working, and overlap with existing assault and public-order offences.'},
'Health (Scoliosis Treatment Services) Bill 2024':{
 'what':'Would require the HSE to establish and maintain a national inpatient and outpatient scoliosis treatment service for children and adults normally resident in the State.',
 'practical':'The HSE would have duties to provide adequate specialist staff, facilities and resources, and could arrange treatment outside the State where timely care cannot otherwise be provided.',
 'context':'The Bill also requires regular reporting on demand, waiting lists and resources, while the Minister for Health would issue service-time guidelines and lay reports and guidelines before the Oireachtas.',
 'issues':'Second Stage can examine enforceable treatment timelines, staffing and capacity, funding, use of overseas care, reporting obligations and the respective duties of the HSE and Minister.'},
'Planning And Development (Exempted Development - External Wall Insulation) Bill 2025':{
 'what':'Would make external wall insulation exempt from planning permission in most cases by amending the Planning and Development Act 2000.',
 'practical':'Households could generally install exterior insulation without applying for planning permission, except for protected structures, architectural conservation areas and developments that seriously conflict with proper planning and sustainable development.',
 'context':'The Bill is intended to remove a planning barrier to external wall insulation while retaining protections for heritage buildings, conservation areas and exceptional planning impacts.',
 'issues':'Second Stage can examine heritage safeguards, visual and neighbour impacts, ministerial regulation, consistency with building standards and whether the exemption is sufficiently precise.'},
'Health (Postponement of Certain Leave) Bill 2024':{
 'what':'Would allow maternity leave or adoptive leave to be postponed where the person taking the leave is diagnosed with, or undergoing treatment for, cancer.',
 'practical':'A worker could request to return to work and preserve the unused portion of qualifying leave for a later continuous period after being certified fit to return, subject to rules set by legislation and regulation.',
 'context':'The Bill amends both the Maternity Protection Act 1994 and Adoptive Leave Act 1995 so serious cancer treatment does not necessarily consume a person’s statutory family leave entitlement.',
 'issues':'Second Stage can examine eligibility, employer consent and notice, medical evidence, maximum postponement periods, interaction with sick leave and protections when resumed leave begins.'},
'Domestic Violence (Amendment) (No. 3) Bill 2024':{
 'what':'Would broaden who can be treated as a “relevant person” for the offence of coercive control under the Domestic Violence Act 2018.',
 'practical':'The offence could apply across a wider set of relationships, including current or former intimate partners, specified relatives and certain people connected through guardianship of a child.',
 'context':'The Bill responds to the existing relationship definition in section 39 of the 2018 Act and also provides a limited defence for conduct reasonably believed to be in the relevant person’s best interests, except for specified behaviour.',
 'issues':'Second Stage can examine how far the relationship definition should extend, the scope of the defence, evidential thresholds and protection against coercive control outside former partner relationships.'}
}

def read_csv(k):
    rk=resolve_production_key(s3,bucket=BUCKET,production_key=k); obj=s3.get_object(Bucket=BUCKET,Key=rk)
    return pd.read_csv(io.BytesIO(obj['Body'].read()),dtype=str,keep_default_na=False),rk

def clean_text(text:str)->str:
    text=text.replace('\u00ad','').replace('\xa0',' '); return re.sub(r'\s+',' ',text).strip()
def pdf_text(url:str)->str:
    r=requests.get(url,timeout=45,headers={'User-Agent':'EirePolitic research pipeline/1.0'}); r.raise_for_status(); reader=PdfReader(io.BytesIO(r.content)); chunks=[]
    for p in reader.pages[:12]:
        try: chunks.append(p.extract_text() or '')
        except Exception: pass
    return clean_text(' '.join(chunks))
def first_url(row)->str:
    for c in ('format_pdf_url','format_pdf_uri','format_xml_url','format_xml_uri'):
        v=str(row.get(c,'')).strip()
        if v:return v
    return ''
def strip_frontmatter(text:str)->str:
    # Remove repeated bilingual title/frontmatter before substantive memo headings where possible.
    for marker in ('Purpose and Background','Purpose of the Bill','Purpose of the bill','Purpose and effect','Introduction','Background'):
        m=re.search(re.escape(marker),text,re.I)
        if m and m.start()<2500: return text[m.start():]
    return text

def long_title(text:str)->str:
    # English long title normally appears after "Bill entitled An Act ..." and before "Be it enacted".
    m=re.search(r'Bill entitled\s+An Act\s+(.{20,1800}?)(?=\s+Be it enacted)',text,re.I|re.S)
    if not m: return ''
    s=clean_text(m.group(1)).rstrip(' .;')
    return 'Would '+re.sub(r'^(to\s+)', '', s, flags=re.I)+'.'
def memo_purpose(text:str)->str:
    t=strip_frontmatter(text)
    m=re.search(r'(?:Purpose(?: of the (?:Bill|bill))?|Introduction|Purpose and Background)\s+(.{40,1200}?)(?=\s+(?:Provisions|Section 1|Background|Policy Background|PART |Part |General Scheme|$))',t,re.I|re.S)
    return clean_text(m.group(1)) if m else ''
def sentences(text:str)->list[str]:
    parts=re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚ])',text); out=[]
    for s in parts:
        s=clean_text(s)
        if 40<=len(s)<=520 and not re.match(r'^(Section|Head)\s+\d+\b',s,re.I) and 'EXPLANATORY MEMORANDUM' not in s.upper(): out.append(s)
    return out
def shorten(s:str,limit:int=300)->str:
    s=clean_text(s)
    s=re.sub(r'^(The purpose of (this|the) Bill is to|This Bill seeks to|The Bill seeks to|The Bill aims to|The main purpose of (this|the) Bill is to)\s+','',s,flags=re.I)
    if s and not s.endswith(('.',':',';','?','!')): s+='.'
    if len(s)<=limit:return s
    cut=s[:limit].rsplit(' ',1)[0].rstrip(',;:'); return cut+'.'
def best_action_sentence(ss:list[str],exclude:set[str]|None=None)->str:
    exclude=exclude or set(); best=''; score=-10**9
    for i,s in enumerate(ss[:120]):
        if s in exclude: continue
        low=s.lower(); sc=40-i//4
        if any(k in low for k in ['would ','will ','requires ','provides for','provides that','establish','prohibit','entitle','allow','permit','amend','insert','remove','extend','reduce','increase']): sc+=18
        if any(k in low for k in ['section 1','section 2','short title','citation','introduced by','acts referred to']): sc-=25
        if sc>score: best=s; score=sc
    return best
def issues_from(text:str)->str:
    low=text.lower(); items=['scope and definitions','implementation']
    if any(k in low for k in ['minister','authority','commission','regulator','hse']): items.append('oversight')
    if any(k in low for k in ['offence','penalty','fine','enforcement','prohibit']): items.append('enforcement')
    if any(k in low for k in ['cost','fund','payment','levy','tax','charge','financial','resource']): items.append('costs and resourcing')
    if any(k in low for k in ['data','privacy','personal information']): items.append('privacy safeguards')
    if any(k in low for k in ['child','children','minor']): items.append('safeguards for children')
    if any(k in low for k in ['constitutional','constitution','referendum']): items.append('constitutional implications')
    items=list(dict.fromkeys(items))[:4]
    phrase=', '.join(items[:-1])+(', and '+items[-1] if len(items)>1 else items[0])
    return 'Second Stage can test the Bill’s overall approach, including '+phrase+'.'
def build_copy(title:str,text:str,sponsor:str,has_memo:bool)->dict:
    if title in MANUAL:
        return {**MANUAL[title],'confidence':'manual','extracted_chars':len(text)}
    lt=long_title(text)
    purpose=memo_purpose(text)
    body=strip_frontmatter(text); ss=sentences(body)
    what=shorten(lt or purpose or (ss[0] if ss else f'The Bill proposes changes in the area described by its title: {title}.'),300)
    action=best_action_sentence(ss,{purpose} if purpose else set())
    practical=shorten(action or lt or 'The detailed legal effect is set out in the official Bill text.',300)
    if purpose and clean_text(purpose).lower()!=clean_text(what).lower(): context=shorten(purpose,300)
    else: context=f'{sponsor} sponsors the Bill. The official Bill text sets out the proposed legal change; no wider political support is inferred from sponsorship.'
    issues=issues_from(text)
    # Reject obvious frontmatter/header contamination.
    noisy=lambda s: ('EXPLANATORY MEMORANDUM' in s.upper() or 'MAR A TIONSCNAÍODH' in s.upper() or s.lower().startswith('an bille '))
    conf='high' if len(text)>700 and lt and not noisy(what) and not noisy(practical) else 'medium'
    return {'what':what,'practical':practical,'context':context,'issues':issues,'confidence':conf,'extracted_chars':len(text)}
def main():
    f={};resolved={}
    for n,k in KEYS.items(): f[n],resolved[n]=read_csv(k)
    snap=build_bill_content_snapshot(bills=f['bills'],stages=f['stages'],sponsors=f['sponsors'],bill_debate_sections=f['bridge'],speeches=f['speeches'],divisions=f['divisions'],member_votes=f['member_votes'],batch_size=6)
    x=snap[snap['series_bucket'].eq('second_stage')].copy(); x['_last']=pd.to_datetime(x['last_event_date'],errors='coerce')
    def keep(r):
        h=str(r.get('current_stage_house_name','')).lower(); dt=r['_last']
        return pd.notna(dt) and ((h in {'34th dáil','34th dail'} and dt>=pd.Timestamp('2024-12-18')) or (h=='27th seanad' and dt>=pd.Timestamp('2025-02-12')))
    x=x[x.apply(keep,axis=1)].copy().sort_values(['_last','bill_year','bill_no'],ascending=[False,False,False]); docs=f['docs']; vers=f['versions']; rows=[]
    for i,(_,r) in enumerate(x.iterrows(),1):
        bid=r['bill_id']; title=r['title']; sponsor=str(r.get('primary_sponsor_name','')).strip() or str(r.get('primary_sponsor_role_name','')).strip() or 'The recorded sponsor'; dd=docs[docs['bill_id'].eq(bid)]; memo=dd[dd['related_doc_label'].str.contains('Explanatory',case=False,na=False)]; url=''; kind=''; has_memo=False
        if len(memo): url=first_url(memo.iloc[0]); kind='Explanatory Memorandum'; has_memo=True
        if not url:
            vv=vers[(vers['bill_id'].eq(bid)) & (vers['version_label'].str.contains('Initiated',case=False,na=False))]
            if len(vv): url=first_url(vv.iloc[0]); kind='As Initiated'
        text=''; err=''
        if url:
            try:text=pdf_text(url)
            except Exception as e:err=repr(e)
        copy=build_copy(title,text,sponsor,has_memo); rows.append({'bill_id':bid,'title':title,'source_kind':kind,'source_url':url,'source_error':err,**copy}); print(f'[{i:03d}/108] {title} :: {kind} :: {copy["confidence"]} :: {len(text)} chars'); time.sleep(0.03)
    out={'count':len(rows),'manual':sum(r['confidence']=='manual' for r in rows),'high':sum(r['confidence']=='high' for r in rows),'medium':sum(r['confidence']=='medium' for r in rows),'low':sum(r['confidence']=='low' for r in rows),'rows':rows,'resolved':resolved}; (OUT/'copy.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/'review_needed.json').write_text(json.dumps([r for r in rows if r['confidence'] in {'medium','low'}],ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({k:out[k] for k in ['count','manual','high','medium','low']},indent=2))
if __name__=='__main__':main()
