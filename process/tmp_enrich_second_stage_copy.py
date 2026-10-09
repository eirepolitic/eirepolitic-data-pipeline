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

def read_csv(k):
    rk=resolve_production_key(s3,bucket=BUCKET,production_key=k); obj=s3.get_object(Bucket=BUCKET,Key=rk)
    return pd.read_csv(io.BytesIO(obj['Body'].read()),dtype=str,keep_default_na=False),rk

def clean_text(text:str)->str:
    text=text.replace('\u00ad','').replace('\xa0',' ')
    text=re.sub(r'\s+',' ',text)
    return text.strip()

def pdf_text(url:str)->str:
    r=requests.get(url,timeout=45,headers={'User-Agent':'EirePolitic research pipeline/1.0'})
    r.raise_for_status()
    if len(r.content)>25_000_000: raise RuntimeError('document too large')
    reader=PdfReader(io.BytesIO(r.content))
    chunks=[]
    for p in reader.pages[:12]:
        try: chunks.append(p.extract_text() or '')
        except Exception: pass
    return clean_text(' '.join(chunks))

def sentences(text:str)->list[str]:
    parts=re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚ])',text)
    out=[]
    for s in parts:
        s=clean_text(s)
        if 35<=len(s)<=520 and not re.match(r'^(Section|Head)\s+\d+\b',s,re.I): out.append(s)
    return out

def shorten(s:str,limit:int=280)->str:
    s=clean_text(s)
    s=re.sub(r'^(The purpose of (this|the) Bill is to|This Bill seeks to|The Bill seeks to|The Bill aims to|The main purpose of (this|the) Bill is to)\s+','',s,flags=re.I)
    if s: s=s[0].upper()+s[1:]
    if len(s)<=limit:return s
    cut=s[:limit].rsplit(' ',1)[0].rstrip(',;:')
    return cut+'.'

def best_sentence(sents:list[str],groups:list[tuple[list[str],int]],exclude:set[str]|None=None)->str:
    exclude=exclude or set(); best=''; score=-1
    for i,s in enumerate(sents[:140]):
        if s in exclude: continue
        low=s.lower(); sc=max(0,55-i//3)
        for kws,w in groups:
            for kw in kws:
                if kw in low: sc+=w
        if sc>score: best=s; score=sc
    return best

def generic_issues(text:str)->str:
    low=text.lower(); items=['scope and definitions','implementation']
    if any(k in low for k in ['minister','authority','commission','regulator']): items.append('oversight')
    if any(k in low for k in ['offence','penalty','fine','enforcement','prohibit']): items.append('enforcement')
    if any(k in low for k in ['cost','fund','payment','levy','tax','charge','financial']): items.append('costs and funding')
    if any(k in low for k in ['data','privacy','personal information']): items.append('privacy safeguards')
    if any(k in low for k in ['child','children','minor']): items.append('safeguards for children')
    if any(k in low for k in ['constitutional','constitution','referendum']): items.append('constitutional implications')
    items=list(dict.fromkeys(items))[:4]
    if len(items)==1: phrase=items[0]
    elif len(items)==2: phrase=items[0]+' and '+items[1]
    else: phrase=', '.join(items[:-1])+', and '+items[-1]
    return 'Second Stage can test the Bill’s overall approach, including '+phrase+'.'

def build_copy(title:str,text:str,sponsor:str,has_memo:bool)->dict:
    ss=sentences(text)
    p=best_sentence(ss,[(['purpose of this bill','purpose of the bill','bill seeks','bill aims','aim of this bill'],20),(['provide for','provides for','amend','establish','introduce','prohibit','require','enable','extend','create'],9)])
    pe=best_sentence(ss,[(['will provide','will require','will prohibit','will enable','will allow','will establish','would provide','would require','would prohibit','provides that'],15),(['entitle','eligible','offence','penalty','scheme','register','licence','levy','payment','right to','duty'],7)],{p} if p else set())
    why=best_sentence(ss,[(['aim','purpose','ensure','protect','address','improve','reduce','prevent','support','recognise','facilitate'],8),(['background','policy','need for','intended to'],11)],{x for x in [p,pe] if x})
    if not p and ss:p=ss[0]
    if not pe and len(ss)>1:pe=ss[1]
    if not why and len(ss)>2:why=ss[2]
    what=shorten(p or f'The Bill proposes changes in the area described by its title: {title}.')
    practical=shorten(pe or 'The detailed legal effect is set out in the initiated Bill text.',300)
    context=shorten(why or f'{sponsor} sponsors the Bill. The official initiated text sets out the proposed legal change.',300)
    issues=generic_issues(text)
    conf='high' if has_memo and len(text)>900 and p and pe else ('medium' if len(text)>500 else 'low')
    return {'what':what,'practical':practical,'context':context,'issues':issues,'confidence':conf,'extracted_chars':len(text)}

def first_url(row)->str:
    for c in ('format_pdf_url','format_pdf_uri','format_xml_url','format_xml_uri'):
        v=str(row.get(c,'')).strip()
        if v:return v
    return ''

def main():
    f={};resolved={}
    for n,k in KEYS.items(): f[n],resolved[n]=read_csv(k)
    snap=build_bill_content_snapshot(bills=f['bills'],stages=f['stages'],sponsors=f['sponsors'],bill_debate_sections=f['bridge'],speeches=f['speeches'],divisions=f['divisions'],member_votes=f['member_votes'],batch_size=6)
    x=snap[snap['series_bucket'].eq('second_stage')].copy(); x['_last']=pd.to_datetime(x['last_event_date'],errors='coerce')
    def keep(r):
        h=str(r.get('current_stage_house_name','')).lower(); dt=r['_last']
        return pd.notna(dt) and ((h in {'34th dáil','34th dail'} and dt>=pd.Timestamp('2024-12-18')) or (h=='27th seanad' and dt>=pd.Timestamp('2025-02-12')))
    x=x[x.apply(keep,axis=1)].copy().sort_values(['_last','bill_year','bill_no'],ascending=[False,False,False])
    docs=f['docs']; vers=f['versions']; rows=[]
    for i,(_,r) in enumerate(x.iterrows(),1):
        bid=r['bill_id']; title=r['title']; sponsor=str(r.get('primary_sponsor_name','')).strip() or str(r.get('primary_sponsor_role_name','')).strip() or 'The recorded sponsor'
        dd=docs[docs['bill_id'].eq(bid)]
        memo=dd[dd['related_doc_label'].str.contains('Explanatory.*Memorandum|Explanatory Memorandum',case=False,na=False,regex=True)]
        url=''; kind=''; has_memo=False
        if len(memo):
            url=first_url(memo.iloc[0]); kind='Explanatory Memorandum'; has_memo=True
        if not url:
            vv=vers[(vers['bill_id'].eq(bid)) & (vers['version_label'].str.contains('Initiated',case=False,na=False))]
            if len(vv): url=first_url(vv.iloc[0]); kind='As Initiated'
        text=''; err=''
        if url:
            try: text=pdf_text(url)
            except Exception as e: err=repr(e)
        copy=build_copy(title,text,sponsor,has_memo)
        rows.append({'bill_id':bid,'title':title,'source_kind':kind,'source_url':url,'source_error':err,**copy})
        print(f'[{i:03d}/108] {title} :: {kind} :: {copy["confidence"]} :: {len(text)} chars')
        time.sleep(0.04)
    out={'count':len(rows),'high':sum(r['confidence']=='high' for r in rows),'medium':sum(r['confidence']=='medium' for r in rows),'low':sum(r['confidence']=='low' for r in rows),'rows':rows,'resolved':resolved}
    (OUT/'copy.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'review_needed.json').write_text(json.dumps([r for r in rows if r['confidence']!='high'],ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:out[k] for k in ['count','high','medium','low']},indent=2))
if __name__=='__main__': main()
