#!/usr/bin/env python3
from __future__ import annotations
import io, json, os
import boto3, pandas as pd
BUCKET=os.getenv('S3_BUCKET','eirepolitic-data'); s3=boto3.client('s3')
KEYS={'member_profile_2025':'processed/members/member_profile_metrics_2025.csv','speeches':'processed/oireachtas_unified/latest/csv/silver_speeches.csv','divisions':'processed/oireachtas_unified/latest/csv/silver_divisions.csv','member_votes':'processed/oireachtas_unified/latest/csv/silver_member_votes.csv','bills':'processed/oireachtas_unified/latest/csv/silver_bills.csv','bill_stages':'processed/oireachtas_unified/latest/csv/silver_bill_stages.csv','bill_sponsors':'processed/oireachtas_unified/latest/csv/silver_bill_sponsors.csv','classified':'processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv'}
def read(key):
    try:
        o=s3.get_object(Bucket=BUCKET,Key=key); return pd.read_csv(io.BytesIO(o['Body'].read()),dtype=str,keep_default_na=False)
    except Exception as e: print('MISSING',key,type(e).__name__,str(e)[:180]); return pd.DataFrame()
f={k:read(v) for k,v in KEYS.items()}
for n,d in f.items(): print('TABLE',n,'ROWS',len(d),'COLS',json.dumps(list(d.columns)))
sp=f['speeches']
if not sp.empty:
    dates=pd.to_datetime(sp['debate_date'],errors='coerce'); wc=pd.to_numeric(sp['word_count'],errors='coerce').dropna()
    print('SPEECHES_BY_YEAR',json.dumps(dates.dt.year.value_counts().sort_index().dropna().astype(int).to_dict()))
    print('DEBATE_DAY_COUNT',int(dates.dt.date.nunique()))
    print('SPEECH_WORD_STATS',json.dumps(wc.describe(percentiles=[.25,.5,.75,.9,.95]).round(2).to_dict()))
    print('SPEECH_WORD_TOTAL',int(wc.sum())); print('SPEECHES_UNDER_50_WORDS',int((wc<50).sum())); print('SPEECHES_OVER_500_WORDS',int((wc>500).sum()))
cl=f['classified']
if not cl.empty:
    d=pd.to_datetime(cl['Debate Date'],errors='coerce'); w=cl[d.dt.year==2025].copy(); s=w['PoliticalIssues'].fillna('').astype(str).str.strip(); s=s[(s!='')&(s.str.upper()!='NONE')]
    print('ISSUES_2025',json.dumps(s.value_counts().head(25).to_dict())); print('ISSUES_2025_POLICY_TOTAL',int(len(s)))
mv=f['member_votes']; dv=f['divisions']
if not mv.empty:
    w=mv.copy(); raw=w['vote_label'].fillna('').astype(str).str.strip().str.lower()
    def kind(x):
        if any(t in x for t in ['staon','abstain']): return 'abstain'
        if x in ['tá','ta','yes','aye','for']: return 'yes'
        if x in ['níl','nil','no','noe','against']: return 'no'
        return 'other'
    w['_kind']=raw.map(kind); c=w.groupby(['division_id','_kind']).size().unstack(fill_value=0).reset_index()
    for k in ['yes','no','abstain','other']:
        if k not in c:c[k]=0
    c['margin']=(c['yes']-c['no']).abs()
    print('DIVISION_KIND_TOTALS',json.dumps({k:int(c[k].sum()) for k in ['yes','no','abstain','other']})); print('DIVISION_MARGIN_STATS',json.dumps(c['margin'].describe().round(2).to_dict())); print('DIVISIONS_MARGIN_LE_5',int((c['margin']<=5).sum())); print('DIVISIONS_MARGIN_LE_10',int((c['margin']<=10).sum()))
b=f['bills']; st=f['bill_stages']; bs=f['bill_sponsors']
if not b.empty:
    print('BILLS_BY_YEAR',json.dumps(b['bill_year'].replace('',pd.NA).dropna().value_counts().sort_index().to_dict())); print('BILLS_STATUS',json.dumps(b['status'].replace('',pd.NA).dropna().value_counts().to_dict())); print('BILLS_ORIGIN_HOUSE_NAME',json.dumps(b['origin_house_name'].replace('',pd.NA).dropna().value_counts().to_dict()))
if not st.empty: print('BILLS_REACHING_STAGE',json.dumps(st.groupby('stage_name')['bill_id'].nunique().sort_values(ascending=False).to_dict()))
if not bs.empty:
    per=bs.groupby('bill_id').size(); print('SPONSORS_PER_BILL_STATS',json.dumps(per.describe().round(2).to_dict())); print('BILLS_MULTI_SPONSOR',int((per>1).sum()))
