#!/usr/bin/env python3
from __future__ import annotations
import io, json, os
import boto3, pandas as pd

BUCKET=os.getenv('S3_BUCKET','eirepolitic-data')
s3=boto3.client('s3')
KEYS={
'member_profile_2025':'processed/members/member_profile_metrics_2025.csv',
'gold_member_activity_yearly':'processed/oireachtas_unified/latest/csv/gold_member_activity_yearly.csv',
'gold_constituency_activity_yearly':'processed/oireachtas_unified/latest/csv/gold_constituency_activity_yearly.csv',
'current_members':'processed/oireachtas_unified/latest/csv/gold_current_members.csv',
'speeches':'processed/oireachtas_unified/latest/csv/silver_speeches.csv',
'divisions':'processed/oireachtas_unified/latest/csv/silver_divisions.csv',
'member_votes':'processed/oireachtas_unified/latest/csv/silver_member_votes.csv',
'bills':'processed/oireachtas_unified/latest/csv/silver_bills.csv',
'bill_stages':'processed/oireachtas_unified/latest/csv/silver_bill_stages.csv',
'bill_sponsors':'processed/oireachtas_unified/latest/csv/silver_bill_sponsors.csv',
'classified':'processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv',
}

def read(key):
    try:
        obj=s3.get_object(Bucket=BUCKET,Key=key)
        return pd.read_csv(io.BytesIO(obj['Body'].read()),dtype=str,keep_default_na=False)
    except Exception as e:
        print('MISSING',key,type(e).__name__,str(e)[:180]); return pd.DataFrame()

def top(df,col,n=10,extra=None):
    if df.empty or col not in df.columns:return []
    x=df.copy(); x[col]=pd.to_numeric(x[col],errors='coerce'); x=x.dropna(subset=[col]).sort_values(col,ascending=False).head(n)
    cols=[c for c in (extra or [])+[col] if c in x.columns]
    return x[cols].to_dict('records')

frames={k:read(v) for k,v in KEYS.items()}
for name,df in frames.items(): print('TABLE',name,'ROWS',len(df),'COLS',json.dumps(list(df.columns)))

mp=frames['member_profile_2025']
if not mp.empty:
    print('MEMBER_TOP_SPEECH',json.dumps(top(mp,'speech_count_2025',15,['full_name','party','constituency','top_issue_2025','vote_participation_pct_2025'])))
    if 'top_issue_2025' in mp: print('MEMBER_TOP_ISSUES',json.dumps(mp['top_issue_2025'].replace('',pd.NA).dropna().value_counts().head(20).to_dict()))

sp=frames['speeches']
if not sp.empty and 'debate_date' in sp:
    d=pd.to_datetime(sp['debate_date'],errors='coerce'); print('SPEECHES_BY_YEAR',json.dumps(d.dt.year.value_counts().sort_index().dropna().astype(int).to_dict()))

cl=frames['classified']
if not cl.empty:
    issue=next((c for c in ['PoliticalIssues','political_issues','issue','Issue','issue_label','category','label'] if c in cl),None)
    datecol=next((c for c in ['Debate Date','date','speech_date','debate_date'] if c in cl),None)
    w=cl.copy()
    if datecol:
        d=pd.to_datetime(w[datecol],errors='coerce'); w=w[d.dt.year==2025]
    if issue:
        s=w[issue].fillna('').astype(str).str.strip(); s=s[(s!='')&(s.str.upper()!='NONE')]
        print('ISSUES_2025',json.dumps(s.value_counts().head(25).to_dict())); print('ISSUES_2025_POLICY_TOTAL',int(len(s)))

# Divisions and margins from member-level votes.
mv=frames['member_votes']; dv=frames['divisions']
if not mv.empty:
    w=mv.copy(); w['_vote']=w.get('vote_label','').fillna('').astype(str).str.strip().str.lower()
    def vk(x):
        if any(t in x for t in ['staon','abstain']): return 'abstain'
        if x in ['tá','ta','yes','aye','for']: return 'yes'
        if x in ['níl','nil','no','noe','against']: return 'no'
        return 'other'
    w['_kind']=w['_vote'].map(vk)
    c=w.groupby(['division_id','_kind']).size().unstack(fill_value=0).reset_index()
    for k in ['yes','no','abstain','other']:
        if k not in c:c[k]=0
    c['margin']=(c['yes']-c['no']).abs(); c['cast_yes_no']=c['yes']+c['no']; c['total_records']=c[['yes','no','abstain','other']].sum(axis=1)
    if not dv.empty: c=c.merge(dv[[x for x in ['division_id','division_date','subject','outcome'] if x in dv]],on='division_id',how='left')
    print('DIVISION_KIND_TOTALS',json.dumps({k:int(c[k].sum()) for k in ['yes','no','abstain','other']}))
    print('DIVISION_MARGIN_STATS',json.dumps(c['margin'].describe().round(2).to_dict()))
    print('DIVISIONS_MARGIN_LE_5',int((c['margin']<=5).sum())); print('DIVISIONS_MARGIN_LE_10',int((c['margin']<=10).sum())); print('DIVISIONS_UNANIMOUS_YN',int(((c['no']==0)|(c['yes']==0)).sum()))
    cols=[x for x in ['division_id','division_date','subject','outcome','yes','no','abstain','margin'] if x in c]
    print('CLOSEST_DIVISIONS',json.dumps(c.sort_values(['margin','division_date']).head(12)[cols].to_dict('records')))

b=frames['bills']; st=frames['bill_stages']; bs=frames['bill_sponsors']
if not b.empty:
    print('BILLS_BY_YEAR',json.dumps(b['bill_year'].replace('',pd.NA).dropna().value_counts().sort_index().to_dict()))
    for col in ['status','bill_type','origin_house_name']:
        if col in b: print('BILLS_'+col.upper(),json.dumps(b[col].replace('',pd.NA).dropna().value_counts().to_dict()))
if not st.empty:
    unique=st.groupby('stage_name')['bill_id'].nunique().sort_values(ascending=False)
    print('BILLS_REACHING_STAGE',json.dumps(unique.to_dict()))
    per=st.groupby('bill_id').size(); print('STAGE_EVENTS_PER_BILL_STATS',json.dumps(per.describe().round(2).to_dict()))
if not bs.empty:
    per=bs.groupby('bill_id').size(); print('SPONSORS_PER_BILL_STATS',json.dumps(per.describe().round(2).to_dict())); print('BILLS_MULTI_SPONSOR',int((per>1).sum()))
    print('TOP_SPONSOR_NAMES',json.dumps(bs['sponsor_name'].replace('',pd.NA).dropna().value_counts().head(20).to_dict()))
    if 'sponsor_role_name' in bs: print('SPONSOR_ROLES',json.dumps(bs['sponsor_role_name'].replace('',pd.NA).dropna().value_counts().head(20).to_dict()))
