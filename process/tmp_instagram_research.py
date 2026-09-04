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
        print('MISSING',key,type(e).__name__,str(e)[:180])
        return pd.DataFrame()

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
    if 'top_issue_2025' in mp:
        print('MEMBER_TOP_ISSUES',json.dumps(mp['top_issue_2025'].replace('',pd.NA).dropna().value_counts().head(20).to_dict()))
    if 'vote_participation_pct_2025' in mp:
        v=pd.to_numeric(mp['vote_participation_pct_2025'],errors='coerce').dropna(); print('MEMBER_VOTE_PCT_STATS',json.dumps(v.describe().round(2).to_dict()))

for nm in ['gold_member_activity_yearly','gold_constituency_activity_yearly']:
    df=frames[nm]
    if not df.empty and 'year' in df:
        yrs=sorted(df['year'].astype(str).unique()); print(nm.upper()+'_YEARS',json.dumps(yrs))
        y=yrs[-1]; cur=df[df['year'].astype(str)==y].copy()
        print(nm.upper()+'_LATEST_YEAR',y)
        print(nm.upper()+'_TOP_SPEECH',json.dumps(top(cur,'speech_count',15,[c for c in ['member_code','constituency_name','constituency','vote_participation_pct','division_count'] if c in cur.columns])))
        if 'speech_count' in cur: print(nm.upper()+'_SPEECH_SUM',int(pd.to_numeric(cur['speech_count'],errors='coerce').fillna(0).sum()))

sp=frames['speeches']
if not sp.empty:
    datecol=next((c for c in ['debate_date','date','speech_date'] if c in sp),None)
    if datecol:
        d=pd.to_datetime(sp[datecol],errors='coerce'); print('SPEECHES_BY_YEAR',json.dumps(d.dt.year.value_counts().sort_index().dropna().astype(int).to_dict()))
    for c in ['section_heading','debate_section_heading','section_title']:
        if c in sp:
            print('TOP_SECTION_HEADINGS',json.dumps(sp[c].replace('',pd.NA).dropna().value_counts().head(20).to_dict())); break

cl=frames['classified']
if not cl.empty:
    issue=next((c for c in ['PoliticalIssues','political_issues','issue','Issue','issue_label','category','label'] if c in cl),None)
    datecol=next((c for c in ['Debate Date','date','speech_date','debate_date'] if c in cl),None)
    w=cl.copy()
    if datecol:
        d=pd.to_datetime(w[datecol],errors='coerce'); w=w[d.dt.year==2025]
    if issue:
        s=w[issue].fillna('').astype(str).str.strip(); s=s[(s!='')&(s.str.upper()!='NONE')]
        print('ISSUES_2025',json.dumps(s.value_counts().head(25).to_dict()))
        print('ISSUES_2025_POLICY_TOTAL',int(len(s)))

for nm in ['divisions','bills','bill_stages','bill_sponsors']:
    df=frames[nm]
    if df.empty: continue
    for c in ['division_date','date','bill_date','stage_date','event_date']:
        if c in df:
            d=pd.to_datetime(df[c],errors='coerce'); print(nm.upper()+'_BY_YEAR',json.dumps(d.dt.year.value_counts().sort_index().dropna().astype(int).to_dict())); break
    for c in ['result','division_result','status','bill_status','stage','stage_name','sponsor_type']:
        if c in df:
            print(nm.upper()+'_'+c.upper(),json.dumps(df[c].replace('',pd.NA).dropna().value_counts().head(20).to_dict()))

# division margin diagnostics if counts are present
DV=frames['divisions']
if not DV.empty:
    yes=next((c for c in ['ta_count','yes_count','ayes','ayes_count'] if c in DV),None); no=next((c for c in ['nil_count','no_count','noes','noes_count'] if c in DV),None)
    if yes and no:
        x=DV.copy(); x['_y']=pd.to_numeric(x[yes],errors='coerce'); x['_n']=pd.to_numeric(x[no],errors='coerce'); x['_margin']=(x['_y']-x['_n']).abs(); x=x.dropna(subset=['_margin']).sort_values('_margin').head(20)
        cols=[c for c in ['division_id','division_date','title','subject','motion_text',yes,no,'_margin'] if c in x]
        print('CLOSEST_DIVISIONS',json.dumps(x[cols].to_dict('records')))
