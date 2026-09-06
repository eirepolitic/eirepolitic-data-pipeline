#!/usr/bin/env python3
from __future__ import annotations
import io, json, os
from pathlib import Path
import boto3, pandas as pd
from extract.oireachtas.io_s3 import get_bytes

BUCKET=os.getenv('S3_BUCKET','eirepolitic-data'); s3=boto3.client('s3')
def read_csv(key):
    return pd.read_csv(io.BytesIO(get_bytes(s3,bucket=BUCKET,key=key)),dtype=str,keep_default_na=False)

sp=read_csv('processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv')
members=read_csv('processed/oireachtas_unified/latest/csv/silver_members.csv')
mships=read_csv('processed/oireachtas_unified/latest/csv/silver_member_memberships.csv')
offices=read_csv('processed/oireachtas_unified/latest/csv/silver_member_offices.csv')

sp['_date']=pd.to_datetime(sp['Debate Date'],errors='coerce')
sp=sp[sp['_date']>=pd.Timestamp('2024-12-18')].copy()
sp['_words']=sp['Speech Text'].fillna('').astype(str).str.findall(r"\b\w+[\w’'\-]*\b").str.len()
sp=sp[sp['member_code'].astype(str).str.strip().ne('')].copy()

m=mships.copy(); m['_start']=pd.to_datetime(m['membership_start'],errors='coerce'); m['_end']=pd.to_datetime(m['membership_end'],errors='coerce')
m=m[m['house_no'].astype(str).str.strip().eq('34')].copy()
w=sp.merge(m[['member_code','_start','_end']],on='member_code',how='inner')
w=w[(w['_start'].isna()|(w['_date']>=w['_start']))&(w['_end'].isna()|(w['_date']<=w['_end']+pd.Timedelta(days=1)-pd.Timedelta(microseconds=1)))].copy()
w=w.merge(members[['member_code','full_name']],on='member_code',how='left')

# Exact longest interventions.
longest=w.sort_values(['_words','Debate Date','Speech Order'],ascending=[False,True,True]).head(10).copy()
longest_rows=[]
for _,r in longest.iterrows():
    longest_rows.append({
        'rank':len(longest_rows)+1,
        'member_code':r['member_code'],'full_name':r['full_name'],'speech_id':r['speech_id'],
        'debate_date':r['Debate Date'],'speech_order':r['Speech Order'],'word_count':int(r['_words']),
        'classification_status':r.get('classification_status',''),
        'text_preview':str(r['Speech Text'])[:220].replace('\n',' '),
    })

# Office history for leaders in key rankings.
agg=w.groupby(['member_code','full_name']).agg(intervention_count=('_words','size'),total_words=('_words','sum'),avg_words=('_words','mean'),median_words=('_words','median'),longest_words=('_words','max'),speaking_days=('_date',lambda x:x.dt.normalize().nunique())).reset_index()
leaders=set(agg.nlargest(10,'intervention_count')['member_code'])|set(agg.nlargest(10,'total_words')['member_code'])|set(agg.nlargest(10,'avg_words')['member_code'])
o=offices[offices['member_code'].isin(leaders)].copy()
o['_start']=pd.to_datetime(o['office_start'],errors='coerce'); o['_end']=pd.to_datetime(o['office_end'],errors='coerce')
o=o[(o['_start'].isna()|(o['_start']<=w['_date'].max()))&(o['_end'].isna()|(o['_end']>=w['_date'].min()))]
office_rows=o[['member_code','office_name','office_start','office_end','is_current']].sort_values(['member_code','office_start']).to_dict('records')

# Key contrast: intervention leader vs total-word leader.
top_int=agg.sort_values(['intervention_count','total_words'],ascending=False).head(5).copy()
top_words=agg.sort_values(['total_words','intervention_count'],ascending=False).head(5).copy()
for d in (top_int,top_words):
    for c in ['avg_words','median_words']: d[c]=d[c].round(2)

result={
  'coverage':{'start':str(w['_date'].min().date()),'end':str(w['_date'].max().date()),'interventions':len(w),'members':int(w['member_code'].nunique())},
  'longest_interventions':longest_rows,
  'top_interventions':top_int.to_dict('records'),
  'top_total_words':top_words.to_dict('records'),
  'leader_office_history':office_rows,
}
Path('diagnostics').mkdir(exist_ok=True)
out=Path(f"diagnostics/speech_carousel_evidence_{os.getenv('GITHUB_RUN_ID','local')}.json")
out.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')
print(out)
