#!/usr/bin/env python3
from __future__ import annotations
import io, json, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import boto3, pandas as pd
from extract.oireachtas.batch import resolve_production_key

BUCKET=os.getenv('S3_BUCKET','eirepolitic-data')
s3=boto3.client('s3')
KEYS={
'bills':'processed/oireachtas_unified/latest/csv/silver_bills.csv',
'bridge':'processed/oireachtas_unified/latest/metrics/event/bill_debate_sections/csv/bill_debate_sections.csv',
'divisions':'processed/oireachtas_unified/latest/csv/silver_divisions.csv',
'member_votes':'processed/oireachtas_unified/latest/csv/silver_member_votes.csv',
}

def read(logical):
    key=resolve_production_key(s3,bucket=BUCKET,production_key=logical)
    o=s3.get_object(Bucket=BUCKET,Key=key)
    return pd.read_csv(io.BytesIO(o['Body'].read()),dtype=str,keep_default_na=False)

f={k:read(v) for k,v in KEYS.items()}
for n,d in f.items(): print('TABLE',n,len(d),list(d.columns))

sample_titles=[
'Development (Strategic Gas Reserve) Bill 2026',
'Israeli Settlements in the Occupied Palestinian Territory (Prohibition of Importation of Goods) Bill 2026',
'Criminal Law, Civil Law and Defence (Miscellaneous Provisions) Bill 2026',
'Housing and Residential Tenancies (Miscellaneous Provisions) Bill 2026',
'Health (Provision of Contraception Prescribing Service in Retail Pharmacy Businesses) Bill 2026',
'Regulation of Artificial Intelligence Bill 2026',
]
b=f['bills']; br=f['bridge']; dv=f['divisions']; mv=f['member_votes']
sel=b[b['title'].isin(sample_titles) | b['short_title'].isin(sample_titles)].copy()
print('SAMPLE_BILLS',json.dumps(sel[[c for c in ['bill_id','title','short_title','status'] if c in sel]].to_dict('records'),ensure_ascii=False,indent=2))

joined=br.merge(sel[['bill_id','title','short_title']],on='bill_id',how='inner').merge(dv,on='debate_section_id',how='inner',suffixes=('','_division'))
joined=joined.drop_duplicates(['bill_id','division_id'])
for bid,g in joined.groupby('bill_id'):
    title=g.iloc[0]['short_title'] or g.iloc[0]['title']
    print('\nBILL',title,'DIVISIONS',len(g))
    for _,r in g.sort_values('division_date').iterrows():
        did=r['division_id']
        v=mv[mv['division_id']==did].copy()
        raw=v['vote_label'].fillna('').str.strip().str.casefold() if 'vote_label' in v else pd.Series([],dtype=str)
        def kind(x):
            if x in ['yes','ta','tá','aye','for']: return 'for'
            if x in ['no','nil','níl','noe','against']: return 'against'
            if 'abst' in x or 'staon' in x: return 'abstain'
            return 'other'
        if not v.empty: v['_kind']=raw.map(kind)
        counts=v['_kind'].value_counts().to_dict() if not v.empty else {}
        party_col=next((c for c in ['party_name','party','party_at_vote','party_name_at_vote'] if c in v.columns),None)
        party_rows=[]
        if party_col:
            p=v.groupby([party_col,'_kind']).size().unstack(fill_value=0)
            for c in ['for','against','abstain','other']:
                if c not in p.columns:p[c]=0
            p['total']=p[['for','against','abstain','other']].sum(axis=1)
            p=p.sort_values('total',ascending=False)
            party_rows=[{'party':idx,**{c:int(row[c]) for c in ['for','against','abstain','other','total']}} for idx,row in p.iterrows()]
        subj=str(r.get('subject','') or r.get('division_subject','') or '')
        outcome=str(r.get('outcome','') or '')
        low=subj.casefold()
        if 'amendment' in low: ptype='amendment'
        elif 'second stage' in low or 'read a second time' in low: ptype='second_stage'
        elif 'fifth stage' in low or 'do now pass' in low or 'passed' in low: ptype='final_passage_candidate'
        elif 'question put' in low: ptype='generic_question_put'
        else: ptype='other'
        print(json.dumps({
            'division_id':did,
            'date':r.get('division_date',''),
            'house':r.get('house_name',''),
            'subject':subj,
            'outcome':outcome,
            'heuristic_type':ptype,
            'counts':counts,
            'party_rows':party_rows,
        },ensure_ascii=False))
