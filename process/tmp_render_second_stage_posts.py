#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw
import process.tmp_render_second_stage_review as r

ROOT = Path('artifacts/bill-tracker-second-stage-posts')
ROOT.mkdir(parents=True, exist_ok=True)


def make_cover(out: Path, edition: str, bills: list[str], intro: str):
    r.OUT = out
    im = r.base_slide(); d = ImageDraw.Draw(im)
    d.text((r.W//2,126),'BILLS OF THE',font=r.font(28,True),fill=r.ACCENT,anchor='ma')
    d.text((r.W//2,174),'CURRENT SESSION',font=r.font(44,True),fill=r.TEXT,anchor='ma')
    r.rule(d,240,138,942,5)
    d.text((r.W//2,282),f'SECOND STAGE · {edition}',font=r.font(26,True),fill=r.ACCENT,anchor='ma')
    r.wrapped(d,intro,0,345,r.font(22),790,center=True)
    d.text((r.W//2,520),'IN THIS POST',font=r.font(23,True),fill=r.ACCENT,anchor='ma')
    y=570
    for i,b in enumerate(bills,1):
        r.panel(d,(76,y,1004,y+176)); d.ellipse((106,y+58,164,y+116),fill=r.ACCENT); d.text((135,y+87),str(i),font=r.font(23,True),fill=r.BG,anchor='mm')
        f=r.fit(d,b,760,24,18,3,True); lines=r.wrap_text_px(d,b,f,760); total=r.lines_height(d,lines,f,4); cur=y+(176-total)//2
        for line in lines:
            d.text((570,cur),line,font=f,fill=r.TEXT,anchor='ma'); bb=d.textbbox((570,cur),line,font=f,anchor='ma'); cur=bb[3]+4
        y+=198
    r.footer(d,f'EirePolitic · Second Stage · {edition.title()}')
    im.save(out/'00-cover.png')


def finish(out: Path):
    r.OUT=out
    r.process_glossary(); r.second_stage_glossary()
    files=['00-cover.png','01-bill.png','02-bill.png','03-bill.png','04-process-glossary.png','05-second-stage-explainer.png']
    for f in files:
        if Image.open(out/f).size != (1080,1350): raise RuntimeError(f'bad dimensions: {out}/{f}')
    r.contact_sheet([(str(i+1),out/f) for i,f in enumerate(files)],out/'contact-sheet.png',columns=3)


def post1():
    out=ROOT/'post1'; out.mkdir(parents=True,exist_ok=True); r.OUT=out
    make_cover(out,'POST 1',[
        'Anti-Shrinkflation Bill 2026',
        'Broadcasting (Amendment) Bill 2026',
        'Electoral (Postal Voting) (Carers) Bill 2026',
    ],'These Bills are in the Second Stage part of the process, where the House considers a Bill’s general principles before detailed Committee Stage scrutiny.')
    r.explainer('01-bill.png','Anti-Shrinkflation Bill 2026','Private Member’s Bill · Holly Cairns · Dáil Éireann',
      'Would require clearer retail labelling when a product’s quantity is reduced in a way that raises its unit price, so consumers can spot hidden price increases.',
      'Large retailers would have to flag qualifying quantity reductions for a set period. The proposal focuses on price transparency rather than banning smaller packs or setting prices.',
      'The sponsor argues shoppers should be told clearly when they are paying effectively more for less, particularly during a period of cost-of-living pressure.',
      'At Second Stage the House can test the proposal’s general approach: which retailers or products should be covered, how notice rules work, exemptions, enforcement and proportionality.',
      'WHERE IT IS NOW','The Bill is in the Second Stage part of the process, but the available certified record does not show a substantive Second Stage decision yet. It remains at this stage rather than having moved on for detailed Committee Stage scrutiny.')
    r.explainer('02-bill.png','Broadcasting (Amendment) Bill 2026','Government Bill · Dáil Éireann',
      'Would reform governance, transparency, funding and oversight arrangements for RTÉ and TG4, expand Coimisiún na Meán functions and implement parts of the European Media Freedom Act.',
      'The Bill would change how public service media governance, auditing, performance assessment and some public-service-content funding arrangements operate.',
      'Government presented the Bill as implementing recommendations from the Future of Media Commission and the independent RTÉ governance review, alongside EU media-law requirements.',
      'Second Stage debate raised issues including governance, long-term public-service-media funding, Irish-language provision, independent production, geo-blocking and implementation detail.',
      'SECOND STAGE OUTCOME','Second Stage was completed in the Dáil. The Bill was agreed to proceed onward for detailed scrutiny, where its individual provisions and possible amendments could be examined.')
    r.explainer('03-bill.png','Electoral (Postal Voting) (Carers) Bill 2026','Private Member’s Bill · Mark Wall · Dáil Éireann',
      'Would extend eligibility for the postal-voter register to certain people who provide care for others, by amending the Electoral Act 1992.',
      'Qualifying carers who cannot readily attend a polling station because of caring responsibilities could gain a postal-voting route if the Bill eventually becomes law.',
      'The sponsor presented the proposal as a way to reduce barriers to electoral participation faced by family carers whose responsibilities can make in-person voting difficult.',
      'The substantive Second Stage debate would be the point to test the proposal’s general principles, eligibility rules, safeguards and how a carers postal-voting category should operate.',
      'WHERE IT IS NOW','The Bill has been introduced and is positioned for Second Stage, but the available certified data does not show a substantive Second Stage debate or decision yet. The next meaningful step is consideration of its general principles at Second Stage.')
    finish(out)


def post2():
    out=ROOT/'post2'; out.mkdir(parents=True,exist_ok=True); r.OUT=out
    make_cover(out,'POST 2',[
        'Defence (Amendment) (No. 2) Bill 2026',
        'Adult Safeguarding Strategy Bill 2026',
        'Prevention of Energy Wastage Bill 2026',
    ],'Three more Bills in the Second Stage part of the process. At this stage the House considers the broad purpose and approach before detailed amendment work.')
    r.explainer('01-bill.png','Defence (Amendment) (No. 2) Bill 2026','Private Member’s Bill · Paul Murphy · Dáil Éireann',
      'Would prohibit U.S. military aircraft, and civilian aircraft carrying munitions of war, from landing in the State, subject to limited emergency search-and-rescue exceptions.',
      'Routine landings in Ireland by aircraft covered by the prohibition would no longer be permitted if the Bill became law, including the type of military transit associated with Shannon Airport.',
      'The sponsor presented the proposal as a way to end U.S. military use of Shannon and to align airport practice with his view of Irish neutrality.',
      'A Second Stage debate could test the Bill’s broad approach, definitions and exceptions, its relationship with neutrality and foreign policy, enforcement and effects on aviation or diplomatic arrangements.',
      'WHERE IT IS NOW','The Bill has been introduced in the Dáil and is positioned for Second Stage. The available certified record does not show a substantive Second Stage debate or decision yet, so there is no outcome or party vote to display.')
    r.explainer('02-bill.png','Adult Safeguarding Strategy Bill 2026','Private Member’s Bill · Tom Clonan · Seanad Éireann',
      'Would require the Minister for Health to prepare recurring statutory strategies for protecting adults at risk of harm, with implementation, review and reporting requirements.',
      'Adult safeguarding would be placed on a regular statutory planning cycle, with public accountability for what actions are taken, what remains outstanding and how the strategy is reviewed.',
      'The sponsors presented the Bill as a way to strengthen coordination and accountability for protecting adults at risk across health, care and other relevant settings.',
      'A Second Stage debate could examine who should be covered, which bodies and services should have duties, oversight and implementation arrangements, resourcing and interaction with existing safeguarding law and policy.',
      'WHERE IT IS NOW','The Bill has been introduced in the Seanad and is positioned for Second Stage. The available certified record contains its introduction but not a substantive Second Stage debate or decision, so no outcome chart is shown.')
    r.explainer('03-bill.png','Prevention of Energy Wastage Bill 2026','Private Member’s Bill · Jennifer Whitmore · Dáil Éireann',
      'Would create a statutory basis for using renewable electricity that would otherwise be curtailed or constrained, linking that unused energy to climate, just-transition and energy-poverty objectives.',
      'The proposal would enable a scheme for eligible electricity customers to benefit from otherwise-unused renewable power, with a focus on affordability and vulnerable households.',
      'The sponsor argues that renewable electricity should not be wasted while households face energy costs, and that surplus clean power should contribute to climate and energy-poverty goals.',
      'A Second Stage debate could examine electricity-market and grid rules, customer eligibility, cost allocation, EU and State-aid constraints, system operation and how the proposed scheme would work in practice.',
      'WHERE IT IS NOW','The Bill has been introduced in the Dáil and is positioned for Second Stage. The available certified record does not show a substantive Second Stage debate or decision yet, so the next meaningful step is debate on its general principles.')
    finish(out)


if __name__=='__main__':
    post1(); post2(); print(ROOT)
