#!/usr/bin/env python3
from pathlib import Path
import re

src=Path('process/tmp_publish_bill_tracker_full_post_contacts.py')
text=src.read_text(encoding='utf-8')

old="""    panel(d,60,232,460,270,*a,body_size=23); panel(d,560,232,460,270,*b,body_size=23)\n    panel(d,60,526,460,286,*c,body_size=22); panel(d,560,526,460,286,*e,body_size=22)\n"""
new="""    # Use one shared body size across all four explainer panels. Find the\n    # largest size that fits the most constrained panel, then apply it to all.\n    panel_specs=[(60,232,460,270,a),(560,232,460,270,b),(60,526,460,286,c),(560,526,460,286,e)]\n    shared_size=30\n    while shared_size>=18:\n        fits=True\n        for _x,_y,_w,_h,_item in panel_specs:\n            _title,_body=_item\n            _f=font(shared_size)\n            _lines=wrap(d,_body,_f,_w-48)\n            _yy=_y+56\n            for _line in _lines:\n                _,_lh=measure(d,_line,_f); _yy += _lh+4\n            if len(_lines)>8 or _yy>_y+_h-18:\n                fits=False; break\n        if fits: break\n        shared_size-=1\n    for _x,_y,_w,_h,_item in panel_specs:\n        panel(d,_x,_y,_w,_h,*_item,body_size=shared_size)\n"""
if old not in text:
    raise RuntimeError('make_explainer panel block not found')
text=text.replace(old,new,1)

pattern=r"def make_no_division\(path\):\n.*?\ndef make_glossary_terms\(path\):"
replacement="""def make_no_division(path):\n    im=slide(); d=ImageDraw.Draw(im)\n    centered(d,70,'Pharmacy Contraception · How This Passed',font(40,True),TEXT); d.rectangle([110,140,970,145],fill=ACCENT)\n    centered(d,178,'PASSED WITHOUT A RECORDED DIVISION',font(26,True),ACCENT)\n\n    # Three-step parliamentary procedure explainer\n    steps=[\n      ('1 · QUESTION PUT','The Chair puts the question to the House. Members respond Tá or Níl aloud.'),\n      ('2 · RESULT DECLARED','The Chair judges the response and declares which side has carried the question.'),\n      ('3 · DIVISION IF REQUIRED','If the result is challenged and a formal division is taken, individual members’ votes are recorded. That is what produces the party tally used on our other slides.'),\n    ]\n    y=255\n    for title,body in steps:\n        d.rounded_rectangle([90,y,990,y+205],radius=22,fill=PANEL,outline=BORDER,width=2)\n        d.text((120,y+24),title,font=font(23,True),fill=ACCENT)\n        draw_wrapped(d,120,y+70,body,font(23),TEXT,840,gap=7,max_lines=5)\n        y+=230\n\n    d.rounded_rectangle([90,955,990,1185],radius=22,fill=PANEL2,outline=ACCENT,width=3)\n    centered(d,985,'WHAT HAPPENED HERE',font(23,True),ACCENT)\n    draw_centered_wrapped(d,1032,'This Bill passed its Dáil stage without a recorded division. The decision stood, but no member-by-member Tá / Níl tally was produced, so there is no party breakdown to show.',font(22),TEXT,820,gap=7,max_lines=6)\n    footer(d,'EirePolitic · Parliamentary procedure · No recorded division'); im.save(path)\n\ndef make_glossary_terms(path):"""
text2,n=re.subn(pattern,replacement,text,flags=re.S)
if n!=1:
    raise RuntimeError(f'make_no_division replacement count={n}')

Path('process/tmp_publish_bill_tracker_full_post_contacts_patched.py').write_text(text2,encoding='utf-8')
print('patched renderer written')
