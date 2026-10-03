#!/usr/bin/env python3
from pathlib import Path
p=Path('process/tmp_render_second_stage_review.py')
s=p.read_text(encoding='utf-8')
s=s.replace("top=ry+112; bw=456; bh=256; gap=24;", "top=ry+112; bw=456; bh=276; gap=24;")
s=s.replace("bf=shared_font(d,texts,402,175,24,18,7)", "bf=shared_font(d,texts,402,184,22,17,7)")
s=s.replace("; f=fit(d,lbl,bw-12,12,10,2,True); wrapped(d,lbl,0,y+56,f,bw-12,fill=BG if active else ACCENT,center=False); # overwrite centered below", "; f=fit(d,lbl,bw-12,12,10,2,True)")
p.write_text(s,encoding='utf-8')
print('patched Second Stage review renderer')
