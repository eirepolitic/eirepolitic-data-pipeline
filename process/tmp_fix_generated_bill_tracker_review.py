#!/usr/bin/env python3
from pathlib import Path
p=Path('process/tmp_publish_bill_tracker_full_post_contacts_patched.py')
s=p.read_text(encoding='utf-8')
s=s.replace("d.multiline_text((x+bw/2,y0+70),'\n'.join(lines),font=font(13,True),fill=txt,anchor='mm',align='center',spacing=2)", "d.multiline_text((x+bw/2,y0+70),chr(10).join(lines),font=font(13,True),fill=txt,anchor='mm',align='center',spacing=2)")
s=s.replace("block='\n'.join(lines)", "block=chr(10).join(lines)")
p.write_text(s,encoding='utf-8')
print('fixed generated renderer newline joins')
