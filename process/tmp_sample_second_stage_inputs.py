#!/usr/bin/env python3
from pathlib import Path
import json
src=Path('artifacts/second-stage-content-inputs/inputs.json')
d=json.loads(src.read_text(encoding='utf-8'))
rows=d['rows']
out={'columns':d.get('columns',[]),'sample':rows[:12]}
Path('artifacts/second-stage-content-inputs/sample.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
