"""Read NVIDIA device occupancy during this run; exits when report completes."""
import json
from pathlib import Path
import subprocess
import sys
import time

out=Path(sys.argv[1]); started=time.time()
with (out/'GPU_SAMPLES.jsonl').open('a',buffering=1) as stream:
    while time.time()-started<5400:
        output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True)
        rows=[]
        for line in output.strip().splitlines():
            index,used,total,util=[int(v.strip()) for v in line.split(',')]
            rows.append(dict(index=index,used_mib=used,total_mib=total,utilization_percent=util))
        stream.write(json.dumps(dict(unix_time=time.time(),gpus=rows))+'\n')
        if (out/'REPORT.json').exists(): break
        time.sleep(5)
