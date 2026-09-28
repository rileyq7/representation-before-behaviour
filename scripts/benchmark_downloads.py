"""Compare one HTTP transfer with four parallel byte-range transfers."""
import concurrent.futures
import json
import time
import httpx
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
request = json.loads((ROOT / ".cache/checkpoint/requested-step.json").read_text())
url = f"https://huggingface.co/EleutherAI/pythia-1.4b/resolve/{request['sha']}/pytorch_model.bin"

def get_range(bounds):
    start, end = bounds
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        with client.stream("GET", url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}) as r:
            if r.status_code != 206:
                raise RuntimeError(f"Range request returned {r.status_code}, expected 206")
            if not r.headers.get("content-range", "").startswith(f"bytes {start}-{end}/"):
                raise RuntimeError("Incorrect content range")
            count = sum(len(b) for b in r.iter_bytes())
    if count != end-start+1:
        raise RuntimeError("Incorrect byte count")
    return count

records=[]
total_size=128*1024**2
for workers in [1,4]:
    size=total_size//workers
    start=time.monotonic()
    ranges=[(i*size,(i+1)*size-1) for i in range(workers)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        total=sum(pool.map(get_range,ranges))
    duration=time.monotonic()-start
    record=dict(workers=workers,bytes=total,seconds=duration,MB_per_second=total/duration/1e6)
    records.append(record)
    print(json.dumps(record),flush=True)
(ROOT / "results/download-benchmark.json").write_text(json.dumps(records,indent=2))
