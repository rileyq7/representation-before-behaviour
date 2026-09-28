"""Read-only local progress and process liveness check."""
import json
import os
from pathlib import Path
import time
root = Path(__file__).resolve().parents[1]
path = root / "results/status.json"
if not path.exists():
    raise SystemExit("Sweep has not started; no status file yet.")
status = json.loads(path.read_text())
try:
    os.kill(status["pid"],0)
    status["process_alive"] = True
except ProcessLookupError:
    status["process_alive"] = False
except PermissionError:
    status["process_alive"] = "unknown (sandbox permission)"
finished = sorted((root / "results/checkpoints").glob("step*/summary.json"))
status["completed_checkpoints"] = len(finished)
status["remaining_checkpoints"] = 154-len(finished)
if status.get("started") and len(finished) >= 3:
    elapsed = time.time()-status["started"]
    status["rough_remaining_hours_excludes_full_reconstruction"] = round(elapsed/max(1,len(finished)-1)*(154-len(finished))/3600,1)
print(json.dumps(status,indent=2))
log = root / "results/sweep.log"
if log.exists():
    with log.open("rb") as f:
        f.seek(max(0,log.stat().st_size-4000))
        tail=f.read().decode(errors="replace").splitlines()[-8:]
    print("\nRecent log:")
    print("\n".join(tail))
