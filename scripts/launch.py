"""Start the resumable Mac sweep without depending on the terminal/chat."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
(root / "results").mkdir(exist_ok=True)
with open(root / "results/launch.lock", "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Check the real process lock; a stale status file cannot block resumption.
    with open(root / "results/sweep.lock", "a") as sweep_lock:
        try:
            fcntl.flock(sweep_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A sweep is already running. See scripts/status.py.")
        fcntl.flock(sweep_lock, fcntl.LOCK_UN)
    if (root / "STOP").exists():
        raise SystemExit("STOP file exists; remove it to resume.")
    command = ["/usr/bin/caffeinate", "-i", sys.executable, "-u", str(root / "scripts/run.py"),
               "sweep", "--steps", "all", "--device", "mps", "--full-bracket"]
    with open(root / "results/sweep.log", "ab", buffering=0) as log:
        proc = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(launcher_pid=proc.pid, started=time.time(), command=command,
                  log=str(root / "results/sweep.log"))
    (root / "results/launch.json").write_text(json.dumps(record, indent=2))
    time.sleep(1)
    if proc.poll() is not None:
        raise SystemExit(f"Runner exited immediately ({proc.returncode}); inspect results/sweep.log")
    print(json.dumps(record, indent=2))
