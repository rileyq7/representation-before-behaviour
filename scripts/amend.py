#!/usr/bin/env python3
"""Analyse saved checkpoints without interfering with GPU collection."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--watch",action="store_true")
    p.add_argument("--background",action="store_true")
    args=p.parse_args()
    if args.background:
        with open(ROOT/"results/amendment-01.log","ab",buffering=0) as log:
            proc=subprocess.Popen([sys.executable,"-u",str(Path(__file__).resolve()),"--watch"],
                                  cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        time.sleep(1)
        if proc.poll() is not None: raise RuntimeError("Worker exited; inspect results/amendment-01.log")
        print(json.dumps({"pid":proc.pid,"log":"results/amendment-01.log"})); return
    lock=open(ROOT/"results/amendment-01.lock","w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from mech.amendment import analyse_checkpoint,generate_report
    sha=hashlib.sha256((ROOT/"AMENDMENT-01.md").read_bytes()).hexdigest()
    status={"pid":os.getpid(),"amendment_sha256":sha,"started":time.time()}
    def save(**updates):
        status.update(updates,updated=time.time())
        path=ROOT/"results/amendment-01-status.json"
        tmp=path.with_suffix(".tmp"); tmp.write_text(json.dumps(status,indent=2)); tmp.replace(path)
    try:
        while True:
            files=sorted((ROOT/"results/checkpoints").glob("step*/summary.json"))
            changed=False
            for path in files:
                if (path.parent/"amendment-01.json").exists(): continue
                save(state="running",checkpoint=path.parent.name)
                start=time.monotonic()
                result=analyse_checkpoint(path.parent,sha)
                print(json.dumps({"event":"amendment_complete","step":result["step"],
                                  "selectivity":result["selectivity"],"seconds":time.monotonic()-start}),flush=True)
                changed=True
            if changed:
                generate_report(ROOT)
            completed=len(list((ROOT/"results/checkpoints").glob("step*/amendment-01.json")))
            save(state="completed" if completed==154 else "waiting_for_checkpoints",completed=completed)
            if not args.watch or completed==154: return
            time.sleep(30)
    except BaseException as e:
        save(state="failed",error=str(e),traceback=traceback.format_exc())
        raise


if __name__=="__main__": main()
