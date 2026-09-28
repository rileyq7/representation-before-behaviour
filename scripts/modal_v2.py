"""Protocol v2 cloud runner (Modal).

    modal run --detach scripts/modal_v2.py                   # full run, resumable
    modal run scripts/modal_v2.py --smoke                    # two checkpoints, to check timing/cost
    modal run scripts/modal_v2.py --fetch                    # copy summary.json files into results/v2/ (arrays stay on the Volume)

GPU inference (download, verify, forward, ablation) writes arrays to a Volume; a CPU function fits
probes and writes summary.json. Both skip work already present, so reruns resume.
"""
import json
import subprocess
import sys
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"pythia-410m-seed0": "EleutherAI/pythia-410m",
          **{f"pythia-410m-seed{i}": f"EleutherAI/pythia-410m-seed{i}" for i in range(1, 10)},
          "pythia-1.4b": "EleutherAI/pythia-1.4b"}
STEPS = [0] + [2**i for i in range(10)] + list(range(1000, 20001, 1000)) + list(range(30000, 140001, 10000)) + [143000]
assert len(STEPS) == 44

image = (modal.Image.debian_slim(python_version="3.13")
         .uv_pip_install("torch==2.14.0", "transformers==5.17.0", "numpy==2.5.3", "scikit-learn==1.9.1",
                         "scipy==1.18.1", "threadpoolctl==3.7.0", "huggingface_hub==1.32.0", "safetensors==0.8.0",
                         "tokenizers==0.23.2", "accelerate==1.15.0")
         .env({"TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "2"})
         .add_local_dir(ROOT / "src/mech", "/root/src/mech")
         .add_local_file(ROOT / "data/matched-v2.jsonl", "/root/data/matched-v2.jsonl"))
volume = modal.Volume.from_name("mech-v2-results", create_if_missing=True)
app = modal.App("mech-v2", image=image)
RESULTS = Path("/results")


def _rows():
    return [json.loads(x) for x in Path("/root/data/matched-v2.jsonl").read_text().splitlines()]


def _folder(tag, step):
    return RESULTS / tag / f"step{step:06d}"


@app.function(gpu="L4", volumes={"/results": volume}, timeout=3600, max_containers=10,
              retries=modal.Retries(max_retries=3, initial_delay=30.0, backoff_coefficient=2.0))
def infer(tag, step, protocol_commit):
    import hashlib, shutil, time
    import numpy as np
    sys.path.insert(0, "/root/src")
    from huggingface_hub import HfApi, snapshot_download
    from mech import v2
    volume.reload()
    out = _folder(tag, step)
    if (out / "arrays.npz").exists():
        return {"tag": tag, "step": step, "status": "cached"}
    started = time.monotonic()
    repo = MODELS[tag]
    info = HfApi().model_info(repo, revision=f"step{step}", files_metadata=True)
    local = Path(f"/tmp/{tag}-{step}")
    snapshot_download(repo, revision=info.sha, local_dir=local,
                      allow_patterns=["config.json", "*token*.json", "special_tokens_map.json", "pytorch_model.bin"])
    expected = next(f for f in info.siblings if f.rfilename == "pytorch_model.bin").lfs.sha256
    with (local / "pytorch_model.bin").open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError(f"{tag} step {step}: checksum mismatch")
    download_seconds = time.monotonic() - started
    rows = _rows()
    arrays = v2.run_inference(local, rows, device="cuda")
    shutil.rmtree(local)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "arrays.npz", **arrays)
    import torch
    meta = dict(tag=tag, repo=repo, step=step, revision_sha=info.sha, weight_sha256=actual,
                protocol_commit=protocol_commit, data_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                code_sha256=v2.code_sha(), device=torch.cuda.get_device_name(), dtype="float16",
                torch=torch.__version__, download_seconds=download_seconds,
                inference_seconds=float(arrays["seconds"]))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    volume.commit()
    return {"tag": tag, "step": step, "status": "done", "seconds": time.monotonic() - started}


@app.function(cpu=2.0, memory=8192, volumes={"/results": volume}, timeout=3600, max_containers=40,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def summarise(tag, step):
    import time
    import numpy as np
    sys.path.insert(0, "/root/src")
    from mech import v2
    volume.reload()
    out = _folder(tag, step)
    if (out / "summary.json").exists():
        return {"tag": tag, "step": step, "status": "cached"}
    started = time.monotonic()
    meta = json.loads((out / "meta.json").read_text())
    with np.load(out / "arrays.npz") as f:
        arrays = {k: f[k] for k in f.files}
    summary = v2.summarise(_rows(), arrays, meta)
    summary["probe_seconds"] = time.monotonic() - started
    tmp = out / "summary.tmp"
    tmp.write_text(json.dumps(summary, allow_nan=False))
    tmp.rename(out / "summary.json")
    volume.commit()
    return {"tag": tag, "step": step, "status": "done", "seconds": summary["probe_seconds"]}


@app.function(timeout=24 * 3600, volumes={"/results": volume})
def orchestrate(jobs, protocol_commit):
    log = []
    pending = []
    for result in infer.starmap([(t, s, protocol_commit) for t, s in jobs], return_exceptions=True, order_outputs=False):
        if isinstance(result, Exception):
            log.append({"stage": "infer", "error": repr(result)}); print(log[-1], flush=True)
            continue
        print(json.dumps(result), flush=True)
        pending.append(summarise.spawn(result["tag"], result["step"]))
    for call in pending:
        try:
            print(json.dumps(call.get()), flush=True)
        except Exception as error:  # noqa: BLE001 - record and continue
            log.append({"stage": "summarise", "error": repr(error)}); print(log[-1], flush=True)
    volume.reload()
    missing = [(t, s) for t, s in jobs if not (_folder(t, s) / "summary.json").exists()]
    status = {"jobs": len(jobs), "missing": missing, "errors": log}
    (RESULTS / "run-status.json").write_text(json.dumps(status, indent=2))
    volume.commit()
    return status


@app.function(volumes={"/results": volume}, timeout=1800)
def collect():
    volume.reload()
    bundle = {}
    for path in sorted(RESULTS.glob("*/step*/summary.json")):
        bundle[f"{path.parent.parent.name}/{path.parent.name}"] = json.loads(path.read_text())
    status = RESULTS / "run-status.json"
    return bundle, (json.loads(status.read_text()) if status.exists() else None)


@app.local_entrypoint()
def main(smoke: bool = False, fetch: bool = False):
    if fetch:
        bundle, status = collect.remote()
        for key, summary in bundle.items():
            path = ROOT / "results/v2" / key / "summary.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary))
        if status:
            (ROOT / "results/v2/run-status.json").write_text(json.dumps(status, indent=2))
        print(json.dumps({"summaries": len(bundle), "run_status": status is not None}))
        return
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"Commit before launching so the protocol/code commit is exact:\n{dirty}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if smoke:
        jobs = [("pythia-410m-seed1", 143000), ("pythia-1.4b", 143000)]
    else:
        # Device check first, then the primary seeds, then 1.4B.
        first = [("pythia-1.4b", s) for s in (0, 1000, 143000)]
        jobs = first + [(t, s) for t in MODELS if t != "pythia-1.4b" for s in STEPS]
        jobs += [("pythia-1.4b", s) for s in STEPS if ("pythia-1.4b", s) not in first]
    print(json.dumps(orchestrate.remote(jobs, commit), indent=2))
