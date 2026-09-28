#!/usr/bin/env python3
"""Local resumable experiment. Run --help for entrypoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
from mech.data import build, digest


def read_rows(name):
    return [json.loads(line) for line in (ROOT / "data" / name).read_text().splitlines()]


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2))
    temp.replace(path)


def protocol_sha():
    return hashlib.sha256((ROOT / "PROTOCOL.md").read_bytes()).hexdigest()


def code_manifest():
    files = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py")) + [ROOT / "requirements-lock.txt"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def checkpoint(args):
    import numpy as np
    from mech.inference import load, extract, save_extract
    from mech.metrics import apple_metrics, occupation_values, interval
    from mech.probes import fit_all
    out = ROOT / f"results/checkpoints/step{args.step:06d}"
    out.mkdir(parents=True, exist_ok=True)
    model_path = ROOT / ".cache/checkpoint"
    provenance = json.loads((model_path / "provenance.json").read_text())
    if provenance["step"] != args.step:
        raise ValueError("Cached checkpoint step does not match requested step")
    start = time.monotonic()
    model, tokenizer, device, loading = load(model_path, args.device)
    print(json.dumps({"event": "model_loaded", "step": args.step, "device": device}), flush=True)
    if args.apple_full:
        rows = read_rows("apple-full.jsonl")
        acts, outputs = extract(model, tokenizer, rows, device, args.batch_size, False)
        save_extract(out / "apple-full", rows, acts, outputs)
        metrics = apple_metrics(rows, outputs)
        # Test data are reported separately from discovery data.
        test_idx = np.array([i for i, r in enumerate(rows) if r["split"] == "test"])
        test_outputs = {k: (v[test_idx] if k in {"logits", "logprobs", "ranks", "top_ids"} else v) for k, v in outputs.items()}
        metrics["test_only"] = apple_metrics([rows[i] for i in test_idx], test_outputs)
        atomic_json(out / "apple-full/summary.json", dict(step=args.step, provenance=provenance,
                                                         protocol_sha256=protocol_sha(), metrics=metrics,
                                                         seconds=time.monotonic()-start))
        return
    rows = read_rows("matched.jsonl")
    if args.benchmark:
        rows = rows[:args.benchmark]
    acts, outputs = extract(model, tokenizer, rows, device, args.batch_size)
    save_extract(out / ("benchmark" if args.benchmark else "matched"), rows, acts, outputs)
    if args.benchmark:
        atomic_json(ROOT / "results/benchmark.json", dict(step=args.step, device=device, n=len(rows),
                    inference_seconds=float(outputs["seconds"]), total_seconds=time.monotonic()-start,
                    provenance=provenance, loading=loading))
        return
    apple_rows = read_rows("apple-pilot.jsonl")
    _, apple_outputs = extract(model, tokenizer, apple_rows, device, args.batch_size, False)
    save_extract(out / "apple-pilot", apple_rows, {}, apple_outputs)
    del model
    import gc
    gc.collect()
    import torch
    if device == "mps":
        torch.mps.empty_cache()
    probes, predictions = fit_all(acts, rows)
    np.savez_compressed(out / "probes.npz", **predictions)
    occ = [r for r in rows if r["kind"] == "occupation"]
    occ_idx = [i for i, r in enumerate(rows) if r["kind"] == "occupation"]
    labels = np.array([r["label"] for r in occ])
    logodds = outputs["logits"][occ_idx, 1] - outputs["logits"][occ_idx, 0]
    signed = (labels * 2 - 1) * logodds
    names, occ_labels, beh = occupation_values(occ, signed)
    margins = predictions["layer_12_transfer"]
    correct = np.where(margins == 0, .5, (margins > 0) == labels).astype(float)
    _, _, rep = occupation_values(occ, correct)
    _, _, beh_accuracy = occupation_values(occ, np.where(signed == 0, .5, signed > 0))
    report = dict(step=args.step, provenance=provenance, protocol_sha256=protocol_sha(),
                  data_sha256=digest(rows), code_sha256=code_manifest(), device=device, dtype="float16" if device != "cpu" else "float32",
                  layers=[0,6,12,18,24], occupations=names, occupation_labels=occ_labels.tolist(),
                  per_occupation=dict(behaviour=beh.tolist(), transfer_accuracy=rep.tolist(),
                                      behaviour_accuracy=beh_accuracy.tolist()),
                  transfer=interval(rep, occ_labels), behaviour=interval(beh, occ_labels),
                  raw_female_logodds=float(logodds.mean()),
                  female_minus_male_occupation_logodds=float(logodds[labels == 1].mean()-logodds[labels == 0].mean()),
                  probes=probes, apple_pilot=apple_metrics(apple_rows, apple_outputs),
                  timings=dict(total_seconds=time.monotonic()-start,
                               matched_inference_seconds=float(outputs["seconds"]),
                               apple_inference_seconds=float(apple_outputs["seconds"])), loading=loading)
    atomic_json(out / "summary.json", report)
    print(json.dumps({"event": "checkpoint_complete", "step": args.step, "transfer": report["transfer"],
                      "behaviour": report["behaviour"], "seconds": report["timings"]["total_seconds"]}), flush=True)


def wait_for_disk(minimum=8 * 1024**3, attempts=None, delay=30, on_wait=None):
    # Preserve the safety threshold; wait for recovery instead of killing the
    # unattended sweep. A STOP file remains effective while waiting.
    attempt = 0
    while True:
        if (ROOT / "STOP").exists():
            raise InterruptedError("STOP file exists while waiting for disk space")
        free = shutil.disk_usage(ROOT).free
        if free >= minimum:
            return
        if attempts is not None and attempt == attempts - 1:
            raise RuntimeError("Less than 8 GiB free disk after waiting: stopped before checkpoint download")
        if on_wait is not None:
            on_wait(free)
        print(f"Waiting for reclaimed disk space: {free / 1024**3:.1f} GiB free", flush=True)
        time.sleep(delay)
        attempt += 1


def retry_download(step, before_attempt=lambda: None, on_network_wait=lambda delay: None):
    transient_failures = 0
    hard_failures = 0
    command = [sys.executable, str(ROOT / "scripts/download_checkpoint.py"), str(step)]
    while True:
        if (ROOT / "STOP").exists():
            raise InterruptedError("STOP file exists during download recovery")
        before_attempt()
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode == 0:
            return
        if result.returncode == 75:
            transient_failures += 1
            delay = min(30 * 2**min(transient_failures - 1, 4), 300)
            on_network_wait(delay)
            print(f"Network unavailable; retrying automatically in {delay} seconds", flush=True)
        else:
            hard_failures += 1
            if hard_failures >= 3:
                raise subprocess.CalledProcessError(result.returncode, command)
            delay = 30
            print(f"Download failed; retry {hard_failures}/2 in 30 seconds", flush=True)
        for _ in range(delay // 30):
            if (ROOT / "STOP").exists():
                raise InterruptedError("STOP file exists during download recovery")
            time.sleep(30)


def ensure_checkpoint(step):
    path = ROOT / ".cache/checkpoint"
    marker = path / "provenance.json"
    if marker.exists() and json.loads(marker.read_text())["step"] == step:
        return
    # Only dispose of this project's explicitly reserved weight cache.
    assert path.resolve() == ROOT / ".cache/checkpoint"
    requested = path / "requested-step.json"
    resume = requested.exists() and json.loads(requested.read_text())["step"] == step
    if path.exists() and not resume:
        shutil.rmtree(path)
    def disk_status(phase, free):
        status_path = ROOT / "results/status.json"
        if status_path.exists():
            state = json.loads(status_path.read_text())
            state.update(phase=phase, free_disk_gib=free / 1024**3, updated=time.time())
            atomic_json(status_path, state)
    def before_attempt():
        # Hub creates a new per-process partial file; abandoned copies are not
        # reused on subprocess restart. The previous downloader has exited and
        # the sweep lock excludes other writers to this disposable cache.
        for partial in (path / ".cache/huggingface/download").glob("*.incomplete"):
            partial.unlink()
        wait_for_disk(on_wait=lambda free: disk_status("waiting_for_disk", free))
        disk_status("download", shutil.disk_usage(ROOT).free)
    retry_download(step, before_attempt=before_attempt,
                   on_network_wait=lambda delay: disk_status("waiting_for_network", shutil.disk_usage(ROOT).free))


def sweep(args):
    import fcntl
    from mech.analysis import ALL_STEPS, COARSE_STEPS, report
    lock = open(ROOT / "results/sweep.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    build(ROOT)
    steps = (COARSE_STEPS + [s for s in ALL_STEPS if s not in COARSE_STEPS]) if args.steps == "all" else (
        COARSE_STEPS if args.steps == "coarse" else [int(x) for x in args.steps.split(",")])
    status = {"state": "running", "pid": os.getpid(), "started": time.time(), "planned_steps": steps,
              "protocol_sha256": protocol_sha(), "apple_full_bracket": args.full_bracket}
    atomic_json(ROOT / "results/status.json", status)
    try:
        for step in steps:
            summary = ROOT / f"results/checkpoints/step{step:06d}/summary.json"
            if summary.exists():
                previous = json.loads(summary.read_text())
                if previous["protocol_sha256"] != protocol_sha() or previous["data_sha256"] != digest(read_rows("matched.jsonl")):
                    raise RuntimeError("Existing result belongs to a different protocol/data; use a separate results directory")
                continue
            if (ROOT / "STOP").exists():
                status.update(state="paused", reason="STOP file exists")
                break
            status.update(current_step=step, phase="download", updated=time.time())
            atomic_json(ROOT / "results/status.json", status)
            ensure_checkpoint(step)
            status.update(phase="inference_and_probes", updated=time.time())
            atomic_json(ROOT / "results/status.json", status)
            subprocess.run([sys.executable, str(ROOT / "scripts/run.py"), "checkpoint", "--step", str(step),
                            "--batch-size", str(args.batch_size), "--device", args.device], cwd=ROOT, check=True)
            progress = report(ROOT)
            status.update(completed=progress["completed"], updated=time.time())
            atomic_json(ROOT / "results/status.json", status)
        else:
            progress = report(ROOT)
            if args.full_bracket and progress["completed"] == 154:
                bracket = select_bracket()
                atomic_json(ROOT / "reports/apple-bracket.json", bracket)
                for step in bracket["steps"]:
                    if (ROOT / "STOP").exists():
                        status.update(state="paused", reason="STOP file exists before full reconstruction")
                        atomic_json(ROOT / "results/status.json", status)
                        return
                    if (ROOT / f"results/checkpoints/step{step:06d}/apple-full/summary.json").exists():
                        continue
                    status.update(current_step=step, phase="apple_full_reconstruction", updated=time.time())
                    atomic_json(ROOT / "results/status.json", status)
                    ensure_checkpoint(step)
                    subprocess.run([sys.executable, str(ROOT / "scripts/run.py"), "checkpoint", "--step", str(step),
                                    "--batch-size", str(args.batch_size), "--device", args.device, "--apple-full"], cwd=ROOT, check=True)
                report(ROOT)
            status.update(state="completed", finished=time.time())
        atomic_json(ROOT / "results/status.json", status)
    except InterruptedError as e:
        status.update(state="paused", reason=str(e), updated=time.time())
        atomic_json(ROOT / "results/status.json", status)
    except BaseException as e:
        status.update(state="failed", error=str(e), traceback=traceback.format_exc(), updated=time.time())
        atomic_json(ROOT / "results/status.json", status)
        raise


def select_bracket():
    import numpy as np
    files = sorted((ROOT / "results/checkpoints").glob("step*/summary.json"))
    summaries = sorted([json.loads(p.read_text()) for p in files], key=lambda d: d["step"])
    summaries = [d for d in summaries if d["step"] >= 1000]
    delta = np.diff([d["apple_pilot"]["female_minus_male_jsdp"] for d in summaries])
    i = int(np.argmax(np.abs(delta)))
    idx = sorted({max(0, i-1), i, i+1, min(len(summaries)-1, i+2)})
    return {"steps": [summaries[j]["step"] for j in idx], "selected_change": float(delta[i]),
            "method": "Largest absolute adjacent JSD-P-gap change after step1000; endpoints plus one neighbour on each side.",
            "warning": "Exploratory selection is not evidence of a discontinuity. Confirm on test-only full reconstruction."}


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("report")
    c = sub.add_parser("checkpoint")
    c.add_argument("--step", required=True, type=int)
    c.add_argument("--batch-size", type=int, default=4)
    c.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    c.add_argument("--benchmark", type=int)
    c.add_argument("--apple-full", action="store_true")
    s = sub.add_parser("sweep")
    s.add_argument("--steps", default="all", help="all, coarse, or comma-separated step numbers")
    s.add_argument("--batch-size", type=int, default=4)
    s.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    s.add_argument("--full-bracket", action="store_true")
    args = p.parse_args()
    if args.command == "prepare":
        print(json.dumps(build(ROOT), indent=2))
        from mech.probes import lexical_baseline
        atomic_json(ROOT / "results/lexical-baseline.json", lexical_baseline(read_rows("matched.jsonl")))
    elif args.command == "report":
        from mech.analysis import report
        print(json.dumps(report(ROOT), indent=2))
    elif args.command == "checkpoint":
        checkpoint(args)
    else:
        sweep(args)


if __name__ == "__main__":
    main()
