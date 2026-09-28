# Representation before behaviour

**Question:** Does occupational gender bias become linearly accessible before it becomes behaviourally observable?

Local Pythia-1.4B checkpoint experiment. See [PROTOCOL.md](PROTOCOL.md) before interpreting results. The source Apple's ~80k finding concerns **6.9B**, so a jump in 1.4B is a hypothesis, not a supplied ground truth.

## Current results and progress

- **Complete:** [Final findings and limitations](reports/FINAL_REPORT.md) — 154/154 checkpoints and all four follow-up evaluations.
- [Current timing analysis: shared onsets and selectivity](reports/AMENDMENT-01.md), updated from saved checkpoints
- [Original protocol report](reports/REPORT.md), retained for audit with its original absolute thresholds
- [Raw curve figure](reports/curves.png)
- [Machine-readable progress](results/status.json)
- [Analysis and missing checkpoints](reports/analysis.json)
- [Amendment and interpretation commitments](AMENDMENT-01.md)
- [Inference verification](results/inference-verification.json)
- Full log: `results/sweep.log`

An incomplete sweep is explicitly labelled **partial** and does not produce an A/B timing conclusion. The runner does not need an active chat to continue, but the Mac must stay on and have network access. The launch command prevents idle sleep while the process runs; closing the lid or restarting can interrupt it. Completed checkpoints are reused on restart.

Use the shared 25/50/75% endpoint-normalised analysis for the revised timing question. The original report's unequal absolute-threshold analysis remains an audit/sensitivity result, not the revised primary onset comparison. The amendment was requested after the coarse curves had been inspected; it is explicitly labelled accordingly.

## Run locally

The project-local `.venv` contains the locked dependencies. All commands run from this folder.

```bash
.venv/bin/python scripts/run.py prepare
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run.py sweep --steps all --device mps --full-bracket
```

The final command runs all 154 checkpoints, then the full Apple-style reconstruction around the largest discovery-set JSD-P-gap change. `--steps coarse` runs 12 discovery checkpoints; `--steps 0,1000,80000` selects explicit steps. Default batch size is 4. Only one sweep may run at a time (process lock). Do not manually replace the cached model while a sweep is active.

To run independently of the terminal, use `.venv/bin/python scripts/launch.py`. It records launch information and writes `results/sweep.log`. To pause after the current checkpoint, create an empty `STOP` file in the project root (`touch STOP`). To resume, remove that file and rerun the launch command. An immediate stop loses only the currently unfinished checkpoint; use the recorded process IDs if necessary. `scripts/status.py` reports process liveness, progress, recent log lines and a rough ETA.

```bash
.venv/bin/python scripts/status.py
.venv/bin/python scripts/run.py report
```

The post-inspection shared-onset/selectivity analysis is a separate CPU worker: `.venv/bin/python scripts/amend.py --background`. It analyses existing saved activations and follows new completed checkpoints, without re-downloading weights. Progress is in `results/amendment-01-status.json`. Run without flags for a one-off update. The original protocol and reports are retained alongside the amendment.

No paid GPU services are used. Download traffic is approximately **450 GB** for the full sweep plus bracket reconstruction. Only one ~2.93 GB checkpoint is retained, and the runner waits before a new download if less than 8 GiB of disk space is free, then resumes automatically when space recovers. Checkpoint downloads retry temporary network/DNS errors automatically with backoff up to five minutes. A STOP file interrupts the wait. Permanent errors still stop after three attempts. Abandoned per-process partial files are cleared before a retry to bound disk use; completed checkpoints are always reused. Failures are saved in `results/status.json`; the process never reports a failed sweep as completed.

Download optimisation checked on 2026-09-21: an equal-size 128 MiB benchmark measured 5.73 MB/s with one HTTP connection and 5.54 MB/s with four parallel byte-range transfers (`results/download-benchmark.json`). Xet high-performance mode stalled and was not adopted. The runner therefore retains HTTP transfers and one checkpoint at a time. It now waits for reclaimed disk space while enforcing the unchanged 8 GiB floor, refreshing its status every 30 seconds; a STOP file pauses this wait, and verifies downloaded weight files against the model repository's SHA-256 digest. These are operational changes; prompts, probe settings and statistical criteria are unchanged.

## Experiment structure

- 40 original WinoBias occupations × 6 fixed neutral templates (240 rows).
- 10 explicit gender word pairs × 6 templates (120 rows) to train an occupation-label-free gender transfer probe.
- Five fixed layers, primary layer 12; train-fold-only scaling and fixed logistic regression.
- Occupation and template holdouts, gender-word generalisation, label permutations, random initialization, embedding and lexical controls.
- Matched behavioural log-odds and discrete preferences; Apple-style probability/rank/JSD-P reconstruction kept separate.
- Clustered uncertainty, simultaneous full-curve onset bands, sustained-crossing rules and censoring.

Saved arrays are real model outputs, not simulated curves. `tests/` only validates calculations and experimental bookkeeping. The inferred concept is linearly accessible gender–occupation alignment under these prompts; this is not evidence of causal use or proof that earlier representations do not exist.

## Files

`data/`: original downloaded data, derived prompts and manifests. `sources/`: paper figure, source descriptions and download hashes. `src/mech/`: data, inference, probes, statistics and reports. `scripts/`: download, run, verify, launch and status commands. `results/checkpoints/stepNNNNNN/`: per-item outputs, activations, probe predictions and checkpoint commit IDs. `requirements-lock.txt`: exact installed packages. Model inference is float16 on MPS; stored activations and probes use float32 or higher.

## Reproduce in a fresh environment

Use Python 3.13, create a virtual environment, and install `requirements-lock.txt`. Run `scripts/fetch_data.py` to download the same original data; it checks the recorded SHA-256 hashes. On CUDA hardware use `--device cuda`; do not combine results from different precision/device settings without a comparison. Public Hugging Face downloads need network access but no account token.
