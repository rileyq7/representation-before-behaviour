# Representation before behaviour?

Does occupational gender bias become linearly decodable inside a language model before it shows up in the model's outputs?

This project tracks two measures across the training history of Pythia models:

- **Association:** whether occupations line up with a gender direction in the residual stream at layer 12. The probe is trained only on explicit gender words such as mother/father, so it never sees occupation labels.
- **Expression:** whether the model prefers "she" over "he" after stereotypically female occupations, and the reverse after stereotypically male ones. This is the signed next-token logit difference.

If association consistently reached its learned level before expression did, that would suggest the representation forms first and the behaviour follows.

## Results

| Study | Status | Finding |
|---|---|---|
| v1: Pythia-1.4B, all 154 checkpoints | Complete | No robust ordering. Both measures reach 25% and 50% of their learned range in the same checkpoint brackets (steps 512-2,000). Bootstrap intervals for the lead include zero at every threshold. |
| Detectability analysis of v1 | Complete (post hoc) | The v1 null result is not informative. If association truly reached every level 16 times sooner, v1 would have detected it at most 61% of the time. |
| v2: 10 training seeds of Pythia-410M, plus 1.4B | Running | Preregistered replication with a continuous association measure, causal ablation of the gender direction, a 25-layer sweep, the logit lens and 64 additional occupations. |

### v1: Pythia-1.4B

- [Final report](reports/FINAL_REPORT.md): main findings, controls and limitations
- [Shared-onset analysis (Amendment 01)](reports/AMENDMENT-01.md), with [figure](reports/amendment-01.png) and [data](reports/amendment-01.json)
- [Original protocol report](reports/REPORT.md) and [raw curves](reports/curves.png), kept for audit
- [Apple-style follow-up](reports/apple-confirmation.json) (JSD-P gap at steps 2,000-5,000)

The direct occupation-label probe reaches 70.1% accuracy against 49.2% for matched shuffled-label controls, a selectivity of 20.9 percentage points (95% CI 12.8 to 28.1).

| Share of range reached | Association (steps) | Expression (steps) | Lead, 95% CI (steps) |
|---:|---|---|---|
| 25% | 512-1,000 | 512-1,000 | -1,000 to +1,872 |
| 50% | 1,000-2,000 | 1,000-2,000 | -2,000 to +1,000 |
| 75% | 5,000-6,000 | 1,000-2,000 | -7,000 to +2,000 |

### Could v1 have detected a lead?

[Detectability report](reports/DETECTABILITY.md), with [figure](reports/detectability.png) and [data](reports/detectability.json).

The simulation plants known leads into noise taken from v1's own residuals, then reruns the unchanged v1 pipeline. Findings:

- Power to detect a representation-first lead is low for every lead size tested.
- Power is higher for an expression-first lead, because the association curve is noisier.
- Pythia has no checkpoint between steps 512 and 1,000, which is where both curves rise. That limits resolution.
- The rule that compares checkpoint brackets reports a false ordering up to 28% of the time when there is no lead. The bootstrap interval had no false positives.

A [simulation of the v2 design](reports/detectability-v2-prospective.json) estimates at least 93% power for a 1.25x lead. This is an upper bound, because it assumes every seed has the same true timing.

### v2: preregistered multi-seed replication

- [Protocol v2](PROTOCOL-v2.md), committed under the tag [v2-prereg](https://github.com/rileyq7/representation-before-behaviour/tree/v2-prereg) before any v2 output was collected
- Primary question: across 10 independent Pythia-410M training runs ([PolyPythias](https://huggingface.co/EleutherAI/pythia-410m-seed1)), is the mean log-ratio of 50% crossing times for association and expression different from zero? The test is a t confidence interval over seeds.
- Secondary analyses: Pythia-1.4B, all 25 layers, the logit lens, ablation of the gender direction compared with 10 random directions, v1's accuracy measure, and the WinoBias occupations on their own.
- Results will be added to `reports/V2_REPORT.md` and linked here when the run finishes.

## Method

- **Prompts:** 40 WinoBias occupations (v2 adds 64 from the BLS occupation tables) in 6 neutral templates such as "The nurse said that". Ten explicit gender word pairs in the same templates train the gender probe.
- **Probes:** standardised logistic regression (C = 0.01), with occupations and templates held out. Controls are shuffled labels, random initialisation, the embedding layer and a character n-gram baseline.
- **Timing:** each curve is scaled to its own range from step 0 to the final checkpoint. The analysis records when it first reaches 25%, 50% and 75%. Uncertainty comes from a stratified bootstrap over occupations; in v2 the training seed is the unit of replication.
- **Scope:** linearly decodable does not mean causally used. The v2 ablation analysis tests causal use directly.

## Repository layout

| Path | Contents |
|---|---|
| `PROTOCOL.md`, `AMENDMENT-01.md`, `PROTOCOL-v2.md` | Analysis plans, in the order they were fixed |
| `reports/` | Reports, figures and machine-readable results |
| `src/mech/` | Data construction, inference, probes and statistics |
| `scripts/` | Local runner (`run.py`), Modal runner (`modal_v2.py`) and report builders |
| `data/` | Prompts, manifests, WinoBias source files and the BLS source table |
| `results/checkpoints/` | v1 per-checkpoint outputs, probe predictions and provenance (activations are not in git) |
| `results/v2/` | v2 per-checkpoint summaries (arrays are kept on a Modal volume) |
| `sources/` | Source metadata, model file hashes and references |
| `tests/` | Software tests on synthetic data, never used as results |

## Reproducing

Python 3.13. Exact package versions are in `requirements-lock.txt`.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-lock.txt
.venv/bin/python scripts/fetch_data.py     # downloads WinoBias and checks SHA-256 hashes
.venv/bin/python scripts/run.py prepare
.venv/bin/python -m pytest -q
```

v1, locally on Apple Silicon or CUDA (about 450 GB of downloads; several days on a laptop):

```bash
.venv/bin/python scripts/run.py sweep --steps all --device mps --full-bracket
.venv/bin/python scripts/amend.py
.venv/bin/python scripts/run.py report
```

`scripts/launch.py` runs the sweep detached, `touch STOP` pauses it and `scripts/status.py` shows progress. Finished checkpoints are reused on restart, and only one checkpoint (about 2.9 GB) is kept on disk at a time.

Detectability analysis (about 8 minutes on CPU):

```bash
.venv/bin/python scripts/detectability.py
```

v2 on Modal (about 480 checkpoints, roughly $5-10 of compute):

```bash
modal run scripts/modal_v2.py --smoke      # two checkpoints, to check timing and cost
modal run --detach scripts/modal_v2.py     # full run, resumable
modal run scripts/modal_v2.py --fetch      # copy summaries into results/v2/
.venv/bin/python scripts/report_v2.py
```

## Provenance and caveats

- Every checkpoint is resolved to a fixed Hugging Face commit, and its weight file is checked against the published SHA-256.
- v1 ran float16 inference on Apple MPS and v2 runs float16 on NVIDIA L4. At the final 1.4B checkpoint the two agree on expression with correlation 1.0000 (largest difference 0.005 logits), and transfer accuracy is identical for all 40 occupations. v1 and v2 data are never pooled.
- Amendment 01 was written after the coarse v1 curves had been inspected and is labelled that way. The detectability analysis is post hoc. v2 is the preregistered test.
- The transition at about 80,000 steps in [Patel et al., Fairness Dynamics During Training](https://arxiv.org/abs/2506.01709) was reported for Pythia-6.9B, not 1.4B. The Apple-style prompts here are a reconstruction, not the authors' code.
- WinoBias ([Zhao et al., 2018](https://github.com/uclanlp/corefBias)) and BLS CPS Table 11 are redistributed under their original terms. Full references are in `sources/REFERENCES.md`.
