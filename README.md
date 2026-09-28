# Representation before behaviour?

**Does occupational gender bias become linearly decodable inside a language model before it shows up in the model's outputs?**

This project tracks both across the full training history of Pythia models. For each checkpoint it measures:

- **Association:** does the residual stream at layer 12 align occupations with an independently learned gender direction? This is a linear probe trained only on explicit words like *mother/father*, with no occupation labels.
- **Expression:** does the model actually prefer *she* over *he* after stereotypically female occupations, and vice versa? This is the signed logit difference at the next token.

If association consistently reached its learned level earlier than expression, that would suggest the representation forms first and behaviour follows.

## Results at a glance

| Study | Status | Finding |
|---|---|---|
| **v1: Pythia-1.4B, all 154 checkpoints** | ✅ Complete | No robust ordering. Both measures reach 25% and 50% of their learned range in the same checkpoint brackets (steps 512–2,000). Paired bootstrap intervals for the lead include zero at every threshold. |
| **Detectability analysis of v1** | ✅ Complete (post hoc) | v1's null result is **not informative**. Even if association truly reached every level 16× sooner, v1 would have detected it at most 61% of the time. The bracket-ordering rule reports a spurious ordering up to 28% of the time when there is no lead. |
| **v2: 10 training seeds (Pythia-410M) + 1.4B, preregistered** | ⏳ Running | Adds a continuous association measure, causal ablation of the gender direction, a 25-layer sweep, the logit lens, and 64 new BLS occupations. Analysis code was committed before any v2 data was inspected. |

### v1: Pythia-1.4B (complete)

- 📄 **[Final report](reports/FINAL_REPORT.md)**: main findings, controls and limitations.
- 📈 [Shared-onset analysis (Amendment 01)](reports/AMENDMENT-01.md), with its [figure](reports/amendment-01.png) and [data](reports/amendment-01.json).
- 📈 [Original protocol report](reports/REPORT.md) and [raw curves](reports/curves.png), retained for audit.
- 🧪 [Apple-style follow-up (JSD-P gap, steps 2k–5k)](reports/apple-confirmation.json).

Headline numbers:
- **Direct occupation-label probe:** 70.1% accuracy against 49.2% for matched shuffled-label controls, giving **+20.9 pp selectivity** (95% CI 12.8–28.1).
- **Onset brackets (step 0 → final range):**

| Range reached | Association | Expression | Lead 95% CI (steps) |
|---:|---|---|---|
| 25% | 512–1,000 | 512–1,000 | [−1,000, +1,872] |
| 50% | 1,000–2,000 | 1,000–2,000 | [−2,000, +1,000] |
| 75% | 5,000–6,000 | 1,000–2,000 | [−7,000, +2,000] |

### Could v1 have detected a lead at all?

📄 **[Detectability report](reports/DETECTABILITY.md)**, with its [figure](reports/detectability.png) and [data](reports/detectability.json).

This is a simulation that plants known leads into realistic noise taken from v1's own residuals, then reruns the exact v1 pipeline. Main points:
- Power to detect a representation-first lead is low at every lead size tested.
- Power is lopsided: an expression-first lead is much easier to detect, because the association curve is noisier.
- Pythia's checkpoint grid has no checkpoint between steps 512 and 1,000, which is exactly where both curves rise. That gap caps the resolution.
- The bracket rule is not a controlled test. The bootstrap CI is conservative, with 0% false positives in simulation.

A [prospective simulation of the v2 design](reports/detectability-v2-prospective.json) estimates at least 93% power for a 1.25× lead. That is an upper bound, because it assumes all seeds share the same true timing.

### v2: preregistered multi-seed replication (running)

- 📋 **[Protocol v2](PROTOCOL-v2.md)**, committed under tag [`v2-prereg`](https://github.com/rileyq7/representation-before-behaviour/tree/v2-prereg) before any v2 output was collected.
- **Primary question:** across 10 independent Pythia-410M training runs ([PolyPythias](https://huggingface.co/EleutherAI/pythia-410m-seed1)), is the mean log-ratio of 50%-crossing times (association vs expression) different from zero? This is tested with a t-CI over seeds.
- **Secondary analyses:** Pythia-1.4B; all 25 layers; logit lens; gender-direction ablation versus 10 random directions; v1's accuracy measure; the WinoBias-only occupations.
- **Where to look:** results will appear in `reports/V2_REPORT.md`, and this README will link them when the run finishes.

## How it works

```
prompts ──► checkpoint ──► last-token residual stream (layers 0–24) ──► linear probes ──► association
                      └──► next-token logits (" she" − " he") ───────────────────────► expression
```

- **Prompts:** 40 WinoBias occupations (v2 adds 64 BLS occupations) × 6 neutral templates, e.g. *"The nurse said that"*. Also 10 explicit gender word pairs × the same templates, used to train the gender probe.
- **Probes:** standardised logistic regression (C = 0.01) with occupation- and template-held-out folds. Controls: shuffled-label controls, random initialisation, the embedding layer, and a character n-gram lexical baseline.
- **Timing:** each curve is normalised to its own step-0-to-final range, and the analysis asks when it first reaches 25%, 50% and 75%. Uncertainty comes from a stratified occupation bootstrap; in v2, seeds are the unit of replication.
- **Scope:** "linearly accessible" is not the same as "causally used". v2's ablation analysis addresses causal use directly.

## Repository layout

| Path | Contents |
|---|---|
| `PROTOCOL.md`, `AMENDMENT-01.md`, `PROTOCOL-v2.md` | Analysis plans, in the order they were fixed |
| `reports/` | All reports, figures and machine-readable results |
| `src/mech/` | Data construction, inference, probes, statistics (`amendment.py`, `detectability.py`, `v2.py`, `analysis_v2.py`) |
| `scripts/` | Local runner (`run.py`), Modal cloud runner (`modal_v2.py`), report builders |
| `data/` | Prompts, manifests, WinoBias source files, BLS source table |
| `results/checkpoints/` | v1 per-checkpoint outputs, probe predictions and provenance (activations are not in git) |
| `results/v2/` | v2 per-checkpoint summaries (arrays stay on the Modal Volume) |
| `sources/` | Source papers' metadata, model file hashes, references |
| `tests/` | Software tests on synthetic data (never used as results) |

## Reproduce

Requires Python 3.13. Exact package versions are in `requirements-lock.txt`.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-lock.txt
.venv/bin/python scripts/fetch_data.py          # downloads WinoBias; verifies SHA-256 hashes
.venv/bin/python scripts/run.py prepare
.venv/bin/python -m pytest -q
```

**v1 (local, Apple Silicon or CUDA):**

```bash
.venv/bin/python scripts/run.py sweep --steps all --device mps --full-bracket   # ~450 GB of downloads, days locally
.venv/bin/python scripts/amend.py                                              # Amendment 01 analysis
.venv/bin/python scripts/run.py report
```

- Run it detached with `scripts/launch.py`, pause it with `touch STOP`, and check progress with `scripts/status.py`.
- Completed checkpoints are reused on restart.
- Only one checkpoint (~2.9 GB) is kept on disk at a time.

**Detectability analysis:** `.venv/bin/python scripts/detectability.py` (~8 minutes on CPU).

**v2 (Modal cloud; about 480 checkpoints, roughly $5–10 of compute):**

```bash
modal run scripts/modal_v2.py --smoke        # 2 checkpoints: timing and cost check
modal run --detach scripts/modal_v2.py       # full run; resumable
modal run scripts/modal_v2.py --fetch        # copy summaries to results/v2/
.venv/bin/python scripts/report_v2.py
```

## Provenance and caveats

- **Models:** each checkpoint is resolved to an immutable Hugging Face commit, and its weight SHA-256 is verified.
- **Devices:** v1 ran float16 inference on Apple MPS; v2 runs float16 on NVIDIA L4. At the 1.4B final checkpoint the two agree on expression to r = 1.0000 (largest difference 0.005 logits), and transfer accuracy is identical for all 40 occupations. v1 and v2 data are never pooled.
- **Amendment 01** was written *after* inspecting v1's coarse curves and is labelled as such. The detectability analysis is post hoc. v2 is the preregistered test.
- **Relation to Patel et al.** The ~80k-step transition in [Patel et al., *Fairness Dynamics During Training*](https://arxiv.org/abs/2506.01709) was reported for Pythia-6.9B, not 1.4B. The Apple-style prompts here are a reconstruction, not the authors' code.
- **Data terms:** WinoBias ([Zhao et al., 2018](https://github.com/uclanlp/corefBias)) and BLS CPS Table 11 are redistributed under their original terms. Full references are in `sources/REFERENCES.md`.
