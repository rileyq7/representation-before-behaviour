# Protocol v2: multi-seed replication with causal and layer-wise measurements

Written 2026-09-28. It was **committed to git before any v2 model output was collected**; the commit hash is recorded in every v2 result file. It is a dated local plan, not an external registry entry.

What was known when it was written: all v1 results (`reports/FINAL_REPORT.md`) and the post-hoc detectability analysis (`reports/DETECTABILITY.md`). The detectability analysis showed three things. The v1 design had little power for a representation-first lead. The bracket-ordering rule is not a controlled test. The noisy accuracy-based association curve and the coarse checkpoint grid limit resolution. This protocol is designed around those findings. v1 and its amendment are unchanged.

## Question

Across independent training runs, does occupation–gender **association** in the residual stream reach a fixed fraction of its learned range at a different time from behavioural **expression** of the stereotype?

## Models and checkpoints

- **Primary:** Pythia-410M, 10 training seeds. These are `EleutherAI/pythia-410m` (the original run, "seed 0") and `EleutherAI/pythia-410m-seed1` … `-seed9` (PolyPythias; the seeds differ in initialisation and data order). All 10 were confirmed on 2026-09-28 to have every checkpoint listed below.
- **Secondary:** Pythia-1.4B (`EleutherAI/pythia-1.4b`, one run), on the same grid with the new measurements.
- **Grid (44 steps):** 0, 1, 2, 4, …, 512; every 1,000 from 1,000 to 20,000; every 10,000 from 30,000 to 140,000; and 143,000. This is denser where v1's curves changed and coarser on the plateau. Each revision is resolved to an immutable commit hash and recorded.
- **Inference:** Modal cloud, CUDA float16 on NVIDIA L4. Activations are stored as float16, and probes are fitted in float32. v1 ran on Apple MPS. The two devices are compared at 1.4B steps 0, 1,000 and 143,000 on v1 prompts. v1 and v2 data are **never pooled**.

## Prompts

- **WinoBias set:** v1's 40 occupations × 6 templates, byte-identical to `data/matched.jsonl`.
- **BLS set:** 64 additional occupations × the same 6 templates. The source is BLS CPS 2025 annual averages, Table 11 (`data/raw-v2/occupations-v2.json`, with the SHA-256 of the source page). The selection rules were fixed before any v2 output existed:
  - at least 100k employed;
  - at most 30% women (label 0) or at least 70% women (label 1);
  - no "other/all other" categories and no supervisor titles;
  - no inherently gendered nouns (waitress, maid, hostess, clergy) and no ambiguous nouns (server, dishwasher, interviewer);
  - no noun containing a WinoBias occupation word, and no BLS category that already contains a WinoBias occupation;
  - classes balanced at 32/32 by dropping the lowest-employment members of the larger class (the dropped nouns are recorded).
- **Combined set:** 104 occupations = WinoBias + BLS. Labels are WinoBias's own for its 40 and the BLS rule above for the other 64.
- **Gender set:** v1's 10 explicit word pairs × 6 templates, unchanged.

## Measurements (per checkpoint)

All measurements use the final prompt token. Layers are 0 (embedding) through 23 (block outputs) and 24 (the final post-norm representation, as in v1). The probe specification is unchanged from v1: train-fold StandardScaler, logistic regression with C = 0.01 (liblinear), and the same template and occupation folds.

1. **Expression:** signed stereotype logit difference, (logit " she" − logit " he") × (+1 for female-labelled occupations, −1 otherwise), averaged over templates within each occupation.
2. **Association, primary:** at layer 12, the **signed transfer margin**. The gender probe is trained on explicit gender words with template-held-out folds (as in v1) and applied to occupations. Its decision function is multiplied by the same sign as expression and averaged over templates within each occupation. This is the continuous counterpart of v1's transfer accuracy and uses no occupation labels in fitting.
3. **Association, v1 measure:** transfer accuracy at layer 12, as in v1.
4. **Layer sweep:** transfer margin and accuracy at all 25 layers.
5. **Logit lens:** at every layer, apply the final layer norm and the unembedding to the residual, then take the signed he/she difference.
6. **Causal ablation:** fit the layer-12 explicit-gender probe on all gender rows and map it to a unit direction *u* in raw activation space. At the output of block 12, at every token position, set the component along *u* to the mean projection over occupation prompts. Recompute expression. The control is 10 fixed random unit directions (seed 20260928), ablated the same way. Ablation effect = (unablated expression − gender-ablated expression) − mean over random directions of (unablated − random-ablated).
7. **Direct occupation probe and selectivity:** as in Amendment 01 (3 split seeds × 20 balanced label permutations), at layer 12, for the WinoBias and combined sets.
8. **Explicit-gender decoding:** accuracy per word pair at every layer, with pair × template held out, as in v1.

## Primary analysis (confirmatory)

- **Data:** Pythia-410M, 10 seeds, combined set.
- **Curves:** for each seed and measure (expression; layer-12 signed transfer margin), average over occupations, then normalise z(t) = [m(t) − m(0)] / [m(143,000) − m(0)] as in Amendment 01. Values are not clipped or made monotone.
- **Crossing time:** t_x is the first grid step where z ≥ 0.5. It is **linearly interpolated in log(1 + step)** between that checkpoint and the one before. This interpolation assumes the curve is monotone between neighbouring checkpoints.
- **Per-seed lead:** L = log(1 + t_expression) − log(1 + t_association). Positive means association first; L = log k corresponds to "association reaches 50% k× sooner".
- **Estimate and CI:** the mean of L over seeds, with a **one-sample t 95% confidence interval** (df = number of included seeds − 1). The seed is the unit of replication.
- **Decision:**
  - CI entirely above 0: association reaches 50% of its range before expression in this model family and measurement.
  - CI entirely below 0: expression reaches it first.
  - Otherwise: unresolved. The report states the largest leads the CI rules out, as exp(CI bounds).
- **Inclusion:** a seed is excluded from a measure if its endpoint range is ≤ 0, if the curve never reaches 0.5 (right-censored), or if held-out explicit-gender decoding at layer 12 at step 143,000 is below 0.8 (for association measures). Exclusions are reported per seed. If more than 3 of the 10 seeds are excluded, the primary result is **inconclusive**.
- **Interpretation limits:** no causal claim follows from the primary analysis, and no claim that a representation was absent before crossing. An ordering is a statement about relative progress toward each measure's own endpoint.

**Prospective power** (`reports/detectability-v2-prospective.json`) comes from simulations with v1 1.4B noise: accuracy-based association, 104 occupations, 10 seeds, 200 realisations. At the 50% threshold it gives at least 93% power for a 1.25× lead in either direction, with a false-positive rate of 5% or less. This is an **upper bound**: it assumes no seed-to-seed variation in true timing, and 410M noise may differ.

## Secondary analyses (prespecified; reported whatever the result, in this order)

1. The primary analysis at the 25% and 75% thresholds.
2. The primary analysis with v1's transfer accuracy in place of the margin.
3. The primary analysis on the WinoBias-only set (the replication most comparable to v1).
4. **Hierarchical bootstrap** of the primary: 2,000 draws, resampling seeds, and occupations within label class within each seed. The occupation draws are shared across steps and both measures within a draw.
5. **Pythia-1.4B:** the same per-model lead L, with a 95% interval from a stratified occupation bootstrap (single run, so there is no seed-level CI). Compared descriptively with the 410M seed distribution.
6. **Layer sweep:** mean L (with t-CI over seeds) for the transfer margin at each of the 25 layers. This is **descriptive**; no layer is selected as a new primary.
7. **Logit lens:** mean L over seeds for the layer-12 logit lens against expression, and a per-layer crossing-time profile. The question is whether a readout of layer 12 through the model's own unembedding leads expression. This separates "the information is at layer 12" from "the probe finds it".
8. **Causal ablation:** per seed, the 50% crossing time of the ablation effect's own step-0-to-final range, compared with expression (mean L with t-CI). Also reported: the final-step ablation effect with its t-CI over seeds, and the proportion of random directions whose effect exceeds the gender direction's at each step. A meaningful causal-use onset requires the final gender-direction effect to exceed every random direction in at least 8 of 10 seeds. Otherwise the ablation is reported as not showing specific causal use.
9. Selectivity (real minus shuffled-label accuracy) curves per seed, as in Amendment 01.
10. **Continuity with v1:** first-crossing brackets and the paired occupation bootstrap of observed-grid leads, per seed. The bracket rule is **descriptive only**; it carries no ordering claim (see the detectability analysis).

Any secondary result that disagrees with the primary is reported as such and does not replace it.

## Software and deviations

- **Code:** `src/mech/v2.py` (measurements), `scripts/modal_v2.py` (cloud runner) and `src/mech/analysis_v2.py` (analysis). The analysis code is written and tested on synthetic curves before v2 curves are inspected. Software tests are in `tests/`; synthetic data never appear as results.
- **Deviations:** any change after data collection begins, whether to code affecting results, inclusion rules, or estimands, is recorded as a dated amendment and labelled as post-data. Operational fixes (retries, infrastructure) are logged but are not amendments.
- **Failure handling:** if a checkpoint cannot be processed after retries, the missing step is reported. If the missing step is 0 or 143,000, that seed is excluded. The primary analysis requires all 44 steps for an included seed.
