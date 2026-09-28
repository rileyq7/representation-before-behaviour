# Amendment 01: shared onset criteria and selectivity

Requested 2026-09-21 after inspecting the coarse curves. At recording, checkpoints 0, 1000, 10000, 20000, 40000, 60000, 70000, 80000, 90000, 100000, 120000 and 143000 were complete. This is a prospective commitment for the remaining analysis, **not a claim of preregistration before seeing data**. `PROTOCOL.md` and its existing results stay intact.

## Identical onset rule

For each curve, use z(t) = [m(t) - m(0)] / [m(143000) - m(0)]. Use the actual final checkpoint, not the maximum observed value. Do not clip overshoots or negative values and do not enforce monotonicity. A nonpositive or effectively zero denominator makes this normalisation undefined. Report the raw floor, endpoint and range alongside every normalised curve.

The three primary curves are held-out explicit-gender decoding accuracy, gender-probe transfer accuracy on occupations (association), and mean signed stereotype logit difference (expression). Use layer 12 as already specified; expression is the existing he/she continuous score. First observed crossing of **25%, 50%, and 75%** of the endpoint range defines three onset estimates, with the identical rule for all curves. No significance/accuracy cutoff applies to only one curve. Also report the same three-consecutive-checkpoint crossing sensitivity for every curve.

Report checkpoint brackets, not interpolated exact times. Until all checkpoints are present, crossings are provisional on the observed grid and can miss earlier transient crossings. Missing endpoints mean no normalised analysis. Threshold-stable ordering is evidence of robustness to these thresholds, not proof of an underlying causal order. Overlapping brackets mean no resolved ordering; differing order across thresholds is reported without selecting a preferred threshold.

Uncertainty: 2000 bootstrap draws, resampling 40 occupations within stereotype class for association and expression using the **same draws at every step**, including both endpoints. Resample the 10 explicit-gender word pairs for gender decoding. Recompute endpoint normalisation inside each bootstrap draw. Report the proportion of nonpositive endpoint ranges, pointwise ratio intervals, and bootstrap distributions of observed-grid association-versus-expression crossing-step differences. Intervals condition on the fixed fitted probes and do not include probe refitting uncertainty. Normalisation can amplify noise and an unusually low step-0 accuracy; retain the original raw/control-aware analysis alongside it. These descriptive post-amendment intervals do not replace the original simultaneous-band analysis.

## Matched control task and selectivity

For the directly trained occupation stereotype probe, use its existing C=0.01 logistic regression, scaling, three occupation-fold seeds and joint occupation/template holdouts. For each split seed, generate **20** occupation-level label permutations, fixed across templates and checkpoints. Shuffle within each held-out occupation fold, preserving exactly four labels per class among its eight occupations, and hence preserving real train/test class balance. Train and evaluate each control with the same folds, feature matrices and budget as the real classifier.

Selectivity = real accuracy minus mean control accuracy, averaging the same three split seeds for both. Report real accuracy, mean/dispersion of controls, selectivity in percentage points and stratified occupation bootstrap intervals. Save per-item control predictions and labels. Do not compare real labels against predictions trained for shuffled labels: each control is scored on its own shuffled targets.

Show raw direct-probe accuracy and control-adjusted selectivity as additional endpoint-normalised association analyses. Keep occupation-label-free gender transfer as the primary association curve; its existing gender-word-label permutation control is a separate diagnostic. A random-label control addresses probe capacity/memorisation; it cannot exclude structured domain/frequency/formality confounds that correlate with true stereotypes. This is an adaptation of Hewitt & Liang's control-task idea to held-out occupation identities, not an exact reproduction of their lexical memorisation setting.

Reference: John Hewitt and Percy Liang (2019), [Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/), EMNLP-IJCNLP.

## Interpretation commitments

- Association before expression: evidence that the measured linearly accessible alignment precedes this measured output preference, conditional on uncertainty and control results. No causal-use claim.
- Expression before association: output preference is detected earlier by this measurement. Compatible with nonlinear or distributed encoding, another layer/position, a poorly matched probe, or differing measurement sensitivity. It does **not** establish any of those explanations without follow-up nonlinear/multi-position/intervention tests.
- Overlapping onset brackets: no resolved ordering at the available checkpoint resolution, not proof of simultaneous emergence.
- Ordering changes across thresholds or association definitions: report threshold/measurement dependence; do not compress it into a universal lead.
- Flat/reversed endpoint range, unstable bootstrap denominator, or weak controls: inconclusive for the affected normalised/representational claim.

The collection protocol, checkpoint order, model weights, prompt data, layer selection, probe regularisation and original results are unchanged. The amendment is computed in a separate resumable CPU worker from saved data.
