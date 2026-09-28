# Does occupational gender bias become linearly accessible before it becomes behaviourally observable?

**Completed 2026-09-23: all 154 Pythia-1.4B checkpoints, all matched controls, and four full Apple-style follow-up evaluations.**

## Main result

We did **not** find a robust representation-first ordering under the revised, shared onset definition. Gender-probe transfer to occupations and continuous behavioural stereotype preference reach 25% and 50% of their respective step-0-to-final ranges in the same checkpoint brackets. Expression reaches 75% earlier in the point estimates, but the bootstrap interval for the timing difference includes zero. These results do not establish either exact co-emergence or a reliable behavioural lead.

| Fraction of endpoint range | Explicit-gender decoding | Occupation association (gender-probe transfer) | Expression (signed he/she logit difference) |
|---:|---|---|---|
| 25% | 512–1,000 | 512–1,000 | 512–1,000 |
| 50% | 512–1,000 | 1,000–2,000 | 1,000–2,000 |
| 75% | 512–1,000 | 5,000–6,000 | 1,000–2,000 |

These brackets locate first observed crossings between saved checkpoints; they are not confidence intervals. Define a positive lead as expression's crossing step minus association's crossing step. The paired bootstrap 95% intervals for this observed-grid difference are **[−1,000, +1,872]**, **[−2,000, +1,000]**, and **[−7,000, +2,000]** steps at 25%, 50%, and 75%, respectively. All include zero. Bootstrap endpoint ranges were positive in all 2,000 draws for each reported curve.

The thresholds measure relative progress toward the final value, not the first physical existence of a representation. They retain dependence on measurement sensitivity, the chosen baseline and endpoint, and the sampled prompts.

## Controls and sensitivity

The directly trained occupation-label probe reaches **70.14% accuracy**, versus **49.22%** for its matched shuffled-label controls. Final selectivity is **20.92 percentage points**, with a stratified occupation-bootstrap 95% interval of **12.78–28.10 pp**. At initialization, selectivity is **2.67 pp**, interval **−1.91–6.86 pp**. Each control preserves occupation-level labels across templates and exact class balance in every train/test fold; 20 permutations are evaluated under each of three split seeds.

That supports non-arbitrary decodability of the provided stereotype labels. It does not exclude domain, frequency, formality or other structured semantic correlates. The direct occupation classifier and the occupation-label-free gender-transfer classifier are different measurements and must not be merged into one claim.

Onset is also sensitive to persistence. Requiring three consecutive crossings moves the transfer probe's 75% bracket to **12,000–13,000**, while expression remains at **1,000–2,000**. The direct probe and its selectivity have early crossings that do not persist: their sustained 75% brackets are **27,000–28,000**. These are operational sensitivity results, not evidence that a representation was absent earlier.

The original unequal absolute-threshold analysis reports behaviour earlier by 25,000–27,000 steps. It is retained for audit, **not used as the revised primary result**. Its different practical thresholds and extra representation-specific control requirements cannot support a threshold-independent lead claim.

## Apple-style follow-up

The original published ~80k finding concerns **Pythia-6.9B**, not 1.4B. This project is an extension to 1.4B and a reconstruction of the paper's prompting, not an exact replication of its code or seeds.

The 128-prompt development discovery set selected the largest adjacent change in the female-minus-male correct-answer JSD-P gap between steps **3,000 and 4,000**. We evaluated the full reconstruction at steps 2,000, 3,000, 4,000 and 5,000: **15,840 prompts per checkpoint**, including **7,920 held-out test prompts** each.

| Step | Test-set female − male correct-answer JSD-P gap (nats) |
|---:|---:|
| 2,000 | +0.0753 |
| 3,000 | +0.1374 |
| 4,000 | −0.0051 |
| 5,000 | −0.0262 |

The selected 3,000→4,000 change also appears on the test set. The equal-sentence-pair-weighted change is **−0.1426 nats**, with a paired-bootstrap 95% interval of **−0.1450 to −0.1401**. This is evidence of a change in this evaluation's gender performance gap; it is not proof of a discontinuity or the emergence of occupation-specific bias. The bracket was selected exploratorily, and these descriptive intervals are not selection-adjusted. The Apple-style task and the neutral occupation-completion task measure different constructs.

## Scope and audit

The shared-onset amendment was requested **after the coarse curves were inspected**. It is transparently recorded as a post-inspection amendment, not retrospectively described as a preregistration. All original results are preserved.

The experiment uses one model training run, 40 occupations, six completion templates, and a prespecified primary layer (12). Bootstrap uncertainty conditions on fitted probes; it does not include refitting uncertainty or variation across model training seeds. A behavioural lead could reflect nonlinear/distributed encoding, another layer or position, probe mismatch, or differing sensitivity. None of those mechanisms is established here.

Final audit: all 154 expected steps are present; all share the same prompt-data and original-protocol hashes; all 154 amended analyses share the same amendment hash; no probe convergence warnings were recorded; expected activation, output and prediction artifacts exist; all four selected follow-ups completed.

## Evidence

- [Shared-onset curves, selectivity table and full amended analysis](AMENDMENT-01.md)
- [Machine-readable amended onsets and uncertainty](amendment-01.json)
- [Apple test-set paired comparisons](apple-confirmation.json)
- [Original protocol analysis, retained for audit](REPORT.md)
- [Original protocol](../PROTOCOL.md) and [dated amendment / interpretation commitments](../AMENDMENT-01.md)
- Original study: [Patel et al., Fairness Dynamics During Training](https://arxiv.org/html/2506.01709v1)

The defensible conclusion is: **in this setup, representation-first emergence is not supported; the apparent timing relationship depends on the operational measurement and threshold, and uncertainty does not resolve a consistent lead.**
