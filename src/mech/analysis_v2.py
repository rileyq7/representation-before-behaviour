"""Protocol v2 analysis, as preregistered in PROTOCOL-v2.md (written before v2 curves were inspected)."""
import json
from pathlib import Path
import numpy as np
from scipy import stats
from .amendment import crossing, normalise, relation
from .detectability import interpolated_crossing
from .metrics import bootstrap_indices

THRESHOLDS = [.25, .50, .75]
PRIMARY_THRESHOLD = .5
SEEDS = [f"pythia-410m-seed{i}" for i in range(10)]
STEPS = [0] + [2**i for i in range(10)] + list(range(1000, 20001, 1000)) + list(range(30000, 140001, 10000)) + [143000]


def load(root):
    data = {}
    for folder in sorted(Path(root, "results/v2").glob("*")):
        files = sorted(folder.glob("step*/summary.json"))
        if files:
            recs = sorted((json.loads(p.read_text()) for p in files), key=lambda d: d["step"])
            data[folder.name] = recs
    return data


def curves(records, occupation_set, layer=12):
    """checkpoints x occupations matrices for every prespecified measure."""
    s = lambda key: np.array([r["sets"][occupation_set][key] for r in records])
    behaviour = s("behaviour")
    abl = behaviour - s("ablation_gender")
    abl_random = (behaviour[:, None] - s("ablation_random")).mean(1)
    return {"expression": behaviour,
            "association_margin": s("transfer_margin_by_layer")[:, layer],
            "association_accuracy": s("transfer_accuracy_by_layer")[:, layer],
            "logit_lens": s("logit_lens_by_layer")[:, layer],
            "ablation_effect": abl - abl_random,
            "selectivity": s("selectivity")}


def layer_curves(records, occupation_set, key):
    return np.array([r["sets"][occupation_set][key] for r in records])  # steps x 25 x occupations


def seed_lead(steps, early, late, threshold):
    """log(1+t_late) - log(1+t_early) with interpolated crossings; NaN plus reason if excluded."""
    za, va = normalise(early.mean(1)); ze, ve = normalise(late.mean(1))
    if not va or not ve:
        return np.nan, "nonpositive_endpoint_range"
    ta, te = interpolated_crossing(steps, za, threshold), interpolated_crossing(steps, ze, threshold)
    if not np.isfinite(ta) or not np.isfinite(te):
        return np.nan, "not_crossed"
    return float(np.log1p(te) - np.log1p(ta)), None


def t_summary(values):
    v = np.asarray([x for x in values if np.isfinite(x)])
    n = len(v)
    if n < 2:
        return dict(n=n, mean=float(v.mean()) if n else None, ci95=None, decision="insufficient_seeds")
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(n)
    lo, hi = m + np.array([-1, 1]) * stats.t.ppf(.975, n - 1) * se
    decision = "association_first" if lo > 0 else "expression_first" if hi < 0 else "unresolved"
    return dict(n=n, mean=float(m), ci95=[float(lo), float(hi)], ratio=float(np.exp(m)),
                ratio_ci95=[float(np.exp(lo)), float(np.exp(hi))], decision=decision,
                per_seed=v.tolist(), sign_consistency=float(np.mean(np.sign(v) == np.sign(m))))


def lead_across_seeds(data, occupation_set, early_key, late_key="expression", threshold=PRIMARY_THRESHOLD,
                      layer=12, gate=True):
    leads, exclusions = [], {}
    for tag in SEEDS:
        recs = data.get(tag, [])
        steps = np.array([r["step"] for r in recs])
        if steps.tolist() != STEPS:
            exclusions[tag] = f"incomplete_grid ({len(steps)}/44)"; leads.append(np.nan); continue
        if gate and early_key.startswith("association"):
            gender = np.mean(recs[-1]["per_gender_pair_accuracy_by_layer"][layer])
            if gender < .8:
                exclusions[tag] = f"gender_decoding_{gender:.2f}"; leads.append(np.nan); continue
        c = curves(recs, occupation_set, layer)
        value, reason = seed_lead(steps, c[early_key], c[late_key], threshold)
        if reason:
            exclusions[tag] = reason
        leads.append(value)
    out = t_summary(leads)
    out["exclusions"] = exclusions
    if len(exclusions) > 3:
        out["decision"] = "inconclusive_too_many_exclusions"
    return out


def hierarchical_bootstrap(data, occupation_set, threshold=PRIMARY_THRESHOLD, draws=2000, seed=20260928):
    rng = np.random.default_rng(seed)
    per_seed = []
    for tag in SEEDS:
        recs = data.get(tag, [])
        if [r["step"] for r in recs] != STEPS:
            continue
        c = curves(recs, occupation_set)
        labels = np.array(recs[0]["sets"][occupation_set]["occupation_labels"])
        per_seed.append((np.array(STEPS), c["association_margin"], c["expression"], labels))
    if len(per_seed) < 2:
        return None
    means = []
    for _ in range(draws):
        chosen = rng.integers(0, len(per_seed), len(per_seed))
        vals = []
        for i in chosen:
            steps, a, e, labels = per_seed[i]
            cols = np.concatenate([rng.choice(np.flatnonzero(labels == c), (labels == c).sum()) for c in (0, 1)])
            v, _ = seed_lead(steps, a[:, cols], e[:, cols], threshold)
            vals.append(v)
        vals = np.array(vals)
        if np.isfinite(vals).any():
            means.append(np.nanmean(vals))
    lo, hi = np.quantile(means, [.025, .975])
    return dict(draws=len(means), ci95=[float(lo), float(hi)], ratio_ci95=[float(np.exp(lo)), float(np.exp(hi))])


def single_model_lead(records, occupation_set, threshold, early_key="association_margin"):
    steps = np.array([r["step"] for r in records])
    if steps.tolist() != STEPS:
        return {"status": f"incomplete_grid ({len(steps)}/44)"}
    c = curves(records, occupation_set)
    point, reason = seed_lead(steps, c[early_key], c["expression"], threshold)
    labels = np.array(records[0]["sets"][occupation_set]["occupation_labels"])
    boots = np.array([seed_lead(steps, c[early_key][:, d], c["expression"][:, d], threshold)[0]
                      for d in bootstrap_indices(labels)])
    boots = boots[np.isfinite(boots)]
    return dict(lead=point, excluded=reason, ci95=np.quantile(boots, [.025, .975]).tolist() if len(boots) else None,
                valid_fraction=len(boots) / 2000)


def continuity_v1(records, occupation_set):
    """Descriptive only: v1-style first-crossing brackets and paired observed-grid lead."""
    steps = np.array([r["step"] for r in records])
    c = curves(records, occupation_set)
    a, e = c["association_accuracy"], c["expression"]
    labels = np.array(records[0]["sets"][occupation_set]["occupation_labels"])
    draws = bootstrap_indices(labels)
    za, _ = normalise(a.mean(1)); ze, _ = normalise(e.mean(1))
    ba, bva = normalise(a[:, draws].mean(2)); be, bve = normalise(e[:, draws].mean(2))
    ok = bva & bve
    rows = []
    for t in THRESHOLDS:
        ca, ce = crossing(steps, za, t), crossing(steps, ze, t)
        lead = steps[np.argmax(be[:, ok] >= t, 0)] - steps[np.argmax(ba[:, ok] >= t, 0)]
        rows.append(dict(threshold=t, association=ca["interval"], expression=ce["interval"],
                         bracket_rule_descriptive=relation(ca, ce),
                         observed_grid_lead_ci95=np.quantile(lead, [.025, .975]).tolist() if len(lead) else None))
    return rows


def ablation_specificity(data, occupation_set):
    """Final-step gender-direction ablation effect vs each random direction, per seed."""
    rows = {}
    for tag in SEEDS + ["pythia-1.4b"]:
        recs = data.get(tag, [])
        if not recs or recs[-1]["step"] != 143000:
            continue
        s = recs[-1]["sets"][occupation_set]
        b = np.array(s["behaviour"])
        gender = float((b - np.array(s["ablation_gender"])).mean())
        randoms = (b[None] - np.array(s["ablation_random"])).mean(1)
        rows[tag] = dict(gender_effect=gender, random_effects=randoms.tolist(),
                         exceeds_all_random=bool(gender > randoms.max()))
    seeds = [rows[t] for t in SEEDS if t in rows]
    return dict(per_model=rows, seeds_exceeding_all_random=sum(r["exceeds_all_random"] for r in seeds),
                seeds_available=len(seeds), final_effect_t=t_summary([r["gender_effect"] - np.mean(r["random_effects"]) for r in seeds]),
                specific=sum(r["exceeds_all_random"] for r in seeds) >= 8)


def analyse(root):
    data = load(root)
    out = {"models": {k: len(v) for k, v in data.items()}}
    primary = lead_across_seeds(data, "combined", "association_margin")
    out["primary"] = dict(description="410M, 10 seeds, combined set, layer-12 signed transfer margin vs expression, 50% threshold",
                          **primary, hierarchical_bootstrap=hierarchical_bootstrap(data, "combined"))
    sec = {}
    sec["1_thresholds"] = {str(t): lead_across_seeds(data, "combined", "association_margin", threshold=t) for t in (.25, .75)}
    sec["2_accuracy_measure"] = {str(t): lead_across_seeds(data, "combined", "association_accuracy", threshold=t) for t in THRESHOLDS}
    sec["3_winobias_only"] = {str(t): lead_across_seeds(data, "winobias", "association_margin", threshold=t) for t in THRESHOLDS}
    sec["4_hierarchical_bootstrap"] = out["primary"]["hierarchical_bootstrap"]
    if "pythia-1.4b" in data:
        sec["5_pythia_1.4b"] = {f"{s}:{t}": single_model_lead(data["pythia-1.4b"], s, t) for s in ("combined", "winobias") for t in THRESHOLDS}
    sec["6_layer_sweep"] = {str(l): lead_across_seeds(data, "combined", "association_margin", layer=l, gate=False) for l in range(25)}
    sec["7_logit_lens"] = {str(l): lead_across_seeds(data, "combined", "logit_lens", layer=l, gate=False) for l in range(25)}
    sec["8_ablation"] = dict(onset={str(t): lead_across_seeds(data, "combined", "ablation_effect", threshold=t, gate=False) for t in THRESHOLDS},
                             specificity=ablation_specificity(data, "combined"))
    sec["9_selectivity"] = {str(t): lead_across_seeds(data, "combined", "selectivity", threshold=t, gate=False) for t in THRESHOLDS}
    sec["10_v1_continuity"] = {tag: continuity_v1(data[tag], "winobias") for tag in SEEDS + ["pythia-1.4b"]
                               if tag in data and [r["step"] for r in data[tag]] == STEPS}
    out["secondary"] = sec
    return out
