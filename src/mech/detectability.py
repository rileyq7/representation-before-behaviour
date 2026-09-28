"""Detectability of association/expression leads under the Amendment 01 pipeline.

Exploratory, post-hoc power analysis on saved curves. Synthetic "truth" curves share one
smoothed empirical shape (expression's, or association's as a sensitivity check); one measure
reaches every level k times earlier (time rescaling keeps step 0 at the initialisation floor).
Leverage-corrected residuals from a local-linear log-time smooth are re-signed (Rademacher) to
create new realisations, keeping checkpoint-level shocks shared across occupations and paired
across both measures.
The amendment's crossing rule, endpoint normalisation and paired occupation bootstrap
are then applied unchanged.
"""
import json
from pathlib import Path
import numpy as np
from .amendment import THRESHOLDS, crossing, normalise, relation
from .metrics import bootstrap_indices

LEAD_FACTORS = [1, 1.25, 1.5, 2, 3, 4, 6, 8, 12, 16]


def load_curves(root):
    files = sorted(Path(root, "results/checkpoints").glob("step*/summary.json"))
    summaries = sorted((json.loads(p.read_text()) for p in files), key=lambda d: d["step"])
    steps = np.array([d["step"] for d in summaries])
    assoc = np.array([d["per_occupation"]["transfer_accuracy"] for d in summaries])
    expr = np.array([d["per_occupation"]["behaviour"] for d in summaries])
    return steps, assoc, expr, np.array(summaries[0]["occupation_labels"])


def local_linear(u_eval, u, values, h):
    """Local-linear Gaussian-kernel smoother in u = log1p(step); values may be 2-D (steps x columns)."""
    d = u_eval[:, None] - u[None, :]
    w = np.exp(-.5 * (d / h) ** 2)
    s0, s1, s2 = w.sum(1), (w * d).sum(1), (w * d * d).sum(1)
    weights = w * (s2[:, None] - d * s1[:, None]) / (s0 * s2 - s1 ** 2)[:, None]
    return weights @ values, weights


def choose_bandwidth(u, mean):
    best = None
    for h in np.linspace(.08, 1.2, 57):
        fit, weights = local_linear(u, u, mean, h)
        if np.diag(weights).max() > .95:
            continue
        loo = ((mean - fit) / (1 - np.diag(weights))) ** 2
        if best is None or loo.mean() < best[0]:
            best = (loo.mean(), h)
    return best[1]


def fit_measure(steps, values):
    """Smooth each occupation; residuals are leverage-corrected, split into checkpoint-common and item parts."""
    u = np.log1p(steps)
    h = choose_bandwidth(u, values.mean(1))
    smooth, weights = local_linear(u, u, values, h)
    residual = (values - smooth) / np.sqrt(1 - np.diag(weights))[:, None]
    common = residual.mean(1, keepdims=True)
    return dict(bandwidth=h, smooth=smooth, common=common, item=residual - common)


def shape_function(steps, values, h):
    """Endpoint-normalised smooth mean curve as a function of (possibly rescaled) training step."""
    u = np.log1p(steps); mean = values.mean(1)
    lo, hi = local_linear(u[[0, -1]], u, mean, h)[0]
    def f(t):
        t = np.clip(t, 0, steps[-1])
        out = (local_linear(np.log1p(t), u, mean, h)[0] - lo) / (hi - lo)
        return np.where(t == 0, 0., out)
    return f


def truth(steps, values, shape, factor):
    """Per-occupation floor/amplitude on the shared shape; the measure reaches each level `factor` times earlier."""
    basis = np.column_stack([np.ones(len(steps)), shape(steps)])
    coef, *_ = np.linalg.lstsq(basis, values, rcond=None)
    return coef[0] + np.outer(shape(steps * factor), coef[1])


def crossing_time(shape, steps, threshold):
    grid = np.unique(np.concatenate([np.arange(0, 20001, 5), steps]))
    above = np.flatnonzero(shape(grid) >= threshold)
    return float(grid[above[0]]) if len(above) else np.nan


def lead_statistics(steps, assoc, expr, draws):
    """Point-estimate ordering and paired-bootstrap observed-grid lead (expression minus association)."""
    za, va = normalise(assoc.mean(1)); ze, ve = normalise(expr.mean(1))
    ba, bva = normalise(assoc[:, draws].mean(2)); be, bve = normalise(expr[:, draws].mean(2))
    valid = bva & bve
    out = []
    for threshold in THRESHOLDS:
        ca = crossing(steps, za if va else np.full(len(steps), np.nan), threshold)
        ce = crossing(steps, ze if ve else np.full(len(steps), np.nan), threshold)
        aidx = np.argmax(ba[:, valid] >= threshold - 1e-12, axis=0)
        eidx = np.argmax(be[:, valid] >= threshold - 1e-12, axis=0)
        lead = steps[eidx] - steps[aidx]
        ci = np.quantile(lead, [.025, .975]) if len(lead) else [np.nan, np.nan]
        out.append(dict(ordering=relation(ca, ce), ci=[float(ci[0]), float(ci[1])]))
    return out


def simulate(root, realisations=500, seed=20260928):
    steps, assoc, expr, labels = load_curves(root)
    draws = bootstrap_indices(labels)
    fits = {"association": fit_measure(steps, assoc), "expression": fit_measure(steps, expr)}
    raw = {"association": assoc, "expression": expr}
    observed = lead_statistics(steps, assoc, expr, draws)
    rng = np.random.default_rng(seed)
    results = {}
    for shape_name in ("expression", "association"):
        shape = shape_function(steps, raw[shape_name], fits[shape_name]["bandwidth"])
        rows = []
        for direction in ("association_first", "expression_first"):
            for k in LEAD_FACTORS:
                if direction == "expression_first" and k == 1:
                    continue
                truth_a = truth(steps, assoc, shape, k if direction == "association_first" else 1)
                truth_e = truth(steps, expr, shape, k if direction == "expression_first" else 1)
                tallies = [dict(before=0, after=0, overlap=0, ci_pos=0, ci_neg=0, widths=[]) for _ in THRESHOLDS]
                sub = np.random.default_rng(rng.integers(2**63))
                for _ in range(realisations):
                    s_step = sub.choice([-1., 1.], (len(steps), 1))
                    s_item = sub.choice([-1., 1.], assoc.shape)
                    sim_a = truth_a + s_step * fits["association"]["common"] + s_item * fits["association"]["item"]
                    sim_e = truth_e + s_step * fits["expression"]["common"] + s_item * fits["expression"]["item"]
                    for tally, stat in zip(tallies, lead_statistics(steps, sim_a, sim_e, draws)):
                        tally["before"] += stat["ordering"] == "association_before_expression"
                        tally["after"] += stat["ordering"] == "expression_before_association"
                        tally["overlap"] += stat["ordering"] == "overlapping_brackets"
                        tally["ci_pos"] += stat["ci"][0] > 0
                        tally["ci_neg"] += stat["ci"][1] < 0
                        tally["widths"].append(stat["ci"][1] - stat["ci"][0])
                for threshold, tally in zip(THRESHOLDS, tallies):
                    t_late = crossing_time(shape, steps, threshold)
                    rows.append(dict(direction=direction, lead_factor=k, threshold=threshold,
                                     true_crossing_later_measure=t_late, true_crossing_earlier_measure=t_late / k,
                                     true_lead_steps=t_late * (1 - 1 / k),
                                     p_bracket_association_first=tally["before"] / realisations,
                                     p_bracket_expression_first=tally["after"] / realisations,
                                     p_bracket_overlap=tally["overlap"] / realisations,
                                     p_ci_excludes_zero_association_first=tally["ci_pos"] / realisations,
                                     p_ci_excludes_zero_expression_first=tally["ci_neg"] / realisations,
                                     median_ci_width_steps=float(np.median(tally["widths"]))))
        results[shape_name] = rows
    return dict(realisations=realisations, seed=seed, lead_factors=LEAD_FACTORS,
                noise={k: dict(bandwidth_log_steps=float(v["bandwidth"]), common_shock_sd=float(v["common"].std()),
                               item_sd=float(v["item"].std())) for k, v in fits.items()},
                observed=[dict(threshold=t, **o) for t, o in zip(THRESHOLDS, observed)],
                simulations=results)


def positive_control(root):
    """Explicit-gender decoding vs expression: an ordering the pipeline should resolve if resolution allows."""
    files = sorted(Path(root, "results/checkpoints").glob("step*/amendment-01.json"))
    recs = sorted((json.loads(p.read_text()) for p in files), key=lambda d: d["step"])
    steps, _, expr, labels = load_curves(root)
    gender = np.array([r["per_gender_pair_accuracy"] for r in recs])
    gdraws = np.random.default_rng(20260921).integers(0, 10, (2000, 10))
    odraws = bootstrap_indices(labels)
    zg, _ = normalise(gender.mean(1)); ze, _ = normalise(expr.mean(1))
    bg, vg = normalise(gender[:, gdraws].mean(2)); be, ve = normalise(expr[:, odraws].mean(2))
    valid = vg & ve
    out = []
    for threshold in THRESHOLDS:
        cg, ce = crossing(steps, zg, threshold), crossing(steps, ze, threshold)
        gi = np.argmax(bg[:, valid] >= threshold - 1e-12, axis=0); ei = np.argmax(be[:, valid] >= threshold - 1e-12, axis=0)
        lead = steps[ei] - steps[gi]
        out.append(dict(threshold=threshold, gender=cg["interval"], expression=ce["interval"],
                        ordering=relation(cg, ce).replace("association", "gender"),
                        lead_ci95=np.quantile(lead, [.025, .975]).tolist(), valid_fraction=float(valid.mean())))
    return out


def interpolated_crossing(steps, z, threshold):
    """First crossing time, linearly interpolated in log1p(step) between the bracketing checkpoints."""
    above = np.flatnonzero(z >= threshold)
    if not len(above):
        return np.nan
    i = above[0]
    if i == 0:
        return 0.
    u0, u1 = np.log1p(steps[i - 1]), np.log1p(steps[i])
    frac = (threshold - z[i - 1]) / (z[i] - z[i - 1])
    return float(np.expm1(u0 + frac * (u1 - u0)))


def prospective_v2(root, seeds=10, occupations_per_class=52, realisations=400, seed=20260929):
    """Power of the planned v2 primary analysis: per-seed log-ratio of interpolated crossing times,
    mean over seeds, one-sample t 95% CI. Uses 1.4B v1 residual noise (accuracy-based association, conservative),
    occupation columns resampled within class to the v2 count, and no extra between-seed timing variation."""
    from scipy import stats
    steps, assoc, expr, labels = load_curves(root)
    fits = {"association": fit_measure(steps, assoc), "expression": fit_measure(steps, expr)}
    shape = shape_function(steps, expr, fits["expression"]["bandwidth"])
    groups = [np.flatnonzero(labels == c) for c in (0, 1)]
    rng = np.random.default_rng(seed)
    tcrit = stats.t.ppf(.975, seeds - 1)
    rows = []
    for direction in ("association_first", "expression_first"):
        for k in LEAD_FACTORS:
            if direction == "expression_first" and k == 1:
                continue
            truth_a = truth(steps, assoc, shape, k if direction == "association_first" else 1)
            truth_e = truth(steps, expr, shape, k if direction == "expression_first" else 1)
            hits = {t: [0, 0] for t in THRESHOLDS}
            for _ in range(realisations):
                leads = {t: [] for t in THRESHOLDS}
                for _ in range(seeds):
                    cols = np.concatenate([rng.choice(g, occupations_per_class) for g in groups])
                    s_step = rng.choice([-1., 1.], (len(steps), 1)); s_item = rng.choice([-1., 1.], (len(steps), len(cols)))
                    sa = truth_a[:, cols] + s_step * fits["association"]["common"] + s_item * fits["association"]["item"][:, cols]
                    se = truth_e[:, cols] + s_step * fits["expression"]["common"] + s_item * fits["expression"]["item"][:, cols]
                    za, _ = normalise(sa.mean(1)); ze, _ = normalise(se.mean(1))
                    for t in THRESHOLDS:
                        ta, te = interpolated_crossing(steps, za, t), interpolated_crossing(steps, ze, t)
                        leads[t].append(np.log1p(te) - np.log1p(ta))
                for t in THRESHOLDS:
                    v = np.array(leads[t]); m, se_ = v.mean(), v.std(ddof=1) / np.sqrt(seeds)
                    hits[t][0] += m - tcrit * se_ > 0
                    hits[t][1] += m + tcrit * se_ < 0
            for t in THRESHOLDS:
                rows.append(dict(direction=direction, lead_factor=k, threshold=t,
                                 p_association_first=hits[t][0] / realisations, p_expression_first=hits[t][1] / realisations))
    return dict(seeds=seeds, occupations_per_class=occupations_per_class, realisations=realisations, rows=rows)
