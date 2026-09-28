"""Protocol v2 report: python scripts/report_v2.py  (after `modal run scripts/modal_v2.py --fetch`)."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mech import analysis_v2 as a  # noqa: E402
from mech.amendment import normalise  # noqa: E402


def short(tag):
    return tag.replace("pythia-410m-", "").replace("pythia-", "")


def fmt(r):
    excl = r.get("exclusions") or {}
    tail = f"; excluded {', '.join(f'{short(k)} ({v})' for k, v in excl.items())}" if excl else ""
    if r.get("ci95") is None:
        return f"n={r['n']}; {r['decision'].replace('_', ' ')}{tail}"
    return (f"{r['ratio']:.2f}x [{r['ratio_ci95'][0]:.2f}, {r['ratio_ci95'][1]:.2f}]; n={r['n']}; "
            f"{r['decision'].replace('_', ' ')}{tail}")


def figure(data, result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    ax = axes[0, 0]
    for tag in a.SEEDS:
        if len(data.get(tag, [])) != 44:
            continue
        steps = np.array([r["step"] for r in data[tag]]); c = a.curves(data[tag], "combined")
        ax.plot(steps + 1, normalise(c["association_margin"].mean(1))[0], color="#196a8c", alpha=.45, lw=1)
        ax.plot(steps + 1, normalise(c["expression"].mean(1))[0], color="#ae5032", alpha=.45, lw=1)
    ax.plot([], [], color="#196a8c", label="Association (layer-12 transfer margin)")
    ax.plot([], [], color="#ae5032", label="Expression (signed logit difference)")
    ax.axhline(.5, color="grey", ls=":", lw=.8); ax.set_xscale("log"); ax.set_xlim(1, 1.5e5); ax.set_ylim(-.3, 1.4)
    ax.set_title("Pythia-410M, 10 seeds: normalised curves", loc="left", fontsize=10)
    ax.set_xlabel("Training step + 1"); ax.set_ylabel("Fraction of step-0 to final range"); ax.legend(fontsize=8)
    ax = axes[0, 1]
    for key, label, color in [("6_layer_sweep", "Transfer margin", "#196a8c"), ("7_logit_lens", "Logit lens", "#6750a4")]:
        rows = result["secondary"][key]
        ls = [int(l) for l, r in rows.items() if r.get("ci95")]
        ax.errorbar(ls, [rows[str(l)]["ratio"] for l in ls],
                    yerr=np.array([[rows[str(l)]["ratio"] - rows[str(l)]["ratio_ci95"][0], rows[str(l)]["ratio_ci95"][1] - rows[str(l)]["ratio"]] for l in ls]).T,
                    fmt="o-", ms=3, color=color, label=label, capsize=2)
    ax.axhline(1, color="grey", lw=.8); ax.set_yscale("log")
    ax.set_title("Lead ratio vs expression at 50%, by layer (>1 = earlier)", loc="left", fontsize=10)
    ax.set_xlabel("Layer"); ax.set_ylabel("exp(mean log-ratio), t-CI over seeds"); ax.legend(fontsize=8)
    ax = axes[1, 0]
    for tag in a.SEEDS:
        if len(data.get(tag, [])) != 44:
            continue
        steps = np.array([r["step"] for r in data[tag]]); c = a.curves(data[tag], "combined")
        ax.plot(steps + 1, c["ablation_effect"].mean(1), color="#2e7d32", alpha=.5, lw=1)
    ax.axhline(0, color="grey", lw=.8); ax.set_xscale("log"); ax.set_xlim(1, 1.5e5)
    ax.set_title("Gender-direction ablation effect on expression (minus random directions)", loc="left", fontsize=10)
    ax.set_xlabel("Training step + 1"); ax.set_ylabel("Δ signed logit difference")
    ax = axes[1, 1]
    labels, vals = [], []
    for name, r in [("Primary 50%", result["primary"])] + [(f"{t} threshold", result["secondary"]["1_thresholds"][t]) for t in ("0.25", "0.75")] + \
                   [("Accuracy measure 50%", result["secondary"]["2_accuracy_measure"]["0.5"]), ("WinoBias-only 50%", result["secondary"]["3_winobias_only"]["0.5"]),
                    ("Logit lens L12 50%", result["secondary"]["7_logit_lens"]["12"]), ("Ablation onset 50%", result["secondary"]["8_ablation"]["onset"]["0.5"])]:
        if r.get("ci95"):
            labels.append(name); vals.append((r["ratio"], *r["ratio_ci95"]))
    y = np.arange(len(vals))[::-1]
    for yi, (m, lo, hi) in zip(y, vals):
        ax.plot([lo, hi], [yi, yi], color="#333"); ax.plot(m, yi, "o", color="#333")
    ax.set_yticks(y, labels, fontsize=8); ax.axvline(1, color="grey", lw=.8); ax.set_xscale("log")
    ax.set_title("Lead ratios with 95% CI (>1 = before expression)", loc="left", fontsize=10)
    fig.savefig(out / "v2.png", dpi=170); plt.close(fig)


SUMMARY = [
    "## Summary", "",
    "- **Primary result: unresolved.** At the 50% threshold the mean lead ratio is 0.50x (95% CI 0.16 to 1.55, 7 seeds). "
    "The point estimate favours expression first, but the interval includes 1. Three seeds were excluded by preregistered rules: "
    "seeds 3 and 4 failed the explicit-gender decoding gate at layer 12 (0.64 and 0.63, required 0.8), and seed 9 is missing a checkpoint.",
    "- **The ordering depends on the threshold.** Association reaches 25% of its range first (1.31x, CI 1.03 to 1.66), "
    "while expression reaches 75% first (0.31x, CI 0.16 to 0.58). Association starts rising earlier; expression finishes sooner.",
    "- **Ablating the gender direction has a specific effect.** Removing it at layer 12 reduces stereotyped output more than every one of 10 random "
    "directions in 9 of 10 seeds, meeting the preregistered criterion. The ablation effect reaches 25% of its range before expression does (1.54x).",
    "- **Layer matters.** The transfer margin leads expression at layers 2 to 7 but not at the primary layer 12. "
    "The layer-12 logit lens leads expression (5.61x, CI 1.15 to 27.3). These layer results are descriptive and involve many comparisons.",
    "- **Pythia-1.4B** (one run): no interval excludes 1 at any threshold.",
    "- Secondary results are reported in full below, in the preregistered order. None of them replaces the primary result.",
]

DEVIATIONS = [
    "## Deviations and notes", "",
    "- `EleutherAI/pythia-410m-seed9` has no weights on its `step40000` branch (only tokenizer files), so that checkpoint could not be run. "
    "As preregistered, seed 9 is excluded from analyses that need the full grid. A post-hoc sensitivity analysis that keeps seed 9 on its "
    "remaining 43 steps gives the same conclusions (25%: 1.29x, CI 1.05 to 1.58; 50%: 0.54x, CI 0.20 to 1.40; 75%: 0.35x, CI 0.19 to 0.64). "
    "See `v2-sensitivity-seed9.json`.",
    "- The two smoke-test checkpoints (seed 1 and 1.4B at step 143,000) record protocol commit `c951ae6`; all others record `2303e75`. "
    "The only difference between those commits is the analysis code and a runner check, not the measurement code.",
    "- On synthetic data with a known 1.5x lead, the interpolated crossing estimator recovered 1.41x to 1.56x, so it can shrink ratios slightly toward 1.",
    "- The logit lens and ablation curves are noisier than the primary measures, and several seeds are excluded from them for nonpositive endpoint ranges.",
]


def main():
    data = a.load(ROOT)
    result = a.analyse(ROOT)
    out = ROOT / "reports"
    (out / "v2.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    figure(data, result, out)
    p, s = result["primary"], result["secondary"]
    abl = s["8_ablation"]["specificity"]
    lines = ["# Protocol v2 results", "",
             f"Preregistered in [PROTOCOL-v2.md](../PROTOCOL-v2.md) (git tag `v2-prereg`). Models loaded: {result['models']}.", "",
             "Lead ratio r = exp(mean over seeds of log(1+t_expression) - log(1+t_association)). "
             "r > 1 means association reaches the threshold first (r times sooner on a log-step clock). Intervals are one-sample t 95% CIs over seeds unless stated.", "",
             *SUMMARY, "",
             "## Primary (confirmatory)", "", f"**{p['description']}:** {fmt(p)}", ""]
    if p.get("per_seed"):
        lines.append("Per-seed log-ratios: " + ", ".join(f"{v:+.2f}" for v in p["per_seed"]) + f" (sign consistency {p['sign_consistency']:.0%}).")
    if p.get("hierarchical_bootstrap"):
        lines.append(f"Hierarchical bootstrap (seeds + occupations) ratio 95% interval: {p['hierarchical_bootstrap']['ratio_ci95'][0]:.2f} to {p['hierarchical_bootstrap']['ratio_ci95'][1]:.2f}.")
    lines += ["", "![v2 results](v2.png)", "", "## Secondary (prespecified order)", "", "| Analysis | Result |", "|---|---|"]
    for t, r in s["1_thresholds"].items(): lines.append(f"| 1. Threshold {float(t):.0%} | {fmt(r)} |")
    for t, r in s["2_accuracy_measure"].items(): lines.append(f"| 2. Transfer accuracy, {float(t):.0%} | {fmt(r)} |")
    for t, r in s["3_winobias_only"].items(): lines.append(f"| 3. WinoBias-only, {float(t):.0%} | {fmt(r)} |")
    for k, r in s.get("5_pythia_1.4b", {}).items():
        ci = r.get("ci95")
        lines.append(f"| 5. Pythia-1.4B {k} (occupation bootstrap) | " + (f"ratio {np.exp(r['lead']):.2f}x [{np.exp(ci[0]):.2f}, {np.exp(ci[1]):.2f}]" if ci and r.get("lead") is not None and np.isfinite(r["lead"]) else str(r)) + " |")
    lines.append(f"| 7. Logit lens layer 12 vs expression, 50% | {fmt(s['7_logit_lens']['12'])} |")
    for t, r in s["8_ablation"]["onset"].items(): lines.append(f"| 8. Ablation-effect onset, {float(t):.0%} | {fmt(r)} |")
    lines.append(f"| 8. Ablation specificity at 143k | gender direction beats all 10 random directions in {abl['seeds_exceeding_all_random']}/{abl['seeds_available']} seeds "
                 f"(criterion at least 8: {'met' if abl['specific'] else 'not met'}); mean excess effect {abl['final_effect_t']['mean']:.3f} logits "
                 f"[{abl['final_effect_t']['ci95'][0]:.3f}, {abl['final_effect_t']['ci95'][1]:.3f}] |")
    for t, r in s["9_selectivity"].items(): lines.append(f"| 9. Selectivity onset, {float(t):.0%} | {fmt(r)} |")
    lines += ["", "### 6 and 7. Layer profile (50% threshold, descriptive; no layer is selected)", "", "| Layer | Transfer margin | Logit lens |", "|---:|---|---|"]
    for l in range(25):
        lines.append(f"| {l} | {fmt(s['6_layer_sweep'][str(l)])} | {fmt(s['7_logit_lens'][str(l)])} |")
    lines += ["", "### 10. Continuity with v1 (WinoBias set, transfer accuracy; descriptive only)", "", "| Model | Threshold | Association | Expression | Bracket rule | Observed-grid lead CI |", "|---|---:|---|---|---|---|"]
    for tag, rows in s["10_v1_continuity"].items():
        for r in rows:
            lines.append(f"| {tag} | {r['threshold']:.0%} | {r['association']} | {r['expression']} | {r['bracket_rule_descriptive']} | {r['observed_grid_lead_ci95']} |")
    lines += ["", *DEVIATIONS]
    (out / "V2_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"primary": {k: p.get(k) for k in ("ratio", "ratio_ci95", "decision", "n")}}))


if __name__ == "__main__":
    main()
