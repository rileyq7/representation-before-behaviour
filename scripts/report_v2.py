"""Protocol v2 report: python scripts/report_v2.py  (after `modal run scripts/modal_v2.py --fetch`)."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mech import analysis_v2 as a  # noqa: E402
from mech.amendment import normalise  # noqa: E402


def fmt(r):
    if r.get("ci95") is None:
        return f"n={r['n']}; {r['decision']}"
    return (f"{r['ratio']:.2f}× [{r['ratio_ci95'][0]:.2f}, {r['ratio_ci95'][1]:.2f}] · n={r['n']} · "
            f"{r['decision'].replace('_', ' ')}" + (f" · excluded: {r['exclusions']}" if r.get("exclusions") else ""))


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
    ax.set_xlabel("Training step + 1"); ax.set_ylabel("Fraction of step-0 → final range"); ax.legend(fontsize=8)
    ax = axes[0, 1]
    for key, label, color in [("6_layer_sweep", "Transfer margin", "#196a8c"), ("7_logit_lens", "Logit lens", "#6750a4")]:
        rows = result["secondary"][key]
        ls = [int(l) for l, r in rows.items() if r.get("ci95")]
        ax.errorbar(ls, [rows[str(l)]["ratio"] for l in ls],
                    yerr=np.array([[rows[str(l)]["ratio"] - rows[str(l)]["ratio_ci95"][0], rows[str(l)]["ratio_ci95"][1] - rows[str(l)]["ratio"]] for l in ls]).T,
                    fmt="o-", ms=3, color=color, label=label, capsize=2)
    ax.axhline(1, color="grey", lw=.8); ax.set_yscale("log")
    ax.set_title("Lead ratio vs expression at 50%, by layer (>1 = earlier than expression)", loc="left", fontsize=10)
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
             "Lead ratio r = exp(mean over seeds of log(1+t_expression) − log(1+t_association)). "
             "r > 1 means association reaches the threshold first (r× sooner on the log-step clock). Intervals are one-sample t 95% CIs over seeds unless stated.", "",
             "## Primary (confirmatory)", "", f"**{p['description']}:** {fmt(p)}", ""]
    if p.get("per_seed"):
        lines.append("Per-seed log-ratios: " + ", ".join(f"{v:+.2f}" for v in p["per_seed"]) + f" (sign consistency {p['sign_consistency']:.0%}).")
    if p.get("hierarchical_bootstrap"):
        lines.append(f"Hierarchical bootstrap (seeds + occupations) ratio 95% interval: {p['hierarchical_bootstrap']['ratio_ci95'][0]:.2f}–{p['hierarchical_bootstrap']['ratio_ci95'][1]:.2f}.")
    lines += ["", "![v2 results](v2.png)", "", "## Secondary (prespecified order)", "", "| Analysis | Result |", "|---|---|"]
    for t, r in s["1_thresholds"].items(): lines.append(f"| 1. Threshold {float(t):.0%} | {fmt(r)} |")
    for t, r in s["2_accuracy_measure"].items(): lines.append(f"| 2. Transfer accuracy, {float(t):.0%} | {fmt(r)} |")
    for t, r in s["3_winobias_only"].items(): lines.append(f"| 3. WinoBias-only, {float(t):.0%} | {fmt(r)} |")
    for k, r in s.get("5_pythia_1.4b", {}).items():
        ci = r.get("ci95")
        lines.append(f"| 5. Pythia-1.4B {k} (occupation bootstrap) | " + (f"ratio {np.exp(r['lead']):.2f}× [{np.exp(ci[0]):.2f}, {np.exp(ci[1]):.2f}]" if ci and r.get("lead") is not None and np.isfinite(r["lead"]) else str(r)) + " |")
    lines.append(f"| 7. Logit lens layer 12 vs expression, 50% | {fmt(s['7_logit_lens']['12'])} |")
    for t, r in s["8_ablation"]["onset"].items(): lines.append(f"| 8. Ablation-effect onset, {float(t):.0%} | {fmt(r)} |")
    lines.append(f"| 8. Ablation specificity at 143k | gender direction beats all 10 random directions in {abl['seeds_exceeding_all_random']}/{abl['seeds_available']} seeds "
                 f"(criterion ≥8: {'met' if abl['specific'] else 'not met'}); mean excess effect {abl['final_effect_t'].get('mean', float('nan')):.3f} "
                 f"{abl['final_effect_t'].get('ci95')} logits |")
    for t, r in s["9_selectivity"].items(): lines.append(f"| 9. Selectivity onset, {float(t):.0%} | {fmt(r)} |")
    lines += ["", "### 6–7. Layer profile (50% threshold, descriptive; no layer is selected)", "", "| Layer | Transfer margin | Logit lens |", "|---:|---|---|"]
    for l in range(25):
        lines.append(f"| {l} | {fmt(s['6_layer_sweep'][str(l)])} | {fmt(s['7_logit_lens'][str(l)])} |")
    lines += ["", "### 10. Continuity with v1 (WinoBias set, transfer accuracy; descriptive only)", "", "| Model | Threshold | Association | Expression | Bracket rule | Observed-grid lead CI |", "|---|---:|---|---|---|---|"]
    for tag, rows in s["10_v1_continuity"].items():
        for r in rows:
            lines.append(f"| {tag} | {r['threshold']:.0%} | {r['association']} | {r['expression']} | {r['bracket_rule_descriptive']} | {r['observed_grid_lead_ci95']} |")
    (out / "V2_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"primary": {k: p.get(k) for k in ("ratio", "ratio_ci95", "decision", "n")}}))


if __name__ == "__main__":
    main()
