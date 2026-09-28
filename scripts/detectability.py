"""Exploratory post-hoc detectability analysis: python scripts/detectability.py [--realisations N]"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mech.detectability import positive_control, simulate  # noqa: E402


def summary_lines(result):
    lines = ["## Summary", ""]
    for shape, rows in result["simulations"].items():
        a = [r for r in rows if r["direction"] == "association_first" and r["lead_factor"] > 1]
        e = [r for r in rows if r["direction"] == "expression_first"]
        null = [r for r in rows if r["lead_factor"] == 1]
        best_a = max(a, key=lambda r: r["p_ci_excludes_zero_association_first"])
        best_e = max(e, key=lambda r: r["p_ci_excludes_zero_expression_first"])
        fp_ci = max(max(r["p_ci_excludes_zero_association_first"], r["p_ci_excludes_zero_expression_first"]) for r in null)
        fp_br = max(max(r["p_bracket_association_first"], r["p_bracket_expression_first"]) for r in null)
        lines.append(f"- **{shape.capitalize()} shape as the truth.** The highest power to resolve an association-first lead with the paired CI is "
                     f"{best_a['p_ci_excludes_zero_association_first']:.0%} (k = {best_a['lead_factor']:g}, {best_a['threshold']:.0%} threshold). "
                     f"For an expression-first lead it is {best_e['p_ci_excludes_zero_expression_first']:.0%} (k = {best_e['lead_factor']:g}, {best_e['threshold']:.0%}). "
                     f"Under no lead, the CI excludes zero in at most {fp_ci:.0%} of realisations, while the bracket rule reports a spurious ordering in up to {fp_br:.0%}.")
    lines += ["- The v1 design is therefore poorly powered to detect a representation-first lead of any size tested (up to 16× on log-time). "
              "Its overlapping-bracket result should be read as *not informative about moderate leads*, not as evidence of co-emergence.",
              "- On Pythia's checkpoint grid, the largest resolvable lead at a threshold is bounded by the later measure's crossing time. "
              "Expression crosses 25% near step 660, and there is no checkpoint between 512 and 1,000.",
              "- The point-estimate bracket ordering is not a controlled test. Only bootstrap-based comparisons should carry an ordering claim.",
              "- Power is asymmetric because the association curve (accuracy of six templates per occupation) is noisier relative to its range than the expression curve.",
              "- Positive control: explicit-gender decoding visibly precedes expression, and the 75% bootstrap CI excludes zero. The bracket rule still reports overlap at every threshold, because adjacent brackets share an endpoint."]
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--realisations", type=int, default=500)
    parser.add_argument("--report-only", action="store_true", help="rebuild figure and markdown from saved JSON")
    args = parser.parse_args()
    out = ROOT / "reports"
    if args.report_only:
        result = json.loads((out / "detectability.json").read_text())
        args.realisations = result["realisations"]
    else:
        result = simulate(ROOT, args.realisations)
        result["positive_control_gender_vs_expression"] = positive_control(ROOT)
        (out / "detectability.json").write_text(json.dumps(result, indent=2, allow_nan=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained", sharey=True)
    for row, shape in enumerate(("expression", "association")):
        rows = result["simulations"][shape]
        for col, threshold in enumerate((.25, .5, .75)):
            ax = axes[row, col]
            for direction, key, color in [("association_first", "p_ci_excludes_zero_association_first", "#196a8c"),
                                          ("expression_first", "p_ci_excludes_zero_expression_first", "#ae5032")]:
                sel = [r for r in rows if r["threshold"] == threshold and (r["direction"] == direction or r["lead_factor"] == 1)]
                sel.sort(key=lambda r: r["lead_factor"])
                ax.plot([r["lead_factor"] for r in sel], [r[key] for r in sel], "o-", color=color, ms=3,
                        label=f"{direction.replace('_', ' ')}: CI excludes 0")
                bkey = "p_bracket_" + direction
                ax.plot([r["lead_factor"] for r in sel], [r[bkey] for r in sel], "o:", color=color, ms=2, alpha=.6,
                        label=f"{direction.replace('_', ' ')}: brackets separate")
            ax.axhline(.8, color="grey", lw=.8, ls="--"); ax.axhline(.05, color="grey", lw=.6, ls=":")
            ax.set_xscale("log", base=2); ax.set_ylim(0, 1)
            ax.set_title(f"{threshold:.0%} threshold · {shape} shape", loc="left", fontsize=10)
            if row == 1: ax.set_xlabel("True lead factor k (earlier measure reaches each level k× sooner)")
            if col == 0: ax.set_ylabel("Proportion of realisations")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Detectability of timing leads under the Amendment 01 pipeline ({args.realisations} noise realisations)")
    fig.savefig(out / "detectability.png", dpi=170); plt.close(fig)

    lines = ["# Detectability of timing leads (exploratory, post hoc)", "",
             "**Question:** if one measure truly led the other, would the Amendment 01 pipeline have detected it?", "",
             "This analysis was run on 2026-09-28, after all results were known. It is a power analysis of the existing design. It does not re-test the hypothesis.", "",
             "## Method", "",
             "- Each measure's per-occupation curves are smoothed with a local-linear smoother in log(1+step); the bandwidth is chosen by leave-one-out CV.",
             "- Leverage-corrected residuals are split into a checkpoint-wide shock (shared by all occupations) and an item part.",
             "- Both synthetic measures share one true shape (the smoothed expression curve; the association curve as a sensitivity check), each with its own per-occupation floor and amplitude.",
             "- A lead is created by time rescaling: the earlier measure reaches every level k× sooner. Step 0 stays at the initialisation floor.",
             "- New noise realisations re-sign the real residuals (Rademacher), using the same signs for both measures at each checkpoint and occupation.",
             "- The unchanged amendment pipeline is then applied: endpoint normalisation, first-crossing brackets, and a paired stratified occupation bootstrap with the same 2,000 draws.",
             f"- Realisations per condition: {args.realisations}.", "",
             "Calibration check: at k = 1, the simulated median bootstrap CI widths are comparable to those observed in the real study (see JSON).", "",
             *summary_lines(result), "",
             "![Detectability](detectability.png)", ""]
    for shape in ("expression", "association"):
        lines += [f"## Power with the {shape} shape as the truth", "",
                  "| k | Threshold | True lead (steps) | P(CI excludes 0, correct direction) | P(brackets separate, correct direction) | P(brackets separate, wrong direction) | Median CI width |",
                  "|---:|---:|---:|---:|---:|---:|---:|"]
        for r in result["simulations"][shape]:
            if r["lead_factor"] == 1:
                correct = max(r["p_ci_excludes_zero_association_first"], r["p_ci_excludes_zero_expression_first"])
                lines.append(f"| 1 (null) | {r['threshold']:.0%} | 0 | {correct:.2f} (false positive, either direction) | {r['p_bracket_association_first']:.2f} (assoc) | {r['p_bracket_expression_first']:.2f} (expr) | {r['median_ci_width_steps']:.0f} |")
                continue
            first = r["direction"]; other = "expression_first" if first == "association_first" else "association_first"
            lines.append(f"| {r['lead_factor']:g} ({first.split('_')[0]} first) | {r['threshold']:.0%} | {r['true_lead_steps']:.0f} | "
                         f"{r['p_ci_excludes_zero_' + first]:.2f} | {r['p_bracket_' + first]:.2f} | {r['p_bracket_' + other]:.2f} | {r['median_ci_width_steps']:.0f} |")
        lines.append("")
    lines += ["## Positive control: explicit-gender decoding vs expression (real data)", "",
              "| Threshold | Gender bracket | Expression bracket | Bracket rule | Gender lead, 95% bootstrap CI (steps) |", "|---:|---|---|---|---|"]
    for r in result["positive_control_gender_vs_expression"]:
        lines.append(f"| {r['threshold']:.0%} | {r['gender']} | {r['expression']} | {r['ordering']} | {r['lead_ci95']} |")
    lines += ["", "The two measures use different bootstrap units (word pairs vs occupations), so these draws are independent rather than paired.", ""]
    (out / "DETECTABILITY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"written": ["reports/detectability.json", "reports/DETECTABILITY.md", "reports/detectability.png"]}))


if __name__ == "__main__":
    main()
