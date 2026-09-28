"""Descriptive curves and conservative, prespecified onset comparisons."""
import json
import os
from pathlib import Path
import numpy as np
from .metrics import bootstrap_indices

ALL_STEPS = [0] + [2**i for i in range(10)] + list(range(1000, 143001, 1000))
COARSE_STEPS = [143000, 0, 1000, 10000, 20000, 40000, 60000, 70000, 80000, 90000, 100000, 120000]


def simultaneous_band(values, labels, baseline=False):
    """values: checkpoints x occupations; preserve paired draws across time."""
    values = np.asarray(values, float)
    if baseline:
        values = values - values[0]
    point = values.mean(1)
    draws = bootstrap_indices(labels)
    boots = values[:, draws].mean(2)
    radius = np.quantile(np.abs(boots - point[:, None]).max(0), .95)
    return point - radius, point + radius


def onset(steps, passed, sustained=3):
    if passed[0]:
        return {"status": "left_censored", "interval": [None, int(steps[0])]}
    for i in range(1, len(steps) - sustained + 1):
        if all(passed[i:i+sustained]):
            return {"status": "detected", "interval": [int(steps[i-1]), int(steps[i])]}
    return {"status": "right_censored", "interval": [int(steps[-1]), None]}


def compare_onsets(rep, beh):
    if rep["status"] != "detected" or beh["status"] != "detected":
        return {"conclusion": "inconclusive: at least one onset is censored"}
    rlo, rhi = rep["interval"]
    blo, bhi = beh["interval"]
    lead = [blo - rhi, bhi - rlo]
    if lead[0] > 0:
        conclusion = "representation precedes behaviour under prespecified operational criteria"
    elif lead[1] < 0:
        conclusion = "behaviour precedes representation under prespecified operational criteria"
    else:
        conclusion = "no resolved lead at checkpoint resolution; not proof of simultaneous emergence"
    return {"conclusion": conclusion, "lead_steps_interval": lead}


def report(root):
    root = Path(root)
    results = root / "results/checkpoints"
    summaries = sorted([json.loads(p.read_text()) for p in results.glob("step*/summary.json")], key=lambda d: d["step"])
    out = root / "reports"
    out.mkdir(exist_ok=True)
    if not summaries:
        return {"status": "no_completed_checkpoints"}
    steps = [d["step"] for d in summaries]
    labels = np.array(summaries[0]["occupation_labels"])
    names = summaries[0]["occupations"]
    assert all(d["occupations"] == names and d["occupation_labels"] == labels.tolist() for d in summaries)
    behaviour = np.array([d["per_occupation"]["behaviour"] for d in summaries])
    representation = np.array([d["per_occupation"]["transfer_accuracy"] for d in summaries])
    complete = steps == ALL_STEPS
    analysis = {"status": "complete_matched_sweep" if complete else "partial",
                "completed": len(steps), "planned": len(ALL_STEPS), "steps": steps,
                "missing_steps": sorted(set(ALL_STEPS) - set(steps)),
                "conclusion": "No timing conclusion: checkpoint sweep is incomplete."}
    if complete:
        rlow, _ = simultaneous_band(representation, labels)
        blow, _ = simultaneous_band(behaviour, labels)
        rdelta, _ = simultaneous_band(representation, labels, baseline=True)
        bdelta, _ = simultaneous_band(behaviour, labels, baseline=True)
        valid = np.array([d["probes"]["layer_12"]["gender_validation"]["accuracy"] > .8 for d in summaries])
        null_max = np.array([max(p["accuracy"] for p in d["probes"]["layer_12"]["transfer_null"]) for d in summaries])
        control = valid & (representation.mean(1) > null_max)
        def decide_rep(threshold):
            passed = (rlow > threshold) & (rdelta > 0) & control
            passed[0] = rlow[0] > threshold and control[0]
            return onset(steps, passed)
        def decide_beh(threshold):
            passed = (blow > threshold) & (bdelta > 0)
            passed[0] = blow[0] > threshold
            return onset(steps, passed)
        rep = decide_rep(.55)
        beh = decide_beh(.10)
        analysis.update(compare_onsets(rep, beh))
        analysis.update(representation_onset=rep, behaviour_onset=beh)
        analysis["sensitivity"] = []
        for rt in [.50, .55, .60]:
            for bt in [.0, .1, .2]:
                r = decide_rep(rt)
                b = decide_beh(bt)
                analysis["sensitivity"].append(dict(representation_threshold=rt, behaviour_threshold=bt,
                                                    representation_onset=r, behaviour_onset=b, **compare_onsets(r, b)))
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, layout="constrained")
    draws = bootstrap_indices(labels)
    for values, ax, title, color in [(representation, axes[0], "Gender-probe transfer to occupations (layer 12)", "#196a8c"),
                                   (behaviour, axes[1], "Behaviour: signed stereotype log-odds", "#ae5032")]:
        lo, hi = np.quantile(values[:, draws].mean(2), [.025, .975], axis=1)
        ax.plot(steps, values.mean(1), "o-", ms=3, color=color)
        ax.fill_between(steps, lo, hi, color=color, alpha=.15, label="Pointwise 95% occupation bootstrap")
        ax.set_ylabel("Accuracy" if ax is axes[0] else "Mean log-odds")
        ax.set_title(title, loc="left")
        ax.legend(fontsize=8)
    axes[0].axhline(.5, ls=":", color="gray")
    axes[1].axhline(0, ls=":", color="gray")
    apple = [d.get("apple_pilot", {}).get("female_minus_male_jsdp", np.nan) for d in summaries]
    axes[2].plot(steps, apple, "o-", color="#715a9a", ms=3)
    axes[2].set_title("Apple-style discovery subset: female − male correct-answer JSD-P", loc="left")
    axes[2].set_ylabel("JSD-P difference (nats)")
    axes[2].set_xlabel("Training step (2,097,152 tokens per step)")
    fig.suptitle(f"Pythia-1.4B · {len(steps)}/154 checkpoints · " + ("complete matched sweep" if complete else "PARTIAL — no timing conclusion"), fontsize=14)
    fig.savefig(out / "curves.png", dpi=180)
    fig.savefig(out / "curves.pdf")
    plt.close(fig)
    rows = ["| Step | Transfer accuracy | Direct probe accuracy | Signed log-odds | Gender validation |",
            "|---:|---:|---:|---:|---:|"]
    for d in summaries:
        p = d["probes"]["layer_12"]
        rows.append(f'| {d["step"]:,} | {p["transfer"]["accuracy"]:.3f} | {np.mean([x["accuracy"] for x in p["direct"]]):.3f} | {np.mean(d["per_occupation"]["behaviour"]):.3f} | {p["gender_validation"]["accuracy"]:.3f} |')
    text = f'''# Representation before behaviour: running report

**Status: {analysis["status"]}. {len(steps)}/154 matched checkpoints completed.**

{analysis["conclusion"]}

The Apple paper's reported ~80k transition is for Pythia-6.9B. This experiment studies Pythia-1.4B; its transition is not assumed. The matched occupation task, gender-direction transfer, and Apple-style pronoun-resolution task measure distinct constructs.

![Checkpoint curves](curves.png)

{"\n".join(rows)}

Primary probe: gender-word-trained linear classifier, tested on occupations with templates held out. Direct occupation-label probe is secondary. Intervals on plots are pointwise, not evidence of an onset; the full-sweep decision uses simultaneous bands, random initialization and label-permutation controls. Read [the protocol](../PROTOCOL.md) for thresholds and limitations.

The Apple curve uses a 128-prompt development discovery subset, not the full paper evaluation. Full reconstruction is separately saved when run. No causality claim follows from decodability. No result establishes that a representation was previously absent.

See `analysis.json`, per-checkpoint `summary.json`, `probes.npz`, `matched/outputs.npz`, and immutable checkpoint provenance for machine-readable evidence.
'''
    full_paths = sorted(results.glob("step*/apple-full/summary.json"))
    if full_paths:
        text += "\n## Full Apple-style reconstruction\n\nThe bracket was selected using the development discovery subset. The table below uses only held-out test sentences.\n\n| Step | Test prompts | Female − male JSD-P gap |\n|---:|---:|---:|\n"
        full_records = []
        from scipy.special import softmax
        from .metrics import jsd_parts
        for p in full_paths:
            full = json.loads(p.read_text())
            metrics = full["metrics"]["test_only"]
            text += f'| {full["step"]:,} | {metrics["n"]:,} | {metrics["female_minus_male_jsdp"]:.4f} |\n'
            items = [json.loads(line) for line in (p.parent / "rows.jsonl").read_text().splitlines()]
            logits = np.load(p.parent / "outputs.npz")["logits"][:,2:5]
            target = np.array([r["label"] for r in items])
            parts = jsd_parts(softmax(logits,axis=1), target)
            groups = sorted(set(r["pair_id"] for r in items if r["split"] == "test"))
            per_pair = []
            for group in groups:
                f = [i for i,r in enumerate(items) if r["pair_id"] == group and r["label"] == 1]
                m = [i for i,r in enumerate(items) if r["pair_id"] == group and r["label"] == 0]
                per_pair.append(float(parts[f,1].mean()-parts[m,0].mean()))
            full_records.append(dict(step=full["step"], groups=groups, gap_per_pair=per_pair))
        if len(full_records) >= 2:
            assert all(r["groups"] == full_records[0]["groups"] for r in full_records)
            rng = np.random.default_rng(20260921)
            n = len(full_records[0]["groups"])
            draws = rng.integers(0,n,size=(2000,n))
            changes = []
            for a,b in zip(full_records,full_records[1:]):
                delta = np.array(b["gap_per_pair"])-np.array(a["gap_per_pair"])
                changes.append(dict(from_step=a["step"],to_step=b["step"],mean=float(delta.mean()),
                                    ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()))
            atomic = dict(per_checkpoint=full_records, adjacent_changes=changes,
                          note="Paired sentence-pair bootstrap on test-only data; pointwise, post-selection descriptive intervals.")
            (out / "apple-confirmation.json").write_text(json.dumps(atomic,indent=2))
            text += "\nPaired test-sentence bootstrap changes are saved in `apple-confirmation.json`. These descriptive intervals do not establish a discontinuity or correct for bracket selection.\n"
    (out / "REPORT.md").write_text(text)
    return analysis
