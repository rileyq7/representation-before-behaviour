"""Post-inspection amendment: shared normalised onsets and matched controls."""
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .probes import classifier, direct_probe, direct_splits
from .metrics import bootstrap_indices

THRESHOLDS = [.25,.50,.75]


def correct(y, margin):
    return np.where(margin == 0,.5,(margin > 0) == np.asarray(y)).astype(float)


def normalise(values):
    values=np.asarray(values,float)
    delta=values[-1]-values[0]
    valid=np.isfinite(delta) & (delta > 1e-8)
    with np.errstate(divide="ignore",invalid="ignore"):
        scaled=(values-values[0])/delta
    return np.where(valid,scaled,np.nan),valid


def crossing(steps, values, threshold, sustained=1):
    if not np.isfinite(values).all():
        return {"status":"undefined_range","interval":None}
    for i in range(len(steps)-sustained+1):
        if np.all(np.asarray(values[i:i+sustained]) >= threshold-1e-12):
            return {"status":"crossed","interval":[int(steps[max(0,i-1)]),int(steps[i])],"first_observed_step":int(steps[i])}
    return {"status":"not_crossed","interval":[int(steps[-1]),None]}


def relation(a,b):
    if a["status"] != "crossed" or b["status"] != "crossed":
        return "unresolved"
    if a["interval"][1] < b["interval"][0]: return "association_before_expression"
    if b["interval"][1] < a["interval"][0]: return "expression_before_association"
    return "overlapping_brackets"


def balanced_control_labels(rows, split_seed, permutation_seed):
    y=np.array([r["label"] for r in rows])
    result=y.copy()
    rng=np.random.default_rng(10000+100*split_seed+permutation_seed)
    # Every third joint split starts a new occupation fold.
    for _,test in list(direct_splits(rows,split_seed))[::3]:
        subjects=sorted(set(rows[i]["subject"] for i in test))
        shuffled=np.array([next(r["label"] for r in rows if r["subject"]==s) for s in subjects])
        rng.shuffle(shuffled)
        for subject,label in zip(subjects,shuffled):
            for i,r in enumerate(rows):
                if r["subject"]==subject: result[i]=label
    return result


def gender_margins(x,rows):
    pred=np.full(len(rows),np.nan)
    y=np.array([r["label"] for r in rows])
    for pairs in ({0,1},{2,3},{4,5},{6,7},{8,9}):
        for templates in ({0,1},{2,3},{4,5}):
            train=[i for i,r in enumerate(rows) if r["pair"] not in pairs and r["template"] not in templates]
            test=[i for i,r in enumerate(rows) if r["pair"] in pairs and r["template"] in templates]
            pred[test]=classifier().fit(x[train],y[train]).decision_function(x[test])
    assert np.isfinite(pred).all()
    return pred


@threadpool_limits.wrap(limits=2)
def analyse_checkpoint(folder, amendment_sha):
    folder=Path(folder)
    target=folder/"amendment-01.json"
    if target.exists():
        previous=json.loads(target.read_text())
        if previous["amendment_sha256"] != amendment_sha:
            raise RuntimeError("Amendment specification changed; preserve old analysis before rerunning")
        return previous
    summary=json.loads((folder/"summary.json").read_text())
    rows=[json.loads(x) for x in (folder/"matched/rows.jsonl").read_text().splitlines()]
    occ=[r for r in rows if r["kind"]=="occupation"]
    gender=[r for r in rows if r["kind"]=="gender"]
    with np.load(folder/"matched/activations.npz") as f: x=f["layer_12"]
    ox=x[[i for i,r in enumerate(rows) if r["kind"]=="occupation"]]
    gx=x[[i for i,r in enumerate(rows) if r["kind"]=="gender"]]
    y=np.array([r["label"] for r in occ])
    with np.load(folder/"probes.npz") as p: real=p["layer_12_direct"]
    control_margins=[]; control_targets=[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",ConvergenceWarning)
        for split_seed in (0,1,2):
            margins=[]; labels=[]
            for perm in range(20):
                yp=balanced_control_labels(occ,split_seed,perm)
                margins.append(direct_probe(ox,occ,split_seed,yp)); labels.append(yp)
            control_margins.append(margins); control_targets.append(labels)
        gp=gender_margins(gx,gender)
    control_margins=np.array(control_margins); control_targets=np.array(control_targets)
    real_correct=correct(y,real)
    null_correct=correct(control_targets,control_margins)
    per_item_selectivity=real_correct.mean(0)-null_correct.mean((0,1))
    names=summary["occupations"]
    per_occ=lambda v: np.array([np.asarray(v)[[i for i,r in enumerate(occ) if r["subject"]==s]].mean() for s in names])
    selectivity=per_occ(per_item_selectivity)
    gy=np.array([r["label"] for r in gender]); gc=correct(gy,gp)
    pair_acc=np.array([gc[[i for i,r in enumerate(gender) if r["pair"]==pair]].mean() for pair in range(10)])
    assert np.isclose(pair_acc.mean(),summary["probes"]["layer_12"]["gender_validation"]["accuracy"])
    boot=bootstrap_indices(np.array(summary["occupation_labels"]))
    result=dict(step=summary["step"],amendment_sha256=amendment_sha,
                real_accuracy=float(real_correct.mean()),control_accuracy=float(null_correct.mean()),
                control_accuracy_by_split_and_permutation=null_correct.mean(-1).tolist(),
                selectivity=float(selectivity.mean()),selectivity_ci95=np.quantile(selectivity[boot].mean(1),[.025,.975]).tolist(),
                per_occupation_selectivity=selectivity.tolist(),per_occupation_direct_accuracy=per_occ(real_correct.mean(0)).tolist(),
                per_gender_pair_accuracy=pair_acc.tolist(),convergence_warnings=[str(w.message) for w in caught])
    np.savez_compressed(folder/"amendment-01-predictions.npz",control_margins=control_margins,
                        control_targets=control_targets,real_margins=real,gender_margins=gp)
    tmp=target.with_suffix(".tmp"); tmp.write_text(json.dumps(result,indent=2)); tmp.replace(target)
    return result


def generate_report(root):
    from .analysis import ALL_STEPS
    root=Path(root)
    files=sorted((root/"results/checkpoints").glob("step*/amendment-01.json"))
    records=[json.loads(p.read_text()) for p in files]
    originals=[json.loads((p.parent/"summary.json").read_text()) for p in files]
    steps=[r["step"] for r in records]
    out=root/"reports"; out.mkdir(exist_ok=True)
    if not steps or steps[0]!=0 or steps[-1]!=143000:
        return {"status":"awaiting_endpoints","completed":len(steps)}
    occ_draws=bootstrap_indices(np.array(originals[0]["occupation_labels"]))
    gender_draws=np.random.default_rng(20260921).integers(0,10,(2000,10))
    raw={
        "gender":np.array([r["per_gender_pair_accuracy"] for r in records]),
        "association":np.array([r["per_occupation"]["transfer_accuracy"] for r in originals]),
        "expression":np.array([r["per_occupation"]["behaviour"] for r in originals]),
        "direct_accuracy":np.array([r["per_occupation_direct_accuracy"] for r in records]),
        "selectivity":np.array([r["per_occupation_selectivity"] for r in records]),
    }
    result=dict(status="complete" if steps==ALL_STEPS else "partial_observed_grid",steps=steps,
                amendment_sha256=records[0]["amendment_sha256"],curves={},thresholds=[])
    boot_curves={}
    for name,values in raw.items():
        mean=values.mean(1); z,valid=normalise(mean)
        draws=gender_draws if name=="gender" else occ_draws
        bz,bvalid=normalise(values[:,draws].mean(2)); boot_curves[name]=(bz,bvalid)
        ci=np.quantile(bz[:,bvalid],[.025,.975],axis=1).T.tolist() if bvalid.any() else None
        result["curves"][name]=dict(raw=mean.tolist(),floor=float(mean[0]),endpoint=float(mean[-1]),
                                 endpoint_range=float(mean[-1]-mean[0]),normalised=z.tolist() if valid else None,
                                 valid_bootstrap_fraction=float(bvalid.mean()),ci95_pointwise=ci)
    for threshold in THRESHOLDS:
        cross={name:crossing(steps,c["normalised"] if c["normalised"] is not None else [np.nan]*len(steps),threshold)
               for name,c in result["curves"].items()}
        sustained={name:crossing(steps,c["normalised"] if c["normalised"] is not None else [np.nan]*len(steps),threshold,3)
                   for name,c in result["curves"].items()}
        ab,av=boot_curves["association"]; eb,ev=boot_curves["expression"]
        valid=av & ev
        # Endpoints are 0 and 1, so every valid replicate crosses each threshold.
        aidx=np.argmax(ab[:,valid]>=threshold-1e-12,axis=0); eidx=np.argmax(eb[:,valid]>=threshold-1e-12,axis=0)
        lead=np.asarray(steps)[eidx]-np.asarray(steps)[aidx]
        result["thresholds"].append(dict(threshold=threshold,crossings=cross,sustained_three=sustained,
             ordering=relation(cross["association"],cross["expression"]),
             direct_ordering=relation(cross["direct_accuracy"],cross["expression"]),
             selectivity_ordering=relation(cross["selectivity"],cross["expression"]),
             paired_bootstrap_valid_fraction=float(valid.mean()),
             observed_grid_lead_steps_ci95=np.quantile(lead,[.025,.975]).tolist() if len(lead) else None))
    result["threshold_stable_ordering"]=len(set(r["ordering"] for r in result["thresholds"]))==1
    (out/"amendment-01.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,8),layout="constrained")
    for name,label,color in [("gender","Explicit gender","#6750a4"),("association","Occupation association (transfer)","#196a8c"),("expression","Expression (signed logit difference)","#ae5032")]:
        c=result["curves"][name]
        if c["normalised"] is None: continue
        axes[0].plot(steps,c["normalised"],"o-",label=label,color=color,ms=3)
    for t in THRESHOLDS: axes[0].axhline(t,color="grey",ls=":",lw=.8)
    axes[0].set_ylabel("Fraction of step-0 → final range"); axes[0].legend(fontsize=8)
    axes[0].set_title("Identical 25%, 50%, 75% onset criteria",loc="left")
    axes[0].set_xlim(0,143000)
    for key,label,color in [("real_accuracy","Real occupation-label accuracy","#196a8c"),("control_accuracy","Mean shuffled-label accuracy","#999999"),("selectivity","Selectivity (real − control)","#ae5032")]:
        axes[1].plot(steps,[r[key] for r in records],"o-",ms=3,label=label,color=color)
    axes[1].set_ylabel("Accuracy / accuracy difference"); axes[1].legend(fontsize=8)
    axes[1].set_xlabel("Training step"); axes[1].set_xlim(0,143000)
    fig.suptitle(f"Amendment 01 · {len(steps)}/154 checkpoints · {result['status']}")
    fig.savefig(out/"amendment-01.png",dpi=180); fig.savefig(out/"amendment-01.pdf"); plt.close(fig)
    lines=["# Shared onset criteria and selectivity", "", f"**{result['status']}: {len(steps)}/154 checkpoints.**", "",
           "Post-inspection amendment; original protocol and results retained. Crossing brackets are point estimates; interpret their ordering alongside bootstrap uncertainty and checkpoint resolution.","",
           "![Normalised curves and selectivity](amendment-01.png)","",
           "| Range crossed | Explicit gender | Association (transfer) | Expression | Ordering |",
           "|---:|---|---|---|---|"]
    for row in result["thresholds"]:
        def fmt(name):
            c=row["crossings"][name]
            return str(c["interval"]) if c["status"]=="crossed" else c["status"]
        lines.append(f"| {row['threshold']:.0%} | {fmt('gender')} | {fmt('association')} | {fmt('expression')} | {row['ordering']} |")
    lines += ["", "Brackets are first observed crossings, not confidence intervals. Do not interpolate missing checkpoints. Endpoint uncertainty is propagated in the bootstrap; see JSON for valid-denominator fractions, pointwise intervals, paired onset differences, and identical sustained-crossing sensitivity.","",
              "| Step | Real occupation probe | Mean control | Selectivity (pp) | 95% interval (pp) |","|---:|---:|---:|---:|---|"]
    for r in records:
        lo,hi=np.array(r["selectivity_ci95"])*100
        lines.append(f"| {r['step']} | {r['real_accuracy']:.3f} | {r['control_accuracy']:.3f} | {100*r['selectivity']:.2f} | [{lo:.2f}, {hi:.2f}] |")
    lines += ["", "Shuffled controls preserve each fold's class balance and occupation labels across templates. They test arbitrary-label decoding capacity, not all structured semantic confounds. All uncertainty conditions on fitted probes.","",
              "An earlier expression crossing is compatible with nonlinear/distributed encoding but does not establish it. Robustness across thresholds is not causal proof. See [interpretation commitments](../AMENDMENT-01.md)."]
    (out/"AMENDMENT-01.md").write_text("\n".join(lines)+"\n")
    return result
