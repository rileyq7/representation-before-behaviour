import numpy as np
from pathlib import Path
from mech.amendment import normalise,crossing,relation,balanced_control_labels,correct
from mech.probes import direct_splits
from mech.data import matched_rows


def test_normalisation_invariant_to_positive_scale_and_offset():
    values=np.array([.4,.5,.6,.8])
    a,valid=normalise(values); b,_=normalise(values*17+9)
    assert valid and np.allclose(a,[0,.25,.5,1]) and np.allclose(a,b)
    assert crossing([0,1,2,3],a,.5)["interval"]==[1,2]


def test_flat_and_reversed_ranges_are_not_forced_to_emerge():
    for x in [[1,1,1],[1,.7,.5]]:
        z,valid=normalise(x)
        assert not valid and np.isnan(z).all()
        assert crossing([0,1,2],z,.5)["status"]=="undefined_range"


def test_overshoots_and_threshold_reversals_preserved():
    a,_=normalise([0,.4,.4,.8,1]); b,_=normalise([0,.1,.7,.7,1])
    steps=[0,1,2,3,4]
    assert crossing(steps,a,.25)["first_observed_step"]<crossing(steps,b,.25)["first_observed_step"]
    assert crossing(steps,a,.5)["first_observed_step"]>crossing(steps,b,.5)["first_observed_step"]
    z,_=normalise([0,2,1]); assert z[1]==2
    assert crossing([0,1,2,3,4],[0,1,0,1,1],.5,3)["status"]=="not_crossed"


def test_each_control_preserves_exact_fold_balance_and_type_labels():
    raw=Path(__file__).resolve().parents[1]/"data/raw"
    rows=[r for r in matched_rows(raw) if r["kind"]=="occupation"]
    y=np.array([r["label"] for r in rows])
    for split in range(3):
        for perm in range(20):
            yp=balanced_control_labels(rows,split,perm)
            for train,test in direct_splits(rows,split):
                assert yp[train].sum()==y[train].sum()
                assert yp[test].sum()==y[test].sum()
            for s in {r["subject"] for r in rows}:
                assert len(set(yp[i] for i,r in enumerate(rows) if r["subject"]==s))==1


def test_control_scored_against_its_own_labels():
    target=np.array([1,0]); margin=np.array([1,-1])
    assert correct(target,margin).mean()==1
    assert correct(1-target,margin).mean()==0


def test_bootstrap_denominators_validated_separately():
    values=np.array([[0,1,2],[.2,1,1],[1,1,0]])
    z,valid=normalise(values)
    assert valid.tolist()==[True,False,False]
    assert np.allclose(z[:,0],[0,.2,1]) and np.isnan(z[:,1:]).all()


def test_report_uses_same_crossings_and_exposes_selectivity(tmp_path,monkeypatch):
    import json
    from mech.amendment import generate_report
    monkeypatch.setenv("MPLCONFIGDIR",str(tmp_path/"mpl"))
    for i,step in enumerate([0,1000,10000,143000]):
        folder=tmp_path/f"results/checkpoints/step{step:06d}"; folder.mkdir(parents=True)
        real=[.5,.6,.7,.8][i]; sel=real-.5
        record=dict(step=step,amendment_sha256="synthetic",real_accuracy=real,control_accuracy=.5,
                    selectivity=sel,selectivity_ci95=[sel,sel],per_occupation_selectivity=[sel]*40,
                    per_occupation_direct_accuracy=[real]*40,per_gender_pair_accuracy=[.5 if i==0 else 1]*10)
        original=dict(step=step,occupation_labels=[0]*20+[1]*20,
                      per_occupation={"transfer_accuracy":[[.4,.55,.7,.8][i]]*40,
                                      "behaviour":[[-.1,.2,.6,.7][i]]*40})
        (folder/"amendment-01.json").write_text(json.dumps(record))
        (folder/"summary.json").write_text(json.dumps(original))
    result=generate_report(tmp_path)
    assert result["status"]=="partial_observed_grid"
    assert [r["threshold"] for r in result["thresholds"]]==[.25,.5,.75]
    assert result["thresholds"][0]["crossings"]["association"]["interval"]==[0,1000]
    assert result["thresholds"][1]["crossings"]["expression"]["interval"]==[1000,10000]
    assert result["curves"]["selectivity"]["valid_bootstrap_fraction"]==1
    assert (tmp_path/"reports/AMENDMENT-01.md").exists()
