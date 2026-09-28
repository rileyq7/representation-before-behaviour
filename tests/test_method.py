import numpy as np
from pathlib import Path
from mech.data import matched_rows, apple_rows
from mech.probes import direct_splits, permuted_labels, score
from mech.metrics import jsd_parts, bootstrap_indices
from mech.analysis import ALL_STEPS, onset, compare_onsets, simultaneous_band

ROOT = Path(__file__).resolve().parents[1]


def test_data_balance_and_no_gender_leakage():
    rows = matched_rows(ROOT / "data/raw")
    occ = [r for r in rows if r["kind"] == "occupation"]
    assert len(occ) == 240 and sum(r["label"] for r in occ) == 120
    assert len(set(r["subject"] for r in occ)) == 40
    for r in occ:
        assert not set(r["prompt"].lower().split()) & {"he", "she", "male", "female"}


def test_joint_holdouts_cover_each_item_once():
    rows = [r for r in matched_rows(ROOT / "data/raw") if r["kind"] == "occupation"]
    for seed in [0, 1, 2]:
        covered = np.zeros(len(rows), dtype=int)
        for train, test in direct_splits(rows, seed):
            assert not {rows[i]["subject"] for i in train} & {rows[i]["subject"] for i in test}
            assert not {rows[i]["template"] for i in train} & {rows[i]["template"] for i in test}
            covered[test] += 1
        assert np.all(covered == 1)


def test_apple_construction_and_pairing():
    rows = apple_rows(ROOT / "data/raw", limit_pairs=16)
    assert len(rows) == 128
    assert sum(r["label"] == 2 for r in rows) == 64
    assert sum(r["label"] == 0 for r in rows) == 32
    assert sum(r["label"] == 1 for r in rows) == 32
    assert all(r["prompt"].endswith("gender is") for r in rows)
    full = apple_rows(ROOT / "data/raw", option_seeds=range(5))
    assert len(full) == 7920


def test_jsd_matches_known_cases():
    assert np.allclose(jsd_parts([[1,0,0]], [0]), 0)
    assert np.isclose(jsd_parts([[0,1,0]], [0]).sum(), np.log(2))
    assert np.isfinite(jsd_parts([[0,0,1]], [0])).all()


def test_permutations_preserve_group_and_balance():
    rows = matched_rows(ROOT / "data/raw")
    y = permuted_labels(rows, 10)
    assert y.sum() == len(y) // 2
    for s in {r["subject"] for r in rows}:
        assert len(set(y[i] for i, r in enumerate(rows) if r["subject"] == s)) == 1


def test_ties_are_chance_not_gender_prior():
    assert score([0,1], np.zeros(2))["accuracy"] == .5


def test_checkpoints_and_sustained_onset():
    assert len(ALL_STEPS) == 154 and len(set(ALL_STEPS)) == 154
    assert onset([0,1000,2000,3000,4000], [False,True,False,True,True])["status"] == "right_censored"
    assert onset([0,1000,2000,3000], [False,True,True,True])["interval"] == [0,1000]
    assert onset([0,1000], [True,True])["status"] == "left_censored"


def test_lead_intervals_and_pairing():
    r = {"status":"detected", "interval":[1000,2000]}
    b = {"status":"detected", "interval":[4000,5000]}
    assert compare_onsets(r,b)["lead_steps_interval"] == [2000,4000]
    assert "no resolved" in compare_onsets(r,r)["conclusion"]
    labels = np.array([0,0,1,1])
    vals = np.array([[.2,.4,.6,.8],[.2,.4,.6,.8]])
    lo,hi = simultaneous_band(vals, labels, baseline=True)
    assert np.all(lo == 0) and np.all(hi == 0)
    draws = bootstrap_indices(labels, n=10)
    assert np.all(labels[draws].sum(1) == 2)
