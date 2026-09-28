"""Synthetic-curve checks of the preregistered v2 analysis (software validation only)."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mech import analysis_v2 as a  # noqa: E402


def logistic(steps, midpoint):
    return 1 / (1 + np.exp(-(np.log1p(steps) - np.log(midpoint)) / .3))


def fake_run(root, tag, lead_factor, rng, n=104):
    labels = [0] * (n // 2) + [1] * (n // 2)
    for step in a.STEPS:
        e = 0.8 * logistic(step, 1200) + rng.normal(0, .02, n)
        m = 1.5 * logistic(step * lead_factor, 1200) + rng.normal(0, .03, n)
        layer = lambda v: [list(v)] * 25
        s = dict(occupations=[f"o{i}" for i in range(n)], occupation_labels=labels, behaviour=list(e),
                 transfer_margin_by_layer=layer(m), transfer_accuracy_by_layer=layer(.5 + m / 4),
                 logit_lens_by_layer=layer(e), direct_accuracy=list(.5 + m / 4), control_accuracy=.5,
                 selectivity=list(m / 4), ablation_gender=list(e * .5), ablation_random=[list(e)] * 10)
        rec = dict(step=step, sets={"combined": s, "winobias": s}, per_gender_pair_accuracy_by_layer=[[1.0] * 10] * 25)
        path = Path(root, "results/v2", tag, f"step{step:06d}", "summary.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec))


def test_recovers_known_lead(tmp_path):
    rng = np.random.default_rng(0)
    for tag in a.SEEDS:
        fake_run(tmp_path, tag, 2.0, rng)
    result = a.lead_across_seeds(a.load(tmp_path), "combined", "association_margin")
    assert result["decision"] == "association_first"
    assert 1.6 < result["ratio"] < 2.5 and not result["exclusions"]


def test_null_is_unresolved_and_exclusions_counted(tmp_path):
    rng = np.random.default_rng(1)
    for tag in a.SEEDS[:6]:
        fake_run(tmp_path, tag, 1.0, rng)
    result = a.lead_across_seeds(a.load(tmp_path), "combined", "association_margin")
    assert len(result["exclusions"]) == 4
    assert result["decision"] == "inconclusive_too_many_exclusions"


def test_full_analysis_runs(tmp_path):
    rng = np.random.default_rng(2)
    for tag in a.SEEDS + ["pythia-1.4b"]:
        fake_run(tmp_path, tag, 1.0, rng)
    out = a.analyse(tmp_path)
    assert out["primary"]["decision"] == "unresolved"
    assert out["secondary"]["8_ablation"]["specificity"]["seeds_available"] == 10
    json.dumps(out, allow_nan=False)
