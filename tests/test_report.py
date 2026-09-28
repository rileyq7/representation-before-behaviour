"""Synthetic integration fixtures stay in pytest temporary directories."""
import json
import os
import numpy as np
from mech.analysis import ALL_STEPS, report


def synthetic_summary(step):
    # Deliberately deterministic occupation values isolate analysis logic.
    r = .5 if step < 10000 else .9
    b = 0.0 if step < 30000 else .5
    return {
        "step": step, "occupations": [f"synthetic_{i}" for i in range(40)],
        "occupation_labels": [0]*20+[1]*20,
        "per_occupation": {"transfer_accuracy": [r]*40, "behaviour": [b]*40},
        "probes": {"layer_12": {"transfer": {"accuracy":r},
                    "direct":[{"accuracy":r}]*3, "gender_validation":{"accuracy":1.0},
                    "transfer_null":[{"accuracy":.5}]*20}},
        "apple_pilot":{"female_minus_male_jsdp": b},
    }


def test_partial_report_never_claims_timing(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR",str(tmp_path / "mpl"))
    folder=tmp_path / "results/checkpoints/step143000"
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps(synthetic_summary(143000)))
    result=report(tmp_path)
    assert result["status"] == "partial"
    assert "incomplete" in result["conclusion"]
    assert "representation_onset" not in result


def test_complete_report_recovers_synthetic_lead(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR",str(tmp_path / "mpl"))
    for step in ALL_STEPS:
        folder=tmp_path / f"results/checkpoints/step{step:06d}"
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text(json.dumps(synthetic_summary(step)))
    result=report(tmp_path)
    assert result["status"] == "complete_matched_sweep"
    assert result["representation_onset"]["interval"] == [9000,10000]
    assert result["behaviour_onset"]["interval"] == [29000,30000]
    assert result["lead_steps_interval"] == [19000,21000]
    assert (tmp_path / "reports/curves.png").exists()
