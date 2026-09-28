import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location("experiment_runner",Path(__file__).resolve().parents[1]/"scripts/run.py")
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_disk_guard_waits_for_reclaimed_space(monkeypatch):
    readings=iter([3,7,9])
    sleeps=[]
    monkeypatch.setattr(runner.shutil,"disk_usage",lambda _:SimpleNamespace(free=next(readings)))
    monkeypatch.setattr(runner.time,"sleep",sleeps.append)
    runner.wait_for_disk(minimum=8,attempts=3,delay=10)
    assert sleeps == [10,10]


def test_disk_guard_preserves_threshold(monkeypatch):
    monkeypatch.setattr(runner.shutil,"disk_usage",lambda _:SimpleNamespace(free=7))
    monkeypatch.setattr(runner.time,"sleep",lambda _:None)
    with pytest.raises(RuntimeError,match="stopped before checkpoint"):
        runner.wait_for_disk(minimum=8,attempts=3,delay=0)


def test_default_wait_survives_long_recovery(monkeypatch):
    readings=iter([6]*20+[9])
    heartbeats=[]
    monkeypatch.setattr(runner.shutil,"disk_usage",lambda _:SimpleNamespace(free=next(readings)))
    monkeypatch.setattr(runner.time,"sleep",lambda _:None)
    runner.wait_for_disk(minimum=8,delay=0,on_wait=heartbeats.append)
    assert heartbeats==[6]*20


def test_stop_file_interrupts_disk_wait(monkeypatch,tmp_path):
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    monkeypatch.setattr(runner.shutil,"disk_usage",lambda _:SimpleNamespace(free=6))
    monkeypatch.setattr(runner.time,"sleep",lambda _:(tmp_path/"STOP").touch())
    with pytest.raises(InterruptedError,match="STOP file"):
        runner.wait_for_disk(minimum=8,delay=0)
