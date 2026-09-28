import importlib.util
from pathlib import Path
from types import SimpleNamespace
import subprocess
import pytest
import httpx

ROOT=Path(__file__).resolve().parents[1]
def load(name, filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/"scripts"/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
runner=load("retry_runner","run.py")
downloader=load("retry_downloader","download_checkpoint.py")


def test_network_errors_are_distinguished_from_permanent_failures():
    assert downloader.transient_network_error(httpx.ConnectError("DNS failure"))
    inner=httpx.ReadTimeout("interrupted")
    outer=RuntimeError("wrapped");outer.__cause__=inner
    assert downloader.transient_network_error(outer)
    assert not downloader.transient_network_error(RuntimeError("checksum mismatch"))
    request=httpx.Request("GET","https://example.com")
    for code,expected in [(429,True),(503,True),(403,False),(404,False)]:
        error=httpx.HTTPStatusError("test",request=request,response=httpx.Response(code,request=request))
        assert downloader.transient_network_error(error)==expected


def test_transient_failures_resume_after_more_than_three_attempts(monkeypatch,tmp_path):
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    codes=iter([75]*6+[0]); waits=[]; checks=[]
    monkeypatch.setattr(runner.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=next(codes)))
    monkeypatch.setattr(runner.time,"sleep",lambda _:None)
    runner.retry_download(71000,before_attempt=lambda:checks.append(1),on_network_wait=waits.append)
    assert waits==[30,60,120,240,300,300] and len(checks)==7


def test_permanent_failure_still_stops(monkeypatch,tmp_path):
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    monkeypatch.setattr(runner.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=1))
    monkeypatch.setattr(runner.time,"sleep",lambda _:None)
    with pytest.raises(subprocess.CalledProcessError):runner.retry_download(71000)


def test_stop_file_interrupts_network_wait(monkeypatch,tmp_path):
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    monkeypatch.setattr(runner.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=75))
    monkeypatch.setattr(runner.time,"sleep",lambda _:(tmp_path/"STOP").touch())
    with pytest.raises(InterruptedError):runner.retry_download(71000)
