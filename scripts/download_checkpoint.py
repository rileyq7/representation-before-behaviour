"""Fetch one official checkpoint, with bounded project-local storage."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
# Xet stalled in this environment's benchmark; HTTP remains the fallback.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# If Xet is explicitly enabled, do not keep an additional chunk cache.
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
from huggingface_hub import HfApi, snapshot_download
import httpx
from huggingface_hub.errors import HfHubHTTPError


def transient_network_error(error):
    # Hub errors may wrap a transport failure when a file isn't cached.
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, httpx.TransportError):
            return True
        if isinstance(error, (HfHubHTTPError, httpx.HTTPStatusError)):
            response = getattr(error, "response", None)
            if response is not None and (response.status_code in {408, 429} or response.status_code >= 500):
                return True
        error = error.__cause__ or error.__context__
    return False


def download(step, destination):
    started = time.monotonic()
    repo = "EleutherAI/pythia-1.4b"
    info = HfApi().model_info(repo, revision=f"step{step}", files_metadata=True)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "requested-step.json").write_text(json.dumps({"step": step, "sha": info.sha}))
    # Official .bin is ~2.93 GB; converted safetensors are ~5.66 GB.
    # Modern torch loads this with weights_only=True through Transformers.
    snapshot_download(repo, revision=info.sha, local_dir=destination,
                      allow_patterns=["config.json", "*token*.json", "special_tokens_map.json", "pytorch_model.bin"],
                      max_workers=2)
    weight_info = next(f for f in info.siblings if f.rfilename == "pytorch_model.bin")
    expected = weight_info.lfs.sha256
    with (destination / "pytorch_model.bin").open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError(f"Checkpoint checksum mismatch: expected {expected}, got {actual}")
    metadata = {"repo": repo, "step": step, "sha": info.sha, "format": "pytorch_model.bin",
                "weight_sha256": actual, "download_and_verify_seconds": time.monotonic()-started,
                "parallel_xet_enabled": os.environ["HF_HUB_DISABLE_XET"] == "0"}
    (destination / "provenance.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", type=int)
    p.add_argument("--destination", default=str(ROOT / ".cache/checkpoint"))
    args = p.parse_args()
    try:
        download(args.step, args.destination)
    except Exception as error:
        if transient_network_error(error):
            print(json.dumps({"event": "transient_network_error", "type": type(error).__name__}), file=sys.stderr, flush=True)
            sys.exit(75)
        raise
