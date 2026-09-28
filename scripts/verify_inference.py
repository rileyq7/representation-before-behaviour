"""Verify optimized last-token extraction against the model's full forward."""
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
from mech.inference import load, extract, answer_ids
from mech.data import matched_rows

model, tokenizer, device, loading = load(ROOT / ".cache/checkpoint", "mps")
rows = matched_rows(ROOT / "data/raw")
sample = [rows[i] for i in [0, 1, 13, 29, 95, 124, 220, 359]]
with torch.inference_mode():
    acts, values = extract(model, tokenizer, sample, device, 4)
    _, single = extract(model, tokenizer, sample, device, 1)
    encoded = tokenizer([r["prompt"] for r in sample], padding=True, return_tensors="pt", add_special_tokens=False)
    encoded = {k:v.to(device) for k,v in encoded.items()}
    output = model(**encoded, use_cache=False, output_hidden_states=True)
    batch = torch.arange(len(sample), device=device)
    last = encoded["attention_mask"].sum(1)-1
    reference = output.logits[batch,last][:,answer_ids(tokenizer)].float().cpu().numpy()
    max_forward_error = float(np.abs(reference-values["logits"]).max())
    max_batch_error = float(np.abs(single["logits"]-values["logits"]).max())
    assert max_forward_error < .04 and max_batch_error < .04, (max_forward_error,max_batch_error)
    assert all(a.shape == (len(sample),2048) for a in acts.values())
    # Verify the residual stream indexing explicitly, including post-norm final layer.
    for layer in [0,6,12,18,24]:
        expected = output.hidden_states[layer][batch,last].float().cpu().numpy()
        assert np.allclose(acts[f"layer_{layer}"], expected, atol=.05, rtol=.01), layer
    del output, encoded
    model.to(device="cpu", dtype=torch.float32)
    torch.mps.empty_cache()
    _, fp32 = extract(model, tokenizer, sample, "cpu", 4, False)
    max_fp32_error = float(np.abs(fp32["logits"]-values["logits"]).max())
    logodds_error = float(np.abs((fp32["logits"][:,1]-fp32["logits"][:,0]) -
                                (values["logits"][:,1]-values["logits"][:,0])).max())
    assert max_fp32_error < .10 and logodds_error < .05, (max_fp32_error,logodds_error)
result = dict(status="passed", device=device, max_full_forward_logit_error=max_forward_error,
              max_single_vs_batch_logit_error=max_batch_error, answer_ids=answer_ids(tokenizer),
              max_fp32_cpu_logit_error=max_fp32_error, max_fp32_cpu_logodds_error=logodds_error,
              loading=loading, n=len(sample), checked_layers=[0,6,12,18,24])
(ROOT / "results/inference-verification.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
