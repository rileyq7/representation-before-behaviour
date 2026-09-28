"""Checkpoint inference with last-token-only logits and bounded memory."""
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYERS = [0, 6, 12, 18, 24]
ANSWER_STRINGS = [" he", " she", " male", " female", " not"]


def load(model_path, device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = torch.float32 if device == "cpu" else torch.float16
    model, loading = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, use_safetensors=False,
        dtype=dtype, attn_implementation="sdpa", output_loading_info=True)
    # Old NeoX checkpoints contain nonpersistent attention masks; no trainable
    # parameters may be absent, unexpected, or mismatched.
    bad = {k: v for k, v in loading.items() if v and k in {"missing_keys", "mismatched_keys", "error_msgs"}}
    unexpected = [k for k in loading.get("unexpected_keys", [])
                  if not k.endswith(("attention.bias", "attention.masked_bias", "rotary_emb.inv_freq"))]
    if bad or unexpected:
        raise RuntimeError(f"Checkpoint did not load completely: {bad}; unexpected={unexpected}")
    assert model.config.num_hidden_layers == 24 and model.config.hidden_size == 2048
    model.eval().to(device)
    model.requires_grad_(False)
    torch.set_num_threads(4)
    loading = {k: sorted(v) if isinstance(v, set) else v for k, v in loading.items()}
    return model, tokenizer, device, loading


def answer_ids(tokenizer):
    ids = []
    for text in ANSWER_STRINGS:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"Answer {text!r} is not one token: {tokens}")
        ids.append(tokens[0])
    assert len(set(ids)) == len(ids)
    return ids


@torch.inference_mode()
def extract(model, tokenizer, rows, device, batch_size=4, save_activations=True):
    ids = answer_ids(tokenizer)
    activations = {f"layer_{l}": [] for l in LAYERS} if save_activations else {}
    values, ranks, logprobs, top_ids = [], [], [], []
    start = time.monotonic()
    for offset in range(0, len(rows), batch_size):
        texts = [r["prompt"] for r in rows[offset:offset + batch_size]]
        encoded = tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False)
        if encoded.input_ids.shape[1] > 256:
            raise ValueError("Unexpected long prompt: no silent truncation allowed")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        last = encoded["attention_mask"].sum(1) - 1
        batch = torch.arange(len(texts), device=device)
        output = model.gpt_neox(**encoded, use_cache=False, output_hidden_states=save_activations)
        logits = model.get_output_embeddings()(output.last_hidden_state[batch, last]).float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite logits")
        chosen = logits[:, ids]
        values.append(chosen.cpu().numpy())
        logprobs.append((chosen - torch.logsumexp(logits, dim=-1, keepdim=True)).cpu().numpy())
        ranks.append((1 + (logits[:, :, None] > chosen[:, None, :]).sum(1)).cpu().numpy())
        top_ids.append(logits.argmax(-1).cpu().numpy())
        if save_activations:
            assert len(output.hidden_states) == 25
            for layer in LAYERS:
                # The final layer's representation is explicitly post-norm.
                h = output.last_hidden_state if layer == 24 else output.hidden_states[layer]
                a = h[batch, last].float().cpu().numpy()
                if not np.isfinite(a).all():
                    raise FloatingPointError("Non-finite activations")
                activations[f"layer_{layer}"].append(a)
        del output, logits
        if offset % (batch_size * 10) == 0 or offset + batch_size >= len(rows):
            print(json.dumps({"event": "inference", "done": min(offset + batch_size, len(rows)),
                              "total": len(rows), "seconds": round(time.monotonic() - start, 2)}), flush=True)
    return ({k: np.concatenate(v) for k, v in activations.items()},
            {"logits": np.concatenate(values), "logprobs": np.concatenate(logprobs),
             "ranks": np.concatenate(ranks), "top_ids": np.concatenate(top_ids),
             "answer_ids": np.array(ids), "seconds": np.array(time.monotonic() - start)})


def save_extract(out, rows, acts, outputs):
    from .data import digest, write_jsonl
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "rows.jsonl", rows)
    if acts:
        np.savez_compressed(out / "activations.npz", **acts)
    np.savez_compressed(out / "outputs.npz", **outputs)
    (out / "data-sha256.txt").write_text(digest(rows) + "\n")
