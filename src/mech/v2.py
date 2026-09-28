"""Protocol v2 measurements: all-layer activations, logit lens, gender-direction ablation, expanded occupations.

v1 modules are left byte-identical (their hashes are recorded in v1 results). v2 reuses their
probe specification (classifier, folds, controls) and adds measurements; see PROTOCOL-v2.md.
"""
import hashlib
import json
import time
import warnings
from pathlib import Path
import numpy as np
from .data import GENDER_PAIRS, TEMPLATES, matched_rows, write_jsonl, digest

PRIMARY_LAYER = 12
RANDOM_DIRECTIONS = 10


def v2_rows(root):
    """v1 matched rows unchanged (set=winobias implied), then BLS occupations x the same six templates."""
    root = Path(root)
    rows = matched_rows(root / "data/raw")
    spec = json.loads((root / "data/raw-v2/occupations-v2.json").read_text())
    for occ in spec["occupations"]:
        for t, template in enumerate(TEMPLATES):
            rows.append(dict(id=f"bls:{occ['subject']}:{t}", kind="occupation", set="bls", subject=occ["subject"],
                             label=occ["label"], percent_women=occ["percent_women"], template=t,
                             prompt=template.format(subject=occ["subject"])))
    return rows


def build_v2(root):
    rows = v2_rows(root)
    write_jsonl(Path(root) / "data/matched-v2.jsonl", rows)
    return {"n": len(rows), "sha256": digest(rows)}


# ---------------------------------------------------------------- inference (GPU)

def load_model(path, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model, loading = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float16,
                                                          attn_implementation="sdpa", output_loading_info=True)
    bad = {k: v for k, v in loading.items() if v and k in {"missing_keys", "mismatched_keys", "error_msgs"}}
    unexpected = [k for k in loading.get("unexpected_keys", [])
                  if not k.endswith(("attention.bias", "attention.masked_bias", "rotary_emb.inv_freq"))]
    if bad or unexpected:
        raise RuntimeError(f"Checkpoint did not load completely: {bad}; unexpected={unexpected}")
    assert model.config.num_hidden_layers == 24
    model.eval().to(device).requires_grad_(False)
    return model, tokenizer


def _answer_ids(tokenizer):
    ids = [tokenizer.encode(t, add_special_tokens=False) for t in (" he", " she")]
    assert all(len(i) == 1 for i in ids)
    return [i[0] for i in ids]


def forward(model, tokenizer, rows, device, batch_size=32, hook=None):
    """Last-token hidden states for all 25 layers (24 = post final norm), he/she logits, logit-lens he/she."""
    import torch
    he, she = _answer_ids(tokenizer)
    norm, unembed = model.gpt_neox.final_layer_norm, model.get_output_embeddings()
    acts, logits, lens = [], [], []
    handle = model.gpt_neox.layers[PRIMARY_LAYER - 1].register_forward_hook(hook) if hook else None
    try:
        with torch.inference_mode():
            for offset in range(0, len(rows), batch_size):
                enc = tokenizer([r["prompt"] for r in rows[offset:offset + batch_size]], padding=True,
                                return_tensors="pt", add_special_tokens=False).to(device)
                last = enc["attention_mask"].sum(1) - 1
                b = torch.arange(len(last), device=device)
                out = model.gpt_neox(**enc, use_cache=False, output_hidden_states=True)
                hs = out.hidden_states
                assert len(hs) == 25
                # As in v1, layer 24 is explicitly the post-norm final representation.
                stack = torch.stack([h[b, last] for h in hs[:24]] + [out.last_hidden_state[b, last]], 1)
                full = unembed(stack[:, 24]).float()
                lens_in = torch.cat([norm(stack[:, :24]), stack[:, 24:]], 1)
                lens_logits = unembed(lens_in)[..., [he, she]].float()
                if not (torch.isfinite(full).all() and torch.isfinite(lens_logits).all()):
                    raise FloatingPointError("Non-finite logits")
                logits.append(full[:, [he, she]].cpu().numpy())
                lens.append((lens_logits[..., 1] - lens_logits[..., 0]).cpu().numpy())
                if hook is None:
                    acts.append(stack.float().cpu().numpy())
    finally:
        if handle:
            handle.remove()
    return (np.concatenate(acts) if acts else None), np.concatenate(logits), np.concatenate(lens)


def gender_direction(acts, rows):
    """Layer-12 explicit-gender probe (all gender rows) mapped back to raw activation space, unit norm."""
    from .probes import classifier
    idx = [i for i, r in enumerate(rows) if r["kind"] == "gender"]
    y = np.array([rows[i]["label"] for i in idx])
    clf = classifier().fit(acts[idx, PRIMARY_LAYER], y)
    scaler, lr = clf.steps[0][1], clf.steps[1][1]
    w = lr.coef_[0] / scaler.scale_
    return w / np.linalg.norm(w)


def ablation_hook(direction, target, torch_device):
    """Set the component along `direction` to `target` at every position of the block output."""
    import torch
    u = torch.tensor(direction, dtype=torch.float32, device=torch_device)
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        hf = h.float()
        hf = hf - ((hf @ u) - target)[..., None] * u
        h = hf.to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return hook


def run_inference(model_path, rows, device="cuda", seed=20260928):
    started = time.monotonic()
    model, tokenizer = load_model(model_path, device)
    acts, logits, lens = forward(model, tokenizer, rows, device)
    occ = [i for i, r in enumerate(rows) if r["kind"] == "occupation"]
    occ_rows = [rows[i] for i in occ]
    direction = gender_direction(acts, rows)
    # Mean-projection ablation: every occupation prompt gets the average occupation projection.
    d = acts.shape[-1]
    rng = np.random.default_rng(seed)
    randoms = rng.standard_normal((RANDOM_DIRECTIONS, d)); randoms /= np.linalg.norm(randoms, axis=1, keepdims=True)
    ablated = []
    for u in [direction, *randoms]:
        target = float((acts[occ, PRIMARY_LAYER] @ u).mean())
        _, lg, _ = forward(model, tokenizer, occ_rows, device, hook=ablation_hook(u, target, device))
        ablated.append(lg)
    return dict(activations=acts.astype(np.float16), logits=logits, logit_lens=lens,
                ablated_logits=np.stack(ablated), gender_direction=direction.astype(np.float32),
                random_directions=randoms.astype(np.float32), seconds=np.array(time.monotonic() - started))


# ---------------------------------------------------------------- probes and summary (CPU)

def _per_subject(rows, values, subjects):
    values = np.asarray(values)
    return np.array([values[[i for i, r in enumerate(rows) if r["subject"] == s]].mean(0) for s in subjects])


def summarise(rows, arrays, meta):
    from threadpoolctl import threadpool_limits
    from sklearn.exceptions import ConvergenceWarning
    from .probes import transfer_probe, direct_probe
    from .amendment import balanced_control_labels, correct, gender_margins
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        acts = arrays["activations"].astype(np.float32)
        occ_idx = [i for i, r in enumerate(rows) if r["kind"] == "occupation"]
        g_idx = [i for i, r in enumerate(rows) if r["kind"] == "gender"]
        occ, gender = [rows[i] for i in occ_idx], [rows[i] for i in g_idx]
        y = np.array([r["label"] for r in occ]); gy = np.array([r["label"] for r in gender])
        sets = {"winobias": [i for i, r in enumerate(occ) if r.get("set", "winobias") == "winobias"],
                "combined": list(range(len(occ)))}
        signed = (2 * y - 1) * (arrays["logits"][occ_idx, 1] - arrays["logits"][occ_idx, 0])
        lens_signed = (2 * y - 1)[:, None] * arrays["logit_lens"][occ_idx]
        abl = arrays["ablated_logits"]
        abl_signed = (2 * y - 1)[None] * (abl[..., 1] - abl[..., 0])
        out = dict(meta, layers=list(range(25)), sets={})
        transfer_by_layer = np.stack([transfer_probe(acts[g_idx, l], gender, acts[occ_idx, l], occ) for l in range(25)])
        gender_pair_acc = []
        for l in range(25):
            gc = correct(gy, gender_margins(acts[g_idx, l], gender))
            gender_pair_acc.append([gc[[i for i, r in enumerate(gender) if r["pair"] == p]].mean() for p in range(len(GENDER_PAIRS))])
        out["per_gender_pair_accuracy_by_layer"] = np.array(gender_pair_acc).tolist()
        for name, sel in sets.items():
            srows = [occ[i] for i in sel]
            subjects = sorted(set(r["subject"] for r in srows))
            sy = y[sel]
            labels = [next(r["label"] for r in srows if r["subject"] == s) for s in subjects]
            ox = acts[[occ_idx[i] for i in sel], PRIMARY_LAYER]
            real = np.stack([direct_probe(ox, srows, seed) for seed in (0, 1, 2)])
            controls = []
            for split_seed in (0, 1, 2):
                for perm in range(20):
                    yp = balanced_control_labels(srows, split_seed, perm)
                    controls.append(correct(yp, direct_probe(ox, srows, split_seed, yp)))
            controls = np.array(controls)
            real_correct = correct(sy, real)
            selectivity = real_correct.mean(0) - controls.mean(0)
            out["sets"][name] = dict(
                occupations=subjects, occupation_labels=labels,
                behaviour=_per_subject(srows, signed[sel], subjects).tolist(),
                transfer_accuracy_by_layer=np.stack([_per_subject(srows, correct(sy, transfer_by_layer[l, sel]), subjects)
                                                     for l in range(25)]).tolist(),
                transfer_margin_by_layer=np.stack([_per_subject(srows, (2 * sy - 1) * transfer_by_layer[l, sel], subjects)
                                                   for l in range(25)]).tolist(),
                logit_lens_by_layer=_per_subject(srows, lens_signed[sel], subjects).T.tolist(),
                direct_accuracy=_per_subject(srows, real_correct.mean(0), subjects).tolist(),
                control_accuracy=float(controls.mean()),
                selectivity=_per_subject(srows, selectivity, subjects).tolist(),
                ablation_gender=_per_subject(srows, abl_signed[0, sel], subjects).tolist(),
                ablation_random=np.stack([_per_subject(srows, abl_signed[k, sel], subjects)
                                          for k in range(1, abl.shape[0])]).tolist())
            if name == "combined":
                pw = [next(r.get("percent_women") for r in srows if r["subject"] == s) for s in subjects]
                out["sets"][name]["percent_women"] = pw
        out["convergence_warnings"] = [str(w.message) for w in caught]
    return out


def code_sha():
    root = Path(__file__).resolve().parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
