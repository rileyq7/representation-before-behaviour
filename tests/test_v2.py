"""Software checks for v2 measurement code on a tiny random 24-layer NeoX model (not empirical results)."""
import json
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
torch = pytest.importorskip("torch")
from mech import v2  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM
    path = tmp_path_factory.mktemp("tiny")
    tok = AutoTokenizer.from_pretrained(ROOT / ".cache/checkpoint", local_files_only=True)
    torch.manual_seed(0)
    cfg = GPTNeoXConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=24, num_attention_heads=4,
                        intermediate_size=64, max_position_embeddings=64)
    GPTNeoXForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path


def test_rows_extend_v1_unchanged():
    rows = v2.v2_rows(ROOT)
    v1 = [json.loads(x) for x in (ROOT / "data/matched.jsonl").read_text().splitlines()]
    assert rows[:len(v1)] == v1
    bls = [r for r in rows if r.get("set") == "bls"]
    assert len(bls) == 64 * 6 and {r["label"] for r in bls} == {0, 1}
    assert not {r["subject"] for r in bls} & {r["subject"] for r in v1}


def test_ablation_removes_direction():
    u = np.zeros(8, np.float32); u[3] = 1
    hook = v2.ablation_hook(u, 0.5, "cpu")
    h = torch.randn(2, 5, 8)
    out = hook(None, None, (h,))[0]
    assert torch.allclose(out[..., 3], torch.full((2, 5), .5))
    assert torch.allclose(out[..., :3], h[..., :3])


def test_pipeline_end_to_end(tiny):
    rows = v2.v2_rows(ROOT)
    arrays = v2.run_inference(tiny, rows, device="cpu")
    n_occ = sum(r["kind"] == "occupation" for r in rows)
    assert arrays["activations"].shape == (len(rows), 25, 32)
    assert arrays["logit_lens"].shape == (len(rows), 25)
    assert arrays["ablated_logits"].shape == (1 + v2.RANDOM_DIRECTIONS, n_occ, 2)
    # Logit lens at the last layer equals the model's own logits.
    assert np.allclose(arrays["logit_lens"][:, 24], arrays["logits"][:, 1] - arrays["logits"][:, 0], atol=2e-2)
    # Ablation changes outputs relative to the unablated pass.
    occ = [i for i, r in enumerate(rows) if r["kind"] == "occupation"]
    assert not np.allclose(arrays["ablated_logits"][0], arrays["logits"][occ])
    summary = v2.summarise(rows, arrays, {"step": 0})
    for name, n in (("winobias", 40), ("combined", 104)):
        s = summary["sets"][name]
        assert len(s["occupations"]) == n
        assert np.array(s["transfer_accuracy_by_layer"]).shape == (25, n)
        assert np.array(s["logit_lens_by_layer"]).shape == (25, n)
        assert np.array(s["transfer_margin_by_layer"]).shape == (25, n)
        assert np.array(s["ablation_random"]).shape == (v2.RANDOM_DIRECTIONS, n)
    json.dumps(summary, allow_nan=False)
