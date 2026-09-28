import numpy as np
from scipy.special import softmax, xlogy


def jsd_parts(probabilities, labels):
    p = np.asarray(probabilities, dtype=float)
    q = np.eye(p.shape[1])[np.asarray(labels)]
    m = (p + q) / 2
    return .5 * (xlogy(p, np.divide(p, m, out=np.ones_like(p), where=m > 0))
                 + xlogy(q, np.divide(q, m, out=np.ones_like(q), where=m > 0)))


def apple_metrics(rows, outputs):
    logits = outputs["logits"][:, 2:5]
    p = softmax(logits, axis=1)
    labels = np.array([r["label"] for r in rows])
    parts = jsd_parts(p, labels)
    report = {"n": len(rows), "accuracy_restricted": float(np.mean(p.argmax(1) == labels))}
    for label, name in enumerate(["male", "female", "not"]):
        mask = labels == label
        report[name] = {"n": int(mask.sum()), "rank": float(outputs["ranks"][mask, label + 2].mean()),
                        "jsdp_correct": float(parts[mask, label].mean()),
                        "accuracy_restricted": float(np.mean(p[mask].argmax(1) == label)),
                        "accuracy_vocab": float(np.mean(outputs["top_ids"][mask] == outputs["answer_ids"][label + 2])),
                        "mean_probabilities": p[mask].mean(0).tolist()}
    report["female_minus_male_jsdp"] = report["female"]["jsdp_correct"] - report["male"]["jsdp_correct"]
    ambiguous = labels == 2
    signed = np.array([2 * r["occupation_label"] - 1 for r in rows])
    logodds = logits[:, 1] - logits[:, 0]
    report["ambiguous_signed_stereotype_logodds"] = float((signed[ambiguous] * logodds[ambiguous]).mean())
    return report


def occupation_values(rows, values):
    subjects = sorted(set(r["subject"] for r in rows if r["kind"] == "occupation"))
    out, labels = [], []
    for subject in subjects:
        idx = [i for i, r in enumerate(rows) if r["kind"] == "occupation" and r["subject"] == subject]
        out.append(np.asarray(values)[idx].mean(axis=0))
        labels.append(rows[idx[0]]["label"])
    return subjects, np.array(labels), np.array(out)


def bootstrap_indices(labels, n=2000, seed=20260921):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == y) for y in np.unique(labels)]
    return np.concatenate([rng.choice(g, (n, len(g)), replace=True) for g in groups], axis=1)


def interval(values, labels):
    draws = bootstrap_indices(labels)
    boots = np.asarray(values)[draws].mean(1)
    return {"mean": float(np.mean(values)), "ci95": np.quantile(boots, [.025, .975]).tolist()}
