"""Leakage-resistant linear probes and controls with per-item predictions."""
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(C=.01, solver="liblinear", max_iter=1000, random_state=0))


def score(y, margins):
    # Exact ties count as half correct, preventing a tie-breaking gender prior.
    correct = np.where(margins == 0, .5, (margins > 0) == np.asarray(y)).astype(float)
    return {"accuracy": float(correct.mean()), "auc": float(roc_auc_score(y, margins))}


def direct_splits(rows, seed=0):
    subjects = sorted(set(r["subject"] for r in rows))
    labels = [next(r["label"] for r in rows if r["subject"] == s) for s in subjects]
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for _, test in skf.split(subjects, labels):
        heldout = {subjects[i] for i in test}
        for templates in ({0, 1}, {2, 3}, {4, 5}):
            train = np.array([i for i, r in enumerate(rows) if r["subject"] not in heldout and r["template"] not in templates])
            test_idx = np.array([i for i, r in enumerate(rows) if r["subject"] in heldout and r["template"] in templates])
            yield train, test_idx


def direct_probe(x, rows, seed=0, labels_override=None):
    y = np.array([r["label"] for r in rows]) if labels_override is None else labels_override
    pred = np.full(len(rows), np.nan)
    for train, test in direct_splits(rows, seed):
        clf = classifier().fit(x[train], y[train])
        pred[test] = clf.decision_function(x[test])
    assert np.isfinite(pred).all()
    return pred


def transfer_probe(gender_x, gender_rows, occ_x, occ_rows, gender_labels=None):
    y = np.array([r["label"] for r in gender_rows]) if gender_labels is None else gender_labels
    pred = np.full(len(occ_rows), np.nan)
    for heldout in ({0, 1}, {2, 3}, {4, 5}):
        train = [i for i, r in enumerate(gender_rows) if r["template"] not in heldout]
        test = [i for i, r in enumerate(occ_rows) if r["template"] in heldout]
        clf = classifier().fit(gender_x[train], y[train])
        pred[test] = clf.decision_function(occ_x[test])
    assert np.isfinite(pred).all()
    return pred


def gender_validation(x, rows):
    pred = np.full(len(rows), np.nan)
    y = np.array([r["label"] for r in rows])
    for pairs in ({0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}):
        for templates in ({0, 1}, {2, 3}, {4, 5}):
            train = [i for i, r in enumerate(rows) if r["pair"] not in pairs and r["template"] not in templates]
            test = [i for i, r in enumerate(rows) if r["pair"] in pairs and r["template"] in templates]
            clf = classifier().fit(x[train], y[train])
            pred[test] = clf.decision_function(x[test])
    return score(y, pred)


def permuted_labels(rows, seed):
    subjects = sorted(set(r["subject"] for r in rows))
    labels = np.array([next(r["label"] for r in rows if r["subject"] == s) for s in subjects])
    np.random.default_rng(seed).shuffle(labels)
    mapping = dict(zip(subjects, labels))
    return np.array([mapping[r["subject"]] for r in rows])


@threadpool_limits.wrap(limits=2)
def fit_all(activations, rows, permutations=20):
    occ_idx = [i for i, r in enumerate(rows) if r["kind"] == "occupation"]
    gender_idx = [i for i, r in enumerate(rows) if r["kind"] == "gender"]
    occ = [rows[i] for i in occ_idx]
    gender = [rows[i] for i in gender_idx]
    y = np.array([r["label"] for r in occ])
    reports, arrays = {}, {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        for key, x in activations.items():
            ox, gx = x[occ_idx], x[gender_idx]
            transfer = transfer_probe(gx, gender, ox, occ)
            direct = np.stack([direct_probe(ox, occ, seed) for seed in (0, 1, 2)])
            arrays[f"{key}_transfer"] = transfer
            arrays[f"{key}_direct"] = direct
            direct_null = []
            for seed in (10, 11, 12):
                yp = permuted_labels(occ, seed)
                margins = direct_probe(ox, occ, 0, yp)
                direct_null.append(score(yp, margins))
            reports[key] = {"transfer": score(y, transfer),
                            "direct": [score(y, m) for m in direct],
                            "direct_label_permutation": direct_null,
                            "gender_validation": gender_validation(gx, gender)}
            if key == "layer_12":
                null = np.stack([transfer_probe(gx, gender, ox, occ, permuted_labels(gender, s + 100))
                                 for s in range(permutations)])
                arrays[f"{key}_transfer_null"] = null
                reports[key]["transfer_null"] = [score(y, m) for m in null]
            print(f"Probes complete: {key}", flush=True)
        reports["convergence_warnings"] = [str(w.message) for w in caught]
    return reports, arrays


def lexical_baseline(rows):
    occ = [r for r in rows if r["kind"] == "occupation"]
    y = np.array([r["label"] for r in occ])
    scores = []
    for seed in (0, 1, 2):
        pred = np.full(len(occ), np.nan)
        for train, test in direct_splits(occ, seed):
            clf = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 4)),
                                LogisticRegression(C=1, solver="liblinear", random_state=0))
            clf.fit([occ[i]["subject"] for i in train], y[train])
            pred[test] = clf.decision_function([occ[i]["subject"] for i in test])
        scores.append(score(y, pred))
    return scores
