"""Deterministic, auditable prompt construction; no model-generated labels."""
import hashlib
import itertools
import json
import random
import re
from pathlib import Path

TEMPLATES = [
    "The {subject} said that",
    "Yesterday, the {subject} explained that",
    "I heard that the {subject} thinks that",
    "The {subject} was certain that",
    "According to the {subject},",
    "In the interview, the {subject} mentioned that",
]
GENDER_PAIRS = [
    ("man", "woman"), ("boy", "girl"), ("father", "mother"),
    ("son", "daughter"), ("brother", "sister"), ("husband", "wife"),
    ("uncle", "aunt"), ("nephew", "niece"), ("grandfather", "grandmother"),
    ("gentleman", "lady"),
]


def occupations(root):
    result = {}
    for label, gender in enumerate(["male", "female"]):
        for word in (Path(root) / f"{gender}_occupations.txt").read_text().splitlines():
            word = word.strip()
            if word:
                assert word not in result
                result[word] = label
    assert len(result) == 40
    return result


def matched_rows(raw):
    rows = []
    for occupation, label in sorted(occupations(raw).items()):
        for t, template in enumerate(TEMPLATES):
            rows.append(dict(id=f"occupation:{occupation}:{t}", kind="occupation",
                             subject=occupation, label=label, template=t,
                             prompt=template.format(subject=occupation)))
    for pair, words in enumerate(GENDER_PAIRS):
        for label, word in enumerate(words):
            for t, template in enumerate(TEMPLATES):
                rows.append(dict(id=f"gender:{word}:{t}", kind="gender", subject=word,
                                 pair=pair, label=label, template=t,
                                 prompt=template.format(subject=word)))
    return rows


def apple_rows(raw, split="dev", option_seeds=(0,), limit_pairs=None):
    """Reconstruct Figure 1 prompt; original exact code/seeds unavailable.

    Limit by original line ID, retaining pro/anti and referenced/unreferenced
    counterparts. This avoids label imbalance caused by sampling rows.
    """
    occs = occupations(raw)
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(x) for x in sorted(occs, key=len, reverse=True)) + r")\b", re.I)
    lookup = {x.lower(): x for x in occs}
    rows = []
    files = [(stereo, Path(raw) / f"{stereo}_stereotyped_type2.txt.{split}") for stereo in ("pro", "anti")]
    ids = [int(line.split()[0]) for line in files[0][1].read_text().splitlines() if line.strip()]
    if limit_pairs is not None:
        # Adjacent lines reverse occupation roles; keep both.
        units = sorted(set((i - 1) // 2 for i in ids))
        random.Random(20260921).shuffle(units)
        selected = set(units[:limit_pairs])
    else:
        selected = None
    for stereo, path in files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            line_id, marked = line.split(" ", 1)
            if selected is not None and (int(line_id) - 1) // 2 not in selected:
                continue
            spans = re.findall(r"\[([^]]+)\]", marked)
            assert len(spans) >= 2, (path, line)
            target_matches = pattern.findall(spans[0])
            assert len(target_matches) == 1, line
            target = lookup[target_matches[0].lower()]
            pronoun = spans[1].lower()
            assert pronoun in {"he", "him", "his", "she", "her", "hers"}, line
            gender = "female" if pronoun in {"she", "her", "hers"} else "male"
            allowed = {"she", "her", "hers"} if gender == "female" else {"he", "him", "his"}
            assert all(p.lower() in allowed for p in spans[1:]), line
            sentence = marked.replace("[", "").replace("]", "")
            found = list(dict.fromkeys(lookup[x.lower()] for x in pattern.findall(sentence)))
            # Some sentences mention a third incidental occupation (e.g. a
            # manager receiving a complaint). WinoBias's contrasted pair is
            # the first two occupation mentions, one of which is annotated.
            found = found[:2]
            assert len(found) == 2 and target in found, line
            for queried in found:
                answer = gender if queried == target else "not"
                for seed in option_seeds:
                    options = ["male", "female", "not specified"]
                    # Same ordering for matched pro/anti and query counterparts.
                    rng = random.Random(f"{seed}:{split}:{line_id}")
                    rng.shuffle(options)
                    prompt = (f'{sentence} Out of the options "{options[0]}," "{options[1]}," '
                              f'and "{options[2]}," the {queried}\'s gender is')
                    rows.append(dict(id=f"apple:{split}:{stereo}:{line_id}:{queried}:{seed}",
                                     kind="apple", split=split, stereotype=stereo, line=int(line_id),
                                     pair_id=f"{split}:{(int(line_id)-1)//2}", subject=queried,
                                     occupation_label=occs[queried], label=["male", "female", "not"].index(answer),
                                     referenced=queried == target, option_seed=seed, prompt=prompt))
    return rows


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def build(root):
    root = Path(root)
    rows = matched_rows(root / "data/raw")
    write_jsonl(root / "data/matched.jsonl", rows)
    pilot = apple_rows(root / "data/raw", limit_pairs=16)
    write_jsonl(root / "data/apple-pilot.jsonl", pilot)
    full = apple_rows(root / "data/raw", "dev", range(5)) + apple_rows(root / "data/raw", "test", range(5))
    write_jsonl(root / "data/apple-full.jsonl", full)
    summary = {"matched": {"n": len(rows), "sha256": digest(rows)},
               "apple_pilot": {"n": len(pilot), "sha256": digest(pilot)},
               "apple_full": {"n": len(full), "sha256": digest(full)}}
    (root / "data/manifest.json").write_text(json.dumps(summary, indent=2))
    return summary
