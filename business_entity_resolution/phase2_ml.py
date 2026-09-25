"""Train and apply the Phase-2 LightGBM matcher using bounded layered blocks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import pickle
import re
from collections import defaultdict
from pathlib import Path

import jellyfish
import numpy as np
from lightgbm import LGBMClassifier
from rapidfuzz.fuzz import ratio
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

try:
    from .src.metrics import macro_f0_5
    from .src.preprocessing import core_name, normalize_address, normalize_name
except ImportError:
    from src.metrics import macro_f0_5
    from src.preprocessing import core_name, normalize_address, normalize_name

# A cap of five was too aggressive on the 10% real-data subset: frequent
# address-number and phonetic signatures were discarded before any pair could
# reach the classifier.  One hundred remains bounded while preserving useful
# evidence for the diagnostic training run.
K = 100


def ph(core: str) -> str:
    token = next((x for x in core.split() if len(x) >= 3), "")
    return jellyfish.soundex(token) if token else ""


def keys(row):
    n = normalize_name(row.get("business_name", "")); c = core_name(row.get("business_name", "")); a = normalize_address(row.get("business_address", ""))
    number = " ".join(re.findall(r"\d+", a))
    return n, c, a, row.get("country", "").strip().casefold(), number


def add(index, key, sid):
    values = index.get(key)
    if values is None:
        if key not in index: index[key] = [sid]
    elif len(values) < K: values.append(sid)
    else: index[key] = None


def feat(left, right):
    n, c, a, country, _ = left; tn, tc, ta, tcountry, _ = right
    ta_set, a_set = set(c.split()), set(tc.split())
    jac = len(ta_set & a_set) / len(ta_set | a_set) if ta_set | a_set else 0.0
    return [ratio(n, tn) / 100, ratio(c, tc) / 100, ratio(a, ta) / 100, float(n == tn and bool(n)), float(a == ta and bool(a)), float(country == tcountry and bool(country)), jac]


def sample_records(train_dir):
    records = {}; strata = {}
    with (train_dir / "train_source1.tsv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sid = row["entity_id"]
            if int(hashlib.blake2b(sid.encode(), digest_size=8).hexdigest(), 16) % 10 == 0:
                records[sid] = keys(row); strata[sid] = row["country"]
    labels = {sid: set() for sid in records}
    with (train_dir / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sid = row["source1_entity_id"]
            if sid in labels: labels[sid] = set(filter(None, row["matched_entity_ids"].split(",")))
    return records, labels


def generate(records, source_paths):
    indexes = [dict() for _ in range(5)]
    indexes.append(dict())
    for sid, (n, c, a, country, number) in records.items():
        for idx, key in enumerate((n, c, a, f"{country}:{c[:5]}", f"{country}:{ph(c)}", number if len(number) >= 5 else "")): add(indexes[idx], key, sid)
    found = defaultdict(dict)
    for path in source_paths:
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                target = keys(row); n, c, a, country, number = target
                sids = set()
                for index, key in zip(indexes, (n, c, a, f"{country}:{c[:5]}", f"{country}:{ph(c)}", number if len(number) >= 5 else "")):
                    sids.update(index.get(key) or ())
                for sid in sids: found[sid][row["entity_id"]] = target
    return found


def make_xy(records, labels, candidates, use_ids):
    rows=[]; y=[]; pairs=[]
    for sid in use_ids:
        for tid, right in candidates.get(sid, {}).items():
            rows.append(feat(records[sid], right)); y.append(int(tid in labels[sid])); pairs.append((sid, tid))
    return np.asarray(rows, dtype=np.float32), np.asarray(y, dtype=np.int8), pairs


def preds_at(probs, pairs, threshold, truth):
    out={sid:set() for sid in truth}
    for (sid, tid), prob in zip(pairs, probs):
        if prob >= threshold: out[sid].add(tid)
    return out


def train(train_dir, model_path, report_path):
    records, labels = sample_records(train_dir)
    train_ids=[sid for sid in records if int(hashlib.blake2b(sid.encode(),digest_size=8).hexdigest(),16)%5]
    train_id_set=set(train_ids)
    valid_ids=[sid for sid in records if sid not in train_id_set]
    candidates=generate(records, [train_dir/'train_source2.tsv', train_dir/'train_source3.tsv'])
    Xtr,ytr,_=make_xy(records,labels,candidates,train_ids); Xv,yv,pairs=make_xy(records,labels,candidates,valid_ids)
    model=LGBMClassifier(objective='binary',n_estimators=350,learning_rate=.05,num_leaves=31,min_child_samples=30,class_weight='balanced',random_state=42,verbosity=-1,n_jobs=-1).fit(Xtr,ytr)
    p=model.predict_proba(Xv)[:,1]; truth={sid:labels[sid] for sid in valid_ids}
    sweep=[]; best=(-1,.9)
    for t in np.linspace(.3,.9,61):
        score=macro_f0_5(truth,preds_at(p,pairs,t,truth),valid_ids); sweep.append((float(t),score))
        if score>best[0]: best=(score,float(t))
    precision,recall,_,_=precision_recall_fscore_support(yv,p>=.5,average='binary',zero_division=0)
    auc=roc_auc_score(yv,p)
    with model_path.open('wb') as f: pickle.dump((model,best[1]),f)
    with report_path.open('w') as f:
        f.write(f'sample_source1={len(records)} train_pairs={len(ytr)} validation_pairs={len(yv)}\n')
        f.write(f'validation_auc={auc:.6f} precision_at_0.5={precision:.6f} recall_at_0.5={recall:.6f}\n')
        f.write(f'best_threshold={best[1]:.2f} validation_macro_f0_5={best[0]:.6f}\n')
        for t,s in sweep: f.write(f'{t:.2f}\t{s:.6f}\n')


def score(test_dir, candidate_path, matching_path, model_path):
    with model_path.open('rb') as f: model, threshold=pickle.load(f)
    by_target=defaultdict(list); source_ids=set()
    with candidate_path.open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            sid=row['source1_entity_id']; source_ids.add(sid)
            for tid in filter(None,row['candidate_entity_ids'].split(',')): by_target[tid].append(sid)
    left={}
    with (test_dir/'test_source1.tsv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            if row['entity_id'] in source_ids: left[row['entity_id']]=keys(row)
    out=defaultdict(list)
    for source in (2,3):
        with (test_dir/f'test_source{source}.tsv').open(encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f,delimiter='\t'):
                sids=by_target.get(row['entity_id'])
                if not sids: continue
                right=keys(row); X=np.asarray([feat(left[sid],right) for sid in sids],dtype=np.float32)
                for sid,prob in zip(sids,model.predict_proba(X)[:,1]):
                    if prob>=threshold: out[sid].append(row['entity_id'])
    with matching_path.open('w',encoding='utf-8',newline='') as dst:
        dst.write('source1_entity_id\tmatched_entity_ids\n')
        with (test_dir/'test_source1.tsv').open(encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f,delimiter='\t'):
                dst.write(f"{row['entity_id']}\t{','.join(sorted(set(out[row['entity_id']])))}\n")


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('train','score')); p.add_argument('--train-dir',type=Path); p.add_argument('--test-dir',type=Path); p.add_argument('--candidate',type=Path); p.add_argument('--matching',type=Path); p.add_argument('--model',type=Path,required=True); p.add_argument('--report',type=Path)
    a=p.parse_args()
    if a.mode=='train': train(a.train_dir,a.model,a.report)
    else: score(a.test_dir,a.candidate,a.matching,a.model)
