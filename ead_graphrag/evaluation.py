"""Scoring and paired inference. Gold data never enter inference functions."""
from __future__ import annotations
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np


def read_rows(path):
    rows = [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    ids = [r['question_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate IDs: {path}')
    return rows


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows), encoding='utf-8')


def csv_write(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def unique(values):
    if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
        raise ValueError('Ranking must be a list of strings')
    return list(dict.fromkeys(x.strip() for x in values if x.strip()))


def ndcg(pred, gold, k=10):
    gold = set(gold)
    if not gold: return None
    return sum((x in gold)/math.log2(i+2) for i,x in enumerate(unique(pred)[:k])) / sum(1/math.log2(i+2) for i in range(min(k,len(gold))))


def path_triples(path):
    # Exact direction, relation and adjacent endpoints; no cross-path substring hits.
    v = [x.strip() for x in path.split('→')]
    return [(v[i],v[i+1],v[i+2]) for i in range(0,len(v)-2,2)] if len(v)%2 else []


def before_candidates(p):
    if 'initial_candidate_diseases' in p: return unique(p['initial_candidate_diseases'])
    trace = p.get('differential_trace')
    if isinstance(trace, dict) and 'candidates_before_fusion' in trace:
        return [x['disease'] for x in trace['candidates_before_fusion']]
    if isinstance(trace, list): return [x['disease'] for x in trace]
    return None


def clusters(golds):
    # Repeated inputs and disease families are connected, not independent replicates.
    parent = {}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x: parent[x]=find(parent[x])
        return parent[x]
    def union(x,y): parent[find(x)]=find(y)
    for g in golds:
        union('f:'+str(g.get('case_family_id') or g.get('case_id') or g['question_id']),
              'q:'+hashlib.sha256(g['question'].encode()).hexdigest())
    return {g['question_id']:find('f:'+str(g.get('case_family_id') or g.get('case_id') or g['question_id'])) for g in golds}


def score(golds, predictions):
    lookup={p['question_id']:p for p in predictions}
    if len(lookup)!=len(predictions): raise ValueError('Duplicate predictions')
    if set(lookup)-{g['question_id'] for g in golds}: raise ValueError('Unexpected prediction IDs')
    grouping=clusters(golds); out=[]
    for g in golds:
        p=lookup.get(g['question_id'], {'error':'missing_prediction'})
        if 'question' in p and p['question']!=g['question']: raise ValueError('Question mismatch: '+g['question_id'])
        failed=bool(p.get('error')); rank=unique(p.get('candidate_diseases',[])) if not failed else []
        target=set(g.get('gold_diseases',[])); initial=before_candidates(p) if not failed else []
        r=dict(question_id=g['question_id'], cluster=grouping[g['question_id']], subset=g.get('subset'),
               crop=g.get('crop',[]), failed=failed, initial_candidate_count=len(initial) if initial is not None else None,
               final_candidate_count=len(rank), gold_count=len(target), candidate_missing=None)
        if target:
            hits=[i+1 for i,x in enumerate(rank[:3]) if x in target]
            r.update(**{'Acc@1':float(bool(rank) and rank[0] in target),
                        'Recall@3':len(target.intersection(rank[:3]))/len(target),
                        'MRR@3':1/min(hits) if hits else 0.})
            if initial is not None:
                r['Candidate Recall@3']=float(bool(target.intersection(initial[:3])))
                r['Candidate Coverage@all']=float(bool(target.intersection(initial)))
                r['candidate_missing']=not bool(target.intersection(initial))
                r['Candidate GoldFraction@3']=len(target.intersection(initial[:3]))/len(target)
        chunks=g.get('gold_chunk_ids',[])
        if chunks and ('retrieved_chunk_ids' in p or failed):
            docs=unique(p.get('retrieved_chunk_ids',[])) if not failed else []
            r['Chunk Recall@10']=len(set(docs[:10]).intersection(chunks))/len(set(chunks))
            r['nDCG@10_text']=ndcg(docs,chunks)
            r['Text MRR']=next((1/(i+1) for i,x in enumerate(docs) if x in chunks),0.)
        triples={tuple(t) for t in g.get('gold_kg_triples',[])}
        if triples and ('kg_paths' in p or failed):
            paths=p.get('kg_paths',[]) if not failed else []
            found={t for path in paths[:10] for t in path_triples(path)}
            r['Triple Recall@10']=len(found.intersection(triples))/len(triples)
            # Keep historical weak metric explicitly separate for manuscript reconciliation.
            hay='\n'.join(paths[:10])
            r['Legacy Triple Substring Recall@10']=sum(all(str(z) in hay for z in t) for t in triples)/len(triples)
        if 'uncertain' in p or failed:
            positive=g.get('subset') in ('ambiguous','ood_unanswerable') or bool(g.get('requires_clarification'))
            r['Decision Accuracy']=float(not failed and p.get('uncertain') is positive)
            r['Withhold Sensitivity']=float(not failed and p.get('uncertain') is True) if positive else None
            r['Diagnose Specificity']=float(not failed and p.get('uncertain') is False) if not positive else None
        out.append(r)
    return out


METRICS=['Acc@1','Recall@3','MRR@3','Candidate Recall@3','Candidate Coverage@all','Candidate GoldFraction@3',
         'Chunk Recall@10','nDCG@10_text','Text MRR','Triple Recall@10','Legacy Triple Substring Recall@10',
         'Decision Accuracy','Withhold Sensitivity','Diagnose Specificity']


def interval(values, labels, reps, seed):
    vals=np.asarray(values,dtype=float); labels=np.asarray(labels)
    groups=list(dict.fromkeys(labels)); totals=np.array([vals[labels==c].sum() for c in groups]); sizes=np.array([(labels==c).sum() for c in groups])
    rng=np.random.default_rng(seed); draws=[]
    for start in range(0,reps,256):
        idx=rng.integers(0,len(groups),(min(256,reps-start),len(groups)))
        draws.extend((totals[idx].sum(1)/sizes[idx].sum(1)).tolist())
    return tuple(float(x) for x in np.quantile(draws,[.025,.975]))


def signflip(differences, labels, reps, seed):
    totals=defaultdict(float)
    for d,c in zip(differences,labels): totals[c]+=d
    # Zero-difference clusters have no influence on any randomization.
    v=np.array([x for x in totals.values() if abs(x)>1e-14]); observed=abs(sum(v)); n=len(v)
    if not n: return 1.,'exact_cluster_signflip',0
    if n<=18:
        extreme=sum(abs(float(np.dot(signs,v)))>=observed-1e-12 for signs in itertools.product((-1,1),repeat=n))
        return extreme/(2**n),'exact_cluster_signflip',2**n
    rng=np.random.default_rng(seed); extreme=0
    for start in range(0,reps,256):
        signs=rng.choice([-1,1],size=(min(256,reps-start),n))
        extreme+=int((np.abs(signs@v)>=observed-1e-12).sum())
    return (extreme+1)/(reps+1),'monte_carlo_cluster_signflip',reps


def summarize(cohort, variant, rows, reps, seed, status):
    result=[]
    for metric in METRICS:
        rr=[r for r in rows if r.get(metric) is not None]
        if not rr: continue
        values=[r[metric] for r in rr]; labels=[r['cluster'] for r in rr]
        lo,hi=interval(values,labels,reps,seed)
        result.append(dict(cohort=cohort,ablation_id=variant,status=status,metric=metric,n=len(rr),
                           cluster_n=len(set(labels)),estimate=float(np.mean(values)),ci95_low=lo,ci95_high=hi,
                           ci_method='paired_cluster_percentile_bootstrap',bootstrap_replicates=reps,seed=seed))
    return result


def compare(cohort, base, other, left, right, metrics, reps, seed):
    if [x['question_id'] for x in left]!=[x['question_id'] for x in right]: raise ValueError('Unpaired ID order')
    if not left:return []
    out=[]
    for metric in metrics:
        if not all(a.get(metric) is not None and b.get(metric) is not None for a,b in zip(left,right)):
            continue  # Undefined endpoints are not replaced with zero.
        d=[a[metric]-b[metric] for a,b in zip(left,right)]; labels=[r['cluster'] for r in left]
        lo,hi=interval(d,labels,reps,seed); pv,test,draws=signflip(d,labels,reps,seed)
        out.append(dict(cohort=cohort,reference=base,comparison=other,metric=metric,n=len(d),cluster_n=len(set(labels)),
                        effect_reference_minus_comparison=float(np.mean(d)),ci95_low=lo,ci95_high=hi,p_raw=pv,
                        test=test,randomization_draws=draws,bootstrap_replicates=reps,seed=seed))
    return out


def holm(rows):
    # One family includes ALL reported contrasts and endpoints, without selecting p values.
    current=0.; n=len(rows)
    for i,r in enumerate(sorted(rows,key=lambda r:r['p_raw'])):
        current=max(current,min(1.,(n-i)*r['p_raw']));r['p_holm']=current
        r['reject_holm_0.05']=current<.05;r['multiplicity_family']='all_reported_contrasts_and_endpoints';r['family_size']=n
    return rows
