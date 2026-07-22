"""Second-order self-consistency: per-hand statistics, then the DISTRIBUTION of them.

u195's real edge is density: it computes a statistic for each hand, then summarises the
spread of those statistics across the chunk (std, q10, q50, q90). A bot repeats itself
so its per-hand statistics cluster tightly (low std, narrow quantile gap); a human
varies. Crucially this is size-free -- it asks how much a subject varies from itself,
which a stack/seat/pot shift leaves untouched. Built here on the hero's own stream and
validated against BOTH graded chunks so a feature must transfer twice.
"""
import copy, gzip, json, sys, math
import numpy as np
from collections import Counter
from scipy.stats import rankdata
sys.path.insert(0,"E:/BIT/127/rekop-99")
from poker44.validator.payload_view import prepare_hand_for_miner as mirror

_ACT={"fold":"F","check":"K","call":"C","bet":"B","raise":"R","allin":"A","all_in":"A"}
_ST={"preflop":"p","flop":"f","turn":"t","river":"r"}
BUCK=[0,0.5,1,1.5,2,3,4,6,8,12,16,24,36,56,84,126]
def bi(v): return min(range(len(BUCK)),key=lambda i:abs(BUCK[i]-v))

def _hand_stats(hand):
    """One hand's own summary, all size-free."""
    hero=(hand.get("metadata") or {}).get("hero_seat")
    acts=[a for a in (hand.get("actions") or []) if a.get("actor_seat")==hero]
    n=len(acts)
    if n==0: return None
    types=[str(a.get("action_type","")).lower() for a in acts]
    streets=[_ST.get(str(a.get("street","")).lower(),"?") for a in acts]
    buckets=[bi(float(a.get("normalized_amount_bb") or 0)) for a in acts if float(a.get("normalized_amount_bb") or 0)>0]
    aggr=sum(1 for t in types if t in ("bet","raise"))/n
    passive=sum(1 for t in types if t in ("check","call","fold"))/n
    # per-hand entropy of its own action types
    c=Counter(types); ent=-sum((v/n)*math.log(v/n) for v in c.values())/math.log(max(len(c),2))
    nstreets=len(set(streets))
    run=1; maxrun=1
    for a,b in zip(types,types[1:]):
        run=run+1 if a==b else 1; maxrun=max(maxrun,run)
    return dict(nact=n, aggr=aggr, passive=passive, ent=ent, nstreets=nstreets,
                maxrun=maxrun/n, nbuck=len(set(buckets)), meanbuck=(np.mean(buckets) if buckets else 0))

def _dist(vals):
    """The spread of a per-hand statistic across the chunk -- the second order."""
    if len(vals)<2: return {"std":0.0,"q10":0.0,"q50":0.0,"q90":0.0,"iqr":0.0,"cv":0.0}
    a=np.array(vals,dtype=float); s=np.std(a); m=np.mean(a)
    q10,q50,q90=np.percentile(a,[10,50,90])
    return {"std":float(s),"q10":float(q10),"q50":float(q50),"q90":float(q90),
            "iqr":float(q90-q10),"cv":float(s/m) if abs(m)>1e-9 else 0.0}

def second_order_features(chunk):
    hs=[_hand_stats(h) for h in chunk]
    hs=[h for h in hs if h is not None]
    f={}
    if len(hs)<2:
        for k in ("nact","aggr","passive","ent","nstreets","maxrun","nbuck","meanbuck"):
            for s in ("std","q10","q50","q90","iqr","cv"): f[f"so_{k}_{s}"]=0.0
        return f
    for stat in ("nact","aggr","passive","ent","nstreets","maxrun","nbuck","meanbuck"):
        vals=[h[stat] for h in hs]
        for sk,sv in _dist(vals).items():
            f[f"so_{stat}_{sk}"]=sv
    return f

