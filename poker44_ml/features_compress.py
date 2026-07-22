"""Explore compression-family features across BOTH exam chunks.

u195's single strongest transferable feature is schema_rp_gzip_ratio (overlap 0.30,
discrimination 0.78). Compression measures repetition directly: a scripted policy
replays sequences so its stream compresses hard, a human's does not, and that holds
whatever the stack depth or table size. We build the family in several granularities
and test each against BOTH graded chunks so a feature must survive twice, not once.
"""
import copy, glob, gzip, json, sys, math
import numpy as np
from collections import Counter
from scipy.stats import rankdata
sys.path.insert(0,"E:/BIT/127/rekop-99")
from poker44.validator.payload_view import prepare_hand_for_miner as mirror

_ACT={"fold":"F","check":"K","call":"C","bet":"B","raise":"R","allin":"A","all_in":"A"}
_ST={"preflop":"p","flop":"f","turn":"t","river":"r"}
BUCK=[0,0.5,1,1.5,2,3,4,6,8,12,16,24,36,56,84,126]
def bi(v): return min(range(len(BUCK)), key=lambda i:abs(BUCK[i]-v))

def gz(text):
    if not text: return 1.0
    raw=text.encode("utf-8"); return len(gzip.compress(raw,6))/max(len(raw),1)

def compress_features(chunk):
    """Many views of 'how repetitive is this subject'."""
    hero_seqs=[]; hero_amt=[]; hero_rich=[]; all_actor=[]
    for h in chunk:
        meta=h.get("metadata") or {}; hero=meta.get("hero_seat"); bb=float(meta.get("bb") or 0)
        toks=[]; amts=[]; rich=[]
        for a in h.get("actions") or []:
            code=_ACT.get(str(a.get("action_type","")).lower(),"?")
            st=_ST.get(str(a.get("street","")).lower(),"?")
            all_actor.append(str(a.get("actor_seat")))
            if a.get("actor_seat")!=hero: continue
            toks.append(st+code)
            v=float(a.get("normalized_amount_bb") or 0)
            b=bi(v) if v>0 else 0
            amts.append(str(b))
            rich.append(f"{st}{code}{b}")
        if toks:
            hero_seqs.append("".join(toks)); hero_amt.append("".join(amts)); hero_rich.append("".join(rich))
    f={}
    def add(name, seqs):
        if not seqs: f[f"cx_{name}_gzip"]=1.0; f[f"cx_{name}_uniq"]=0.5; f[f"cx_{name}_top1"]=0.0; return
        joined="|".join(seqs)
        f[f"cx_{name}_gzip"]=gz(joined)                       # overall compressibility
        f[f"cx_{name}_gzip_sorted"]=gz("|".join(sorted(seqs)))# order-free baseline
        f[f"cx_{name}_gzip_gap"]=f[f"cx_{name}_gzip"]-f[f"cx_{name}_gzip_sorted"]
        c=Counter(seqs); n=len(seqs)
        f[f"cx_{name}_uniq"]=len(c)/n
        f[f"cx_{name}_top1"]=max(c.values())/n
        f[f"cx_{name}_dup_frac"]=sum(v for v in c.values() if v>1)/n   # fraction in repeated groups
    add("act", hero_seqs)      # action+street tokens
    add("amt", hero_amt)       # bet-bucket tokens
    add("rich", hero_rich)     # action+street+bucket
    add("actor", ["".join(all_actor)])  # actor rotation (whole chunk)
    return f

# load exam A and B
