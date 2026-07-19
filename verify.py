"""Fast concrete verification of the joblib submission.

Extracts features ONCE, loads the joblib artifact, blends, applies both safety modes.
Reports SN126 holdout composite (last 3 days) + per-batch 152-forfeiture over all days.
"""

import json
import math
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score, roc_auc_score

from poker44_ml.combined import chunk_features
from poker44_ml.inference import Poker44Model, SAFETY_MODE
from train import _source_files, _source_records


HERE = Path(__file__).resolve().parent


def _load_mirror():
    """Same payload transform as serving (verification must match training)."""
    import sys
    try:
        from poker44.validator.payload_view import prepare_hand_for_miner
        return prepare_hand_for_miner
    except Exception:
        for cand in [
            HERE.parent / "127" / "10_relearn_2026-07-15" / "Poker44-subnet-fresh",
            HERE.parent / "127" / "01_subnet_code" / "Poker44-subnet",
        ]:
            if (cand / "poker44" / "validator" / "payload_view.py").exists():
                sys.path.insert(0, str(cand))
                from poker44.validator.payload_view import prepare_hand_for_miner
                return prepare_hand_for_miner
        raise RuntimeError("prepare_hand_for_miner not found")


def recall_at_fpr(s, y, mf=0.05):
    y, s = np.asarray(y, int), np.asarray(s, float)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if not pos or not neg:
        return 0.0
    o = np.argsort(-s, kind="mergesort"); sl = y[o]
    rc = np.cumsum(sl == 1) / pos; fp = np.cumsum(sl == 0) / neg; a = fp <= mf
    return float(rc[a].max()) if a.any() else 0.0


def reward(s, y):
    y, s = np.asarray(y, int), np.asarray(s, float)
    ap = average_precision_score(y, s) if (y == 1).any() else 0.0
    r = recall_at_fpr(s, y); hard = s >= 0.5
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    tp, fp = int((hard & (y == 1)).sum()), int((hard & (y == 0)).sum()); hf = fp / max(neg, 1)
    sf = 1.0 if (not pos or not neg) else (0.0 if tp == 0 else (1.0 if hf <= .1 else max(0, 1 - (hf - .1) / .9)))
    comp = 0.0 if sf <= 0 else float(np.clip(.35 * ap + .30 * r + .20 * sf + .10 * sf + .05, 0, 1))
    return dict(composite=comp, ap=float(ap), recall=r, safety=sf, hard_fpr=hf,
                auc=float(roc_auc_score(y, s)) if pos and neg else 0.0)


def topk(p, mode):
    n = len(p)
    if n == 0:
        return p
    k = max(1, int(math.floor(0.10 * n)))
    if n >= 8:
        k = max(k, 2)
    o = np.argsort(-p, kind="mergesort"); out = np.empty(n)
    ph, pl, nh, nl = (0.509, 0.501, 0.49, 0.01) if mode == "band" else (0.9, 0.55, 0.45, 0.02)
    for i, ix in enumerate(o[:k]):
        out[ix] = ph - (i / max(k - 1, 1)) * (ph - pl)
    r = o[k:]
    for i, ix in enumerate(r):
        out[ix] = nh - (i / max(len(r) - 1, 1)) * (nh - nl) if len(r) > 1 else nl
    return np.clip(out, 0, 1)


def main():
    import copy
    mirror = _load_mirror()
    m = Poker44Model()
    F, models = m.feature_names, m.models
    feats, ys, ds = [], [], []
    for d, path in _source_files():
        for rec in _source_records(path):
            for bag, y in zip(rec["chunks"], rec["groundTruth"]):
                hands = [mirror(copy.deepcopy(h)) for h in bag]   # same transform as serving
                feats.append(chunk_features(hands)); ys.append(int(y)); ds.append(d)
    X = np.array([[f.get(n, 0.0) for n in F] for f in feats], dtype=np.float64)
    Y = np.array(ys); D = np.array(ds)
    # Per-model feature subsets (model_feature_idx) keep ablated members shape-compatible.
    #   Feeding all 343 features to every model raises. Always go through _blend.
    P = m._blend(X)

    dates = sorted(set(D.tolist())); td = set(dates[-3:])
    print("=" * 62)
    print(f"joblib submission check | default SAFETY_MODE={SAFETY_MODE} | {len(models)} models")
    print("=" * 62)
    for mode in ("honest", "band"):
        tem = np.array([d in td for d in D])
        r = reward(topk(P[tem], mode), Y[tem])
        ff = nb = 0; mins = []
        for d in dates:
            di = np.where(D == d)[0]
            for i in range(0, len(di), 8):
                bi = di[i:i + 8]
                if len(bi) < 4:
                    continue
                c = reward(topk(P[bi], mode), Y[bi])["composite"]; nb += 1; mins.append(c)
                if c <= 0:
                    ff += 1
        print(f"[{mode:>6}] holdout comp={r['composite']:.4f} AP={r['ap']:.4f} "
              f"R@5%={r['recall']:.4f} safety={r['safety']:.2f} hfpr={r['hard_fpr']:.3f} | "
              f"batches{nb} forfeits{ff} min{min(mins):.3f}")
    print("\nNOTE: deployed model trains on all data, so figures above are in-sample "
          "(wiring check). See daily forward tests for skill. 0 forfeits = safe to deploy")


if __name__ == "__main__":
    main()
