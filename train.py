"""Solution C (rank ensemble, uid152) - final recipe v3 (2026-07-19).

Key points (all measurement-backed):
  - payload mirroring: without it, serving collapses 0.92 -> 0.00
  - mixed-shape training: native (30-40 hands) + merged-100 batches. Live queries use
    100-hand batches; native-only training scores 0.78 there, mixed scores 0.95 (+0.17)
  - merged boost 1.3 and recency halflife 3: strongest recent time-holdout recipe
  - weighted member-rank blending: +0.0046 holdout reward over probability blending
  - no logistic member: it did not converge and slightly reduced holdout reward
Usage: python train.py --all   (final training on all data)
"""
import copy
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from poker44_ml.combined import chunk_features


OUT = HERE / "model"; OUT.mkdir(exist_ok=True)

MIN_DATE = "2026-07-06"        # current-generator era only
HALFLIFE = 3.0                 # prioritize the current generator regime
MERGED_BOOST = 1.3             # live requests use the merged 100-hand shape
MAX_MERGED_PER_DAY_LABEL = 60


def _source_files():
    """Return one source per date from bundled JSON and downloaded JSONL data."""
    bundled_dirs = [
        HERE / "02_benchmark_data" / "chunks",
        HERE.parent / "127" / "02_benchmark_data" / "chunks",
    ]
    download_dirs = [
        HERE / "downloads" / "poker44_benchmark",
        HERE.parent / "sn21" / "downloads" / "poker44_benchmark",
    ]
    by_date = {}
    for chunk_dir in bundled_dirs:
        if not chunk_dir.exists():
            continue
        for path in sorted(chunk_dir.glob("*.json")):
            if path.stem >= MIN_DATE:
                by_date.setdefault(path.stem, path)
    for download_dir in download_dirs:
        if not download_dir.exists():
            continue
        for date_dir in sorted(download_dir.iterdir()):
            path = date_dir / "chunks.jsonl"
            if date_dir.is_dir() and date_dir.name >= MIN_DATE and path.exists():
                by_date.setdefault(date_dir.name, path)
    if not by_date:
        raise RuntimeError(
            "No benchmark chunks found. Expected 02_benchmark_data/chunks or "
            "sn21/downloads/poker44_benchmark beside this project."
        )
    return [(date, by_date[date]) for date in sorted(by_date)]


def _source_records(path):
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    yield from payload["chunks"]


def _mirror():
    try:
        from poker44.validator.payload_view import prepare_hand_for_miner
        return prepare_hand_for_miner
    except Exception:
        for c in [HERE.parent / "127" / "10_relearn_2026-07-15" / "Poker44-subnet-fresh",
                  HERE.parent / "127" / "01_subnet_code" / "Poker44-subnet"]:
            if (c / "poker44" / "validator" / "payload_view.py").exists():
                sys.path.insert(0, str(c))
                from poker44.validator.payload_view import prepare_hand_for_miner
                return prepare_hand_for_miner
    raise RuntimeError("prepare_hand_for_miner not found")


def load(mirror):
    """Native batches + merged-100 batches (3 same-day same-label batches concatenated)."""
    feats, ys, dates, shapes = [], [], [], []   # shape 0=native, 1=merged100
    for d, path in _source_files():
        per_label = {0: [], 1: []}
        for rec in _source_records(path):
            for bag, y in zip(rec["chunks"], rec["groundTruth"]):
                hands = [mirror(copy.deepcopy(h)) for h in bag]   # same transform as serving
                feats.append(chunk_features(hands)); ys.append(int(y))
                dates.append(d); shapes.append(0)
                per_label[int(y)].append(hands)
        for lab in (0, 1):
            same = per_label[lab]
            cnt = 0
            for i in range(len(same)):
                if cnt >= MAX_MERGED_PER_DAY_LABEL or len(same) < 3:
                    break
                merged = (same[i] + same[(i + 1) % len(same)] + same[(i + 2) % len(same)])[:100]
                if len(merged) < 80:
                    continue
                feats.append(chunk_features(merged)); ys.append(lab)
                dates.append(d); shapes.append(1); cnt += 1
    return feats, np.array(ys, np.int32), np.array(dates), np.array(shapes)


def main(train_on_all=True):
    mirror = _mirror()
    print("Solution C: mirrored + mixed shapes, halflife 3, weighted-rank ensemble")
    feats, y, dates, shapes = load(mirror)
    cols = sorted({k for f in feats for k in f})
    X = np.array([[f.get(n, 0.0) for n in cols] for f in feats], np.float64)
    print(f"rows={len(y)} (native {(shapes==0).sum()}, merged100 {(shapes==1).sum()}) features={len(cols)} bot_rate={y.mean():.3f}")

    held = [] if train_on_all else sorted(set(dates.tolist()))[-1:]
    mask = np.ones(len(y), bool) if train_on_all else np.array([d not in set(held) for d in dates])
    uniq = sorted(set(dates.tolist())); dpos = {d: i for i, d in enumerate(uniq)}
    parr = np.array([dpos[d] for d in dates], float)
    sw = (np.power(0.5, (parr.max() - parr) / HALFLIFE) *
          np.where(shapes == 1, MERGED_BOOST, 1.0))[mask]
    print(f"train rows={mask.sum()} halflife={HALFLIFE} merged_boost={MERGED_BOOST}")

    models, weights = [], []
    for seed in range(5):
        m = lgb.LGBMClassifier(n_estimators=700, learning_rate=0.03, num_leaves=63,
                               feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                               min_child_samples=20, random_state=seed, verbose=-1)
        m.fit(X[mask], y[mask], sample_weight=sw); models.append(m); weights.append(1.0)
    for M, w in ((RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1), 1.2),
                 (ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1), 1.2),
                 (HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, random_state=0), 1.2)):
        M.fit(X[mask], y[mask], sample_weight=sw); models.append(M); weights.append(w)

    # Axis-ablated diversity members - counter 'humanized bots' (a second view that cannot
    # see autocorr/rand/state axes; missed bots faked exactly those. Measured +recall +AP)
    feature_idx = [None] * len(models)
    EXCL = {i for i, c in enumerate(cols) if ("autocorr" in c) or c.startswith("rand_") or c.startswith("state_")}
    KEEP = [i for i in range(len(cols)) if i not in EXCL]
    for s in (10, 11):
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
                               feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                               min_child_samples=25, random_state=s, verbose=-1)
        m.fit(X[mask][:, KEEP], y[mask], sample_weight=sw)
        models.append(m); weights.append(0.7); feature_idx.append(KEEP)

    art = {"models": models, "model_weights": weights, "feature_names": cols, "calibrator": None,
           "model_feature_idx": feature_idx,
           "metadata": {"name": "poker44-C-rank", "version": "5", "blend": "weighted_rank",
                        "payload_mirrored": True, "recency_weighted_halflife_days": HALFLIFE,
                        "merged_boost": MERGED_BOOST, "mixed_shapes": True,
                        "safety": "topk_cap", "n_models": len(models),
                        "strategy": "recent live-shape ensemble with weighted member ranks",
                        "trained_on_all": train_on_all, "holdout_dates": held,
                        "training_dates": uniq, "n_train": int(mask.sum())}}
    path = OUT / "poker44_model.joblib"
    joblib.dump(art, path, compress=3)
    print(f"saved {path.name} ({path.stat().st_size/1e6:.1f}MB, {len(models)} models)")


if __name__ == "__main__":
    main(train_on_all="--all" in sys.argv)
