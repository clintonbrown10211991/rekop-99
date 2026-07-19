"""Creative features - additional features on top of the combined312 set.

Design principles (based on verified noise analysis):
  - Use only categorical axes (action type, street, player) - noise-free.
  - Avoid bet-size and non-contiguous-sequence axes - they collapse under noise.
  - Bot = mechanical regularity -> capture self-similarity, low variance, high correlation.

creative_features(chunk) -> dict  (chunk = mirrored miner-visible hand list)
"""
from __future__ import annotations
import math
from collections import Counter

ACTS = ("fold", "check", "call", "bet", "raise")
STREETS = ("preflop", "flop", "turn", "river")
AGG = {"bet", "raise"}


def _mean(a): return sum(a) / len(a) if a else 0.0
def _var(a):
    if len(a) < 2: return 0.0
    m = _mean(a); return sum((x - m) ** 2 for x in a) / len(a)
def _std(a): return math.sqrt(_var(a))


def _skew(a):
    if len(a) < 3: return 0.0
    m, s = _mean(a), _std(a)
    if s < 1e-9: return 0.0
    return sum(((x - m) / s) ** 3 for x in a) / len(a)


def _kurt(a):
    if len(a) < 4: return 0.0
    m, s = _mean(a), _std(a)
    if s < 1e-9: return 0.0
    return sum(((x - m) / s) ** 4 for x in a) / len(a) - 3.0


def _entropy(counts):
    tot = sum(counts)
    if tot <= 0: return 0.0
    return -sum((c / tot) * math.log(c / tot + 1e-12) for c in counts if c > 0)


def _hand_profile(hand):
    """Categorical profile of one hand (noise-free axes only)."""
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    c = Counter(a.get("action_type") for a in actions)
    n = max(1, sum(c.get(k, 0) for k in ACTS))
    prof = {k: c.get(k, 0) / n for k in ACTS}
    prof["_agg"] = (c.get("bet", 0) + c.get("raise", 0)) / n
    prof["_nact"] = len(actions)
    prof["_nstreet"] = len(streets)
    prof["_nplayer"] = len(players)
    prof["_seq"] = "".join((a.get("action_type") or "?")[:1] for a in actions)
    return prof


def _cosine(v1, v2):
    keys = ACTS
    a = [v1[k] for k in keys]; b = [v2[k] for k in keys]
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9: return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def creative_features(chunk):
    out = {}
    n = len(chunk)
    out["cr_n_hands"] = float(n)
    if n == 0:
        return out
    profs = [_hand_profile(h) for h in chunk]

    # ---- 1) multi-metric self-similarity (bots resemble themselves) ----
    # (a) mean pairwise cosine similarity of action profiles (higher = bot)
    sims = []
    for i in range(min(n, 40)):
        for j in range(i + 1, min(n, 40)):
            sims.append(_cosine(profs[i], profs[j]))
    out["cr_sim_action_mean"] = _mean(sims)
    out["cr_sim_action_std"] = _std(sims)      # variance of similarity itself (bot = low)
    # (b) sequence uniqueness (lower = repetitive = bot)
    seqs = [p["_seq"] for p in profs]
    out["cr_seq_uniq_ratio"] = len(set(seqs)) / n
    grams = Counter()
    for s in seqs:
        for k in range(len(s) - 1):
            grams[s[k:k + 2]] += 1
    tot = sum(grams.values()) or 1
    out["cr_seq_bigram_entropy"] = _entropy(list(grams.values()))
    out["cr_seq_top_bigram_share"] = max(grams.values()) / tot if grams else 0.0

    # ---- 2) higher distribution moments (beyond the mean) ----
    for key, name in [("_agg", "agg"), ("call", "call"), ("fold", "fold"),
                      ("raise", "raise"), ("_nstreet", "street"), ("_nplayer", "player")]:
        vals = [p[key] for p in profs]
        out[f"cr_{name}_std"] = _std(vals)
        out[f"cr_{name}_skew"] = _skew(vals)
        out[f"cr_{name}_kurt"] = _kurt(vals)

    # ---- 3) feature interactions (products/ratios of top axes) ----
    sim = out["cr_sim_action_mean"]
    agg_std = out["cr_agg_std"]
    uniq = out["cr_seq_uniq_ratio"]
    out["cr_x_sim_by_aggstd"] = sim / (agg_std + 1e-6)      # high similarity + low variance = strong bot signal
    out["cr_x_sim_times_uniqinv"] = sim * (1.0 - uniq)       # similar AND repetitive
    out["cr_x_uniq_by_streetstd"] = uniq / (out["cr_street_std"] + 1e-6)
    out["cr_x_aggstd_times_streetstd"] = agg_std * out["cr_street_std"]

    # ---- 4) cross-hand regularity (bot = consistent) ----
    # lag-1 autocorrelation of per-hand aggression - bots repeat patterns
    agg_seq = [p["_agg"] for p in profs]
    if len(agg_seq) >= 3 and _std(agg_seq) > 1e-9:
        a, b = agg_seq[:-1], agg_seq[1:]
        ma, mb = _mean(a), _mean(b)
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / len(a)
        out["cr_agg_autocorr"] = cov / (_std(a) * _std(b) + 1e-12)
    else:
        out["cr_agg_autocorr"] = 0.0
    # first-half vs second-half drift (bot = no drift)
    if n >= 6:
        h = n // 2
        out["cr_half_agg_drift"] = abs(_mean(agg_seq[:h]) - _mean(agg_seq[h:]))
        out["cr_half_uniq_drift"] = abs(
            len(set(seqs[:h])) / h - len(set(seqs[h:])) / (n - h))
    else:
        out["cr_half_agg_drift"] = out["cr_half_uniq_drift"] = 0.0

    return out


if __name__ == "__main__":
    import sys, json, copy
    sys.path.insert(0, "E:/BIT/127/127/10_relearn_2026-07-15/Poker44-subnet-fresh")
    from poker44.validator.payload_view import prepare_hand_for_miner
    d = json.load(open("E:/BIT/127/127/02_benchmark_data/chunks/2026-07-15.json", encoding="utf-8"))
    sc = d["chunks"][0]
    print("=== creative features: bot vs human discrimination test ===")
    import numpy as np
    feats, ys = [], []
    for bag, y in zip(sc["chunks"], sc["groundTruth"]):
        hands = [prepare_hand_for_miner(copy.deepcopy(h)) for h in bag]
        feats.append(creative_features(hands)); ys.append(y)
    cols = sorted(feats[0].keys())
    ys = np.array(ys)
    M = np.array([[f.get(c, 0.0) for c in cols] for f in feats])
    print(f"creative feature count: {len(cols)}")
    print("top8 most discriminating:")
    disc = []
    for j, c in enumerate(cols):
        b, h = M[ys == 1, j], M[ys == 0, j]
        if len(b) and len(h):
            sd = M[:, j].std() + 1e-9
            disc.append((abs(b.mean() - h.mean()) / sd, c, b.mean(), h.mean()))
    for s, c, bm, hm in sorted(disc, reverse=True)[:8]:
        print(f"  {c:28} bot{bm:7.3f} human{hm:7.3f} disc{s:.2f}")
