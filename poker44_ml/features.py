"""Unified miner-visible feature extractor for Poker44 SN126.

chunk_features(chunk) -> dict[str, float]

Operates on a *chunk* = list of miner-visible hand dicts (the DetectionSynapse payload:
streets=[], outcome zeroed, 5-8 sampled actions, amounts bucketed+noised). It computes:

  schema_*  : table/hand structure + action-rate + entropy + pot-size stats
  hero_*    : the labeled seat's own action profile
  rand_*    : randomization / serial-dependence tests   (Walker-Wooders AER 2001)
  potodds_* : pot-odds threshold sharpness               (US8360838)
  state_*   : cross-hand state dependence / tilt proxy    (Wei-Yan 2016)
  grid_*    : bet-size grid geometry                       (Libratus/Pluribus)
  simil_*   : self-similarity / compressibility           (Lee NDSS 2016)

The SAME function is used at training time (on benchmark chunks) and at inference time
(on live DetectionSynapse chunks) so features are identical in both regimes.
"""

from __future__ import annotations

import gzip
import math
from collections import Counter

ACTS = ("fold", "check", "call", "bet", "raise")
STREETS = ("preflop", "flop", "turn", "river")
AGG = {"bet", "raise"}
SOLVER_FRACTIONS = (0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.25, 1.5, 2.0)


def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _entropy(counts):
    tot = sum(counts)
    if tot <= 0:
        return 0.0
    return -sum((c / tot) * math.log(c / tot + 1e-12) for c in counts if c > 0)


def _mean(a):
    return sum(a) / len(a) if a else 0.0


def _std(a):
    if len(a) < 2:
        return 0.0
    m = _mean(a)
    return math.sqrt(sum((x - m) ** 2 for x in a) / len(a))


def _stats(prefix, values, out):
    a = [x for x in values if x is not None]
    if not a:
        for k in ("mean", "std", "min", "max", "cv", "nuniq"):
            out[f"{prefix}_{k}"] = 0.0
        return
    m, s = _mean(a), _std(a)
    out[f"{prefix}_mean"] = m
    out[f"{prefix}_std"] = s
    out[f"{prefix}_min"] = min(a)
    out[f"{prefix}_max"] = max(a)
    out[f"{prefix}_cv"] = s / (abs(m) + 1e-9)
    out[f"{prefix}_nuniq"] = float(len({round(x, 2) for x in a}))


def _runs_z(bits):
    n = len(bits)
    n1 = sum(bits)
    n0 = n - n1
    if n1 == 0 or n0 == 0 or n < 3:
        return 0.0
    runs = 1 + sum(1 for i in range(1, n) if bits[i] != bits[i - 1])
    mu = 1 + 2 * n1 * n0 / n
    var = 2 * n1 * n0 * (2 * n1 * n0 - n) / (n * n * (n - 1))
    if var <= 0:
        return 0.0
    return (runs - mu) / math.sqrt(var)


def _lag1_autocorr(x):
    if len(x) < 3:
        return 0.0
    a, b = x[:-1], x[1:]
    if _std(a) < 1e-9 or _std(b) < 1e-9:
        return 0.0
    ma, mb = _mean(a), _mean(b)
    cov = sum((ai - ma) * (bi - mb) for ai, bi in zip(a, b)) / len(a)
    return cov / (_std(a) * _std(b) + 1e-12)


def _cond_entropy_drop(labels):
    if len(labels) < 4:
        return 0.0

    def H(counter):
        tot = sum(counter.values())
        return _entropy(list(counter.values())) if tot else 0.0

    marg = Counter(labels)
    joint = Counter(zip(labels[:-1], labels[1:]))
    prev = Counter(labels[:-1])
    return H(marg) - (H(joint) - H(prev))


def _z_to_p(z):
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def _fisher(pvals):
    pv = [min(max(p, 1e-6), 1 - 1e-6) for p in pvals if p is not None]
    return -2.0 * sum(math.log(p) for p in pv) if pv else 0.0


def _logistic_slope(x, y):
    """Newton fit P(call)=sigmoid(a+b*x); return slope b (threshold sharpness) and resid."""
    if len(x) < 5 or len(set(y)) < 2:
        return 0.0, 0.0
    mx = _mean(x)
    sx = _std(x) + 1e-9
    xm = [(xi - mx) / sx for xi in x]
    a, b = 0.0, 0.0
    for _ in range(20):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for xi, yi in zip(xm, y):
            p = 1.0 / (1.0 + math.exp(-max(-30, min(30, a + b * xi))))
            w = max(p * (1 - p), 1e-6)
            g0 += (yi - p)
            g1 += (yi - p) * xi
            h00 += w
            h01 += w * xi
            h11 += w * xi * xi
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-9:
            break
        a += (h11 * g0 - h01 * g1) / det
        b += (-h01 * g0 + h00 * g1) / det
        if abs(b) > 60:
            break
    resid = 0.0
    for xi, yi in zip(xm, y):
        p = 1.0 / (1.0 + math.exp(-max(-30, min(30, a + b * xi))))
        resid += (yi - p) ** 2
    return b, resid / len(xm)


def chunk_features(chunk):
    """chunk = list of miner-visible hand dicts. Returns a flat feature dict."""
    out = {}
    n_hands = len(chunk)
    out["schema_n_hands"] = float(n_hands)
    if n_hands == 0:
        return out

    all_acts, hero_acts = Counter(), Counter()
    all_street, hero_street = Counter(), Counter()
    all_amt, all_potb, all_pota = [], [], []
    hero_amt, hero_ratio, hero_potb = [], [], []
    act_per_hand, players_per_hand, hero_per_hand = [], [], []
    hero_agg_bits, callfold_x, callfold_y = [], [], []
    per_hand_agg, size_ratios, sizes = [], [], []
    action_strings = []
    hero_act_seq = []
    h_tot_players = 0

    for hand in chunk:
        meta = hand.get("metadata") or {}
        hero = meta.get("hero_seat") or 0
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        players_per_hand.append(len(players))
        h_tot_players += len(players)
        act_per_hand.append(len(actions))

        seq = []
        mine = 0
        h_agg = 0
        for a in actions:
            at = (a.get("action_type") or "")
            street = (a.get("street") or "").lower()
            amt = _f(a.get("normalized_amount_bb"))
            pb = _f(a.get("pot_before"))
            pa = _f(a.get("pot_after"))
            all_acts[at] += 1
            all_street[street] += 1
            all_amt.append(amt)
            all_potb.append(pb)
            all_pota.append(pa)
            seq.append(at[:1])
            if a.get("actor_seat") == hero:
                mine += 1
                hero_acts[at] += 1
                hero_street[street] += 1
                hero_act_seq.append(at[:1])
                hero_amt.append(amt)
                hero_potb.append(pb)
                hero_ratio.append(amt / (pb + 1e-6))
                hero_agg_bits.append(1 if at in AGG else 0)
                if at in AGG:
                    h_agg += 1
                    if pb > 0:
                        sizes.append(amt)
                        size_ratios.append(amt / (pb + 1e-6))
                if at in ("call", "fold"):
                    callfold_x.append(pb)
                    callfold_y.append(1 if at == "call" else 0)
        hero_per_hand.append(mine)
        per_hand_agg.append(h_agg)
        action_strings.append("".join(seq))

    a_tot = max(1, sum(all_acts.values()))
    h_tot = max(1, sum(hero_acts.values()))

    # ---- schema: action rates + entropy ----
    for k in ACTS:
        out[f"schema_all_{k}"] = all_acts[k] / a_tot
        out[f"schema_hero_{k}"] = hero_acts[k] / h_tot
    for s in STREETS:
        out[f"schema_all_street_{s}"] = all_street[s] / max(1, sum(all_street.values()))
        out[f"schema_hero_street_{s}"] = hero_street[s] / max(1, sum(hero_street.values()))
    out["schema_all_act_entropy"] = _entropy([all_acts[k] for k in ACTS])
    out["schema_hero_act_entropy"] = _entropy([hero_acts[k] for k in ACTS])
    out["schema_hero_action_share"] = h_tot / a_tot
    out["schema_hero_aggression"] = (hero_acts["bet"] + hero_acts["raise"]) / (hero_acts["call"] + 1e-6)
    out["schema_hero_vpip_proxy"] = 1.0 - out["schema_hero_fold"]

    # ---- schema: numeric summaries ----
    _stats("schema_all_amt", all_amt, out)
    _stats("schema_all_potb", all_potb, out)
    _stats("schema_all_pota", all_pota, out)
    _stats("schema_act_per_hand", act_per_hand, out)
    _stats("schema_players", players_per_hand, out)
    _stats("schema_hero_amt", hero_amt, out)
    _stats("schema_hero_potb", hero_potb, out)
    _stats("schema_hero_per_hand", hero_per_hand, out)

    # ---- (D) randomization / serial dependence ----
    z = _runs_z(hero_agg_bits)
    out["rand_runs_z"] = z
    out["rand_runs_absz"] = abs(z)
    out["rand_lag1_autocorr"] = _lag1_autocorr(hero_agg_bits)
    out["rand_cond_entropy_drop"] = _cond_entropy_drop(hero_act_seq)
    pv = [_z_to_p(_runs_z([1 if c in ("b", "r") else 0 for c in s])) for s in action_strings if len(s) >= 4]
    out["rand_fisher_pooled"] = _fisher(pv)
    out["rand_iid_gap"] = abs(z) - abs(out["rand_lag1_autocorr"])

    # ---- (B) pot-odds rationality ----
    if len(callfold_x) >= 5:
        slope, resid = _logistic_slope(callfold_x, callfold_y)
        out["potodds_slope"] = slope
        out["potodds_abs_slope"] = abs(slope)
        out["potodds_resid"] = resid
    else:
        out["potodds_slope"] = out["potodds_abs_slope"] = out["potodds_resid"] = 0.0

    # ---- (E) cross-hand state dependence / tilt ----
    out["state_lag1_agg_autocorr"] = _lag1_autocorr(per_hand_agg)
    if len(per_hand_agg) >= 6:
        half = len(per_hand_agg) // 2
        out["state_half_drift"] = abs(_mean(per_hand_agg[:half]) - _mean(per_hand_agg[half:]))
        out["state_agg_var"] = _std(per_hand_agg) ** 2
    else:
        out["state_half_drift"] = out["state_agg_var"] = 0.0

    # ---- (A) bet-size grid geometry ----
    if size_ratios:
        dists = [min(abs(r - g) for g in SOLVER_FRACTIONS) for r in size_ratios]
        out["grid_mean_dist"] = _mean(dists)
        out["grid_frac_on_grid"] = _mean([1.0 if d < 0.05 else 0.0 for d in dists])
        out["grid_n_unique"] = float(len({round(x, 2) for x in sizes}))
        out["grid_size_cv"] = _std(size_ratios) / (abs(_mean(size_ratios)) + 1e-9)
    else:
        out["grid_mean_dist"] = out["grid_frac_on_grid"] = out["grid_n_unique"] = out["grid_size_cv"] = 0.0

    # ---- (F) self-similarity / compressibility ----
    blob = "|".join(action_strings).encode()
    out["simil_gzip_ratio"] = (len(gzip.compress(blob)) / len(blob)) if len(blob) > 4 else 1.0
    grams = Counter()
    for s in action_strings:
        for i in range(len(s) - 1):
            grams[s[i:i + 2]] += 1
    tot = sum(grams.values())
    out["simil_top_bigram_share"] = (max(grams.values()) / tot) if tot else 0.0
    out["simil_uniq_hand_share"] = len(set(action_strings)) / max(1, len(action_strings))

    return out
