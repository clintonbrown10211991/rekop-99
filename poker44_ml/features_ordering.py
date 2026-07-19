"""Ordering-quality features: axes that humanized bots cannot fake cheaply.

Four families (research-backed, absent from every scanned competitor repo):
  od_*   overdispersion: RNG-mixed strategies give exactly-binomial frequency
         variance across time blocks; human propensities wander (rho > 0).
  dr_*   within-batch non-stationarity: humans tilt/drift (JS halves, CUSUM,
         trend); bots are stationary noise around a fixed mean.
  mo_*   human random-generation signatures: repetition-avoidance, adjacency
         steps, momentum - present in humans, absent in RNG mixing.
  pr_*   pot-relative sizing (scale-free, defeats the bb-scale benchmark->live
         shift) incl. solver-grid proximity.

All computed from mirrored miner-visible hands only.
"""
from __future__ import annotations

import math

AGG = {"bet", "raise"}
SOLVER_GRID = (0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.5, 2.0)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _hand_hero_actions(hand):
    hero = (hand.get("metadata") or {}).get("hero_seat")
    out = []
    for a in hand.get("actions") or []:
        if hero is not None and a.get("actor_seat") == hero:
            out.append(a)
    return out


def _spot_key(action, prior_agg):
    street = str(action.get("street") or "").lower()
    pre = street == "preflop"
    return (0 if pre else 1, 1 if prior_agg else 0)


def _collect_spots(chunk):
    """Per hand: hero responses tagged by (street-phase, facing-aggression)."""
    per_hand = []  # list of (hand_idx, spot_key, is_aggressive, pot_ratio or None)
    for hi, hand in enumerate(chunk):
        hero = (hand.get("metadata") or {}).get("hero_seat")
        prior_agg = False
        for a in hand.get("actions") or []:
            at = str(a.get("action_type") or "").lower()
            if hero is not None and a.get("actor_seat") == hero:
                key = _spot_key(a, prior_agg)
                ratio = None
                try:
                    amt = float(a.get("amount") or 0.0)
                    pot = float(a.get("pot_before") or 0.0)
                    if amt > 0 and pot > 1e-9:
                        ratio = amt / pot
                except Exception:
                    ratio = None
                per_hand.append((hi, key, 1 if at in AGG else 0, ratio))
            if at in AGG:
                prior_agg = True
    return per_hand


# ---------- od_* overdispersion ----------
def _dispersion(events, n_hands, blocks=3):
    """events: list of (hand_idx, y). Chi-square/df of block frequencies vs
    pooled binomial. RNG mixing -> ~1.0; human drift -> > 1."""
    if len(events) < 12 or n_hands < blocks:
        return None
    edges = [n_hands * b / blocks for b in range(1, blocks + 1)]
    buckets = [[] for _ in range(blocks)]
    for hi, y in events:
        for b, e in enumerate(edges):
            if hi < e:
                buckets[b].append(y)
                break
    buckets = [b for b in buckets if len(b) >= 3]
    if len(buckets) < 2:
        return None
    p_all = _mean([y for b in buckets for y in b])
    if p_all <= 0.02 or p_all >= 0.98:
        return None
    chi = 0.0
    for b in buckets:
        pb = _mean(b)
        chi += len(b) * (pb - p_all) ** 2 / (p_all * (1 - p_all))
    return chi / (len(buckets) - 1)


# ---------- dr_* drift ----------
def _js_halves(seqs, n_hands):
    """JS divergence of action-type distributions, first vs second half."""
    half = n_hands / 2.0
    c1, c2 = {}, {}
    for hi, key, y, _ in seqs:
        (c1 if hi < half else c2)[(key, y)] = (c1 if hi < half else c2).get((key, y), 0) + 1
    if sum(c1.values()) < 5 or sum(c2.values()) < 5:
        return 0.0
    keys = set(c1) | set(c2)
    t1, t2 = sum(c1.values()), sum(c2.values())
    js = 0.0
    for k in keys:
        p = c1.get(k, 0) / t1
        q = c2.get(k, 0) / t2
        m = (p + q) / 2
        if p > 0:
            js += 0.5 * p * math.log(p / m)
        if q > 0:
            js += 0.5 * q * math.log(q / m)
    return js


def _cusum_max(series):
    if len(series) < 6:
        return 0.0
    m = _mean(series)
    s = _std(series)
    if s < 1e-9:
        return 0.0
    c, worst = 0.0, 0.0
    for x in series:
        c += x - m
        worst = max(worst, abs(c))
    return worst / (s * math.sqrt(len(series)))


def _trend_t(series):
    n = len(series)
    if n < 6:
        return 0.0
    xm = (n - 1) / 2.0
    ym = _mean(series)
    sxx = sum((i - xm) ** 2 for i in range(n))
    sxy = sum((i - xm) * (series[i] - ym) for i in range(n))
    if sxx < 1e-9:
        return 0.0
    b = sxy / sxx
    resid = [series[i] - (ym + b * (i - xm)) for i in range(n)]
    se = (_std(resid) + 1e-9) / math.sqrt(sxx)
    return b / se


# ---------- mo_* momentum ----------
def _repeat_gap(events):
    """Observed same-spot consecutive repeat rate minus independence expectation.
    Humans avoid repetition less than RNG (momentum) -> gap > 0; RNG -> ~0."""
    by_spot = {}
    for hi, key, y, _ in events:
        by_spot.setdefault(key, []).append(y)
    num = den = 0
    exp_sum = 0.0
    for key, ys in by_spot.items():
        if len(ys) < 6:
            continue
        p = _mean(ys)
        e_rep = p * p + (1 - p) * (1 - p)
        for a, b in zip(ys, ys[1:]):
            num += 1 if a == b else 0
            den += 1
            exp_sum += e_rep
    if den < 8:
        return 0.0
    return (num / den) - (exp_sum / den)


def _adjacency_step(events):
    """Mean |step| between consecutive hero bet pot-ratio grid positions.
    Humans slide to nearby sizes (small steps); grid-drawing bots jump."""
    ratios = [r for _, _, _, r in events if r is not None]
    if len(ratios) < 5:
        return -1.0
    def gpos(r):
        return min(range(len(SOLVER_GRID)), key=lambda i: abs(SOLVER_GRID[i] - r))
    pos = [gpos(r) for r in ratios]
    steps = [abs(a - b) for a, b in zip(pos, pos[1:])]
    return _mean(steps)


def ordering_features(chunk):
    out = {}
    n_hands = len(chunk)
    events = _collect_spots(chunk)

    # od_*: dispersion per coarse spot + pooled
    by_spot = {}
    for hi, key, y, r in events:
        by_spot.setdefault(key, []).append((hi, y))
    disps = []
    for si, key in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        d = _dispersion(by_spot.get(key, []), n_hands)
        out[f"od_disp_spot{si}"] = math.log(d) if d else 0.0
        if d:
            disps.append(d)
    all_events = [(hi, y) for hi, key, y, r in events]
    d_all = _dispersion(all_events, n_hands)
    out["od_disp_all"] = math.log(d_all) if d_all else 0.0
    out["od_disp_mean"] = _mean([math.log(d) for d in disps]) if disps else 0.0
    out["od_n_spots_scored"] = float(len(disps))

    # dr_*: within-batch drift
    per_hand_agg = {}
    for hi, key, y, r in events:
        per_hand_agg.setdefault(hi, []).append(y)
    series = [(hi, _mean(ys)) for hi, ys in sorted(per_hand_agg.items())]
    vals = [v for _, v in series]
    out["dr_js_halves"] = _js_halves(events, n_hands)
    out["dr_cusum"] = _cusum_max(vals)
    out["dr_trend_t"] = _trend_t(vals)
    if len(vals) >= 8:
        h = len(vals) // 2
        out["dr_half_absdiff"] = abs(_mean(vals[:h]) - _mean(vals[h:]))
        thirds = [vals[: len(vals) // 3], vals[len(vals) // 3: 2 * len(vals) // 3], vals[2 * len(vals) // 3:]]
        out["dr_block_std"] = _std([_mean(t) for t in thirds if t])
    else:
        out["dr_half_absdiff"] = 0.0
        out["dr_block_std"] = 0.0

    # mo_*: momentum / adjacency
    out["mo_repeat_gap"] = _repeat_gap(events)
    out["mo_adjacency_step"] = _adjacency_step(events)

    # pr_*: pot-relative sizing (scale-free)
    ratios = [r for _, _, _, r in events if r is not None and 0.01 < r < 10]
    if len(ratios) >= 4:
        dists = [min(abs(r - g) for g in SOLVER_GRID) for r in ratios]
        out["pr_grid_dist_mean"] = _mean(dists)
        out["pr_grid_on_share"] = _mean([1.0 if d < 0.04 else 0.0 for d in dists])
        out["pr_ratio_cv"] = _std(ratios) / (_mean(ratios) + 1e-9)
        out["pr_ratio_uniq"] = len({round(r, 2) for r in ratios}) / len(ratios)
        out["pr_ratio_median"] = sorted(ratios)[len(ratios) // 2]
    else:
        out["pr_grid_dist_mean"] = -1.0
        out["pr_grid_on_share"] = -1.0
        out["pr_ratio_cv"] = -1.0
        out["pr_ratio_uniq"] = -1.0
        out["pr_ratio_median"] = -1.0

    out["ord_n_events"] = float(len(events))
    return out
