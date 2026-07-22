"""Self-consistency features: measure a player against themselves, never against a norm.

Measured 2026-07-22, the public benchmark and the chunk validators actually grade are
perfectly separable (discriminator AUC 1.0000): live runs 100bb stacks, 6-9 seats and
bets around a third of the pot, while the benchmark is 239bb, 6-max only, betting
three quarters. Of 705 features, only six survived both a distribution-overlap test
and a discrimination test, and every survivor measured the same kind of thing --
dispersion, autocorrelation, first-half-versus-second-half divergence, compressibility.
Nothing that asked "what does this player do" transferred; only "how consistent is this
player with themselves" did.

That is the whole design rule here. A scripted policy repeats itself, and repetition
looks the same whether the stacks are 100bb or 239bb, whether five sit at the table or
nine. Every feature below is computed from one subject's own stream and compared only
to that subject's own distribution, so a population shift cannot move it.

Deliberately excluded: absolute amounts, stack depths, pot sizes, seat identity, table
size, and any rate compared against a fixed threshold. Those are exactly the columns
that failed to transfer.
"""
from __future__ import annotations

import gzip
import math
from collections import Counter

_ACT = {"fold": "F", "check": "K", "call": "C", "bet": "B",
        "raise": "R", "allin": "A", "all_in": "A"}
_STREET = {"preflop": "p", "flop": "f", "turn": "t", "river": "r"}


def _num(value, default=0.0):
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _entropy(counts):
    """Shannon entropy normalised by its own maximum, so length does not leak in."""
    total = sum(counts)
    if total <= 0 or len(counts) <= 1:
        return 0.0
    p = [c / total for c in counts if c > 0]
    h = -sum(x * math.log(x) for x in p)
    return h / math.log(len(p)) if len(p) > 1 else 0.0


def _gini(values):
    """Dispersion that needs no location: scale-free by construction."""
    v = sorted(abs(x) for x in values)
    n = len(v)
    s = sum(v)
    if n == 0 or s <= 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(v))
    return (2.0 * cum) / (n * s) - (n + 1.0) / n


def _run_lengths(seq):
    if not seq:
        return []
    runs, cur = [], 1
    for a, b in zip(seq, seq[1:]):
        if a == b:
            cur += 1
        else:
            runs.append(cur); cur = 1
    runs.append(cur)
    return runs


def _autocorr(values, lag=1):
    n = len(values)
    if n <= lag + 1:
        return 0.0
    mu = sum(values) / n
    var = sum((x - mu) ** 2 for x in values)
    if var <= 1e-12:
        return 0.0
    cov = sum((values[i] - mu) * (values[i + lag] - mu) for i in range(n - lag))
    return cov / var


def _js_divergence(a: Counter, b: Counter):
    keys = set(a) | set(b)
    ta, tb = sum(a.values()), sum(b.values())
    if not keys or ta <= 0 or tb <= 0:
        return 0.0
    out = 0.0
    for k in keys:
        p, q = a.get(k, 0) / ta, b.get(k, 0) / tb
        m = 0.5 * (p + q)
        if p > 0:
            out += 0.5 * p * math.log(p / m)
        if q > 0:
            out += 0.5 * q * math.log(q / m)
    return out / math.log(2)


def _compress_ratio(text: str):
    """How much of the stream is redundant. A replayed policy compresses hard."""
    if not text:
        return 1.0
    raw = text.encode("utf-8")
    return len(gzip.compress(raw, 6)) / max(len(raw), 1)


def _hero_stream(chunk):
    """Per-hand token strings for the subject seat only, plus pot-relative sizes.

    Sizes are expressed as a fraction of the pot before the action, which was the
    best-transferring scale-free quantity measured (distribution overlap 0.24 against
    0.04 for raw bb), and then bucketed, so the exact number never reaches a feature.
    """
    seqs, sizes, streets = [], [], []
    for hand in chunk:
        meta = hand.get("metadata") or {}
        hero = meta.get("hero_seat")
        bb = _num(meta.get("bb"))
        toks, hand_sizes = [], []
        for a in hand.get("actions") or []:
            if a.get("actor_seat") != hero:
                continue
            code = _ACT.get(str(a.get("action_type", "")).lower(), "?")
            st = _STREET.get(str(a.get("street", "")).lower(), "?")
            toks.append(st + code)
            amt = _num(a.get("normalized_amount_bb"))
            pot_bb = _num(a.get("pot_before")) / bb if bb > 0 else 0.0
            if amt > 0 and pot_bb > 0:
                hand_sizes.append(amt / pot_bb)
        if toks:
            seqs.append("".join(toks))
            streets.append(len({t[0] for t in toks}))
            sizes.extend(hand_sizes)
    return seqs, sizes, streets


# Pot-fraction buckets. Boundaries sit where both distributions have mass: live
# clusters near a third of the pot, the benchmark spreads from a half upward.
_POT_EDGES = (0.15, 0.25, 0.35, 0.5, 0.75, 1.25)


def _bucket(fraction):
    for i, edge in enumerate(_POT_EDGES):
        if fraction <= edge:
            return i
    return len(_POT_EDGES)


def selfconsistency_features(chunk):
    """Everything here answers 'how much does this subject repeat themselves?'"""
    f = {}
    seqs, sizes, streets = _hero_stream(chunk)
    n = len(seqs)
    if n == 0:
        return {f"sc_{k}": 0.0 for k in (
            "seq_uniq", "seq_top1", "seq_entropy", "seq_gzip", "seq_gzip_shuffled_gap",
            "run_mean", "run_max", "run_gini", "act_entropy", "act_gini",
            "size_gini", "size_entropy", "size_autocorr", "size_bucket_top1",
            "half_js_seq", "half_js_act", "half_js_size", "streets_entropy",
            "len_gini", "len_autocorr", "novelty_rate", "repeat_gap_mean")}

    # --- sequence-level repetition -----------------------------------------
    cs = Counter(seqs)
    f["sc_seq_uniq"] = len(cs) / n
    f["sc_seq_top1"] = max(cs.values()) / n
    f["sc_seq_entropy"] = _entropy(list(cs.values()))
    joined = "|".join(seqs)
    f["sc_seq_gzip"] = _compress_ratio(joined)
    # Compare against the same tokens in sorted order: isolates ORDER redundancy
    # from mere symbol frequency, which differs between the two distributions.
    f["sc_seq_gzip_shuffled_gap"] = _compress_ratio("|".join(sorted(seqs))) - f["sc_seq_gzip"]

    runs = _run_lengths(seqs)
    f["sc_run_mean"] = (sum(runs) / len(runs)) / n if runs else 0.0
    f["sc_run_max"] = (max(runs) / n) if runs else 0.0
    f["sc_run_gini"] = _gini(runs)

    # --- action-token level -------------------------------------------------
    acts = Counter(t for s in seqs for t in (s[i:i + 2] for i in range(0, len(s), 2)))
    f["sc_act_entropy"] = _entropy(list(acts.values()))
    f["sc_act_gini"] = _gini(list(acts.values()))

    # --- sizing, only ever as pot fraction and bucket ------------------------
    if sizes:
        f["sc_size_gini"] = _gini(sizes)
        bc = Counter(_bucket(x) for x in sizes)
        f["sc_size_entropy"] = _entropy(list(bc.values()))
        f["sc_size_bucket_top1"] = max(bc.values()) / len(sizes)
        f["sc_size_autocorr"] = _autocorr(sizes, 1)
    else:
        f["sc_size_gini"] = f["sc_size_entropy"] = 0.0
        f["sc_size_bucket_top1"] = f["sc_size_autocorr"] = 0.0

    # --- first half against second half: drift within the subject's own play --
    mid = max(1, n // 2)
    f["sc_half_js_seq"] = _js_divergence(Counter(seqs[:mid]), Counter(seqs[mid:]))
    ha = Counter(t for s in seqs[:mid] for t in (s[i:i+2] for i in range(0, len(s), 2)))
    hb = Counter(t for s in seqs[mid:] for t in (s[i:i+2] for i in range(0, len(s), 2)))
    f["sc_half_js_act"] = _js_divergence(ha, hb)
    if sizes:
        smid = max(1, len(sizes) // 2)
        f["sc_half_js_size"] = _js_divergence(
            Counter(_bucket(x) for x in sizes[:smid]),
            Counter(_bucket(x) for x in sizes[smid:]))
    else:
        f["sc_half_js_size"] = 0.0

    # --- hand shape -----------------------------------------------------------
    f["sc_streets_entropy"] = _entropy(list(Counter(streets).values()))
    lens = [len(s) // 2 for s in seqs]
    f["sc_len_gini"] = _gini(lens)
    f["sc_len_autocorr"] = _autocorr([float(x) for x in lens], 1)

    # --- how fast does the subject stop inventing new patterns? ---------------
    seen, first_seen, gaps = set(), 0, []
    last_pos = {}
    for i, s in enumerate(seqs):
        if s not in seen:
            seen.add(s); first_seen += 1
        else:
            gaps.append(i - last_pos[s])
        last_pos[s] = i
    f["sc_novelty_rate"] = first_seen / n
    f["sc_repeat_gap_mean"] = (sum(gaps) / len(gaps)) / n if gaps else 1.0
    return f
