"""Offline transfer-gap analyzer for captured live payloads.

Runs OUTSIDE the miner process (zero score impact): reads the gzip captures
written by poker44_ml.live_capture, extracts features with the SAME extractor
the model uses, and reports the live feature distribution. Feed it a benchmark
stats file (optional) to get per-feature z-scores that surface which features
drift benchmark->live (the ones worth dropping / normalizing).

Usage:
    python diag_capture.py                       # dump live feature stats
    python diag_capture.py --bench bench_stats.json   # add z-scores vs benchmark
    python diag_capture.py --top 40 --only-bb    # focus on raw-magnitude _bb cols

Build a benchmark stats file once from your training matrix:
    {"feature_name": {"mean": <float>, "std": <float>}, ...}
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
from pathlib import Path

import numpy as np

from poker44_ml.combined import chunk_features


def _load_captures(cap_dir: Path) -> list:
    chunks = []
    for path in sorted(glob.glob(str(cap_dir / "chunks_*.json.gz"))):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
            for chunk in data:
                if isinstance(chunk, list) and chunk:
                    chunks.append(chunk)
        except Exception as exc:
            print(f"[skip] {path}: {exc}")
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.getenv("POKER44_CAPTURE_DIR", "capture"))
    ap.add_argument("--bench", default="", help="benchmark stats json {feat:{mean,std}}")
    ap.add_argument("--top", type=int, default=30, help="show N most-drifted features")
    ap.add_argument("--only-bb", action="store_true", help="restrict to raw _bb magnitude cols")
    args = ap.parse_args()

    chunks = _load_captures(Path(args.dir))
    if not chunks:
        print(f"No captures under {args.dir!r}. (Miner writes them per unique daily paper.)")
        return

    feats = [chunk_features(c) for c in chunks]
    names = sorted({k for f in feats for k in f})
    X = np.array([[f.get(n, 0.0) for n in names] for f in feats], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    live_mean, live_std = X.mean(axis=0), X.std(axis=0)

    bench = {}
    if args.bench:
        try:
            bench = json.loads(Path(args.bench).read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] could not read benchmark stats: {exc}")

    rows = []
    for i, name in enumerate(names):
        if args.only_bb and "_bb" not in name:
            continue
        z = float("nan")
        if name in bench:
            b = bench[name]
            denom = float(b.get("std", 0.0)) or 1e-9
            z = abs(live_mean[i] - float(b.get("mean", 0.0))) / denom
        rows.append((name, live_mean[i], live_std[i], z))

    # Rank by drift z-score if we have a benchmark, else by |live_mean| (magnitude cols surface).
    has_bench = bool(bench)
    rows.sort(key=lambda r: (r[3] if not math.isnan(r[3]) else -1.0) if has_bench else abs(r[1]),
              reverse=True)

    print(f"captured_chunks={len(chunks)}  features={len(names)}  "
          f"{'z-scored vs benchmark' if has_bench else '(no benchmark: showing by magnitude)'}")
    print(f"{'feature':45s} {'live_mean':>12s} {'live_std':>12s} {'z_drift':>9s}")
    for name, m, s, z in rows[: args.top]:
        zs = f"{z:9.2f}" if not math.isnan(z) else "     n/a"
        flag = "  DRIFT" if (has_bench and not math.isnan(z) and z > 5.0) else ""
        print(f"{name:45s} {m:12.4f} {s:12.4f} {zs}{flag}")

    if has_bench:
        drifted = [r[0] for r in rows if not math.isnan(r[3]) and r[3] > 5.0]
        print(f"\n{len(drifted)} features with z>5 (drop candidates): {drifted[:60]}")


if __name__ == "__main__":
    main()
