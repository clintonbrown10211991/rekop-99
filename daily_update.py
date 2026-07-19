"""Daily adaptation loop for SN126 — because the live eval data changes every day.

Run once per day (cron), BEFORE the daily eval window closes. It:
  1. pulls the newest public benchmark release (catches generator/schema changes)
  2. retrains the model on all available data (benchmark recency test: full history > recent)
  3. snapshots OUR UID's live windowCompositeScores  (the ONLY real live-perf feedback)
  4. flags drift: new schemaVersion, big handCount jump, or our live score dropping

Set POKER44_MY_UID to your registered miner UID to enable live-score tracking.

    POKER44_MY_UID=123 python daily_update.py

Why full-history retrain (not recent-only): measured on the benchmark, training on all
days (0.9558) beats last-7-days (0.9530) and last-2-days (0.9326) — the benchmark's daily
drift is shallow. The real daily signal is our live window score, not the benchmark.
"""

import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def _find_repo_root(start):
    p = start
    for _ in range(7):
        if (p / "02_benchmark_data").exists():
            return p
        p = p.parent
    return start

HERE = Path(__file__).resolve().parent
_ROOT = None  # set below
ROOT = _find_repo_root(HERE)
LOG = HERE / "daily_log.jsonl"
STATE = HERE / "daily_state.json"
BENCH = "https://api.poker44.net/api/v1/benchmark"
LEADERBOARD = "https://api.poker44.net/api/v1/competition/leaderboard"
MY_UID = os.environ.get("POKER44_MY_UID", "").strip()


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(entry):
    entry["t"] = now()
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps(entry, ensure_ascii=False))


def main():
    prev = json.loads(STATE.read_text()) if STATE.exists() else {}
    warnings = []

    # 1) newest benchmark release + drift detection ---------------------------
    status = get(BENCH)["data"]
    latest = status.get("latestSourceDate")
    schema = status.get("schemaVersion")
    releases = status.get("releases") or []
    hand_counts = [r.get("handCount", 0) for r in releases[:5]]

    if prev.get("schema") and schema != prev["schema"]:
        warnings.append(f"SCHEMA CHANGED {prev['schema']} -> {schema} (may need reparsing)")
    if prev.get("hand_counts") and hand_counts and prev["hand_counts"]:
        if hand_counts[0] and prev["hand_counts"][0] and \
           abs(hand_counts[0] - prev["hand_counts"][0]) / max(prev["hand_counts"][0], 1) > 0.5:
            warnings.append(f"HANDCOUNT JUMP {prev['hand_counts'][0]} -> {hand_counts[0]} (generator may have changed)")

    # 2) pull latest chunks (incremental) + retrain ---------------------------
    new_data = latest and latest != prev.get("latest_pulled")
    if new_data or not (ROOT / "02_benchmark_data" / f"chunks/{latest}.json").exists():
        subprocess.run([sys.executable, str(ROOT / "06_analysis" / "pull_benchmark.py")],
                       check=False)
    retrain = subprocess.run([sys.executable, str(HERE / "train.py"), "--all"],
                             capture_output=True, text=True)
    retrain_ok = retrain.returncode == 0
    train_tail = (retrain.stdout or "").strip().splitlines()[-1:] or [""]

    # 3) our live window scores (the only real feedback) ----------------------
    my_live = None
    if MY_UID:
        try:
            rows = get(LEADERBOARD)["data"]["rows"]
            me = next((r for r in rows if str(r.get("uid")) == MY_UID), None)
            if me:
                wc = me.get("windowCompositeScores") or []
                live = [{"start": w.get("windowStart"), "score": w.get("compositeScore"),
                         "state": w.get("state")} for w in wc]
                my_live = {"uid": int(MY_UID), "rank": me.get("rank"),
                           "incentive_pct": round((me.get("incentive") or 0) * 100, 3),
                           "active": me.get("active"), "windows": live}
                # drift alert: our latest completed window dropped vs the prior one
                done = [w for w in live if w["state"] in ("completed", "active") and w["score"] is not None]
                if len(done) >= 2 and done[-1]["score"] < done[-2]["score"] - 0.03:
                    warnings.append(f"OUR LIVE SCORE DROPPED {done[-2]['score']:.3f} -> {done[-1]['score']:.3f}")
        except Exception as exc:
            warnings.append(f"live-score fetch failed: {exc}")

    STATE.write_text(json.dumps({
        "latest_pulled": latest, "schema": schema, "hand_counts": hand_counts,
    }, indent=2), encoding="utf-8")

    log({
        "event": "daily_update",
        "latest_release": latest, "schema": schema,
        "new_data": bool(new_data), "retrain_ok": retrain_ok, "train": train_tail[0],
        "my_live": my_live,
        "warnings": warnings or None,
    })
    if warnings:
        print("\n⚠️  " + "\n⚠️  ".join(warnings))


if __name__ == "__main__":
    main()
