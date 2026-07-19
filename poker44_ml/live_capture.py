"""Diagnostic capture of live validator payloads (local-only, gitignored).

ATTESTATION: captures exist solely to diagnose the benchmark->live
distribution gap (feature drift). They are NEVER used as training input;
training uses only the public benchmark releases.

Design:
  * Fail-safe: every path is wrapped - a capture error can never affect
    serving or scoring.
  * Dedup: at most one capture per unique chunk-set sha256 (the served
    paper rotates ~daily, so this stays tiny: ~1 gzip file per round).
  * Local-only: output dir is gitignored; nothing is transmitted.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
from pathlib import Path

_DIR = Path(os.getenv("POKER44_CAPTURE_DIR", "capture"))
_LOCK = threading.Lock()
_seen: set | None = None
_MAX_CAPTURES = int(os.getenv("POKER44_CAPTURE_MAX", "40"))


def _seen_path() -> Path:
    return _DIR / "seen.txt"


def _load_seen() -> set:
    global _seen
    if _seen is None:
        try:
            _seen = set(_seen_path().read_text(encoding="utf-8").split())
        except Exception:
            _seen = set()
    return _seen


def capture(chunks, sha: str | None = None) -> bool:
    """Persist one unique chunk-set; returns True if a new capture was written."""
    try:
        if not chunks:
            return False
        _DIR.mkdir(parents=True, exist_ok=True)
        if sha is None:
            sha = hashlib.sha256(
                json.dumps(chunks, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
        with _LOCK:
            seen = _load_seen()
            if sha in seen or len(seen) >= _MAX_CAPTURES:
                return False
            seen.add(sha)
            with open(_seen_path(), "a", encoding="utf-8") as f:
                f.write(sha + "\n")
        out = _DIR / f"chunks_{sha[:16]}.json.gz"
        with gzip.open(out, "wt", encoding="utf-8") as f:
            json.dump(chunks, f, default=str)
        return True
    except Exception:
        return False
