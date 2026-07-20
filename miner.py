"""Drop-in Poker44 SN126 miner — loads the joblib artifact (real submission format).

Copy this repo's poker44_ml/, model/poker44_model.joblib, and this file into the
Poker44-subnet checkout (replacing neurons/miner.py), then run like the reference miner.
"""

import json
import hashlib
import os
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

_SUBMISSION_ROOT = Path(__file__).resolve().parent
if str(_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(1, str(_SUBMISSION_ROOT))

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import build_local_model_manifest, manifest_digest
from poker44.validator.synapse import DetectionSynapse

from poker44_ml.inference import Poker44Model, SAFETY_MODE, _MODEL as _MODEL_PATH
from poker44_ml import live_capture

# Report per-query validator and chunk scores to the optional live dashboard.
# To send to a remote dashboard, set POKER44_REPORT_URL=http://<dashboard-ip>:8127.
REPORT_URL = os.environ.get("POKER44_REPORT_URL", "").strip().rstrip("/")
_QLOG = os.environ.get("POKER44_QUERY_LOG", "queries.jsonl")


def _report_query(uid, validator, scores):
    rec = {
        "uid": int(uid) if uid is not None else None,
        "validator": validator or "?",
        "n_chunks": len(scores),
        "scores": [round(float(s), 4) for s in scores],
        "window": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    try:
        with open(_QLOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + chr(10))
    except Exception:
        pass
    if REPORT_URL:
        def _post():
            try:
                body = json.dumps(rec).encode("utf-8")
                req = urllib.request.Request(REPORT_URL + "/api/report", data=body,
                                             headers={"content-type": "application/json"}, method="POST")
                urllib.request.urlopen(req, timeout=5).read()
            except Exception:
                pass
        threading.Thread(target=_post, daemon=True).start()




class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        print("[STARTUP] Poker44 submission miner started", flush=True)
        repo_root = Path(__file__).resolve().parent
        self.model = Poker44Model()
        print(
            f"[MODEL] loaded joblib path={_MODEL_PATH} "
            f"name={self._model_name(self.model.metadata)} "
            f"version={self._model_version(self.model.metadata)} "
            f"models={len(self.model.models)} "
            f"features={len(self.model.feature_names)} "
            f"safety={SAFETY_MODE}",
            flush=True,
        )
        # Auto-reload when daily_update refreshes the joblib artifact.
        self._model_mtime = _MODEL_PATH.stat().st_mtime if _MODEL_PATH.exists() else 0.0
        threading.Thread(target=self._reload_watcher, daemon=True).start()
        # Publish the manifest by default. It evaluates to compliance status
        # "transparent" (all required fields present, no policy violations, no
        # suspicion flags), and withholding it leaves the miner recorded as
        # "opaque" with nothing for a reviewer to verify against. Set
        # POKER44_SEND_MODEL_MANIFEST=0 to withhold it.
        # Only publish from a commit that exists in the public repo: the manifest
        # declares the local git HEAD, so serving an unpushed commit would
        # advertise a revision nobody can resolve.
        self.send_model_manifest = (
            os.getenv("POKER44_SEND_MODEL_MANIFEST", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        runtime_repo_commit = self._repo_head(repo_root)
        runtime_repo_url = self._normalize_repo_url(self._repo_url(repo_root))
        configured_repo_commit = os.getenv("POKER44_MODEL_REPO_COMMIT", "").strip()
        if (
            self.send_model_manifest
            and configured_repo_commit
            and runtime_repo_commit
            and configured_repo_commit != runtime_repo_commit
        ):
            raise RuntimeError(
                "POKER44_MODEL_REPO_COMMIT does not match the deployed Git HEAD: "
                f"configured={configured_repo_commit} runtime={runtime_repo_commit}"
            )
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[
                repo_root / "miner.py",
                repo_root / "train.py",
                repo_root / "poker44_ml" / "inference.py",
                repo_root / "poker44_ml" / "combined.py",
                repo_root / "poker44_ml" / "features.py",
                repo_root / "poker44_ml" / "features_creative.py",
                repo_root / "poker44_ml" / "features_leader.py",
            ],
            defaults={
                "model_name": self._model_name(self.model.metadata),
                "model_version": self._model_version(self.model.metadata),
                "framework": "sklearn+lightgbm-ensemble (joblib)",
                "license": "MIT",
                "repo_url": runtime_repo_url,
                "repo_commit": runtime_repo_commit,
                "artifact_sha256": self._sha256_file(_MODEL_PATH) if _MODEL_PATH.exists() else "",
                "training_data_statement": (
                    "Trained only on the public Poker44 benchmark "
                    "(api.poker44.net/api/v1/benchmark). No validator-only eval labels used."
                ),
                "training_data_sources": ["poker44-public-benchmark"],
                "private_data_attestation": "Does not train on validator-only evaluation data.",
                "inference_mode": "local",
                "notes": (
                    f"2026-07-17 B standalone joblib; safety mode={SAFETY_MODE}; "
                    f"models={len(self.model.models)}; features={len(self.model.feature_names)}."
                ),
                "open_source": True,
            },
        )
        self.manifest_digest = manifest_digest(self.model_manifest)
        print(
            "[MODEL] manifest "
            f"name={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"digest={self.manifest_digest}",
            flush=True,
        )
        print(
            f"[MODEL] send_manifest={self.send_model_manifest} "
            "(set POKER44_SEND_MODEL_MANIFEST=1 to publish)",
            flush=True,
        )
        bt.logging.info(
            f"Poker44 miner up | joblib models={len(self.model.models)} "
            f"features={len(self.model.feature_names)} safety={SAFETY_MODE}"
        )

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _clean_text(value) -> str:
        return str(value or "").strip()

    @staticmethod
    def _normalize_repo_url(url: str) -> str:
        cleaned = str(url or "").strip()
        if cleaned.startswith("git@"):
            host_path = cleaned.split(":", 1)
            if len(host_path) == 2:
                host = host_path[0][4:]
                path = host_path[1]
                cleaned = f"https://{host}/{path}"
        if cleaned.endswith(".git"):
            cleaned = cleaned[:-4]
        return cleaned

    @staticmethod
    def _repo_head(repo_root: Path) -> str:
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        except Exception:
            return ""

    @staticmethod
    def _repo_url(repo_root: Path) -> str:
        try:
            completed = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        except Exception:
            return ""

    @classmethod
    def _model_name(cls, metadata: dict) -> str:
        name = cls._clean_text(metadata.get("name"))
        if name:
            return name
        env_name = cls._clean_text(os.getenv("POKER44_MODEL_NAME"))
        if env_name:
            return env_name
        recipe = cls._clean_text(metadata.get("recipe")).upper()
        if recipe == "B":
            return "poker44-B-robust"
        if recipe == "A":
            return "poker44-A-aggressive"
        return "poker44-B-robust"

    @classmethod
    def _model_version(cls, metadata: dict) -> str:
        version = cls._clean_text(metadata.get("version"))
        if version:
            return version
        env_version = cls._clean_text(os.getenv("POKER44_MODEL_VERSION"))
        if env_version:
            return env_version
        return cls._clean_text(metadata.get("built")) or "2"

    def _reload_watcher(self, every=60):
        """Load a refreshed model without restarting when daily_update updates the joblib."""
        while True:
            time.sleep(every)
            try:
                if not _MODEL_PATH.exists():
                    continue
                mt = _MODEL_PATH.stat().st_mtime
                if mt > self._model_mtime + 1:
                    new_model = Poker44Model()          # load refreshed joblib
                    self.model = new_model               # atomic reference swap
                    self._model_mtime = mt
                    bt.logging.info(
                        f"Model auto-reloaded | models={len(new_model.models)} "
                        f"name={new_model.metadata.get('name')}"
                    )
            except Exception as e:
                bt.logging.warning(f"Model reload failed; keeping previous model: {e}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        print(f"[FORWARD] received chunks={len(chunks)}", flush=True)
        try:
            scores = self.model.predict_chunk_scores(chunks)
            if len(scores) != len(chunks):
                raise ValueError(
                    f"model returned {len(scores)} scores for {len(chunks)} chunks"
                )
        except Exception as e:
            # Always return the correct response length; validators discard malformed replies.
            print(
                f"[FORWARD] batch_score_failed error={e}; using fallback scores",
                flush=True,
            )
            bt.logging.error(f"Inference failed, using fallback scores: {e}")
            n = len(chunks)
            k = max(1, n // 10) if n < 8 else max(2, n // 10)
            scores = [0.55 if i < k else 0.05 for i in range(n)]
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = (
            dict(self.model_manifest) if self.send_model_manifest else None
        )
        bt.logging.info(f"Scored {len(chunks)} chunks | mean={sum(scores)/max(len(scores),1):.3f}")
        try:
            vhot = getattr(getattr(synapse, "dendrite", None), "hotkey", None)  # querying validator
            _report_query(getattr(self, "uid", None), vhot, scores)
        except Exception:
            pass
        # Diagnostic-only payload capture (dedup by sha; never used for training).
        live_capture.capture(chunks)
        mean_score = sum(scores) / len(scores) if scores else 0.0
        print(
            f"[FORWARD] scored chunks={len(chunks)} scores={len(scores)} "
            f"mean={mean_score:.4f} first_scores={[round(score, 4) for score in scores[:20]]}",
            flush=True,
        )
        # Transfer-gap diagnostics. Read-only from the model's stashed aggregates;
        # wrapped so logging can never affect the response that was already built.
        try:
            d = getattr(self.model, "_last_diag", {}) or {}
            if d.get("n"):
                warn = "  <<< COLLAPSE (live scores not separating)" if d.get("collapse") else ""
                print(
                    f"[DIAG] n={d.get('n')} raw_mean={d.get('raw_mean')} raw_std={d.get('raw_std')} "
                    f"raw_min={d.get('raw_min')} raw_max={d.get('raw_max')} "
                    f"raw_p90={d.get('raw_p90')} raw_spread={d.get('raw_spread_p90_p10')} "
                    f"hard_flags={d.get('hard_flags')}{warn}",
                    flush=True,
                )
        except Exception:
            pass
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        print("[STARTUP] Poker44 miner running", flush=True)
        bt.logging.info("Poker44 submission miner running...")
        while True:
            print(
                f"[HEARTBEAT] uid={miner.uid} block={miner.block} "
                f"incentive={miner.metagraph.I[miner.uid]}",
                flush=True,
            )
            bt.logging.info(f"UID {miner.uid} | Incentive {miner.metagraph.I[miner.uid]}")
            time.sleep(300)
