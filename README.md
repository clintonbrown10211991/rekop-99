# Solution C Rank Ensemble (uid 152) - 2026-07-19

## Strategy

Recent live-shape scoring with weighted member ranks. The model combines
high-capacity tree ensembles with two feature-ablated members, emphasizes merged
100-hand examples, and preserves rank through the top-K safety mapping.

## Configuration

| Item | Value |
|---|---|
| Data | 2026-07-06 through 2026-07-18, mirrored payload view, original 1906 plus merged100 1560 for 3466 rows |
| Artifact recipe | `poker44-C-rank`, version 5 |
| Main ensemble | LGBM x5, RF500, ExtraTrees500, HistGB500 |
| Diversity members | Two LGBM models trained with autocorr, rand, and state axes excluded, 333 features, weight 0.7 |
| Blend | Weighted per-batch member ranks |
| Weights | recency half-life 3.0 x merged weight 1.3 |
| Safety cap | top-K 10 percent, minimum 2, above 0.5 only |
| Defenses | per-chunk try/except, NaN sanitation, deterministic fallback |

## Verification

```text
Leakage-free time holdout: train through 2026-07-15, evaluate 2026-07-16 through 2026-07-18
Mean composite: 0.9682 (previous recipe: 0.9571)
Production-shape smoke test: 100 inputs -> 100 finite scores in 1.10 seconds
Safety: exactly 10/100 scores above 0.5; 239 verification batches with zero forfeits
Artifact: 10 models, 343 features, joblib 49.4 MB
```

## Key History

| Change | Reason |
|---|---|
| Inference speed fix | Avoids recomputing features 343 times per chunk. |
| Mixed-shape training | Original-only training was weaker on live-shape proxy data. |
| Weighted-rank blend | Improved the recent time holdout and reduces scale disagreement between ensemble families. |
| Logistic removal | The weak non-converged member reduced recent holdout quality. |
| Per-model feature subsets | Adds two diversity models for humanized bot patterns. |

## Deploy

```bash
pip install -r requirements.txt
python verify.py
python neurons/miner.py --netuid 126 --wallet.name student --wallet.hotkey buzz-1 --subtensor.network finney --axon.port 8095 --neuron.uid 152 --blacklist.force_validator_permit --logging.debug
```

## Public Model Manifest

When `POKER44_SEND_MODEL_MANIFEST=1`, the miner publishes the transparency
manifest attached to each validator response.

The public repository identity is derived directly from the deployed checkout:

- `repo_url`: normalized from `git config --get remote.origin.url`
- `repo_commit`: the full hash returned by `git rev-parse HEAD`
- `implementation_files`: `miner.py`, `train.py`, and the complete
  `poker44_ml` inference and feature pipeline
- `artifact_sha256`: SHA-256 of the locally deployed joblib artifact

If `POKER44_MODEL_REPO_COMMIT` is also configured, startup fails when it does
not equal the deployed Git HEAD. This prevents a stale or unpublished commit
from being sent to validators.
