# Solution B Stable Variant (uid 152) - Final v2 - 2026-07-18

This folder is the active uid152 B deployment package for `sn23`.

The model uses the 2026-07-18 refreshed public benchmark data and an 11-member
ensemble. It includes two feature-subset diversity members, so the inference wrapper
must support `model_feature_idx`.

Expected verification:

```text
Original34 comp=0.9698
Merged100 comp=0.9888
90 batches x 100 hands around 2.8 seconds
```

Run:

```bash
python verify.py
python neurons/miner.py --netuid 126 --wallet.name student --wallet.hotkey buzz-1 --subtensor.network finney --axon.port 8095 --neuron.uid 152 --blacklist.force_validator_permit --logging.debug
```
