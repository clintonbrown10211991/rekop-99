"""C-model feature composition (rekop-99): maximally different basis.

  leader-666  : full leader feature pipeline (features_leader_full)
  + ordering  : our unique weapons (od_/dr_/mo_/pr_)
  + honest    : our signals the leaders lack (rand_/potodds_/state_/grid_/simil_)

Distinct from 232 (our-343 basis) and 152 (our-736 basis): this starts from
the #1 lineage feature extractor and adds our differentiators on top.
"""
from __future__ import annotations

from poker44_ml.features_leader_full import chunk_features as _leader_full
from poker44_ml.features_ordering import ordering_features as _ordering_features
from poker44_ml.features import chunk_features as _our_features

_HONEST_PREFIXES = ("rand_", "potodds_", "state_", "grid_", "simil_")


def chunk_features(chunk):
    f = dict(_leader_full(chunk))                     # leader lineage features (666)
    f.update(_ordering_features(chunk))               # + weapons 20 (overdispersion/drift/momentum/pot-grid)
    for k, v in _our_features(chunk).items():         # + honest signals 19 (leaders lack these)
        if k.startswith(_HONEST_PREFIXES):
            f[k] = v
    return f
