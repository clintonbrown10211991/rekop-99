"""Feature composition for rekop-99.

  leader-666  : full leader feature pipeline (features_leader_full)
  + ordering  : our unique weapons (od_/dr_/mo_/pr_)
  + honest    : our signals the leaders lack (rand_/potodds_/state_/grid_/simil_)
  + sc_*      : self-consistency measures (features_selfconsistency)

The self-consistency block was added 2026-07-22 after measuring the public benchmark
against the chunk validators actually grade: a discriminator separates them at AUC
1.0000, and of 705 columns only six kept both their distribution and their
discriminating power across the gap -- every survivor a dispersion, autocorrelation or
divergence measure. Features that ask "what does this player do" did not transfer;
features that ask "how consistent is this player with themselves" did.

Emitting a superset is deliberate and safe. Inference selects only the columns the
loaded artifact declares, so an older model that never saw sc_* simply ignores them,
while a self-consistency model finds what it needs. A missing name would silently
become 0.0 for every chunk, so producing more than is asked for is the safe direction.
"""
from __future__ import annotations

from poker44_ml.features_leader_full import chunk_features as _leader_full
from poker44_ml.features_ordering import ordering_features as _ordering_features
from poker44_ml.features import chunk_features as _our_features
from poker44_ml.features_selfconsistency import selfconsistency_features as _selfcons

_HONEST_PREFIXES = ("rand_", "potodds_", "state_", "grid_", "simil_")


def chunk_features(chunk):
    f = dict(_leader_full(chunk))                     # leader lineage features (666)
    f.update(_ordering_features(chunk))               # + weapons 20 (overdispersion/drift/momentum/pot-grid)
    for k, v in _our_features(chunk).items():         # + honest signals 19 (leaders lack these)
        if k.startswith(_HONEST_PREFIXES):
            f[k] = v
    f.update(_selfcons(chunk))                        # + 22 self-consistency measures
    return f
