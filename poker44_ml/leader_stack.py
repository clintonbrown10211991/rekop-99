"""Picklable stacked ensemble wrapper for the C-model (rekop-99).

Base learners (LightGBM + XGBoost + CatBoost + ExtraTrees + RandomForest)
feed a logistic-regression meta-learner. Stored as the single object in
artifact["models"], so the standard Poker44Model loader blends it as one
member (model_weights=[1.0], model_feature_idx=[None]).

Defined at module level (not __main__) so joblib can unpickle it at serving.
"""
from __future__ import annotations

import numpy as np


class LeaderStack:
    def __init__(self, members, meta):
        self.members = members
        self.meta = meta

    def predict_proba(self, X):
        P = np.vstack([np.clip(m.predict_proba(X)[:, 1], 0, 1) for m in self.members]).T
        return self.meta.predict_proba(P)
