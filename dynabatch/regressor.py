"""
Internal module: XGBoost-backed optimal batch-size selection.

The trained model lives at ``dynabatch/models/regressor.ubj``.  Its expected
feature set is declared in ``_CANDIDATE_STATS`` and ``_BASELINE_STATS`` below.
``_EXPERIMENTAL_STATS`` records the additional statistics that were explored
during model development but are not used by the current artifact — they are
the reference for future retraining, not inline commented-out code.

To experiment with extra features:
  1. Add the desired stat names to ``_CANDIDATE_STATS`` / ``_BASELINE_STATS``.
  2. Re-run the training notebook with the expanded feature set.
  3. Replace ``models/regressor.ubj`` before enabling in production.
"""

import os
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from functools import cache  # Python 3.9+
except ImportError:
    from functools import lru_cache as cache


_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "regressor.ubj")
_regressor: xgb.XGBRegressor | None = None

# ---------------------------------------------------------------------------
# Feature schema — single source of truth for what the model expects
# ---------------------------------------------------------------------------

# Stats computed for each candidate batch (DataFrame suffix: _y).
_CANDIDATE_STATS: dict[str, list[str]] = {
    "token": ["mean", "std", "sum", "max"],
    "word": ["mean", "sum", "max"],
    "char": ["sum"],
}

# Stats computed for the baseline (first/hardest) batch (DataFrame suffix: _x).
_BASELINE_STATS: dict[str, list[str]] = {
    "token": ["mean", "sum", "max"],
    "word": ["mean", "sum", "max"],
    "char": ["sum"],
}

# Experimental stats tracked during model development — not active in the
# current regressor.  To enable, move entries into the dicts above and retrain.
_EXPERIMENTAL_STATS: dict[str, list[str]] = {
    "token": ["min", "median", "mode"],
    "word": ["std", "min", "median", "mode"],
    "char": ["mean", "std", "min", "median", "mode", "max"],
}


# ---------------------------------------------------------------------------
# Regressor access (lazy-loaded to avoid import-time I/O)
# ---------------------------------------------------------------------------


@cache
def get_regressor() -> xgb.XGBRegressor:
    """Return the module-level regressor, loading from disk on first call."""
    global _regressor
    if _regressor is None:
        _regressor = xgb.XGBRegressor()
        _regressor.load_model(_MODEL_PATH)
    return _regressor


# ---------------------------------------------------------------------------
# Feature construction helpers
# ---------------------------------------------------------------------------


def _stat_value(arr: np.ndarray, stat: str) -> float:
    if stat == "mean":
        return float(arr.mean())
    if stat == "std":
        return float(arr.std())
    if stat == "sum":
        return float(arr.sum())
    if stat == "max":
        return float(arr.max())
    if stat == "min":
        return float(arr.min())
    if stat == "median":
        return float(np.median(arr))
    if stat == "mode":
        return float(np.bincount(arr).argmax())
    raise ValueError(f"Unknown stat: {stat!r}")


def build_baseline_features(
    token_lengths: np.ndarray,
    word_lengths: np.ndarray,
    char_lengths: np.ndarray,
    min_batch_size: int,
    n_candidates: int,
) -> dict[str, list]:
    """
    Compute the fixed-baseline feature vectors for the first (hardest) batch.

    Each scalar is broadcast to a list of length ``n_candidates`` so the result
    can be row-aligned with per-candidate features in a single DataFrame.
    """
    tl = token_lengths[:min_batch_size]
    wl = word_lengths[:min_batch_size]
    cl = char_lengths[:min_batch_size]

    arrays = {"token": tl, "word": wl, "char": cl}
    features: dict[str, list] = {"batch_size_x": [min_batch_size] * n_candidates}
    for prefix, arr in arrays.items():
        for stat in _BASELINE_STATS.get(prefix, []):
            features[f"{prefix}_{stat}_x"] = [_stat_value(arr, stat)] * n_candidates
    return features


def _build_candidate_features(
    token_lengths: np.ndarray,
    word_lengths: np.ndarray,
    char_lengths: np.ndarray,
    candidate_batch_sizes: np.ndarray,
) -> dict[str, list]:
    arrays = {"token": token_lengths, "word": word_lengths, "char": char_lengths}
    features: dict[str, list] = {"batch_size_y": list(candidate_batch_sizes)}
    for prefix, arr in arrays.items():
        for stat in _CANDIDATE_STATS.get(prefix, []):
            features[f"{prefix}_{stat}_y"] = [_stat_value(arr[:bs], stat) for bs in candidate_batch_sizes]
    return features


# ---------------------------------------------------------------------------
# Batch-size selection
# ---------------------------------------------------------------------------


def select_optimal_batch_size(
    token_lengths: np.ndarray,
    word_lengths: np.ndarray,
    char_lengths: np.ndarray,
    baseline_features: dict[str, Any],
    threshold: float,
    candidate_batch_sizes: np.ndarray,
) -> int:
    """
    Use the pre-trained regressor to find the largest batch size whose predicted
    memory pressure stays at or below ``threshold`` relative to the first batch.

    Returns the largest passing candidate, or the baseline batch size when no
    candidate passes the threshold.
    """
    regressor = get_regressor()

    candidate_features = _build_candidate_features(token_lengths, word_lengths, char_lengths, candidate_batch_sizes)
    feature_df = pd.DataFrame({**candidate_features, **baseline_features})

    feature_df["batch_size_diff"] = feature_df["batch_size_y"] / feature_df["batch_size_x"]
    feature_df["token_max_diff"] = feature_df["token_max_y"] / feature_df["token_max_x"]
    feature_df["token_mean_diff"] = feature_df["token_mean_y"] / feature_df["token_mean_x"]
    feature_df["token_sum_diff"] = feature_df["token_sum_y"] / feature_df["token_sum_x"]
    feature_df["word_max_diff"] = feature_df["word_max_y"] / feature_df["word_max_x"]
    feature_df["word_mean_diff"] = feature_df["word_mean_y"] / feature_df["word_mean_x"]
    feature_df["word_sum_diff"] = feature_df["word_sum_y"] / feature_df["word_sum_x"]
    feature_df["char_sum_diff"] = feature_df["char_sum_y"] / feature_df["char_sum_x"]

    selected = feature_df[regressor.get_booster().feature_names]
    preds_raw = regressor.predict(selected)
    preds = (preds_raw <= threshold).astype(int)
    optimal = int(np.max(preds * candidate_batch_sizes))
    return optimal if optimal > 0 else int(baseline_features["batch_size_x"][0])
