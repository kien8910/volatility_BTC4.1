from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .losses import exact_qlike_numpy


def prediction_metrics(
    frame: pd.DataFrame,
    spike_threshold: float,
    model_name: str,
) -> dict[str, Any]:
    true = frame["true_rv"].to_numpy(dtype=np.float64)
    predicted = frame["predicted_rv"].to_numpy(dtype=np.float64)
    true_log = frame["true_log_rv"].to_numpy(dtype=np.float64)
    predicted_log = frame["predicted_log_rv"].to_numpy(dtype=np.float64)
    finite = (
        np.isfinite(true)
        & np.isfinite(predicted)
        & np.isfinite(true_log)
        & np.isfinite(predicted_log)
        & (true > 0)
        & (predicted > 0)
    )
    n_bad = int((~finite).sum())
    if not finite.all():
        true = true[finite]
        predicted = predicted[finite]
        true_log = true_log[finite]
        predicted_log = predicted_log[finite]
    qlike = exact_qlike_numpy(true, predicted)
    spike = true > spike_threshold
    error = true_log - predicted_log
    return {
        "model": model_name,
        "n_predictions": int(len(frame)),
        "n_nan_inf_or_nonpositive": n_bad,
        "mean_qlike": float(np.mean(qlike)),
        "sum_qlike": float(np.sum(qlike)),
        "normal_qlike": float(np.mean(qlike[~spike])) if (~spike).any() else None,
        "spike_qlike": float(np.mean(qlike[spike])) if spike.any() else None,
        "normal_n": int((~spike).sum()),
        "spike_n": int(spike.sum()),
        "r2_logrv": float(r2_score(true_log, predicted_log)),
        "rmse_logrv": float(np.sqrt(np.mean(error**2))),
        "mae_logrv": float(np.mean(np.abs(error))),
    }

