from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import GammaRegressor, LinearRegression

from .data import MarketData
from .utils import ensure_finite


@dataclass
class BaselineResult:
    name: str
    predictions: pd.DataFrame
    metadata: dict[str, Any]


@dataclass
class HarQlikeFit:
    intercept: float
    coefficients: np.ndarray
    n_iter: int
    convergence_warnings: list[str]

    def predict_log_rv(
        self,
        market: MarketData,
        dates: list[pd.Timestamp],
    ) -> np.ndarray:
        values = self.intercept + _har_features(market, dates) @ self.coefficients
        ensure_finite("HAR-QLIKE anchor log prediction", values)
        return values

    def metadata(self) -> dict[str, Any]:
        return {
            "intercept": self.intercept,
            "coefficients": self.coefficients.tolist(),
            "n_iter": self.n_iter,
            "convergence_warnings": self.convergence_warnings,
            "fit_scope": "core_train_only",
            "features": ["logRV_d", "logRV_w", "logRV_m"],
        }


def _har_features(market: MarketData, dates: list[pd.Timestamp]) -> np.ndarray:
    rows = []
    for date in dates:
        index = market.date_to_index[date]
        lag = market.log_rv[index - 22 : index]
        if len(lag) != 22 or not np.isfinite(lag).all():
            raise ValueError(f"Invalid HAR lag window for {date}")
        rows.append([lag[-1], float(np.mean(lag[-5:])), float(np.mean(lag))])
    return np.asarray(rows, dtype=np.float64)


def fit_har_qlike(
    market: MarketData,
    core_dates: list[pd.Timestamp],
) -> HarQlikeFit:
    x_core = _har_features(market, core_dates)
    rv_core = np.asarray(
        [market.rv[market.date_to_index[date]] for date in core_dates],
        dtype=np.float64,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        gamma = GammaRegressor(
            alpha=0.0,
            fit_intercept=True,
            solver="lbfgs",
            max_iter=5000,
            tol=1e-8,
            warm_start=False,
        )
        gamma.fit(x_core, rv_core)
    convergence_messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    if convergence_messages:
        raise RuntimeError(
            "HAR-QLIKE did not converge: " + "; ".join(convergence_messages)
        )
    fit = HarQlikeFit(
        intercept=float(gamma.intercept_),
        coefficients=np.asarray(gamma.coef_, dtype=np.float64),
        n_iter=int(gamma.n_iter_),
        convergence_warnings=convergence_messages,
    )
    ensure_finite("HAR-QLIKE anchor coefficients", fit.coefficients)
    if not np.isfinite(fit.intercept):
        raise FloatingPointError("HAR-QLIKE anchor intercept is NaN/Inf")
    return fit


def fit_baselines(
    market: MarketData,
    core_dates: list[pd.Timestamp],
    test_dates: list[pd.Timestamp],
) -> list[BaselineResult]:
    x_core = _har_features(market, core_dates)
    x_test = _har_features(market, test_dates)
    y_core = np.asarray(
        [market.log_rv[market.date_to_index[date]] for date in core_dates],
        dtype=np.float64,
    )
    true_test = np.asarray(
        [market.rv[market.date_to_index[date]] for date in test_dates],
        dtype=np.float64,
    )
    true_log_test = np.log(true_test)

    rw_log = x_test[:, 0]
    rw_rv = np.exp(rw_log)
    ols = LinearRegression(fit_intercept=True).fit(x_core, y_core)
    ols_log = ols.predict(x_test)
    residuals = y_core - ols.predict(x_core)
    smearing = float(np.mean(np.exp(residuals)))
    ols_rv = smearing * np.exp(ols_log)

    gamma = fit_har_qlike(market, core_dates)
    gamma_log = gamma.predict_log_rv(market, test_dates)
    gamma_rv = np.exp(gamma_log)
    for name, values in [
        ("random_walk", rw_rv),
        ("har_ols", ols_rv),
        ("har_qlike", gamma_rv),
    ]:
        ensure_finite(name, values)
    base = pd.DataFrame(
        {
            "target_date": [date.strftime("%Y-%m-%d") for date in test_dates],
            "true_rv": true_test,
            "true_log_rv": true_log_test,
        }
    )

    def result(
        name: str, predicted_rv: np.ndarray, predicted_log_rv: np.ndarray, metadata: dict[str, Any]
    ) -> BaselineResult:
        frame = base.copy()
        frame["predicted_rv"] = predicted_rv
        frame["predicted_log_rv"] = predicted_log_rv
        return BaselineResult(name, frame, metadata)

    return [
        result("random_walk", rw_rv, rw_log, {"fit": "none"}),
        result(
            "har_ols",
            ols_rv,
            ols_log,
            {
                "duan_smearing": smearing,
                "r2_core_logrv": float(ols.score(x_core, y_core)),
            },
        ),
        result(
            "har_qlike",
            gamma_rv,
            gamma_log,
            {
                "alpha": 0.0,
                "solver": "lbfgs",
                "analytic_gradient": True,
                **gamma.metadata(),
            },
        ),
    ]
