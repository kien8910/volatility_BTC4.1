from __future__ import annotations

import warnings
from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import GammaRegressor, LinearRegression

from .config import Fold
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


@dataclass(frozen=True)
class ArchFamilySpec:
    name: str
    vol: str
    p: int
    o: int
    q: int
    distribution: str


ARCH_FAMILY_SPECS = (
    ArchFamilySpec("arch_5_normal", "ARCH", 5, 0, 0, "normal"),
    ArchFamilySpec("garch_1_1_normal", "GARCH", 1, 0, 1, "normal"),
    ArchFamilySpec("gjr_garch_1_1_t", "GARCH", 1, 1, 1, "t"),
    ArchFamilySpec("egarch_1_1_t", "EGARCH", 1, 1, 1, "t"),
)


def _daily_close_returns_percent(
    market: MarketData,
    start: str,
    end: str,
) -> pd.Series:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    mask = (
        market.valid
        & (market.dates >= start_ts)
        & (market.dates <= end_ts)
    )
    values = np.sum(market.raw_returns[mask], axis=1) * 100.0
    ensure_finite("daily close-to-close returns", values)
    return pd.Series(values, index=market.dates[mask], dtype=np.float64)


def fit_arch_family_baselines(
    market: MarketData,
    fold: Fold,
    test_dates: list[pd.Timestamp],
    logger: logging.Logger,
    minimum_core_returns: int = 250,
) -> tuple[list[BaselineResult], list[dict[str, Any]]]:
    """Fit fixed-parameter ARCH-family models on core and filter through OOS.

    Parameters are estimated on core returns only. The fixed parameters are
    then used to update conditional variance sequentially with returns observed
    through t-1. Validation and test outcomes never enter parameter fitting.
    """
    try:
        from arch import arch_model
    except ImportError as error:  # pragma: no cover - exercised on GPU server
        raise RuntimeError(
            "ARCH-family baselines require package 'arch>=7.2,<9'. "
            "Install the project requirements before running the benchmark."
        ) from error

    core_returns = _daily_close_returns_percent(
        market, fold.core_start, fold.core_end
    )
    full_returns = _daily_close_returns_percent(
        market, fold.core_start, fold.test_end
    )
    if len(core_returns) < minimum_core_returns:
        raise RuntimeError(
            f"{fold.name} has only {len(core_returns)} valid core daily returns "
            "for ARCH-family estimation"
        )
    true_rv = np.asarray(
        [market.rv[market.date_to_index[date]] for date in test_dates],
        dtype=np.float64,
    )
    base = pd.DataFrame(
        {
            "target_date": [
                date.strftime("%Y-%m-%d") for date in test_dates
            ],
            "true_rv": true_rv,
            "true_log_rv": np.log(true_rv),
        }
    )
    results: list[BaselineResult] = []
    failures: list[dict[str, Any]] = []
    for spec in ARCH_FAMILY_SPECS:
        logger.info(
            "ECONOMETRIC FIT | fold=%s model=%s core_returns=%d",
            fold.name,
            spec.name,
            len(core_returns),
        )
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                core_model = arch_model(
                    core_returns,
                    mean="Zero",
                    vol=spec.vol,
                    p=spec.p,
                    o=spec.o,
                    q=spec.q,
                    dist=spec.distribution,
                    rescale=False,
                )
                fit = core_model.fit(
                    update_freq=0,
                    disp="off",
                    show_warning=False,
                    tol=1e-8,
                    options={"maxiter": 2000, "ftol": 1e-9},
                )
            warning_messages = [str(item.message) for item in caught]
            if int(fit.convergence_flag) != 0:
                raise RuntimeError(
                    f"optimizer convergence_flag={fit.convergence_flag}"
                )
            full_model = arch_model(
                full_returns,
                mean="Zero",
                vol=spec.vol,
                p=spec.p,
                o=spec.o,
                q=spec.q,
                dist=spec.distribution,
                rescale=False,
            )
            forecast = full_model.forecast(
                params=fit.params,
                horizon=1,
                start=0,
                align="target",
                reindex=True,
            )
            target_index = pd.DatetimeIndex(test_dates)
            predicted_percent_squared = (
                forecast.variance.iloc[:, 0]
                .reindex(target_index)
                .to_numpy(dtype=np.float64)
            )
            predicted_rv = predicted_percent_squared / 10_000.0
            ensure_finite(spec.name, predicted_rv)
            if np.any(predicted_rv <= 0):
                raise FloatingPointError(
                    f"{spec.name} emitted nonpositive variance"
                )
            frame = base.copy()
            frame["predicted_rv"] = predicted_rv
            frame["predicted_log_rv"] = np.log(predicted_rv)
            results.append(
                BaselineResult(
                    spec.name,
                    frame,
                    {
                        "estimation": "maximum_likelihood",
                        "fit_scope": "core_daily_close_returns_only",
                        "state_update": (
                            "fixed core parameters; sequential observed "
                            "returns through t-1"
                        ),
                        "mean": "Zero",
                        "vol": spec.vol,
                        "p": spec.p,
                        "o": spec.o,
                        "q": spec.q,
                        "distribution": spec.distribution,
                        "return_scale": "100 * daily close-to-close log return",
                        "variance_scale_back": 10_000.0,
                        "target": (
                            "next-day realized variance from 5-minute returns"
                        ),
                        "core_return_count": int(len(core_returns)),
                        "loglikelihood": float(fit.loglikelihood),
                        "aic": float(fit.aic),
                        "bic": float(fit.bic),
                        "convergence_flag": int(fit.convergence_flag),
                        "warnings": warning_messages,
                        "parameters": {
                            str(key): float(value)
                            for key, value in fit.params.items()
                        },
                    },
                )
            )
        except Exception as error:
            failure = {
                "fold": fold.name,
                "model": spec.name,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failures.append(failure)
            logger.warning(
                "ECONOMETRIC FAILURE | fold=%s model=%s error=%s",
                fold.name,
                spec.name,
                error,
            )
    return results, failures
