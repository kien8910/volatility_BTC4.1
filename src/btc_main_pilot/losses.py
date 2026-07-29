from __future__ import annotations

import numpy as np
import torch


def exact_qlike_torch(
    true_log_rv: torch.Tensor, predicted_log_rv: torch.Tensor
) -> torch.Tensor:
    u = true_log_rv.double() - predicted_log_rv.double()
    loss = torch.exp(u) - u - 1.0
    return loss.mean()


def exact_qlike_numpy(
    true_rv: np.ndarray, predicted_rv: np.ndarray
) -> np.ndarray:
    true = np.asarray(true_rv, dtype=np.float64)
    predicted = np.asarray(predicted_rv, dtype=np.float64)
    if np.any(true <= 0) or np.any(predicted <= 0):
        raise ValueError("QLIKE requires strictly positive RV")
    u = np.log(true) - np.log(predicted)
    return np.exp(u) - u - 1.0


def gamma_half_deviance(
    true_rv: np.ndarray, predicted_rv: np.ndarray
) -> np.ndarray:
    ratio = np.asarray(true_rv, dtype=np.float64) / np.asarray(
        predicted_rv, dtype=np.float64
    )
    return (2.0 * (ratio - np.log(ratio) - 1.0)) / 2.0


def guarded_gamma_objective_gradient(
    parameters: np.ndarray,
    x: np.ndarray,
    y_rv: np.ndarray,
    alpha: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Mean exact QLIKE + alpha/2 L2; first parameter is unpenalized intercept."""
    beta = np.asarray(parameters, dtype=np.float64)
    design = np.asarray(x, dtype=np.float64)
    target = np.asarray(y_rv, dtype=np.float64)
    eta = beta[0] + design @ beta[1:]
    if not np.isfinite(eta).all():
        return np.inf, np.zeros_like(beta)
    u = np.log(target) - eta
    if not np.isfinite(u).all() or float(np.max(u)) > 700.0:
        return np.inf, np.zeros_like(beta)
    exp_u = np.exp(u)
    loss = np.mean(exp_u - u - 1.0) + 0.5 * alpha * np.dot(beta[1:], beta[1:])
    residual_gradient = 1.0 - exp_u
    gradient = np.empty_like(beta)
    gradient[0] = np.mean(residual_gradient)
    gradient[1:] = (
        design.T @ residual_gradient / len(target) + alpha * beta[1:]
    )
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        return np.inf, np.zeros_like(beta)
    return float(loss), gradient

