import numpy as np
from dataclasses import replace

from btc_main_pilot.config import MainPilotConfig
from btc_main_pilot.losses import (
    exact_qlike_numpy,
    gamma_half_deviance,
    guarded_gamma_objective_gradient,
)
from btc_main_pilot.pipeline import _locked_config_hash


def test_locked_training_configuration():
    config = MainPilotConfig()
    config.validate()
    assert config.optimizer == "AdamW"
    assert config.learning_rate == 3e-4
    assert config.weight_decay == 1e-4
    assert config.warmup_steps == 100
    assert config.max_epochs == 200
    assert config.patience == 20
    assert config.gradient_clip_norm == 1.0
    assert config.amp_grad_scaler_initial_scale == 1024.0
    assert config.amp_grad_scaler_growth_interval == 2000
    assert config.training_loss == "exact_qlike"
    assert config.seed == 11
    assert config.pca_dim == 8
    assert config.verified_maintenance_intervals == (
        (
            "binance_spot_upgrade_2021_08_13",
            "2021-08-13 02:00:00+00:00",
            "2021-08-13 06:25:00+00:00",
        ),
        (
            "binance_spot_upgrade_2021_09_29",
            "2021-09-29 07:00:00+00:00",
            "2021-09-29 08:55:00+00:00",
        ),
    )


def test_scheduler_lock_hash_ignores_only_runtime_batching_controls():
    config = MainPilotConfig()
    runtime_changed = replace(
        config,
        physical_batch_size=8,
        num_workers=4,
        output_dir="another/output",
        embedding_cache_path="another/cache.sqlite",
    )
    assert _locked_config_hash(config) == _locked_config_hash(runtime_changed)
    hyperparameter_changed = replace(config, learning_rate=1e-4)
    assert _locked_config_hash(config) != _locked_config_hash(
        hyperparameter_changed
    )


def test_gamma_half_unit_deviance_equals_exact_qlike_float64():
    true = np.array([1e-8, 0.25, 1.0, 9.0], dtype=np.float64)
    predicted = np.array([2e-8, 0.20, 1.5, 3.0], dtype=np.float64)
    np.testing.assert_allclose(
        gamma_half_deviance(true, predicted),
        exact_qlike_numpy(true, predicted),
        rtol=1e-14,
        atol=1e-14,
    )


def test_guarded_gamma_extreme_u_returns_inf_not_nan():
    x = np.ones((2, 1), dtype=np.float64)
    y = np.array([1.0, 2.0], dtype=np.float64)
    objective, gradient = guarded_gamma_objective_gradient(
        np.array([-1000.0, 0.0]), x, y
    )
    assert np.isinf(objective)
    assert not np.isnan(objective)
    assert np.isfinite(gradient).all()


def test_gamma_gradient_is_analytic_and_intercept_unpenalized():
    x = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    beta = np.array([0.2, 0.3], dtype=np.float64)
    alpha = 0.7
    _, gradient = guarded_gamma_objective_gradient(beta, x, y, alpha)
    epsilon = 1e-7
    numeric = np.empty_like(beta)
    for i in range(len(beta)):
        plus = beta.copy()
        minus = beta.copy()
        plus[i] += epsilon
        minus[i] -= epsilon
        f_plus = guarded_gamma_objective_gradient(plus, x, y, alpha)[0]
        f_minus = guarded_gamma_objective_gradient(minus, x, y, alpha)[0]
        numeric[i] = (f_plus - f_minus) / (2 * epsilon)
    np.testing.assert_allclose(gradient, numeric, rtol=1e-6, atol=1e-7)
