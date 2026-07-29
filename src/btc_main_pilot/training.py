from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from .config import MainPilotConfig
from .losses import exact_qlike_torch
from .model import MainPatchTST, attention_backend_metadata, trainable_parameter_count
from .utils import ensure_finite, stable_hash, utc_now, write_json


@dataclass
class TrainingResult:
    best_epoch: int
    epochs_run: int
    early_stopped: bool
    best_validation_qlike: float
    history: list[dict[str, Any]]
    numerical_failure: bool
    failure_reason: str | None
    training_seconds: float
    peak_gpu_memory_bytes: int


def _optimizer(model: MainPatchTST, config: MainPilotConfig) -> AdamW:
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=config.adam_betas,
        eps=config.adam_epsilon,
    )


def _scheduler(
    optimizer: AdamW,
    config: MainPilotConfig,
    floor_step: int,
) -> LambdaLR:
    if floor_step <= config.warmup_steps:
        raise AssertionError("T_floor must be >100 for a positive cosine phase")
    floor_ratio = config.min_learning_rate / config.learning_rate

    def multiplier(step: int) -> float:
        if step < config.warmup_steps:
            return step / config.warmup_steps
        if step >= floor_step:
            return floor_ratio
        progress = (step - config.warmup_steps) / (
            floor_step - config.warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor_ratio + (1.0 - floor_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=multiplier)


def _move_batch(
    batch: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate_loader(
    model: MainPatchTST,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch_index, raw_batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        batch = _move_batch(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model(batch)
        loss_vector = (
            torch.exp(batch["true_log_rv"].double() - output["predicted_log_rv"])
            - (batch["true_log_rv"].double() - output["predicted_log_rv"])
            - 1.0
        )
        if not bool(torch.isfinite(loss_vector).all()):
            raise FloatingPointError("Validation QLIKE contains NaN/Inf")
        total += float(loss_vector.sum().item())
        count += int(loss_vector.numel())
    if count == 0:
        raise ValueError("Validation loader yielded no observations")
    return total / count


def train_model(
    model: MainPatchTST,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    config: MainPilotConfig,
    horizon_epochs: int,
    checkpoint_dir: Path,
    preprocessor_hash: str,
    scheduler_hash: str,
    fold_name: str,
    logger: logging.Logger,
    device: torch.device,
    resume: bool,
    max_epochs_override: int | None = None,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> TrainingResult:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.physical_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.physical_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    optimizer = _optimizer(model, config)
    steps_per_epoch = math.ceil(len(train_dataset) / config.effective_batch_size)
    floor_step = horizon_epochs * steps_per_epoch
    if floor_step <= config.warmup_steps:
        raise AssertionError(
            f"{fold_name}: T_floor={floor_step} must exceed warmup=100"
        )
    scheduler = _scheduler(optimizer, config, floor_step)
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=config.amp_grad_scaler_initial_scale,
        growth_interval=config.amp_grad_scaler_growth_interval,
        enabled=device.type == "cuda",
    )
    accumulation = max(
        1, math.ceil(config.effective_batch_size / config.physical_batch_size)
    )
    max_epochs = max_epochs_override or config.max_epochs
    start_epoch = 1
    best_epoch = 0
    best_validation = float("inf")
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"
    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["preprocessor_hash"] != preprocessor_hash:
            raise RuntimeError("Resume checkpoint preprocessor hash mismatch")
        if state["scheduler_hash"] != scheduler_hash:
            raise RuntimeError("Resume checkpoint scheduler hash mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["grad_scaler"])
        start_epoch = int(state["epoch"]) + 1
        best_epoch = int(state["best_epoch"])
        best_validation = float(state["best_validation"])
        stale_epochs = int(state["stale_epochs"])
        history = list(state["history"])
        logger.info(
            "RESUME | fold=%s next_epoch=%d best_epoch=%d best_val=%.8f",
            fold_name,
            start_epoch,
            best_epoch,
            best_validation,
        )
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    early_stopped = False
    numerical_failure = False
    failure_reason = None
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(start_epoch, max_epochs + 1):
            model.train()
            train_sum = 0.0
            train_count = 0
            last_gradient_norm = float("nan")
            actual_batches = min(
                len(train_loader),
                max_train_batches if max_train_batches is not None else len(train_loader),
            )
            for batch_index, raw_batch in enumerate(train_loader, start=1):
                if max_train_batches is not None and batch_index > max_train_batches:
                    break
                batch = _move_batch(raw_batch, device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    output = model(batch)
                loss = exact_qlike_torch(
                    batch["true_log_rv"], output["predicted_log_rv"]
                )
                if not bool(torch.isfinite(loss)):
                    logger.error(
                        "NaN/Inf WARNING | fold=%s epoch=%d batch=%d loss=%s",
                        fold_name,
                        epoch,
                        batch_index,
                        loss,
                    )
                    raise FloatingPointError(
                        f"Training loss is NaN/Inf at epoch={epoch} "
                        f"batch={batch_index}"
                    )
                scaled_loss = loss / accumulation
                scaler.scale(scaled_loss).backward()
                batch_size = int(batch["true_log_rv"].numel())
                train_sum += float(loss.item()) * batch_size
                train_count += batch_size
                should_step = batch_index % accumulation == 0 or batch_index == actual_batches
                if should_step:
                    scaler.unscale_(optimizer)
                    last_gradient_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.gradient_clip_norm
                        ).item()
                    )
                    gradients_finite = all(
                        parameter.grad is None
                        or bool(torch.isfinite(parameter.grad).all())
                        for parameter in model.parameters()
                    )
                    if not gradients_finite or not np.isfinite(last_gradient_norm):
                        logger.error(
                            "NaN/Inf WARNING | fold=%s epoch=%d batch=%d gradient",
                            fold_name,
                            epoch,
                            batch_index,
                        )
                        raise FloatingPointError(
                            f"Gradient contains NaN/Inf at epoch={epoch} "
                            f"batch={batch_index} amp_scale={scaler.get_scale():.1f}"
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                logger.info(
                    "TRAIN | fold=%s epoch=%03d/%03d batch=%04d/%04d "
                    "train_qlike=%.8f lr=%.8g grad_norm=%s amp_scale=%.1f",
                    fold_name,
                    epoch,
                    max_epochs,
                    batch_index,
                    actual_batches,
                    train_sum / max(train_count, 1),
                    optimizer.param_groups[0]["lr"],
                    f"{last_gradient_norm:.6f}"
                    if np.isfinite(last_gradient_norm)
                    else "pending",
                    scaler.get_scale(),
                )
            validation = evaluate_loader(
                model, validation_loader, device, max_batches=max_eval_batches
            )
            train_mean = train_sum / max(train_count, 1)
            improved = validation < best_validation - config.min_delta
            if improved:
                best_validation = validation
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            row = {
                "epoch": epoch,
                "train_qlike": train_mean,
                "validation_qlike": validation,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": last_gradient_norm,
                "amp_scale": scaler.get_scale(),
                "improved": improved,
                "stale_epochs": stale_epochs,
            }
            history.append(row)
            metadata = {
                "fold": fold_name,
                "seed": config.seed,
                "objective": config.training_loss,
                "preprocessor_hash": preprocessor_hash,
                "scheduler_hash": scheduler_hash,
                "horizon_epochs": horizon_epochs,
                "T_floor": floor_step,
                "parameter_count": trainable_parameter_count(model),
                "training_config": {
                    "optimizer": config.optimizer,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "adam_betas": config.adam_betas,
                    "adam_epsilon": config.adam_epsilon,
                    "warmup_steps": config.warmup_steps,
                    "scheduler": "linear_warmup_then_cosine_to_floor",
                    "min_learning_rate": config.min_learning_rate,
                    "max_epochs": config.max_epochs,
                    "patience": config.patience,
                    "min_delta": config.min_delta,
                    "gradient_clip_norm": config.gradient_clip_norm,
                    "amp_dtype": "float16_on_cuda",
                    "forecast_head_dtype": "float32",
                    "amp_grad_scaler_initial_scale": (
                        config.amp_grad_scaler_initial_scale
                    ),
                    "amp_grad_scaler_growth_interval": (
                        config.amp_grad_scaler_growth_interval
                    ),
                    "effective_batch_size": config.effective_batch_size,
                },
            }
            state = {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation": best_validation,
                "stale_epochs": stale_epochs,
                "history": history,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "grad_scaler": scaler.state_dict(),
                **metadata,
            }
            torch.save(state, last_path)
            if improved:
                torch.save(state, best_path)
            pd.DataFrame(history).to_csv(
                checkpoint_dir / "training_history.csv", index=False
            )
            write_json(checkpoint_dir / "checkpoint_metadata.json", metadata)
            logger.info(
                "VALIDATE | fold=%s epoch=%03d train_qlike=%.8f "
                "validation_qlike=%.8f best=%.8f patience=%d/%d",
                fold_name,
                epoch,
                train_mean,
                validation,
                best_validation,
                stale_epochs,
                config.patience,
            )
            if stale_epochs >= config.patience:
                early_stopped = True
                logger.info(
                    "EARLY STOP | fold=%s epoch=%d best_epoch=%d best_val=%.8f",
                    fold_name,
                    epoch,
                    best_epoch,
                    best_validation,
                )
                break
    except FloatingPointError as error:
        numerical_failure = True
        failure_reason = str(error)
    epochs_run = history[-1]["epoch"] if history else start_epoch - 1
    if not best_path.exists():
        if numerical_failure:
            raise FloatingPointError(failure_reason)
        raise RuntimeError("No best checkpoint was produced")
    best_state = torch.load(best_path, map_location=device, weights_only=False)
    if best_state["preprocessor_hash"] != preprocessor_hash:
        raise RuntimeError("Best checkpoint feature/preprocessor hash mismatch")
    if best_state["scheduler_hash"] != scheduler_hash:
        raise RuntimeError("Best checkpoint scheduler hash mismatch")
    model.load_state_dict(best_state["model"])
    elapsed = time.monotonic() - started
    peak = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    result = TrainingResult(
        best_epoch=best_epoch,
        epochs_run=int(epochs_run),
        early_stopped=early_stopped,
        best_validation_qlike=best_validation,
        history=history,
        numerical_failure=numerical_failure,
        failure_reason=failure_reason,
        training_seconds=elapsed,
        peak_gpu_memory_bytes=peak,
    )
    write_json(
        checkpoint_dir / "training_result.json",
        {
            **result.__dict__,
            "attention_backend": attention_backend_metadata(device),
            "completed_utc": utc_now(),
        },
    )
    return result


@torch.no_grad()
def predict_dataset(
    model: MainPatchTST,
    dataset: Dataset,
    config: MainPilotConfig,
    device: torch.device,
    scheduler_path: Path,
    expected_scheduler_hash: str,
    max_batches: int | None = None,
) -> pd.DataFrame:
    if not scheduler_path.exists():
        raise RuntimeError(
            "scheduler_horizon.json must exist before any test prediction artifact"
        )
    scheduler_payload = __import__("json").loads(scheduler_path.read_text(encoding="utf-8"))
    if stable_hash(scheduler_payload) != expected_scheduler_hash:
        raise RuntimeError("scheduler_horizon.json hash mismatch before prediction")
    loader = DataLoader(
        dataset,
        batch_size=config.physical_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch_index, raw_batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        batch = _move_batch(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model(batch)
        predicted = output["predicted_rv"].cpu().numpy()
        predicted_log = output["predicted_log_rv"].cpu().numpy()
        true = batch["true_rv"].cpu().numpy()
        true_log = batch["true_log_rv"].cpu().numpy()
        ensure_finite("test predictions", predicted)
        for index, date in enumerate(raw_batch["target_date"]):
            rows.append(
                {
                    "target_date": date,
                    "true_rv": float(true[index]),
                    "true_log_rv": float(true_log[index]),
                    "predicted_rv": float(predicted[index]),
                    "predicted_log_rv": float(predicted_log[index]),
                }
            )
    return pd.DataFrame(rows)


def scheduler_payload(
    e_pilot: int,
    h_cos: int,
    config_hash: str,
    pilot_early_stopped: bool,
) -> dict[str, Any]:
    return {
        "H_cos": int(h_cos),
        "E_pilot": int(e_pilot),
        "config_hash": config_hash,
        "locked_from": "fold_1_core_validation_only_seed_11_exact_qlike",
        "pilot_completed_without_numerical_failure": True,
        "pilot_early_stopped": bool(pilot_early_stopped),
        "pilot_checkpoint_disposition": "destroyed_after_horizon_lock",
        "created_utc": utc_now(),
    }
