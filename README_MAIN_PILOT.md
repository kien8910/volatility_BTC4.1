# BTC PatchTST–BGE–FinBERT main-pilot

This package implements the `main-pilot` workflow defined by
`BTC_PATCHTST_FINBERT_MODEL_SPEC.md` plus a separately labeled,
development-only spike diagnostic authorized after the main-pilot result.

It performs one Fold-1 scheduler pilot (core/validation only), locks `H_cos`,
deletes the pilot checkpoint, retrains the primary model on Fold 5 with seed
11, and evaluates the Fold-5 OOS block against Random Walk, HAR-OLS with Duan
smearing, and HAR-QLIKE.

The market loader causally inserts no-trade bars only for the two verified
Binance spot-maintenance closures locked in the specification
(`2021-08-13 02:00–06:25 UTC` and `2021-09-29 07:00–08:55 UTC`). Every other
gap remains invalid. The synthetic-maintenance mask is audit-only and is never
a model feature.

The full command refuses to run without CUDA. BGE and FinBERT are loaded with
`local_files_only=True`; put both Hugging Face model snapshots in the server
cache before starting.

```powershell
$env:PYTHONPATH="src"
python -m btc_main_pilot --profile main-pilot --smoke
```

Before the first full run, export and inspect the development-only news filter
sample:

```bash
PYTHONPATH=src python -m btc_main_pilot \
  --profile main-pilot \
  --review-news-only
```

Then run on a CUDA server:

```bash
PYTHONPATH=src TOKENIZERS_PARALLELISM=false python -m btc_main_pilot \
  --profile main-pilot \
  --market data/BTCUSDT_5min_2018_2025_present.csv \
  --news data/news_clusters.json \
  --output-dir outputs/main_pilot \
  --physical-batch-size 32 \
  --num-workers 4 \
  --resume \
  --confirm-news-filter-reviewed
```

If batch 32 exceeds GPU memory, reduce only `--physical-batch-size` (for
example to 8). Gradient accumulation retains the locked effective batch size
of 32.

The exact-QLIKE path keeps the forecast head in FP32 while the attention
branches use CUDA FP16 autocast. AMP gradient scaling starts at 1024 and is
recorded in checkpoint metadata. Any non-finite pilot gradient aborts before
`scheduler_horizon.json` is created; any non-finite Fold-5 gradient aborts
before OOS predictions or metrics are written.

After upgrading from a run that ended with `numerical_failure=true`, archive
the old scheduler/checkpoint/prediction/metrics artifacts before restarting.
Keep `outputs/main_pilot/cache/article_embeddings.sqlite`: it remains valid and
prevents BGE/FinBERT from being recomputed.

Smoke output is isolated under `outputs/main_pilot/smoke` and uses a
deterministic 768-dimensional test double for BGE/FinBERT. It is never a
research result and never creates the main `scheduler_horizon.json`.

## Development-only spike diagnostic

The diagnostic runs exactly 12 deep experiments: Fold 1-4, seed 11, exact
QLIKE, crossed with `main`, `market_only`, and `hybrid_har`. It reuses the
successful main-pilot `H_cos` without relocking it. Fold 5, final test, COVID,
MCS, loss reweighting, spike oversampling, and five-seed inference are not
opened or run.

Run a CPU-safe three-variant smoke test:

```bash
PYTHONPATH=src python -m btc_main_pilot \
  --profile development-spike-diagnostic \
  --market data/BTCUSDT_5min_2018_2025_present.csv \
  --news data/news_clusters.json \
  --output-dir outputs/spike_diagnostic \
  --smoke \
  --no-resume
```

Run all 12 experiments on a CUDA server:

```bash
PYTHONPATH=src TOKENIZERS_PARALLELISM=false python -u -m btc_main_pilot \
  --profile development-spike-diagnostic \
  --market data/BTCUSDT_5min_2018_2025_present.csv \
  --news data/news_clusters.json \
  --output-dir outputs/spike_diagnostic \
  --embedding-cache outputs/main_pilot/cache/article_embeddings.sqlite \
  --scheduler-path outputs/main_pilot/scheduler_horizon.json \
  --physical-batch-size 32 \
  --num-workers 4 \
  --resume \
  --confirm-news-filter-reviewed
```

The runner writes per-fold predictions/metrics, pooled Fold 1-4 results,
worst-loss days, and the predeclared 3-of-4-fold spike screen under
`outputs/spike_diagnostic`. A completed run is skipped safely by `--resume`;
partial compatible checkpoints resume normally.

## Development-only HAR-anchor regime diagnostic

This follow-up runs exactly 8 experiments: rolling Fold 1-4 crossed with
`har_anchor_market` and `har_anchor_market_text`, always with seed 11 and
exact QLIKE. For every fold, HAR-QLIKE is fitted on the core block only.
Epoch 0 is exactly that HAR forecast because the neural correction head is
initialized to zero. The neural correction is retained only if validation
QLIKE improves by at least `1e-5`; otherwise the saved best checkpoint remains
epoch 0. This isolates the incremental value of text while limiting damage
from regime shifts.

Run the CPU-safe two-variant smoke test:

```bash
PYTHONPATH=src python -m btc_main_pilot \
  --profile development-regime-anchor-diagnostic \
  --market data/BTCUSDT_5min_2018_2025_present.csv \
  --news data/news_clusters.json \
  --output-dir outputs/regime_anchor_diagnostic \
  --smoke \
  --no-resume
```

Run all 8 development experiments on a CUDA server:

```bash
PYTHONPATH=src TOKENIZERS_PARALLELISM=false python -u -m btc_main_pilot \
  --profile development-regime-anchor-diagnostic \
  --market data/BTCUSDT_5min_2018_2025_present.csv \
  --news data/news_clusters.json \
  --output-dir outputs/regime_anchor_diagnostic \
  --embedding-cache outputs/main_pilot/cache/article_embeddings.sqlite \
  --scheduler-path outputs/main_pilot/scheduler_horizon.json \
  --physical-batch-size 32 \
  --num-workers 4 \
  --resume \
  --confirm-news-filter-reviewed
```

The report is written to
`outputs/regime_anchor_diagnostic/metrics/diagnostic_report.json`. Per-run
validation and development-OOS predictions include both the HAR anchor and
the learned log-RV correction. Fold 5 and the final test remain unopened.
