# BTC PatchTST–BGE–FinBERT main-pilot

## Sampling revision (development continuation)

The text/Gamma/gate development profiles now use a dedicated target sampler:
the target day and its previous 22 daily RV observations must be valid. These
profiles no longer inherit PatchTST intraday-window exclusions.

PatchTST keeps the original 5-minute bars and 6-hour coarse patches, but its
coarse market lookback is 22 days (88 coarse tokens). Its 30-day news lookback
is unchanged, so PatchTST still requires 30 calendar days of initial history.
This revision changes the locked configuration hash: do not reuse scheduler or
model checkpoints produced with the former 60-day coarse lookback.

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

## Development-only news representation audit

This profile audits the news input before any spike gate is introduced. It
creates a deterministic stratified filter-review sample, measures conservative
same-time/title and same-day/content duplicates, yearly source concentration,
and BGE tokenizer truncation. It then runs HAR-QLIKE-anchored linear probes on
rolling Fold 1-4 for eight predeclared representations:

- current 11 daily news scalars;
- FinBERT slow/fast probabilities;
- BGE slow/fast with core-only PCA8 or PCA16;
- semantic/sentiment surprise norms;
- current combined PCA8;
- conservative deduplication plus source-balanced PCA8;
- combined PCA16 plus surprise norms.

Every probe standardizes features on its fold core only. The locked
`lambda_sum` grid is selected by validation exact QLIKE, with the larger
penalty preferred inside `1e-5`. A correction is applied to development OOS
only if it improves validation over the core-only HAR-QLIKE anchor by at least
`1e-5`. This profile does not train a Transformer, add spike gating, open Fold
5/final test, or make a statistical claim.

First export and inspect the new stratified review:

```bash
PYTHONPATH=src python -u -m btc_main_pilot \
  --profile development-news-representation-audit \
  --market data/BTCUSDT_5min_2018_2025_present.csv \
  --news data/news_clusters.json \
  --output-dir outputs/news_representation_audit \
  --review-news-only
```

Run the CPU-safe smoke test:

```bash
PYTHONPATH=src python -u -m btc_main_pilot \
  --profile development-news-representation-audit \
  --market data/BTCUSDT_5min_2018_2025_present.csv \
  --news data/news_clusters.json \
  --output-dir outputs/news_representation_audit \
  --smoke \
  --no-resume
```

After reviewing
`outputs/news_representation_audit/audit/stratified_news_filter_review.csv`,
run the full development audit on CUDA:

```bash
PYTHONPATH=src TOKENIZERS_PARALLELISM=false python -u -m btc_main_pilot \
  --profile development-news-representation-audit \
  --market data/BTCUSDT_5min_2018_2025_present.csv \
  --news data/news_clusters.json \
  --output-dir outputs/news_representation_audit \
  --embedding-cache outputs/main_pilot/cache/article_embeddings.sqlite \
  --confirm-news-filter-reviewed \
  --resume
```

The final screen is written to
`outputs/news_representation_audit/metrics/representation_screen.json`, with
the complete report at
`outputs/news_representation_audit/metrics/diagnostic_report.json`.

## GPT-silver event-aware long-text audit

This development-only profile uses the original locked 366-row review as a
silver evaluation holdout. The remaining rows of the expanded review fit a
high-recall event-aware news policy. The holdout is relabeled blindly by a
second GPT model; disagreements with the first pass are independently
adjudicated. Prompts contain publication time, source, title, and lead only:
market outcomes and future prices are never sent to the model.

If the expanded 1,200-row GPT review does not already exist, create it first
(this also preserves the original review as
`stratified_news_filter_review_original_366.csv`):

```bash
PYTHONPATH=src python -u -m btc_main_pilot \
  --profile development-news-representation-audit \
  --news data/news_clusters.json \
  --output-dir outputs/news_representation_audit \
  --label-news-review \
  --review-target-size 1200 \
  --review-model gpt-5.6-terra
```

Then create the 366-row silver holdout (Windows can reuse the DPAPI credential;
Linux should provide `OPENAI_API_KEY`):

```bash
PYTHONPATH=src python -u -m btc_main_pilot \
  --profile development-event-aware-longtext-audit \
  --news data/news_clusters.json \
  --output-dir outputs/event_aware_longtext_audit \
  --review-audit-dir outputs/news_representation_audit/audit \
  --label-silver-holdout \
  --silver-model gpt-5.6-sol \
  --silver-batch-size 12
```

The call is resumable through its per-pass JSON caches. It writes
`outputs/event_aware_longtext_audit/audit/gpt_silver_holdout_366.csv` and an
agreement report. GPT labels are silver proxy labels, not expert ground truth.

Run the CPU-safe smoke test without making API calls:

```bash
PYTHONPATH=src python -u -m btc_main_pilot \
  --profile development-event-aware-longtext-audit \
  --market data/BTCUSDT_5min_2018_2025_present.csv \
  --news data/news_clusters.json \
  --output-dir outputs/event_aware_longtext_audit \
  --review-audit-dir outputs/news_representation_audit/audit \
  --smoke
```

Then run the full development-only Fold 1-4 audit on CUDA:

```bash
PYTHONPATH=src TOKENIZERS_PARALLELISM=false python -u -m btc_main_pilot \
  --profile development-event-aware-longtext-audit \
  --market data/BTCUSDT_5min_2018_2025_present.csv \
  --news data/news_clusters.json \
  --output-dir outputs/event_aware_longtext_audit \
  --review-audit-dir outputs/news_representation_audit/audit \
  --silver-holdout-path outputs/event_aware_longtext_audit/audit/gpt_silver_holdout_366.csv \
  --longtext-cache outputs/event_aware_longtext_audit/cache/longtext_embeddings.sqlite \
  --resume
```

The predeclared candidates are title only; token-limited title plus lead;
normalized mean aggregation of token-limited article chunks; separate title
and content streams; FinBERT slow/fast; and semantic/sentiment surprise norms.
PCA components are fitted on each fold core only. PCA16 and the previous
combined-PCA representation are intentionally excluded. Fold 5 and the final
test are never opened.

## Adaptive point-in-time gate diagnostic

This follow-up profile is intentionally adaptive: its filter-family choices
were made after inspecting the first GPT-silver development report. Therefore
the 366 GPT-silver rows are calibration diagnostics, no longer an independent
holdout, and remain non-expert proxy labels.

The refined filter retains regulation/ETF, exchange/custody,
macro/liquidity, and mining/energy context. Stablecoin/DeFi, security/hack,
and broad other-crypto context are deferred pending more targeted labels. A
single Bitcoin mention is recovered only when concrete event language is also
present.

For each Fold 1-4, a class-balanced logistic gate is fitted on core only. Its
inputs use market RV lags through `t-1`, core-scaled daily news fields dated
`t-1`, title semantic-surprise norm, and FinBERT sentiment-surprise norm. The
logistic regularization is fixed at `C=0.1`. Hard thresholds are selected from
`{0.2, 0.3, 0.4, 0.5}` on validation exact QLIKE; soft gates have no threshold.
The predeclared normal routes are HAR, FinBERT, and chunk mean, while the spike
route is title-only. A route is deployed OOS only when it beats HAR on
validation by at least `1e-5`.

Run the CPU-safe smoke test:

```bash
PYTHONPATH=src python -u -m btc_main_pilot \
  --profile development-point-in-time-gate-diagnostic \
  --market data/BTCUSDT_5min_2018_2025_present.csv \
  --news data/news_clusters.json \
  --output-dir outputs/point_in_time_gate_diagnostic \
  --smoke \
  --no-resume
```

Run the full adaptive Fold 1-4 diagnostic on CUDA, reusing the prior long-text
cache:

```bash
PYTHONPATH=src TOKENIZERS_PARALLELISM=false python -u -m btc_main_pilot \
  --profile development-point-in-time-gate-diagnostic \
  --market data/BTCUSDT_5min_2018_2025_present.csv \
  --news data/news_clusters.json \
  --output-dir outputs/point_in_time_gate_diagnostic \
  --review-audit-dir outputs/news_representation_audit/audit \
  --silver-holdout-path outputs/event_aware_longtext_audit/audit/gpt_silver_holdout_366.csv \
  --longtext-cache outputs/event_aware_longtext_audit/cache/longtext_embeddings.sqlite \
  --resume
```

The primary decision files are
`outputs/point_in_time_gate_diagnostic/metrics/gate_screen.json`,
`fold_metrics.csv`, `pooled_metrics.csv`, and `diagnostic_report.json`.
Gate probabilities, raw routing weights, deployed weights, and component
predictions are retained in the prediction CSVs. Fold 5, final test, MCS,
five-seed evaluation, realized-spike oracle routing, and PatchTST retraining
are excluded.

## Locked-selection core+validation refit diagnostic

This separate development profile measures whether the same selected text
corrections and gates benefit from a final refit on more recent data. It first
reproduces the point-in-time gate diagnostic exactly: fit on core, select the
Gamma penalty, correction fallback, gate mode, and hard threshold on
validation, then save the original test results under `before_refit/`.

Those choices are then frozen. Fold-local semantic scaling and PCA, HAR,
selected Gamma probes, and the logistic gate are fitted again using the valid
core plus validation targets. The core-only spike threshold used during
selection remains locked so that before/after normal and spike metrics use the
same definition. Test dates are not used by preprocessing, refitting,
calibration, fallback, or threshold selection. Results from this profile are
post-hoc development diagnostics because the Fold 1-4 tests have already been
inspected.

Run the CPU-safe end-to-end smoke test:

```bash
PYTHONPATH=src python -u -m btc_main_pilot \
  --profile development-point-in-time-refit-diagnostic \
  --market data/BTCUSDT_5min_2018_2025_present.csv \
  --news data/news_clusters.json \
  --output-dir outputs/point_in_time_refit_diagnostic \
  --smoke \
  --no-resume
```

Run the full Fold 1-4 refit diagnostic on CUDA while reusing the long-text
cache:

```bash
PYTHONPATH=src TOKENIZERS_PARALLELISM=false python -u -m btc_main_pilot \
  --profile development-point-in-time-refit-diagnostic \
  --market data/BTCUSDT_5min_2018_2025_present.csv \
  --news data/news_clusters.json \
  --output-dir outputs/point_in_time_refit_diagnostic \
  --review-audit-dir outputs/news_representation_audit/audit \
  --silver-holdout-path outputs/event_aware_longtext_audit/audit/gpt_silver_holdout_366.csv \
  --longtext-cache outputs/event_aware_longtext_audit/cache/longtext_embeddings.sqlite \
  --resume
```

Original test results are saved below `before_refit/`; locked-selection refit
results are saved below `after_refit/`. The direct comparison files are
`metrics/before_after_fold_metrics.csv`,
`metrics/before_after_pooled_metrics.csv`, and
`metrics/before_after_summary.json`. The complete report is
`metrics/diagnostic_report.json`. Fold 5, final test, PatchTST retraining,
MCS, five-seed evaluation, realized-spike oracle routing, and any test-based
model selection remain excluded.
