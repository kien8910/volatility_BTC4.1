# BTC PatchTST–BGE–FinBERT main-pilot

This package implements only the `main-pilot` workflow defined by
`BTC_PATCHTST_FINBERT_MODEL_SPEC.md`.

It performs one Fold-1 scheduler pilot (core/validation only), locks `H_cos`,
deletes the pilot checkpoint, retrains the primary model on Fold 5 with seed
11, and evaluates the Fold-5 OOS block against Random Walk, HAR-OLS with Duan
smearing, and HAR-QLIKE.

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

Smoke output is isolated under `outputs/main_pilot/smoke` and uses a
deterministic 768-dimensional test double for BGE/FinBERT. It is never a
research result and never creates the main `scheduler_horizon.json`.

