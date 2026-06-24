# Figure 5 Paired Cross-Fold Analysis

This folder contains the paired cross-fold across-condition transfer analysis for Figure 5.

## Files

- `build_across_cond_disjoint_folds.py`
  - Generates paired disjoint test folds for `multi_cond` and `single_cond`.
  - Target-fish train, validation, and test segment IDs are identical for each SB/DB pair.
  - Target-fish test bins are reused across target train conditions, so `fold_k` means the same held-out test data for every train-condition comparison.
  - Train, validation, and test sample counts are validated to be constant across the six folds. Remainder test segments are left unused when a group's segment count is not divisible by six.
  - Transfer fish are train-only, matching the original Figure 5 design.
  - Fold keys are stored as `<condition>__fold_<k>`.
- `generate_syn_params.py`
  - Generates fitting parameter file (`fit_params_paired.pkl`) for the paired cross-fold pipeline.
- `run_full_transfer_analysis.py`
  - Runs the full `condition x fold x target fish x fit_type` sweep locally by default.
  - Use `--backend lsf` to submit the same sweep as one LSF array job.
  - Default mode is full re-fit + postprocess for the paired folds.
  - Early stopping scope is configurable and defaults to target fish only (`EARLY_STOPPING_SCOPE='target_only'`).
  - Writes a manifest to:
    - `results/publication_results/gnldr/quantification_paired/_manifests/`
  - Writes array logs to:
    - `results/publication_results/gnldr/quantification_paired/_array_logs/`
  - Saves results under:
    - `results/publication_results/gnldr/quantification_paired/<condition>/fold_<k>/subj_<id>/<fit_type>/...`
- `sweep_worker.py`
  - Executes one manifest row for local or LSF-array execution.
  - Supports `fit_only`, `post_only`, and `fit_and_post` modes.
- `make_across_cond_plots.ipynb`
  - Fold-aware plotting and statistics notebook.
  - Per-fish tests: two-sided exact sign-count tests across condition-cell values within each fish, reported separately for diagonal and off-diagonal train/test pairs.
  - Pooled tests: two-sided exact sign-count tests across all valid condition-cell values, reported separately for diagonal and off-diagonal train/test pairs.

## Typical Usage

1. Generate fold structures:

```bash
python figures/figure_5/build_across_cond_disjoint_folds.py \
  --n_folds 6 \
  --min_train_segments 18 \
  --min_validation_segments 4 \
  --min_test_segments 5
```

2. Generate model-fit params:

```bash
python figures/figure_5/generate_syn_params.py
```

3. Run training + postprocess jobs locally:

```bash
python figures/figure_5/run_full_transfer_analysis.py
```

Or submit the same sweep to LSF:

```bash
python figures/figure_5/run_full_transfer_analysis.py --backend lsf
```

4. Open and run:

`figures/figure_5/make_across_cond_plots.ipynb`
