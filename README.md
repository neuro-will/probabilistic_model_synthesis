# Deep Probabilistic Model Synthesis (DPMS)

William E. Bishop, Luuk W. Hesselink, Bernhard Englitz, Misha B. Ahrens, and James E. Fitzgerald

## Overview

Deep Probabilistic Model Synthesis (DPMS) is a probabilistic machine-learning framework for synthesizing models across multiple instances of a general system. In this repository, DPMS is applied to simulated examples and whole-brain light-sheet imaging data from larval zebrafish.

DPMS jointly learns:

- A conditional prior distribution (CPD) capturing structure shared across systems.
- System-specific posterior distributions for model parameters.

Manuscript: **Deep Probabilistic Model Synthesis**  
arXiv preprint DOI: https://doi.org/10.48550/arXiv.2603.14161

Repository: https://github.com/neuro-will/probabilistic_model_synthesis

![DPMS overview figure](./github_figure.png)

## Repository Contents

- `probabilistic_model_synthesis/`: DPMS package code.
- `figures/figure_1/`: Figure 1 example notebooks.
- `figures/figure_2/`: One-dimensional simulated synthesis example.
- `figures/figure_3/`: Same-condition regression transfer analysis.
- `figures/figure_4_and_S2/`: Qualitative factor-analysis synthesis example and supplemental Figure S2.
- `figures/figure_5/`: Cross-condition factor-analysis transfer analysis.
- `figures/figure_S1/`: Supplemental linear-regression synthesis simulation.
- `data/`: Manuscript data downloader and tracked fold/segment inputs.
- `folds_and_segments_generation/`: Notebooks used to generate real-data segment and fold structures.

## System Requirements

### Tested Software Environment

The code has been tested on Linux with:

- Python 3.8
- NumPy 1.24
- Matplotlib 3.7
- PyTorch 2.4
- SciPy 1.10
- scikit-learn 1.3
- h5py 3.11
- pandas 2.0
- POT 0.9
- statsmodels

This repository depends on `janelia_core`, companion custom software used for data handling, model fitting utilities, and visualization:

```bash
git clone https://github.com/neuro-will/janelia_core
```

### Hardware

The package and plotting notebooks can run on CPU. GPU acceleration is required for practical full-scale real-data model fitting.

- Figure 1, Figure S1, and plotting notebooks: CPU is sufficient.
- Figure 2 simulation: CPU is supported; GPU or a high-memory CPU machine is recommended for publication-scale runs.
- Figure 3 and Figure 5 real-data fitting: NVIDIA GPU recommended for each fit.
- Full Figure 3 and Figure 5 sweeps: many independent GPU fits are required, so a cluster or equivalent job-array setup is recommended.

Single real-data fits can be run directly from the command line. The included sweep runners use local execution by default and can optionally submit full sweeps on an LSF cluster.

## Installation

### 1. Clone Repositories

```bash
git clone https://github.com/neuro-will/janelia_core
git clone https://github.com/neuro-will/probabilistic_model_synthesis
```

### 2. Create Environment

Recommended:

```bash
cd probabilistic_model_synthesis
conda env create -f environment.yml
conda activate dpms
```

The Conda environment installs this repository in editable mode using the package metadata in `pyproject.toml`.

Install `janelia_core` into the same environment:

```bash
cd /path/to/janelia_core
pip install -e .
```

If needed, reinstall this repository in editable mode from the repository root:

```bash
cd /path/to/probabilistic_model_synthesis
pip install -e .
```

PyTorch installation can be system-specific. If the `environment.yml` PyTorch build does not match your system, create the environment with the other dependencies and then install the CPU or CUDA PyTorch build recommended at:

https://pytorch.org/get-started/locally

Typical installation time on a standard workstation is a few minutes, excluding PyTorch download time.

## Data Availability and Setup

This project uses whole-brain light-sheet imaging data from larval zebrafish fictively behaving in a virtual environment.

Dataset:

- Chen, Mu, Hu et al., Neuron, 2018
- DOI: https://doi.org/10.1016/j.neuron.2018.09.042
- Data: https://doi.org/10.25378/janelia.7272617

The DPMS real-data scripts expect subject folders under `data/`:

```text
data/
  subject_1/
    data_full.mat
    TimeSeries.h5
  subject_2/
    data_full.mat
    TimeSeries.h5
  fold_and_segment_structures/
    phototaxis_ns_subjects_1_2_5_6_8_9_10_11.json
    omr_l_r_f_ns_across_cond_segments_8_9_10_11.json
  ...
```

### Prepare Manuscript Data

We provide a repository downloader for preparing the manuscript data. It retrieves only the subject data and shared reference files used by these analyses, extracts them into `data/`, and generates or verifies the required fold/segment JSON artifacts. The prepared data directory is approximately 35 GB after extraction.

Download, extract, and prepare all required data files:

```bash
python data/download_data.py --data-dir data
```

Verify an existing data directory without downloading:

```bash
python data/download_data.py --data-dir data --verify-only
```

The subject download is large, and the downloader prints the requested files and total size before downloading. Use `--dry-run` if you only want to inspect the planned download.

Fold and segment JSON files define the train/validation/test splits and contiguous behavioral periods used by the real-data model fits; they are required for Figures 3 and 5. The downloader generates missing artifacts automatically after the subject data are present. They can also be regenerated manually with:

```bash
python data/generate_fold_and_segment_artifacts.py --data-dir data
```

Repository defaults point to:

```text
data/fold_and_segment_structures/phototaxis_ns_subjects_1_2_5_6_8_9_10_11.json
data/fold_and_segment_structures/omr_l_r_f_ns_across_cond_segments_8_9_10_11.json
data/fold_and_segment_structures/fold_str_base_14_tgt_{1,2,4,8,14}.json
data/fold_and_segment_structures/gnldr_paired/ac_an_disjoint_paired_k6_tgt_{8,9,11}_{multi_cond,single_cond}_folds.json
```

## Reproducing Manuscript Figures

The full real-data analyses are computationally intensive. The commands below show the intended entry points and default repository-relative paths. Full reproduction requires the manuscript data under `data/` and JSON fold/segment artifacts under `data/fold_and_segment_structures/`.

### Direct Runs and Full Sweeps

The Figure 3 and Figure 5 sections show direct single-fit commands first. These commands are scheduler-independent and are the clearest way to test that paths, data, and parameters are correct.

For full manuscript-scale sweeps, `figures/figure_3/run_full_transfer_analysis.py` and `figures/figure_5/run_full_transfer_analysis.py` run the same direct fit and post-processing commands across all required folds, subjects, and fit types. They use local execution by default and can also submit to LSF with `--backend lsf`.

### Figure 1

Run the notebooks:

```text
figures/figure_1/2d_example.ipynb
figures/figure_1/optimal_mcpd.ipynb
```

Expected output: the illustrative Figure 1 panels shown in the manuscript.

### Figure 2: Simulated Regression Synthesis

Publication-scale run:

```bash
python figures/figure_2/fit_one_dim_synthesis_example.py \
  --save-folder results/publication_results/gnlr/simulation
```

Plot results:

```text
figures/figure_2/vis_one_dim_synthesis_example.ipynb
```

Expected output includes `full_simulations.pt`, checkpoint folders, and SVG panels generated by the notebook.

### Figure 3: Same-Condition Regression Transfer

Generate parameter file:

```bash
python figures/figure_3/generate_syn_params.py
```

Run one fit directly:

```bash
python figures/figure_3/syn_ahrens_gnlr_mdls.py \
  results/publication_results/gnlr/real_data/fit_params.pkl \
  -results_dir results/publication_results/gnlr/real_data/example_fit \
  -fold_str_file fold_str_base_14_tgt_1.json \
  -fold 0 \
  -subject_filter 8 \
  -save_file fit_results.pt \
  -rand_seed 1
```

Post-process one fit:

```bash
python figures/figure_3/post_process.py \
  results/publication_results/gnlr/real_data/example_fit/fit_results.pt \
  results/publication_results/gnlr/real_data/example_fit/pp_fit_results.pkl \
  -early_stopping_subjects 8 \
  -early_stopping True \
  -rand_seed 1
```

Full sweep runner, using local execution by default:

```bash
python figures/figure_3/run_full_transfer_analysis.py
```

Use multiple local workers if the machine has suitable resources:

```bash
python figures/figure_3/run_full_transfer_analysis.py --max-workers 2
```

Submit the same sweep to LSF:

```bash
python figures/figure_3/run_full_transfer_analysis.py --backend lsf
```

Plot results:

```text
figures/figure_3/generate_transfer_analysis_plots.ipynb
```

Expected output includes `fit_results.pt`, `pp_fit_results.pkl`, and the Figure 3 SVG panels.

### Figure 4 and Figure S2: Qualitative Cross-Condition FA

Run:

```text
figures/figure_4_and_S2/across_cond_synthesis_example.ipynb
```

Expected output: latent-space and anatomical visualization panels for Figure 4 and Figure S2.

### Figure 5: Quantitative Cross-Condition FA

The downloader generates the Figure 5 fold structures automatically. To regenerate them manually:

```bash
python figures/figure_5/build_across_cond_disjoint_folds.py \
  --segment_table_path data/fold_and_segment_structures/omr_l_r_f_ns_across_cond_segments_8_9_10_11.json \
  --save_dir data/fold_and_segment_structures/gnldr_paired \
  --n_folds 6 \
  --min_train_segments 18 \
  --min_validation_segments 4 \
  --min_test_segments 5
```

Generate model-fit parameter file:

```bash
python figures/figure_5/generate_syn_params.py
```

Run one fit directly:

```bash
python figures/figure_5/syn_ahrens_gnldr_mdls.py \
  results/publication_results/gnldr/quantification_paired/fit_params_paired.pkl \
  -results_dir results/publication_results/gnldr/quantification_paired/example_fit \
  -fold_str_file ac_an_disjoint_paired_k6_tgt_8_multi_cond_folds.json \
  -fold omr_f_ns__fold_0 \
  -save_file fit_results.pt \
  -rand_seed 1
```

Post-process one fit:

```bash
python figures/figure_5/post_process.py \
  results/publication_results/gnldr/quantification_paired/example_fit/fit_results.pt \
  results/publication_results/gnldr/quantification_paired/example_fit/pp_fit_results.pt \
  -early_stopping_subjects 8 \
  -test_periods omr_forward,omr_right,omr_left \
  -early_stopping True \
  -rand_seed 1
```

Full sweep runner, using local execution by default:

```bash
python figures/figure_5/run_full_transfer_analysis.py
```

Use multiple local workers if the machine has suitable resources:

```bash
python figures/figure_5/run_full_transfer_analysis.py --max-workers 2
```

Submit the same sweep to LSF:

```bash
python figures/figure_5/run_full_transfer_analysis.py --backend lsf
```

Plot results:

```text
figures/figure_5/make_across_cond_plots.ipynb
```

Expected output includes `fit_results.pt`, post-processed result files, per-fish heatmaps, mean-delta heatmaps, positive-count heatmaps, and statistical summaries.

### Figure S1: Linear-Regression Simulation

Run:

```bash
python figures/figure_S1/linear_regression_synthesis.py
```

Expected output:

```text
results/simulation/reg_with_varying_no_of_example_systems/reg_synthesis_with_varying_n_ex_systems.pkl
results/simulation/reg_with_varying_no_of_example_systems/reg_synthesis_with_varying_n_ex_systems.svg
```

## License

This software is released under the MIT License. See `LICENSE`.

## Citation

If you use this code, please cite the DPMS preprint:

```text
Bishop, W. E., Hesselink, L. W., Englitz, B., Ahrens, M. B., and Fitzgerald, J. E.
Deep Probabilistic Model Synthesis.
arXiv. https://doi.org/10.48550/arXiv.2603.14161
```

Please also cite the original zebrafish dataset:

```text
Chen, Mu, Hu et al., Neuron, 2018.
https://doi.org/10.1016/j.neuron.2018.09.042
```

## Contact

For questions, please contact William Bishop at [willbishop.neuro@gmail.com](mailto:willbishop.neuro@gmail.com).
