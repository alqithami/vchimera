# Reproducibility Guide (V-CHIMERA)

This document provides a minimal, deterministic workflow to reproduce the results and regenerate the manuscript figures/tables.

## 1) Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run a smoke test:

```bash
python scripts/smoke_test.py
```

## 2) Main experiments

Run the main config:

```bash
python scripts/run_experiments.py --config configs/experiments/journal_main.yaml
```

Capture the created run directory (the script prints it), or select the newest run:

```bash
RUN_DIR=$(ls -td runs/journal_main_* | head -n 1)
```

Expected outputs in `$RUN_DIR/`:
- `summary.csv` (aggregate metrics across seeds)
- `summary_by_seed.csv` (seed-level summaries)
- `episode_steps.csv` (per-step trajectories; required for time-series plots)

## 3) Sensitivity sweep

```bash
python scripts/run_sensitivity.py --config configs/experiments/sensitivity.yaml
```

Expected output directory:
- `runs/sensitivity_grid/`
- `runs/sensitivity_grid/sensitivity.csv` (grid metrics used by the heatmaps)

## 4) Generate LaTeX assets for the paper

```bash
python scripts/make_paper_assets.py --run_dir "$RUN_DIR" --paper_dir paper
```

This writes (paths may vary by config, but the paper expects these locations):
- `paper/paper_assets/table_main.tex`
- `paper/paper_assets/table_protocol.tex`
- `paper/paper_assets/table_sensitivity_summary.tex`
- `paper/paper_assets/table_perm_tests.tex` (if enabled)

## 5) Regenerate submission figures (B/W + accent)

```bash
python paper/scripts/regenerate_figures_bw_accent.py \
  --episode_steps "$RUN_DIR/episode_steps.csv" \
  --sensitivity runs/sensitivity_grid/sensitivity.csv \
  --out_dir paper/figures
```

This regenerates the figures referenced by `paper/main.tex` under `paper/figures/`.

## 6) Compile the paper

Requires `pdflatex` + `bibtex`:

```bash
python scripts/compile_paper.py --paper_dir paper --tex main.tex
```

or compile manually from within `paper/`.

## 7) Optional CybORG transfer experiment

```bash
python scripts/run_experiments.py --config configs/experiments/cyborg_transfer.yaml
```

This verifies that the V-CHIMERA coupling bus + protocol monitor can be exercised with a higher-fidelity cyber backend.
