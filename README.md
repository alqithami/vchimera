# V-CHIMERA (Immune-Inspired, Verified Cyber–Social Incident Response)

This repository provides an end-to-end, **reproducible research artifact** for studying **misinformation-aware cyber defense** under **explicit communication governance**.
V-CHIMERA couples **cyber incident response** with **online belief/trust dynamics** and enforces a safety-critical **communication protocol** via a runtime **shield**.

It accompanies the manuscript:

> *Immune-Inspired Verified Coupled Human–Information–Machine Incident Response for Misinformation-Aware Cyber Defense* (Biomimetics, under review / preprint).

---

## What’s included

- **CyberCrisisGym-J**: a coupled cyber–social simulation environment  
  - Cyber incident dynamics (attack-graph default; optional CybORG adapter)
  - Multi-platform social ABM (communities, bots, moderation, sentiment metrics)
  - Bidirectional coupling (cyber → social narrative shocks; social → cyber compliance/reporting modifiers)
- **Policies and ablations** for controlled comparison:
  - `pipeline`, `pipeline+shield`
  - `vchimera`, `vchimera+shield`
  - `vchimera-no-coupling+shield`, `vchimera-no-targeting+shield`
  - immune-inspired coupling controller (IGC / AIS variant; see paper)
- **Protocol shield** with audit signals:
  - attempted vs. executed protocol violations
  - shield edits (runtime interventions)
- **Experiment runners** producing `runs/<RUN_ID>/summary.csv` and (optionally) `episode_steps.csv`

---

## Repository layout

- `vchimera/` — core package (environment, coupling, protocol/shield, metrics, policies, adapters)
- `configs/` — scenario, calibration, and experiment YAML files
- `scripts/` — experiment runners, calibration, sensitivity sweeps, and utilities
- `runs/` — generated outputs (created when you run experiments)

---

## Installation

**Recommended:** Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quick sanity check

```bash
python scripts/smoke_test.py
```

---

## Run the main experiment set

```bash
python scripts/run_experiments.py --config configs/experiments/journal_main.yaml
```

This creates a timestamped run directory under `runs/`, for example:

```text
runs/journal_main_YYYYMMDD_HHMMSS/
```

To grab the latest run directory automatically:

```bash
RUN_DIR=$(ls -td runs/journal_main_* | head -n 1)
echo "$RUN_DIR"
```

---

## Generate paper tables and figures (LaTeX assets)

```bash
python scripts/make_paper_assets.py --run_dir "$RUN_DIR" --paper_dir paper
```

This writes LaTeX tables and figure PDFs into `paper/paper_assets/` (and/or `paper/figures/`, depending on the config).

---

## Compile the manuscript PDF

Compiling requires a TeX distribution (e.g., TeX Live / MacTeX).

```bash
python scripts/compile_paper.py --paper_dir paper --tex main.tex
```

Alternatively:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

---

## Regenerate submission-ready figures

The paper source includes a figure-regeneration script.

First ensure you produced the required CSVs (main run + sensitivity sweep):

```bash
python scripts/run_sensitivity.py --config configs/experiments/sensitivity.yaml
```

Then:

```bash
python paper/scripts/regenerate_figures_bw_accent.py \
  --episode_steps "$RUN_DIR/episode_steps.csv" \
  --sensitivity runs/sensitivity_grid/sensitivity.csv \
  --out_dir paper/figures
```

---

## Optional: CybORG transfer run (external backend)

V-CHIMERA includes an adapter scaffold for the CybORG ecosystem. CybORG is **optional** and installed separately.

```bash
python scripts/run_experiments.py --config configs/experiments/cyborg_transfer.yaml
```

If your CybORG configuration requires additional dependencies, install them in the same environment (see CybORG / CAGE docs).

---

## Reproducibility

A step-by-step reproducibility checklist (including table/figure mapping) is provided in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## Citation

A `CITATION.cff` file is provided for software citation. 
---

## Ethics & safety

The default cyber backend is an **abstract incident simulator** designed for research evaluation. The repository does **not** provide exploit payloads or operational offensive instructions.
