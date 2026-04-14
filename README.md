# V-CHIMERA

**V-CHIMERA** is a reproducible research artifact for studying **organizational cyber crisis response under misinformation**. The repository models cyber operations and public communication as a coupled human-information-machine system, then evaluates how verified communication governance and immune-inspired regulation affect safety and performance.

The project accompanies the paper **"V-CHIMERA: An Immune-Inspired Verified Framework for Organizational Cyber Crisis Response under Misinformation"** and is designed to support transparent experimentation, figure regeneration, and manuscript preparation.

## Graphical Abstract

A publication-quality graphical abstract can be inserted in this section when the repository is updated.

```md
![Graphical Abstract](path/to/graphical_abstract.png)
```

## Why this repository matters

Traditional incident-response pipelines often treat technical containment and public communication as separate activities. V-CHIMERA brings them together in a single coupled framework. It models how cyber incidents shape rumor, trust, and reporting behavior, and how those social responses feed back into defender effectiveness. At the same time, it enforces communication safety constraints at runtime so that unsafe public messages are never executed.

## Core capabilities

| Component | Description |
|---|---|
| **CyberCrisisGym-J** | A coupled cyber-social simulation environment that combines incident progression with online belief, trust, uncertainty, moderation, and reporting dynamics. |
| **V-CHIMERA policies** | Baseline, coupled, shielded, and ablation policies for controlled comparison across scenarios. |
| **Runtime protocol shield** | A safety layer that filters proposed communication actions according to evidence, uncertainty, cooldown, and disclosure rules. |
| **Immune-gated coupling** | An immune-inspired controller that uses danger, tolerance, and memory to regulate when social feedback should influence operational response. |
| **Reproducible experiment scripts** | Utilities for running the main journal experiments, sensitivity sweeps, and manuscript asset generation. |
| **Paper sources** | Manuscript source files, references, figures, and table-generation scripts for paper-ready outputs. |

## Repository structure

| Path | Purpose |
|---|---|
| `vchimera/` | Core package containing environments, coupling logic, protocol enforcement, metrics, policies, and optional adapters. |
| `configs/` | Scenario definitions, calibration files, and experiment YAML configurations. |
| `scripts/` | Experiment runners, sensitivity studies, smoke tests, and supporting utilities. |
| `paper/` | Manuscript source, bibliography, figures, and generated paper assets. |
| `runs/` | Experiment outputs created after execution. |

## Installation

The recommended environment is **Python 3.11 or newer**.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick start

The following commands cover the standard workflow from environment validation to experiment execution and paper asset generation.

| Step | Command | Outcome |
|---|---|---|
| Smoke test | `python scripts/smoke_test.py` | Confirms that the basic environment and package wiring are functioning correctly. |
| Main experiment suite | `python scripts/run_experiments.py --config configs/experiments/journal_main.yaml` | Produces a timestamped run directory under `runs/`. |
| Sensitivity study | `python scripts/run_sensitivity.py --config configs/experiments/sensitivity.yaml` | Generates the data used for robustness and sensitivity figures. |
| Paper assets | `python scripts/make_paper_assets.py --run_dir "$RUN_DIR" --paper_dir paper` | Writes tables and figure assets for manuscript integration. |
| Figure regeneration | `python paper/scripts/regenerate_figures_bw_accent.py --episode_steps "$RUN_DIR/episode_steps.csv" --sensitivity runs/sensitivity_grid/sensitivity.csv --out_dir paper/figures` | Regenerates publication-style figures using the expected run outputs. |

To retrieve the latest journal run directory automatically, use:

```bash
RUN_DIR=$(ls -td runs/journal_main_* | head -n 1)
echo "$RUN_DIR"
```

## Running the main experiments

The default starter pack evaluates three representative crisis settings: `ransomware_rumor`, `outage_rumor`, and `exfiltration_scam`. These scenarios are intended to stress different combinations of cyber disruption, misinformation diffusion, trust erosion, and reporting behavior.

```bash
python scripts/run_experiments.py --config configs/experiments/journal_main.yaml
```

A typical run creates a directory of the form:

```text
runs/journal_main_YYYYMMDD_HHMMSS/
```

Within that directory, the key outputs are usually `summary.csv` and, when enabled, `episode_steps.csv`.

## Generating manuscript outputs

Once a run has completed, paper assets can be generated directly for manuscript integration.

```bash
python scripts/make_paper_assets.py --run_dir "$RUN_DIR" --paper_dir paper
```

This step writes tables and figure assets into the paper workspace so that the manuscript can be rebuilt without manual copy-paste steps.

## Compiling the manuscript

Compiling the paper requires a local TeX distribution such as **TeX Live** or **MacTeX**.

```bash
python scripts/compile_paper.py --paper_dir paper --tex main.tex
```

A manual LaTeX workflow is also supported:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Optional CybORG transfer experiment

The repository includes an adapter scaffold for the **CybORG** ecosystem. This backend is optional and must be installed separately if transfer experiments are required.

```bash
python scripts/run_experiments.py --config configs/experiments/cyborg_transfer.yaml
```

If the CybORG configuration depends on additional packages, install them in the same environment before running the transfer experiment.

## Reproducibility

A dedicated reproducibility guide is available in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). It should be used when recreating tables, figures, and paper outputs from raw runs.

## Citation

A `CITATION.cff` file can be included for software citation, and the accompanying manuscript should also be cited when the repository is used in academic work.

## Safety and scope

The default cyber backend is an **abstract incident simulator** intended for research evaluation. The repository is designed to study organizational response, misinformation-aware coordination, and communication governance. It is **not** a source of offensive payloads or operational exploitation guidance.
