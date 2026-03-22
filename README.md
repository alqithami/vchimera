# V-CHIMERA — Cyber–social crisis-response pipeline

This repository is a runnable artifact for the V-CHIMERA paper:
a coupled **cyber incident response** + **misinformation/trust** simulator with a **verifiable communications protocol shield**.

## What you get (out-of-the-box)
- A coupled simulation environment (cyber + social ABM + bidirectional coupling)
- Policies: `pipeline`, `pipeline+shield`, `vchimera(+shield)` and ablations
- Correct protocol metrics:
  - **attempted violations** (policy intent),
  - **executed violations** (what actually happened),
  - **shield interventions** (how often the shield edited actions)
- A complete experiment runner that produces `runs/<RUN_ID>/summary.csv`
- Paper assets generator: produces LaTeX tables + vector PDF figures
- Calibration script (moment-matching) to keep ABM regimes realistic
- Robustness / sensitivity sweeps

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick sanity check (2–3 minutes)
```bash
python scripts/smoke_test.py
```

## Run the main journal experiment
```bash
python scripts/run_experiments.py --config configs/experiments/journal_main.yaml
```

This creates a run directory like:
`runs/journal_main_YYYYMMDD_HHMMSS/`


## Included run (used for the provided PDF)
The PDF in this bundle was compiled from run directory:
`runs/journal_main_20260210_215043/`

## Generate paper tables/figures (publication-ready assets)
```bash
python scripts/make_paper_assets.py --run_dir runs/<RUN_DIR> --paper_dir paper
```

## Compile the paper (requires `pdflatex` + `bibtex8`)
```bash
python scripts/compile_paper.py --paper_dir paper --tex vchimera_journal_q1.tex
```

## Calibration (recommended before large sweeps)
```bash
python scripts/calibrate_social.py --targets configs/calibration/targets_default.yaml --out configs/calibration/calibrated.yaml
```

Then run experiments using the calibrated override:
```bash
python scripts/run_experiments.py --config configs/experiments/journal_main.yaml --override configs/calibration/calibrated.yaml
```

## Robustness / sensitivity sweep (Q1-style robustness section)
```bash
python scripts/run_sensitivity.py --config configs/experiments/sensitivity.yaml
```

## Optional: benchmark backend transfer (CybORG adapter)
This artifact runs fully without external cyber backends. For **benchmark transfer** you can optionally run with CybORG (CC4 Enterprise scenario).
Install CybORG separately (see their docs / repo), then run:
```bash
python scripts/run_experiments.py --config configs/experiments/cyborg_transfer.yaml
```

If CybORG is not installed, the transfer run will print an actionable error and exit.

## Reproducibility
All experiments are controlled via YAML configs and use explicit RNG seeds.
Outputs include:
- per-step time series (`episode_steps.csv` if enabled),
- per-episode summaries,
- aggregated `summary.csv` with confidence intervals.

---
**Ethics & safety note:** the default cyber backend is an *abstract* attack-graph simulator (no exploit payloads or actionable offensive instructions).


## Quickstart (one command)

```bash
python scripts/run_all.py --compile
```

This runs main experiments + sensitivity + regenerates paper assets + compiles the PDF.
