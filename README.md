# V-CHIMERA

V-CHIMERA is a cyber–social crisis-response simulation and evaluation artifact for studying misinformation-aware cyber defense. The repository contains the **core simulation code**, baseline and ablation policies, experiment runners, calibration and sensitivity tooling, and selected run outputs.

## What the artifact does

V-CHIMERA couples:
1. a cyber incident simulator or external cyber backend,
2. a social multi-agent / ABM misinformation layer,
3. a runtime communication protocol monitor and shield,
4. experiment scripts that produce run logs and aggregate summaries.

The repository supports:
- policy comparisons (`pipeline`, `pipeline+shield`, `vchimera`, and ablations),
- protocol accounting (attempted violations, executed violations, shield interventions),
- calibration of social-dynamics parameters,
- robustness / sensitivity sweeps,
- optional transfer experiments through a CybORG adapter.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick sanity check

```bash
python scripts/smoke_test.py
```

## Main experiment run

```bash
python scripts/run_experiments.py --config configs/experiments/journal_main.yaml
```

This creates a timestamped run directory under `runs/`, for example:

```text
runs/journal_main_YYYYMMDD_HHMMSS/
```

## Calibration

```bash
python scripts/calibrate_social.py   --targets configs/calibration/targets_default.yaml   --out runs/calibration_social_local.yaml
```

You can then pass the resulting calibration file back into your experiment workflow if you maintain a calibrated config branch or local override.

## Sensitivity / robustness

```bash
python scripts/run_sensitivity.py --config configs/experiments/sensitivity.yaml
```

## Statistical comparison

```bash
python scripts/stat_tests.py   --run_dir runs/journal_main_YYYYMMDD_HHMMSS   --out table_stats.tex
```

## Optional CybORG transfer

The repository includes a CybORG adapter, but CybORG itself is **not shipped in this repository**. Install or clone the appropriate CybORG / CAGE Challenge assets separately, then run:

```bash
python scripts/run_experiments.py --config configs/experiments/cyborg_transfer.yaml
```

## Outputs

Typical experiment outputs include:
- per-seed summaries,
- aggregate `summary.csv`,
- optional per-step logs,
- sensitivity outputs under `runs/sensitivity_grid/`.

## License

Code in this repository is provided under the MIT License. 
