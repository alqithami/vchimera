#!/usr/bin/env bash
set -euo pipefail

# Example sweep: 5 seeds baseline vs IML for Cleanup and Harvest.
#
# Usage:
#   bash scripts/run_sweep.sh

SEEDS=(0 1 2 3 4)

for SEED in "${SEEDS[@]}"; do
  python -m iml_ssd.experiments.train --config configs/cleanup_baseline.yaml --seed "$SEED"
  python -m iml_ssd.experiments.train --config configs/cleanup_iml.yaml --seed "$SEED"
  python -m iml_ssd.experiments.train --config configs/harvest_baseline.yaml --seed "$SEED"
  python -m iml_ssd.experiments.train --config configs/harvest_iml.yaml --seed "$SEED"
done

echo "Sweep complete. Aggregate + plot:"
echo "  python -m iml_ssd.analysis.aggregate --runs_dir runs --out_dir results"
echo "  python -m iml_ssd.analysis.plot --results_dir results --out_dir figures"
