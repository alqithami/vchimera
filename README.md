# IML: Institutional Monitoring & Ledger for Sequential Social Dilemmas

This repository provides a reference implementation of an **Institutional Monitoring and Ledger (IML)** wrapper for sequential social dilemma environments (Harvest and Cleanup) from *Sequential Social Dilemma Games* (SSD). The key idea is to keep the base Markov game intact while adding an auditable institutional layer that (i) monitors norm-relevant events, (ii) logs evidence to a ledger, and (iii) applies delayed, contestable settlement (sanctions/remedies).

The extended evaluation includes two established cooperation baselines (**Inequity Aversion** and **Social Influence**), three IML ablation variants, and comprehensive statistical analysis.

<p align="center">
  <img src="figures/graphical_abstract.png" width="500" alt="Simulation Environment">
</p>

---

## Experimental Conditions

| Condition | Description | Config prefix |
|---|---|---|
| **Baseline** | PPO agents, no cooperation mechanism | `*_baseline.yaml` |
| **Inequity Aversion (IA)** | Hughes et al. (NeurIPS 2018) — reward shaping via disadvantageous/advantageous inequity | `*_ia.yaml` |
| **Social Influence (SI)** | Jaques et al. (ICML 2019) — intrinsic reward for influencing peers | `*_si.yaml` |
| **IML Monitor Only** | Ablation: detection + ledger logging, no sanctions | `*_iml_monitor_only.yaml` |
| **IML No Review** | Ablation: detection + sanctions, no contestability/review | `*_iml_sanction_no_review.yaml` |
| **IML Full** | Full IML: detection + sanctions + review + ledger | `*_iml.yaml` |
| **IML High Review** | Ablation: full IML with high review probability (0.8) | `*_iml_high_review.yaml` |

Environments: **Cleanup** and **Harvest** from the SSD suite.

---

## Repository Layout

```
IML-main/
├── configs/                        # YAML configs for all conditions × environments
│   ├── cleanup_baseline.yaml
│   ├── cleanup_ia.yaml
│   ├── cleanup_si.yaml
│   ├── cleanup_iml.yaml
│   ├── cleanup_iml_monitor_only.yaml
│   ├── cleanup_iml_sanction_no_review.yaml
│   ├── cleanup_iml_high_review.yaml
│   ├── harvest_*.yaml              # Same set for Harvest
│   └── ...
├── iml_ssd/
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── inequity_aversion.py    # IA wrapper (Hughes et al., 2018)
│   │   └── social_influence.py     # SI wrapper (Jaques et al., 2019)
│   ├── institution/
│   │   ├── iml_wrapper.py          # Core IML wrapper
│   │   ├── iml_ablation.py         # Ablation preset configurations
│   │   ├── ledger.py               # Accountability ledger
│   │   └── rules.py                # Norm rules (NoPunishmentBeam, etc.)
│   ├── envs/
│   │   └── ssd_env.py              # SSD environment adapter
│   ├── rl/
│   │   ├── networks.py             # SharedCNNActorCritic
│   │   └── ppo.py                  # PPO implementation
│   ├── experiments/
│   │   ├── train.py                # Unified training (all conditions)
│   │   └── evaluate.py             # Unified evaluation (all conditions)
│   ├── analysis/
│   │   ├── aggregate.py            # Result aggregation across runs
│   │   ├── plot.py                 # Publication-ready figures
│   │   └── statistics.py           # Statistical tests (Mann-Whitney, Cohen's d, etc.)
│   └── utils/
│       ├── logging.py              # CSV + TensorBoard logging
│       └── metrics.py              # Gini, mean_ci, etc.
├── robustness/
│   ├── robust_eval_seed.py         # Cross-seed robustness analysis
│   └── sensitivity_cleanup_iml.py  # Hyperparameter sensitivity analysis
├── scripts/
│   ├── install_ssd_no_ray.sh       # Install SSD without Ray/RLlib
│   ├── run_sweep.sh                # Original baseline+IML sweep
│   └── run_full_sweep.sh           # Full sweep (all 7 conditions × 5 seeds × 2 envs)
├── rebuild_eval_seed_sweep.py      # Rebuild eval tables from per-run CSVs
├── results/                        # Aggregated results (generated)
├── figures/                        # Generated figures (generated)
├── runs/                           # Training run directories (generated)
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## System Requirements

- **Python:** 3.9 (SSD is not compatible with newer Python/Gym/NumPy stacks)
- **OS:** macOS or Linux recommended
  - **Windows:** use **WSL2 (Ubuntu)** or another Linux environment; the scripts are `bash`-based.
- **Tools:** `git`, and either **conda/miniforge** (recommended) or a compatible Python environment.

---

## Quickstart

### 1) Clone the repository

```bash
git clone https://github.com/alqithami/IML.git
cd IML
```

### 2) Create and activate a clean environment

```bash
conda create -n imlssd python=3.9 -y
conda activate imlssd
python -m pip install --upgrade pip setuptools wheel
```

### 3) Install SSD (without Ray/RLlib)

```bash
bash scripts/install_ssd_no_ray.sh
```

### 4) Install this package

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 5) Run a smoke test

```bash
python -m iml_ssd.tools.smoke_test --env cleanup --num_agents 5 --steps 50
```

---

## Training

### Single condition

```bash
# Baseline (PPO only)
python -m iml_ssd.experiments.train --config configs/cleanup_baseline.yaml --seed 0

# Full IML
python -m iml_ssd.experiments.train --config configs/cleanup_iml.yaml --seed 0

# Inequity Aversion baseline
python -m iml_ssd.experiments.train --config configs/cleanup_ia.yaml --seed 0

# Social Influence baseline
python -m iml_ssd.experiments.train --config configs/cleanup_si.yaml --seed 0

# IML ablation: monitor only
python -m iml_ssd.experiments.train --config configs/cleanup_iml_monitor_only.yaml --seed 0

# IML ablation: sanctions without review
python -m iml_ssd.experiments.train --config configs/cleanup_iml_sanction_no_review.yaml --seed 0

# IML ablation: high review probability
python -m iml_ssd.experiments.train --config configs/cleanup_iml_high_review.yaml --seed 0
```

### Override config values from CLI

```bash
python -m iml_ssd.experiments.train \
  --config configs/cleanup_iml.yaml \
  --seed 42 \
  --set ppo.total_steps=1000000 iml.sanction=1.0
```

### Full experimental sweep

Run all 7 conditions × 2 environments × 5 seeds = **70 training runs**, followed by evaluation, aggregation, plotting, and statistical analysis:

```bash
bash scripts/run_full_sweep.sh
```

---

## Evaluation

```bash
python -m iml_ssd.experiments.evaluate \
  --run_dir runs/cleanup_iml_agents5_seed0 \
  --episodes 50 \
  --seed 0 \
  --out_suffix "_seed0"
```

---

## Analysis Pipeline

### 1) Aggregate results

```bash
python -m iml_ssd.analysis.aggregate --runs_dir runs --out_dir results
```

### 2) Generate figures

```bash
python -m iml_ssd.analysis.plot --results_dir results --out_dir figures
```

Produces publication-ready PDF and PNG figures:
- Learning curves (welfare + inequality) for all conditions
- Evaluation bar charts
- Ablation study comparison
- Radar charts (multi-dimensional comparison)
- Seed-level robustness plots
- Sanction dynamics over training

### 3) Statistical tests

```bash
python -m iml_ssd.analysis.statistics --results_dir results --out_dir results/statistics
```

Generates:
- `pairwise_return_mean.csv` — Mann-Whitney U, Cohen's d, Cliff's delta for all condition pairs
- `pairwise_gini.csv` — Same for inequality (Gini)
- `summary_return_mean.csv` — Descriptive statistics per condition
- `tab_pairwise_return_mean.tex` — LaTeX tables for the paper
- `tab_summary_return_mean.tex`

### 4) Robustness analysis

```bash
python robustness/robust_eval_seed.py
python robustness/sensitivity_cleanup_iml.py
```

### 5) Rebuild evaluation-seed sweep (optional)

```bash
python rebuild_eval_seed_sweep.py
```

---

## Baseline Implementations

### Inequity Aversion (IA)

Based on [Hughes et al. (NeurIPS 2018)](https://arxiv.org/abs/1803.00991). Modifies agent rewards:

```
r_i' = r_i - α · max(r̄ - r_i, 0) - β · max(r_i - r̄, 0)
```

- `α = 5.0` (disadvantageous inequity aversion)
- `β = 0.05` (advantageous inequity aversion)

### Social Influence (SI)

Based on [Jaques et al. (ICML 2019)](https://arxiv.org/abs/1810.08647). Adds intrinsic reward proportional to the agent's influence on peers:

```
r_i' = r_i + w · influence_i
```

Where `influence_i` measures how much agent `i`'s reward deviates from the group mean (proxy for counterfactual influence).

### IML Ablation Design

| Ablation | Monitor | Sanction | Review | Ledger |
|---|---|---|---|---|
| Monitor Only | ✓ | ✗ | ✗ | ✓ |
| No Review | ✓ | ✓ | ✗ | ✓ |
| Full IML | ✓ | ✓ | ✓ | ✓ |
| High Review | ✓ | ✓ | ✓ (p=0.8) | ✓ |

---

## Statistical Analysis

All comparisons use:
- **Mann-Whitney U test** (non-parametric, appropriate for small n)
- **Welch's t-test** (parametric comparison)
- **Cohen's d** with 95% CI (effect size)
- **Cliff's delta** (non-parametric effect size)
- **Bootstrap confidence intervals** (10,000 resamples)
- **Holm-Bonferroni correction** for multiple comparisons
- **Kruskal-Wallis H test** for omnibus comparison

---

## Compute Backend Notes (CPU / CUDA / Apple Silicon)

```bash
python -c 'import torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("mps", hasattr(torch.backends,"mps") and torch.backends.mps.is_available())'
```

- **CUDA** is used if `torch.cuda.is_available()` is `True`.
- **Apple Silicon (MPS)** is used if `torch.backends.mps.is_available()` is `True`.
- **CPU-only** works, but sweeps are compute-intensive.

---

## Troubleshooting

### `No module named 'social_dilemmas'`
Re-run:
```bash
bash scripts/install_ssd_no_ray.sh
```

### `No module named 'cv2'`
```bash
python -m pip install "numpy<2" "opencv-python<4.13"
```

### Gym / NumPy warnings
SSD depends on the legacy `gym` package, which emits warnings under NumPy 2. This repo constrains NumPy to `<2` for compatibility.

---

## Citation

If you use this code in academic work, please cite the accompanying manuscript and the SSD benchmark:

```bibtex
@article{alqithamiIML2026,
  title   = {Designing Contestable AI Institutions: Interactive Accountability
             Mechanisms for Cooperative Multi-Agent Systems},
  author  = {Alqithami, Saad},
  journal = {Multimodal Technologies and Interaction},
  year    = {2026},
  note    = {Manuscript under review. Code: https://github.com/alqithami/IML}
}

@inproceedings{leibo2017multi,
  title={Multi-agent Reinforcement Learning in Sequential Social Dilemmas},
  author={Leibo, Joel Z and Zambaldi, Vinicius and Lanctot, Marc and Marecki, Janusz and Graepel, Thore},
  booktitle={Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems},
  pages={464--473},
  year={2017}
}
```

---

## License

This repository is released under the **MIT License**. See `LICENSE`.

---

## Acknowledgements

This repository builds on *Sequential Social Dilemma Games* (SSD) and the sequential social dilemmas introduced by Leibo et al. (2017) cited above. The Inequity Aversion baseline follows Hughes et al. (2018) and the Social Influence baseline follows Jaques et al. (2019).
