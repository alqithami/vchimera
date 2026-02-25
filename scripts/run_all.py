#!/usr/bin/env python3
"""One-command runner for the V-CHIMERA artifact.

This script executes:
  1) main experiments (journal_main.yaml)
  2) sensitivity sweep (sensitivity.yaml)
  3) paper asset generation (tables/figs)
  4) paper compilation (optional; requires LaTeX installed)

Usage:
  python scripts/run_all.py

You can override configs via CLI:
  python scripts/run_all.py --main configs/experiments/journal_main.yaml \
                           --sensitivity configs/experiments/sensitivity.yaml \
                           --compile
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(cmd: list[str]) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="configs/experiments/journal_main.yaml")
    ap.add_argument("--sensitivity", default="configs/experiments/sensitivity.yaml")
    ap.add_argument("--paper", default="paper")
    ap.add_argument("--tex", default="vchimera_intr_q1.tex")
    ap.add_argument("--compile", action="store_true", help="Compile LaTeX PDF (requires pdflatex + bibtex8)." )
    return ap.parse_args()

def latest_run(run_root: Path, prefix: str) -> Path | None:
    # pick newest directory that starts with prefix_
    candidates = [p for p in run_root.iterdir() if p.is_dir() and p.name.startswith(prefix + "_")]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

def main() -> None:
    args = parse_args()
    run_root = ROOT / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    # 1) main experiments
    run([sys.executable, "scripts/run_experiments.py", "--config", args.main])
    main_run = latest_run(run_root, "journal_main") or latest_run(run_root, "quick") or latest_run(run_root, "journal_demo")
    if main_run is None:
        raise RuntimeError("No run directory found under runs/ after running experiments.")

    # 2) sensitivity sweep
    run([sys.executable, "scripts/run_sensitivity.py", "--config", args.sensitivity])

    # 3) paper assets
    run([sys.executable, "scripts/make_paper_assets.py", "--run_dir", str(main_run), "--paper_dir", args.paper])

    # 4) compile paper
    if args.compile:
        run([sys.executable, "scripts/compile_paper.py", "--paper_dir", args.paper, "--tex", args.tex])

    print("[OK] Done.")
    print("Main run:", main_run)
    print("Paper assets:", Path(args.paper)/"paper_assets")

if __name__ == "__main__":
    main()
