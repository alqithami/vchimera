#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compile LaTeX paper with bibtex8 (clean build).")
    p.add_argument("--paper_dir", required=True)
    p.add_argument("--tex", required=True)
    return p.parse_args()


def run(cmd, cwd: Path, check: bool = True) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=check)


def clean_aux(stem: str, cwd: Path) -> None:
    for ext in [".aux", ".bbl", ".blg", ".out", ".toc", ".lof", ".lot"]:
        p = cwd / (stem + ext)
        if p.exists():
            p.unlink()


def main() -> None:
    args = parse_args()
    paper_dir = Path(args.paper_dir)
    tex = args.tex
    stem = Path(tex).stem

    # Clean build to avoid stale .bbl issues
    clean_aux(stem, paper_dir)

    # First pass: generate .aux (ignore non-zero as long as PDF is produced)
    run(["pdflatex", "-interaction=nonstopmode", tex], cwd=paper_dir, check=False)

    # Bibtex
    run(["bibtex8", stem], cwd=paper_dir, check=False)

    # Final passes
    run(["pdflatex", "-interaction=nonstopmode", tex], cwd=paper_dir, check=False)
    run(["pdflatex", "-interaction=nonstopmode", tex], cwd=paper_dir, check=False)

    out_pdf = paper_dir / (stem + ".pdf")
    print("[OK] Compiled:", out_pdf)


if __name__ == "__main__":
    main()
