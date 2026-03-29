# Manuscript source (MDPI Biomimetics template)

Main entrypoint: `main.tex`

Compile (example):

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Figures are in `figures/`. Tables are in `paper_assets/`.

A regeneration script is provided in `scripts/regenerate_figures_bw_accent.py`.
