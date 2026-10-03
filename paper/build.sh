#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PDFLATEX=${PDFLATEX:-pdflatex}
if [[ -z "${BIBTEX:-}" ]]; then
  if command -v bibtex >/dev/null 2>&1; then BIBTEX=bibtex
  elif command -v bibtex.original >/dev/null 2>&1; then BIBTEX=bibtex.original
  else echo 'BibTeX is required; use Overleaf or set BIBTEX to a working executable.' >&2; exit 1
  fi
fi
"$PDFLATEX" -interaction=nonstopmode -halt-on-error main.tex
"$BIBTEX" main
"$PDFLATEX" -interaction=nonstopmode -halt-on-error main.tex
"$PDFLATEX" -interaction=nonstopmode -halt-on-error main.tex
