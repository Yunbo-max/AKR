# Manuscript release

Title: **Acoustic-to-KV Regression for Bioacoustic Recognition in Speech Language Models**.

`main.tex` and `references.bib` are restored from the user-approved nine-page RQ-organized
source package. Only the approved method/title naming and portable image extensions
are changed in this migration. All numerical table bodies are checked against the
source. No negative transfer, scale, direct-ridge or class-coverage row is removed.

Run `bash paper/build.sh` from the repository root, or upload this directory to Overleaf
with `main.tex` as the root and pdfLaTeX as the engine. All active figure dependencies
are ordinary files. `main.bbl` is a convenience cache; the bibliography source is present.

`figures/intro.jpg` and `figures/pipeline.jpg` are documented portable derivatives of
the author's PNG assets (see `../migration/figure_renderings.json`). Full-resolution
original PNGs and the previous rendered PDF are retained in the supplied Overleaf
archive, not falsely represented as recovered public bytes. Regeneration prompts are
in `figure_prompts/`, with the latest two specifications outside historical archives.

The source package is not itself an anonymous submission: repository identity,
provenance and historical author metadata need a separate anonymity check. Use the
compiled manuscript only after author review. Code tests establish implementation
contracts, not scientific claims or venue acceptance.
