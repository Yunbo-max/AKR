# AKR manuscript sources

This directory consolidates the supplied nine-page RQ-organized manuscript,
its bibliography/style files, complete numeric tables, plotting data, small
vector PDF figures, and Figure 1/2 prompts. It is an author working draft, not
an assertion of venue acceptance or a new experimental evaluation.

## Build

```bash
cd paper
bash build.sh
```

The release build produces 9 main-text pages and 24 total pages with the
supplied style. It requires pdfLaTeX and BibTeX. The style is unmodified.

Editorial changes in this consolidation: the title and method expansion now
use **Acoustic-to-KV Regression for Bioacoustic Recognition in Speech Language
Models**, as approved in the conversation. No numeric results were changed.
The draft's references are preserved; the bounded literature check is not a
full submission-grade bibliography or novelty audit.

For transport, `figures/intro.jpg` and `figures/pipeline.jpg` are 1000-pixel
JPEG renderings of the supplied PNG images. Layout, labels and numeric content
are unchanged, but the renditions are not pixel-identical. The original
high-resolution PNGs remain in the supplied `AKR_ICLR2027_Final_9Pages_Tables.zip`
and in the full-resolution delivery accompanying this consolidation.
For publication, replace the JPEGs with those original PNGs and change the two
includegraphics extensions. Hashes and provenance are in
`../migration/figure_renderings.json`. No new illustration or data was invented.

`data/latest/` contains the supplied F1/F2/F3 scalar exports. A file named
FINAL_STATUS in that directory belongs to the historical source study; it is
not the status of a new run. New evaluations write to `../results/akr_final/`.

Raw audio, model weights, private repository HEAD, and absent large gradient
arrays are not reconstructed from prose or aggregate scores. Figure prompts
are design instructions, not additional empirical evidence.
