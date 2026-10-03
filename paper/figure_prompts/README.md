# Figure sources and prompts

Use `figure1_prompt.md` and `figure2_prompt.md` for the latest two main figures.
Figure 1 is one horizontal row (a),(b),(c). Figure 2 is a two-stage 16:9 pipeline.
The second prompt corrects the negative-gradient sign, feature normalization,
feature-to-ridge versus gradient-to-SVD routing, target dimensions and query audio input.

`archive_4500_5000/` preserves all 18 historical prompt files and their scalar source data.
`archive_long/` preserves the earlier ~20,000-character method prompt. These are design
history, NOT later empirical evidence. Historical F1/F2/F3 templates must not override
completed results in `../data/latest/`; do not manufacture missing scores.

Active manuscript figures are versioned assets. A new Gemini rendering needs mathematical
review against the current prompt before replacing them. No image model was run during
this migration. Original high-resolution PNG author assets remain in the supplied complete
Overleaf archive; this public tree contains documented portable JPEG derivatives.
