## 1. Figure goal
Create the Dogs source-to-target bandwidth transfer matrix for the AKR paper. Display the deficit of a transferred frozen-feature readout relative to a readout fitted at the target bandwidth. The matrix should reveal condition-specific compatibility, including negative differences, rather than merely reproduce a table of high and low accuracies. This is a readout diagnostic, not an AKR benchmark.

## 2. Canvas and layout
Use a 1:1 square white canvas. The central object is a 6 by 6 square matrix, with generous left and bottom margins for source and target labels. Place one slim vertical color scale on the right, separate from the matrix by a small white gutter. Keep cells genuinely square; do not stretch them to fill a wide page. Use no global title, subtitle banner, decorative animal image, gradient background, or heavy table frame.

## 3. Locked counts and computation
Read additional_analysis/source_data.json, section transfer.Dogs. The denominator is N=139 paired test recordings. Both row and column order is [Full, LP1, LP2, LP4, LP6, LP8]. The saved correct-count matrix C, with rows as readout-training condition and columns as target condition, is:
[129, 56, 83, 113, 128, 127]
[54, 122, 69, 40, 46, 53]
[98, 61, 127, 104, 99, 101]
[121, 48, 102, 127, 122, 122]
[116, 47, 75, 108, 124, 111]
[129, 52, 103, 119, 127, 132]
Compute raw accuracy M[s,t]=100*C[s,t]/N. Plot D[s,t]=M[t,t]-M[s,t] in percentage points, not M itself and not M[s,s]-M[s,t]. Use exact counts for computation. Round only final cell annotations to one decimal. The diagonal is zero by definition; it is not proof of perfect recognition.

## 4. Matrix encoding
Use a restrained diverging scale with a neutral midpoint at zero and the same limits [-10,70] pp for the Dogs and Watkins deficit matrices. Positive cells indicate worse performance than the target-fitted readout; negative cells indicate better observed transfer performance. Never clip negative cells to zero. Print every cell's signed value, with plain 0.0 on the diagonal, and label the color scale 'Deficit (pp)'. Use fine white separators rather than thick black grid lines. Switch annotation tone for contrast where needed; color alone must not carry the numerical result. Avoid a second inset heatmap or clustered dendrogram.

## 5. Orientation and labels
Put 'Readout training bandwidth' on the vertical axis and 'Evaluation bandwidth' on the horizontal axis. Repeat the exact six short condition labels on both axes. Full means the unfiltered model-observable baseband; LP1/2/4/6/8 denote low-pass cutoffs in kHz. Full must not be merged with LP8. Keep source labels readable from top to bottom in the stated order, and target labels left to right. Do not reorder rows to create visually cleaner clusters or to maximize a diagonal pattern. Each column uses the same target recordings and reference M[t,t], making a within-column comparison interpretable.

## 6. Specific checks and emphasis
For Full -> LP1, verify 87.77 - 40.29 = 47.48 pp before rounding the cell. For LP1 -> Full, verify 92.81 - 38.85 = 53.96 pp. These examples use target-matched references, not the source-matched drops of the bidirectional point chart. Keep all measured off-diagonal cells, including unexpected or small negative deficits. Use a regular-weight sans-serif family, compact axis labels, and tabular numerals. A small N=139 note may sit near the legend; task descriptions and selection procedures belong in the external caption.

## 7. Protocol and interpretation
BD-ID denotes BEANS Dogs individual identification across ten dogs. The same 139 test recordings underlie every condition pair. It is not a six-class marmoset task or an animal-intention decoder. Source validation selects the feature layer and ridge regularization; each source-trained readout is then applied unchanged to the paired targets. This target-reference display differs from the source-matched endpoint chart: the baseline changes across columns, not within a column. Do not interpret color intensity as mutual information, a distance between neural boundaries, a causal spectral mechanism, or a cross-family model comparison. No synthetic off-diagonal values may be reconstructed from a single full-input transfer curve.

## 8. Delivery and validation
Export the specified PDF, an editable SVG, and a square high-resolution PNG. Keep a small machine-readable table of D next to the plotting script. Confirm a zero diagonal, preserved signs, correct denominator, and the exact source/target orientation. Preserve the 1:1 page frame on export; accommodate labels within it rather than letting automatic cropping change the ratio. Use the supplied counts without fitting any model or generating additional predictions. Inspect all axis text and cell annotations at final size, and ensure the shared color-scale limits match the companion dataset matrix.
