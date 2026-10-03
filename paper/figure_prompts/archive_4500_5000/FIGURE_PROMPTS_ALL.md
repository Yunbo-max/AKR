# AKR 图示 prompts｜4,500–5,000 英文字符版

18 份独立英文提示：15 张现有论文图 + F1/F2/F3 三份待结果模板。每张只提供一个确定构图。

每份代码块内及独立文件的完整内容均为 4,500–5,000 字符（含空格、LF 换行与小标题）。不要把目录说明一起复制进单图 prompt。

方法流程图固定为 16:9；单幅矩阵、诊断散点与敏感性曲线使用 1:1；并排比较与多面板图使用 16:9。所有长说明和图注均置于图外。


---

## 01｜任务与同监督动机

**画幅：16:9｜完整 prompt：4,857 字符｜已有图示／数值已给出**

随附数据：data/plot_data.json

```text
## 1. Figure goal
Create the opening motivation figure for the AKR bioacoustic recognition paper. Explain one concrete contrast: the same labelled animal recordings support a useful fitted acoustic readout but much weaker native audio in-context learning. The task is recognizing call types within one marmoset species, not distinguishing different species or translating animal intentions.

## 2. Canvas and hierarchy
Use a 16:9 landscape canvas. Keep the background pure white and the artwork flat, precise, and editable. Reserve 4% outer margins and a narrow, consistent gutter between three aligned panels. Use restrained sage, slate, or lavender accents only to organize information; marker shapes must retain meaning in grayscale. No overall title, paper title, decorative portrait, banner, drop shadow, or long explanatory paragraph inside the artwork. The section headings in this prompt are instructions, not labels to print.

## 3. Composition
Allocate approximately 27% of the usable width to panel (a), 35% to (b), and 38% to (c). Align the top edges of all panel bodies and the baselines of the two quantitative plots. Panel (a) contains three compact frequency-versus-time contour sketches, stacked vertically and labelled Phee, Trill, Twitter. Phee is a sustained tonal contour; Trill is a periodically modulated contour; Twitter is a sequence of separated upward sweeps. Treat them as conceptual illustrations, not sampled waveforms. A slim bracket ties the examples to the label 'One species'. Do not draw different animals for different call types.

## 4. Locked measurements
Use data/plot_data.json, section support. Panel (b) compares exactly the same support identities and the same 75 recording-disjoint queries. At k=1 recording per call type, ridge accuracy is 46.67% and audio ICL is 8.00%. At k=2, ridge is 58.67% and audio ICL is 17.33%. The respective paired gaps are 38.67 and 41.33 percentage points. There are six classes: k=1 means six labelled clips, and k=2 means twelve labelled clips, not two animals. Panel (c) uses k=[1,2,4,8,16]. Ridge accuracies are [46.67,58.67,73.33,84.00,82.67]; nearest-centroid accuracies are [29.33,46.67,50.67,78.67,76.00]. Preserve all five points. Do not infer ICL values at k=4,8,16 from these curves or insert separate order-control runs.

## 5. Quantitative design
Panel (b) uses a shared 0-100% accuracy scale, square markers for ridge and triangles for ICL. Place a narrow vertical interval marker between each pair and label only its gap, '38.67 pp' or '41.33 pp'. Panel (c) uses the same 0-100% scale with exact support ticks; use a log2 coordinate or equally spaced doubling positions clearly labelled 1,2,4,8,16. Ridge keeps its square/solid encoding; centroid uses circles and a dashed line. Connect only observed points with straight segments. Do not smooth, extrapolate, or draw a fitted saturation law. Keep quantitative plot regions larger than legends, and avoid the redundant display of every value, gap, and percentage at once.

## 6. Typography and annotation
Use one clean sans-serif family, regular-weight axis labels, and tabular numerals. Design for roughly 8-9 pt labels at a 5.5-inch final composite width; resize the layout before shrinking type. Panel letters (a), (b), (c) are small and consistently positioned. Axis wording is 'Recordings per call type' and 'Accuracy (%)'. Keep the two method names in one compact legend, not oversized headings. The only required prose inside panel (a) is 'Illustrative contours'. Put the 75-query count, recording-disjoint protocol, and distinction between recording and caller identity in the external caption. Let whitespace separate concepts instead of heavy boxes.

## 7. Interpretation boundary
Show that annotation budget alone does not explain the readout/ICL disparity. Do not claim compute-matched training, a proven failure at a particular decoder layer, unseen-caller recognition, or that ICL has no acoustic sensitivity. The ridge curve measures a fitted classifier; it is not AKR performance. The call drawings carry no numerical time or frequency claims. An unchanged or weaker point at k=16 must remain visible rather than being corrected to fit a more attractive trend.

## 8. Production and final check
Generate numeric panels with plotting code from the provided data, not painted approximations. Export editable PDF and SVG plus a high-resolution PNG with the full 16:9 canvas retained. Supply the three component panels for figures/fig1_calls.pdf, figures/fig1_matched.pdf, and figures/fig1_scaling.pdf, together with an optional assembled preview. Keep the caption outside the image. Inspect at final width for collisions, clipped ticks, unreadable subscripts, and inconsistent line weights. Every plotted point must trace to the locked arrays above; no new model run or invented uncertainty is permitted.
```


---

## 02｜AKR 方法主图

**画幅：16:9｜完整 prompt：4,776 字符｜已有图示／数值已给出**

随附数据：方法定义已写在 prompt 中，无需实验数据

```text
## 1. Figure goal
Draw the main architecture of Acoustic-to-KV Correction Regression (AKR): a small regressor learns from labelled support how frozen acoustic features predict corrective K/V increments. At inference, each new recording receives its own additive update before the original speech language model emits a text label.

## 2. Canvas and hierarchy
Use one 16:9 landscape vector canvas with a pure white background, 4% outer margins, and two horizontal lanes. The upper 44% is support fitting; the lower 46% is query inference, separated by a light rule and a generous gutter. Use muted sage for frozen acoustic features and pale lavender for correction objects; use neutral gray for the backbone. All distinctions must survive grayscale. No overall title, numbered algorithm, decorative animal, gradient background, perspective block, or large prose box.

## 3. Support fitting lane
At the upper left place a compact stack of labelled waveform cards (x_i,y_i). The feature branch yields h_i and then support-standardized z_i. The target branch uses the correct-label token cross-entropy L_ans and its negative gradient at selected audio-token K/V projection outputs. Show a short audio/text token strip with only audio positions active. Pool across those audio positions to produce g_i; do not depict a full tokenwise field as the main target. Send the pooled target into 'Centre + SVD', yielding the mean g_bar, basis B, and coefficients a_i. Join z_i and a_i at a small 'Ridge fit' block producing W,b. Label y_i must appear only in this support lane.

## 4. Equations and learned objects
Use three compact equations as typeset labels, not explanatory sentences: a_i = B^T(g_i - g_bar); L_reg = sum_i ||W z_i + b - a_i||^2 + lambda ||W||_F^2; g_hat_q = g_bar + B(W z_q + b). Keep the reconstruction equation in the most prominent correction block. L_ans supplies targets; L_reg fits the auxiliary regressor. The fitted objects are B, g_bar, W,b, and feature normalizers. The language-model weights theta are frozen throughout. The support loss differentiates states without a backbone optimizer step.

## 5. Query inference lane
A new waveform x_q enters a frozen feature pass at lower left, producing z_q using saved support statistics. The saved ridge map yields continuous signed coefficients, which are reconstructed into g_hat_q and unpacked as Delta K and Delta V. A second forward path carries the same recording and its prompt into corrected generation. Draw this as a separate pass through shared frozen weights, not a time-travel connection from future hidden states into an earlier layer of the same pass. End at one short text-label card. A dashed transfer connector from fitted support objects to the query predictor is allowed; no backward arrow or ground-truth query label enters this lane.

## 6. Attention inset and arrow grammar
Enlarge one attention block at lower right. Show separate k_proj and v_proj outputs, plus signs adding predicted increments at audio positions, then ordinary attention and decoding. Place the key addition before head reshaping and RoPE. Place the value addition at its projection output before head reshaping; do not draw RoPE applied to values. Preserve original K/V states through the plus nodes. Within a projection block the pooled increment is broadcast across that recording's audio positions, although the increment differs between recordings. Use solid arrows for forward data and dashed arrows for support supervision or saved objects. Align ports and use orthogonal bends; avoid crossed arrows or ambiguous branch junctions.

## 7. Typography and scientific boundaries
Use a consistent sans-serif label family with properly typeset math. Keep labels about 8-9 pt at final paper width, with most module names below four words. Lane labels may read 'Support fitting' and 'Query inference'; include a small lock icon with 'Frozen' on the backbone and 'Fit' on the regressor. Do not add teacher/student towers, policy distillation, reward models, self-generated training answers, hard class routing, or projections K P and V P. The shared basis describes update coordinates; it does not replace the original KV space. Avoid 'training-free': auxiliary support fitting is supervised. Put longer caveats in the external caption, not on the canvas.

## 8. Production and final check
Deliver figures/fig2_method.pdf, an editable SVG, and a 16:9 PNG preview. Keep the diagram scalable with separate layers for arrows, blocks, math, and tokens. Inspect the exact data dependencies: targets and labels stay in support; feature extraction precedes query correction; original decoding remains intact; only audio positions receive the main-method increment. No recognition percentages belong in this architecture figure.
```


---

## 03｜三任务频谱读出分析

**画幅：16:9｜完整 prompt：4,852 字符｜已有图示／数值已给出**

随附数据：data/plot_data.json

```text
## 1. Figure goal
Create a three-task result figure distinguishing information recoverable by a newly fitted readout from the performance of an unchanged readout after a bandwidth change. Compare condition-fitted probes, full-input probes transferred unchanged, and native generation. This is a supervised-readout diagnostic in the AKR paper, not three curves of AKR itself.

## 2. Canvas and hierarchy
Use a 16:9 white landscape canvas with three aligned, equal-width plotting panels. Use 4% outer margins, a modest horizontal gutter, and one common legend in reserved space below the panels. Panel identifiers are (a) MA-CT, (b) BD-ID, (c) BW-SP; no overall title or slogan. MA-CT means marmoset call types, BD-ID dog individuals, and BW-SP marine-mammal species. Keep plot areas large enough for readable labels at paper width. Avoid colored panel backgrounds, shadows, illustrations, and large text callouts.

## 3. Axes and layout
Each panel uses x ticks 1,2,4,6,8 labelled as low-pass cutoff in kHz, positioned on a linear numerical axis. Each uses a common y range 0-100% and identical vertical tick positions. Align the baselines and top plot boundaries precisely. One left-side 'Accuracy (%)' label can serve the composite, but preserve y tick values where they aid separate-panel export. Put 'Low-pass cutoff (kHz)' below each component or once beneath the shared assembly. Do not add Full at the same x coordinate as 8 kHz: unfiltered observable baseband and the 8-kHz filtered control are distinct conditions.

## 4. Locked data
Read data/plot_data.json, section frequency; use its available precision and round labels only. In cutoff order [1,2,4,6,8], MA-CT native is [12.64,15.20,19.23,23.08,28.39], condition-fitted is [77.29,79.85,82.60,93.96,94.87], and full-trained transfer is [34.98,54.58,59.52,88.64,95.42]. BD-ID native is [2.88,2.88,2.88,2.88,2.88], condition-fitted [87.77,91.37,91.37,89.21,94.96], transfer [40.29,59.71,81.29,92.09,91.37]. BW-SP native is [6.19,6.49,5.60,6.19,5.90], condition-fitted [74.04,77.88,85.84,86.73,84.07], transfer [9.73,24.19,71.09,82.60,86.73]. Preserve dips, crossings, and exact observed conditions; do not impose monotonicity.

## 5. Marks and visual emphasis
Use squares and a solid line for the condition-fitted readout, circles and a dashed line for the unchanged full-input readout, triangles and a dotted line for native generation. Reuse these encodings in every panel. Muted colors may distinguish the three methods, but the markers and line styles must be sufficient without color. Use a light horizontal guide grid only. Emphasize the 1-kHz separation with a thin bracket between the two probes, labelled 42.31 pp, 47.48 pp, and 64.31 pp, respectively, only where it fits without covering a curve. Keep the native curve visibly separate from that bracket. No filled area should imply an information-theoretic quantity.

## 6. Typography and annotation
Use a clean regular-weight sans-serif family and a single hierarchy: compact panel tags, readable axis text, then a small legend. Design at a 5.5-inch composite width with approximately 8-9 pt main labels. Do not label all forty-five points; label selected 1-kHz endpoints or the gap, not both when space is tight. Use short legend phrases: 'Condition-fitted', 'Full-trained transfer', 'Native'. Keep paragraph explanations outside the image. Avoid abbreviating the three methods to cryptic single letters. Place the legend below the axes rather than over high-performing curves.

## 7. Protocol and interpretation
Both probes in a panel score the same target-condition queries. Only the condition-fitted probe is fitted on that condition's training features; the transferred readout is held fixed. MA-CT uses grouped out-of-fold results on 546 clips; BD-ID and BW-SP use their declared 139- and 339-query fixed splits. These are not a common few-shot budget comparison across datasets. High probe performance establishes recoverable category information, not complete native understanding. The figure does not prove that frequency drift causes every ICL error or that within-species recognition is intrinsically harder than species recognition. Keep all such details in the external caption.

## 8. Delivery and validation
Render each numeric panel using plotting code and assemble into the 16:9 figure. Export editable PDFs to figures/fig3_marmaudio.pdf, figures/fig3_dogs.pdf, and figures/fig3_watkins.pdf; also provide SVG components and a composite PNG. Preserve the canvas aspect ratio when exporting the composite, instead of trimming one side to make labels fit. Verify all fifteen values per task against the source arrays, inspect crossings and negative probe gaps without altering them, and check grayscale readability. Do not add unseen bandwidths, confidence ribbons, fictitious spectrograms, or new experimental runs.
```


---

## 04｜连续 AKR 五划分结果

**画幅：16:9｜完整 prompt：4,929 字符｜已有图示／数值已给出**

随附数据：data/plot_data.json

```text
## 1. Figure goal
Show the main historical continuous-AKR recognition result as complete paired outcomes across five recording-group draws, with a complementary macro-F1 summary. Readers should see the difference between a fixed mean update and a recording-specific predicted update, without confusing these results with class-routed variants, Oracle interventions, or the later balanced-support protocol.

## 2. Canvas and hierarchy
Use a 16:9 landscape canvas, pure white background, flat vector marks, and generous outer margins. Allocate about 60% of the usable width to panel (a), the per-draw accuracy plot, and 40% to panel (b), macro-F1. Align plotting baselines and keep a visibly wider gutter than the line widths. Use no overall title, decorative model diagram, shaded panel box, or oversized improvement badge. A compact shared legend should sit below the panels. The plot, not its explanatory text, should dominate the area.

## 3. Locked raw observations
Read data/plot_data.json, section repair. Draw indices 1,2,3,4,5 correspond to seeds 20250813,20250814,20250815,20250816,20250817. Query counts are [39,64,57,68,82]. Native correct counts are [5,9,3,11,11]; fixed-mean counts [2,19,4,5,13]; continuous-AKR counts [12,27,16,16,31]. Calculate accuracy as 100 times correct count divided by that draw's query count. Do not substitute rounded counts, combine all 310 event occurrences as independent cases, or sort draws by improvement. Each draw uses twenty TOTAL error-enriched support recordings, rank four, feature index zero, ridge penalty ten, and raw eta=300.

## 4. Per-draw panel
Use x ticks 1 to 5, in registered order, with a small second line giving n=39,64,57,68,82. Put the long seed strings in the external caption, not on the axis. Set the accuracy axis from 0 to 55%, with clear ticks and a light horizontal grid. Use triangles/dotted lines for Native, circles/dashed lines for Fixed mean, and squares/solid lines for Continuous AKR. Draw straight connectors between observed points only; these connect draw summaries and do not imply a temporal learning curve. Keep the small native/fixed values fully visible near the baseline. Do not remove the high fixed score in draw two or the comparatively small AKR gain in draw four.

## 5. Macro-F1 panel
Plot three mean points with sample-standard-deviation whiskers, not large filled bars. In percent, Native is 5.9154 +/- 1.6922, Fixed mean 6.7399 +/- 4.3591, and Continuous AKR 24.5399 +/- 3.8137. Label the y axis 'Macro-F1 (%)' and use a 0-35% range. Use exactly the same method markers as panel (a). Short x labels are Native, Fixed, AKR. A small line beneath the panel may say 'Mean +/- sample SD'. These error bars describe dispersion across the five draws; they are not confidence intervals from independent query sets. Do not compare the height of this axis directly with accuracy on the other panel.

## 6. Text and scientific emphasis
Use restrained neutral and muted accent colors, with the continuous predictor emphasized by its solid line rather than by a bright background. All labels should remain readable in grayscale at a 5.5-inch composite width. Use regular sans-serif type and consistent decimal precision. A single unobtrusive annotation may report mean accuracy 12.35% to 32.47%, with Fixed at 13.01%; do not repeat this as a large title, paragraph, and badge. Keep actual point values in an accompanying data table when direct labels become crowded. Panel letters and short metric names are sufficient; the paper caption supplies the interpretation.

## 7. Protocol boundaries
Within each draw, support recordings do not overlap with query recordings. Different draws come from the same source pool and may overlap with each other. The support pool is error enriched and twenty-total, not eight or twenty recordings per class. Fixed and predicted increments use the same raw eta, but this is not a matched-relative-norm intervention study. The method is continuous correction regression: a shared basis plus acoustic-feature-dependent coefficients. Do not attribute the Dogs paraphrase gain to this curve or label it cross-family generalization. Showing positive gains in all five draws does not mean all six call types improve uniformly.

## 8. Export and final check
Generate charts directly from the saved counts and F1 statistics. Export figures/fig4_splits.pdf and figures/fig4_f1.pdf as editable vector panels, with SVG versions and a 16:9 composite PNG preview. Keep the external caption separate. Before delivery recompute the three mean accuracies and confirm 12.35%, 13.01%, 32.47% to the displayed precision. Check that the whiskers are sample SD, draw order is unchanged, negative lower-bound issues are not hidden by shifting the axis, and every method uses the same queries in each draw. This is a plotting task only: do not run inference, fit a new regressor, select a better seed, or add simulated uncertainty.
```


---

## 05｜梯度方向与读出漂移

**画幅：1:1｜完整 prompt：4,738 字符｜已有图示／数值已给出**

随附数据：data/plot_data.json

```text
## 1. Figure goal
Create a focused analysis scatterplot relating bandwidth-dependent readout transfer deficits to changes in supervised corrective-gradient orientation. It belongs to the MarmAudio analysis of the AKR paper. It is not a causal mediation test, a training curve, or evidence that the regressor predicts held-out gradients accurately.

## 2. Canvas and hierarchy
Use a 1:1 square white canvas with one large plotting area. Reserve slightly more space at the left and bottom for readable axis labels, and keep a small inner margin around annotations. No overall title, decorative frequency waves, animal silhouettes, shaded quadrant names, or enlarged correlation badge.

## 3. Exact observations
Use data/plot_data.json, section geometry, together with its referenced alignment summary. Each pair below is x,y, where y is in percentage points: LP1=(0.4853696919,42.30769231), LP2=(0.3747367518,25.27472527), LP4=(0.2766326155,23.07692308), LP6=(0.1751379158,5.31135531), LP8=(0.0213397550,-0.54945055). These correspond to cutoffs 1,2,4,6,8 kHz. There are exactly five observations. Source-reported descriptive statistics are Spearman rho=1.00 and Pearson r=0.969. Use the higher-precision arrays for plotting and round only the text labels. Do not duplicate points, jitter them, or turn them into a larger apparent sample.

## 4. Axis definitions
The horizontal coordinate is one minus the mean cosine similarity between a cutoff's corrective gradients and the full-input reference. Label it 'Corrective-target change (1 - cosine)'. It is not an angle in degrees, Euclidean displacement, or gradient magnitude. The vertical coordinate is condition-fitted readout accuracy minus unchanged full-trained readout accuracy on the same filtered queries; label it 'Readout transfer deficit (pp)'. Set x approximately from -0.02 to 0.54 and y from -5 to 48. Preserve the negative LP8 deficit. A thin horizontal zero reference helps interpret that point. The x axis does not represent cutoff or model depth.

## 5. Mark and annotation grammar
Use five equal-size solid markers in one muted accent or neutral dark tone. Label each point directly by cutoff: '1 kHz', '2 kHz', '4 kHz', '6 kHz', '8 kHz'. Do not encode importance by marker area because the five conditions are not frequency-weighted observations. Place the 2- and 4-kHz labels on different sides to prevent collisions. Keep the LP8 label clear of the axes. Use small, light leader segments only when label displacement is necessary. Prefer no connecting line: the observed ordering is visible from the labels and coordinates. In particular, do not add a fitted trend, smooth curve, confidence ellipse, or shaded regression ribbon.

## 6. Typography and visual balance
Use one clean sans-serif family. At a final square width of about 3.5 inches, keep axes and cutoff labels at least comfortably readable, roughly 8-9 pt. Use a slightly heavier stroke for observations than for axes; remove top and right spines if doing so improves clarity. A few faint horizontal guides are enough. Leave open white space above the low-deficit points and around the high-deficit point rather than filling it with interpretation text. Use consistent minus signs and decimal formatting. If rho is shown inside the chart, use one small line '5 cutoff conditions; rho = 1.00', never a celebratory badge.

## 7. Evidence and caption boundary
The targets are correct-label gradients, so their structure is supervised and potentially label dependent. Both axes vary with the same bandwidth manipulation. Their association can motivate studying recording-appropriate corrections, but cannot establish that gradient rotation causes decoder errors. These are not individual recordings, five independent training seeds, or a universal law. Keep that explanation in an external caption, not printed across the plot. Do not rename the horizontal variable 'acoustic semantic drift' or claim a measured physical movement of the decoder's decision boundary. Low projection error and successful acoustic prediction are separate questions not answered here.

## 8. Production and validation
Export figures/appendix_rotation.pdf, an editable SVG, and a square high-resolution PNG. Use plotting code to place the numerical marks exactly and keep the full 1:1 export frame. Recheck the five cutoff labels against their coordinates, ensure the LP8 value remains below zero, and verify that any shown correlations are descriptive source statistics rather than a newly claimed significance test. Put the file source and methodological detail in the caption or companion notes. Do not compute new model outputs, synthesize gradient vectors, hide an inconvenient point, or modify the data to strengthen a trend.
```


---

## 06｜双向频谱迁移

**画幅：16:9｜完整 prompt：4,845 字符｜已有图示／数值已给出**

随附数据：additional_analysis/source_data.json、additional_analysis/bidirectional_transfer.csv

```text
## 1. Figure goal
Show that a fitted acoustic readout can lose accuracy when its input bandwidth changes in either direction. The important comparison is not simply full-to-low degradation: a low-bandwidth readout can also perform poorly on fuller-bandwidth features. This figure concerns frozen-feature classifiers on Dogs and Watkins, not continuous AKR recognition, and should keep the fixed-readout interpretation immediately visible.

## 2. Canvas and hierarchy
Use a 16:9 landscape white canvas with a four-row horizontal paired-point chart. Reserve about 25% of the width for row labels, 64% for the shared accuracy axis, and the remainder for breathing room. Separate the two datasets with a larger inter-row gap rather than a colored background block. Keep a small legend below the axes and no global title or oversized drop-percentage badge. Muted colors may help distinguish paired endpoint roles, but shapes must encode those roles in grayscale. No icons, logos, perspective, gradients, or decorative waveform strips.

## 3. Exact paired scores
Use additional_analysis/source_data.json, section transfer, and bidirectional_transfer.csv. The four rows in this fixed order are: Dogs, Full to LP1: 92.81% to 40.29%; Dogs, LP1 to Full: 87.77% to 38.85%; Watkins, Full to LP1: 88.20% to 9.73%; Watkins, LP1 to Full: 74.04% to 20.35%. More precise values can be calculated from saved correct-count matrices divided by 139 for Dogs or 339 for Watkins. These same query counts apply to both endpoints of each dataset row. Do not append the shared-hyperparameter control as an unlabelled fifth observation or substitute its scores for these source-selected results.

## 4. What each row means
The first endpoint is M[s,s], the source-trained readout evaluated on source-condition test features. The second endpoint is M[s,t], that same unchanged readout evaluated on target-condition features. Thus the connected pair has one classifier and two audio conditions. This is distinct from a target-referenced deficit M[t,t]-M[s,t]. Print concise row labels 'Dogs: Full -> LP1', 'Dogs: LP1 -> Full', 'Watkins: Full -> LP1', and 'Watkins: LP1 -> Full'. Full is unfiltered model-observable baseband; LP1 means a 1-kHz low-pass filter. More bandwidth is not a new species label or a new independent test set.

## 5. Marks and numerical placement
Use one common 0-100% horizontal axis. Draw a neutral thin horizontal segment joining the two scores in each row. Place a filled circle at the source-matched score and an x-shaped marker at the changed-bandwidth score; preserve these roles in all four rows even though the changed score lies to the left. Label both endpoints to two decimals, offset above or below the line so the mark remains visible. Use sparse vertical grid lines at 0,25,50,75,100. Do not reverse the direction text to match spatial ordering. Leftward loss is encoded by the coordinates, not by an extra thick arrow or a red downvote symbol.

## 6. Typography and visual emphasis
Use short method-role names in the legend: 'Source-matched input' and 'Changed-bandwidth input'. Axis text is 'Accuracy (%)'. Keep row text left aligned, numerical annotations tabular, and category spacing even. Type should remain readable at a 5.5-inch figure width, without a microscopic footnote. Emphasize the reciprocal pair structure by giving both directions equal visual weight, not by making the Full-to-LP1 rows brighter. Move explanatory sentences into the external caption. A single small annotation 'Readout fixed within each row' is enough if the geometry alone needs clarification; do not add a large panel heading for each line.

## 7. Scope and interpretation
Readout layer and ridge regularization were selected on source validation; coefficients were fitted at the source condition and transferred unchanged. These are not zero-shot versus supervised gaps and not a direct comparison of biological task difficulty. A low-to-full drop shows that an unchanged readout may not exploit the fuller input representation, not that adding frequencies destroys all recoverable information. The separate target-trained probe can still work well. Do not imply that this figure proves the cause of native language-model collapse or predicts how AKR will transfer.

## 8. Export and validation
Export figures/analysis/02_bidirectional_readout_transfer.pdf, an editable SVG, and a full 16:9 PNG preview. Read numeric values from the copied result files, with rounding applied only at annotation time. Validate every row's source and target labels and its sample count. Keep the source-matched baseline definition in the external caption so this panel cannot be mistaken for the transfer-deficit heatmaps elsewhere in the paper. Do not add fitted trends, uncertainty not present in the sources, new bandwidth conditions, or fresh model evaluations.
```


---

## 07｜Dogs 频谱迁移差距矩阵

**画幅：1:1｜完整 prompt：4,891 字符｜已有图示／数值已给出**

随附数据：additional_analysis/source_data.json

```text
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
```


---

## 08｜Watkins 频谱迁移差距矩阵

**画幅：1:1｜完整 prompt：4,887 字符｜已有图示／数值已给出**

随附数据：additional_analysis/source_data.json

```text
## 1. Figure goal
Create the Watkins source-to-target bandwidth transfer matrix for the AKR paper. Display the deficit of a transferred frozen-feature readout relative to a readout fitted at the target bandwidth. The matrix should reveal condition-specific compatibility, including negative differences, rather than merely reproduce a table of high and low accuracies. This is a readout diagnostic, not an AKR benchmark.

## 2. Canvas and layout
Use a 1:1 square white canvas. The central object is a 6 by 6 square matrix, with generous left and bottom margins for source and target labels. Place one slim vertical color scale on the right, separate from the matrix by a small white gutter. Keep cells genuinely square; do not stretch them to fill a wide page. Use no global title, subtitle banner, decorative animal image, gradient background, or heavy table frame.

## 3. Locked counts and computation
Read additional_analysis/source_data.json, section transfer.Watkins. The denominator is N=339 paired test recordings. Both row and column order is [Full, LP1, LP2, LP4, LP6, LP8]. The saved correct-count matrix C, with rows as readout-training condition and columns as target condition, is:
[299, 33, 82, 241, 280, 294]
[69, 251, 165, 83, 75, 68]
[148, 149, 264, 182, 164, 137]
[231, 38, 133, 291, 278, 216]
[274, 33, 118, 288, 294, 261]
[286, 32, 86, 216, 254, 285]
Compute raw accuracy M[s,t]=100*C[s,t]/N. Plot D[s,t]=M[t,t]-M[s,t] in percentage points, not M itself and not M[s,s]-M[s,t]. Use exact counts for computation. Round only final cell annotations to one decimal. The diagonal is zero by definition; it is not proof of perfect recognition.

## 4. Matrix encoding
Use a restrained diverging scale with a neutral midpoint at zero and the same limits [-10,70] pp for the Dogs and Watkins deficit matrices. Positive cells indicate worse performance than the target-fitted readout; negative cells indicate better observed transfer performance. Never clip negative cells to zero. Print every cell's signed value, with plain 0.0 on the diagonal, and label the color scale 'Deficit (pp)'. Use fine white separators rather than thick black grid lines. Switch annotation tone for contrast where needed; color alone must not carry the numerical result. Avoid a second inset heatmap or clustered dendrogram.

## 5. Orientation and labels
Put 'Readout training bandwidth' on the vertical axis and 'Evaluation bandwidth' on the horizontal axis. Repeat the exact six short condition labels on both axes. Full means the unfiltered model-observable baseband; LP1/2/4/6/8 denote low-pass cutoffs in kHz. Full must not be merged with LP8. Keep source labels readable from top to bottom in the stated order, and target labels left to right. Do not reorder rows to create visually cleaner clusters or to maximize a diagonal pattern. Each column uses the same target recordings and reference M[t,t], making a within-column comparison interpretable.

## 6. Specific checks and emphasis
For Full -> LP1, verify 74.04 - 9.73 = 64.31 pp before rounding the cell. For LP1 -> Full, verify 88.20 - 20.35 = 67.85 pp. Keep the LP8 column separate from Full even when some values are numerically close. Keep all measured off-diagonal cells, including unexpected or small negative deficits. Use a regular-weight sans-serif family, compact axis labels, and tabular numerals. A small N=339 note may sit near the legend; task descriptions and selection procedures belong in the external caption.

## 7. Protocol and interpretation
BW-SP denotes BEANS Watkins recognition across 31 marine-mammal species. The same 339 test recordings underlie every condition pair. This matrix is not evidence of completed 31-species continuous-AKR repair. Source validation selects the feature layer and ridge regularization; each source-trained readout is then applied unchanged to the paired targets. This target-reference display differs from the source-matched endpoint chart: the baseline changes across columns, not within a column. Do not interpret color intensity as mutual information, a distance between neural boundaries, a causal spectral mechanism, or a cross-family model comparison. No synthetic off-diagonal values may be reconstructed from a single full-input transfer curve.

## 8. Delivery and validation
Export the specified PDF, an editable SVG, and a square high-resolution PNG. Keep a small machine-readable table of D next to the plotting script. Confirm a zero diagonal, preserved signs, correct denominator, and the exact source/target orientation. Preserve the 1:1 page frame on export; accommodate labels within it rather than letting automatic cropping change the ratio. Use the supplied counts without fitting any model or generating additional predictions. Inspect all axis text and cell annotations at final size, and ensure the shared color-scale limits match the companion dataset matrix.
```


---

## 09｜读出变化与迁移差距平面

**画幅：1:1｜完整 prompt：4,914 字符｜已有图示／数值已给出**

随附数据：additional_analysis/readout_phase.csv、data/plot_data.json

```text
## 1. Figure goal
Make one diagnostic plane for the AKR paper that separates two observable performance changes: how well a condition-fitted readout performs compared with a full-input readout, and how much additional performance is lost by transferring the full-input readout unchanged. Plot all three animal-recognition datasets and five bandwidth conditions. The coordinates are arithmetic differences in accuracy, not inferred information loss or physical movement of a neural boundary.

## 2. Canvas and composition
Use a 1:1 square white canvas with one large Cartesian plot. Leave about 15% of width for the left-axis title and sufficient bottom margin for a two-line horizontal title. Place a compact three-dataset legend in an unused corner, outside the dense point cluster. Use no overall title, oversized equation banner, explanatory quadrant names, or decorative acoustic drawings. The square must be comfortable at roughly 4 inches in print. Use restrained muted accents, thin reference lines, and visible marker differences that survive grayscale.

## 3. Exact coordinate definitions
Read additional_analysis/readout_phase.csv, checked against data/plot_data.json. For each dataset and cutoff c, compute X=M[Full,Full]-M[c,c] and Y=M[c,c]-M[Full,c], with M in percent. Here M[c,c] is the condition-fitted probe score, and M[Full,c] is the unchanged full-trained probe on condition-c queries. X+Y equals the full-matched minus transferred score; this is an algebraic identity, not a causal error decomposition. Plot the recorded fifteen rows only. Do not add a separate Full point, extrapolated cutoff, or model-size curve.

## 4. Numerical anchors
Cutoff order is [1,2,4,6,8] kHz. MA-CT full-matched reference is 95.42%; condition-fitted scores are [77.29,79.85,82.60,93.96,94.87], transfers [34.98,54.58,59.52,88.64,95.42]. BD-ID full reference is 92.81%; fitted [87.77,91.37,91.37,89.21,94.96], transfers [40.29,59.71,81.29,92.09,91.37]. BW-SP full reference is 88.20%; fitted [74.04,77.88,85.84,86.73,84.07], transfers [9.73,24.19,71.09,82.60,86.73]. Use the copied CSV's precision in calculations. Rounded LP1 anchors are MA-CT (18.13,42.31), BD-ID (5.04,47.48), and BW-SP (14.16,64.31). Preserve negative coordinates at other cutoffs.

## 5. Axes and marks
Label x 'Condition-fitted score change (pp)' and y 'Transfer deficit (pp)'. A suitable view is x from -5 to 22 and y from -7 to 72; adjust only to include all real marks and labels. Show thin horizontal and vertical zero lines plus a faint dotted y=x reference. This diagonal denotes equal numerical differences, not a performance threshold. Give MA-CT circles, BD-ID squares, BW-SP triangles. Use small solid points and light straight connectors within each dataset in cutoff order only. Do not join datasets, draw a regression line, smooth the path, or use arrows that imply training progress. Colors should identify datasets rather than encode success.

## 6. Text and visual emphasis
Label the 1-kHz and 2-kHz observations directly with short dataset/cutoff tags. Keep the remaining points identifiable through a small accompanying data table or light cutoff annotations where they fit. Do not place all three long dataset names beside every point. Use regular sans-serif text, tabular numerals, and a consistent hierarchy; avoid bold paragraph labels inside the axes. Keep the small negative-valued points visible near the origin by offsetting text rather than moving the data. The desired reading is that several narrow-band observations lie well above the y=x reference, without a giant 'readout failure' slogan.

## 7. Protocol and interpretation
MA-CT is MarmAudio call-type recognition, BD-ID is BEANS Dogs individual identification, and BW-SP is BEANS Watkins species recognition. Their evaluation protocols differ, so these fifteen points are not independent replications of one common task. Within each dataset, paired recordings define the differences. The plane can show that a transferred classifier leaves recoverable discrimination unused; it cannot quantify mutual information or establish that all native text errors have this cause. An increased condition-fitted score can make X negative. Keep it rather than clipping it to fit the story. No classification boundary is literally plotted.

## 8. Delivery and checks
Export figures/analysis/03_readout_change_vs_transfer_deficit.pdf, an editable SVG, and a square PNG. Save the fifteen plotted coordinates with dataset and cutoff labels. Recompute X and Y from the source precision, verify the three LP1 anchors, and check X+Y for each row. Keep the full 1:1 frame, including label margins. Inspect at final width for near-origin overlap, cropped negative ticks, and illegible dataset markers. No new model call, refitting, fake scatter samples, confidence region, or causal line belongs in this figure. Keep interpretation and protocol notes in the external caption.
```


---

## 10｜跨 prompt 修复与改错

**画幅：16:9｜完整 prompt：4,948 字符｜已有图示／数值已给出**

随附数据：additional_analysis/source_data.json、additional_analysis/cross_prompt_transitions.csv

```text
## 1. Figure goal
Explain how a positive accuracy gain is composed of corrected errors and damaged previously correct answers. Use the complete Dogs A-J cross-prompt validation results for the class-routed pooled variant. This is an outcome-accounting figure, not evidence that continuous AKR is prompt invariant. The two query prompts are paraphrase and reversed candidate order.

## 2. Canvas and visual hierarchy
Use a 16:9 white landscape canvas with a clean two-row diverging count chart. Reserve a left label column and a small right column for net gains. The central vertical zero line divides damaged answers on the left from corrected errors on the right. Make row spacing generous, and place the legend below rather than over the marks. No global title, colored background stripe, Sankey ribbon, trophy icon, or oversized success badge. Use one muted accent for correction and a neutral contrasting tone for damage; hatch or outline the damage bars so grayscale remains clear.

## 3. Locked counts
Read additional_analysis/source_data.json, cross_prompt, and cross_prompt_transitions.csv. Each prompt evaluates the same 139 validation recordings. Paraphrase: 33 native-wrong/repair-correct, 8 native-correct/repair-wrong, 10 both correct, 88 both wrong. Reversed order: 22 native-wrong/repair-correct, 11 native-correct/repair-wrong, 4 both correct, 102 both wrong. Thus net gains are (33-8)/139*100=17.99 pp and (22-11)/139*100=7.91 pp. The four outcome counts sum to 139 for each prompt. Never aggregate the two prompts into 278 independent recordings.

## 4. Chart construction
Draw the damage bars leftward to -8 and -11, and correction bars rightward to 33 and 22. Negative x is only a visual convention for a nonnegative count. Label the axis 'Paired answer changes (recordings)', with damage-side tick labels displayed as magnitudes if doing so avoids confusion. Use a common extent approximately -15 to 38 for both rows. Put count labels at the free bar ends, positioned outside the fill. Print net gains in the separate right column as '+25/139; +17.99 pp' and '+11/139; +7.91 pp'. Keep this column aligned so it does not appear to be another data series.

## 5. Context and reference scores
The paraphrase native accuracy is 12.95% and the corrected accuracy 30.94%; reversed-order scores are 10.79% and 18.71%. These roundings agree with 18 to 43 correct and 15 to 26 correct out of 139. The bar counts, not a rounded percentage subtraction, define the gain. The same A-J support gradients and classifier router are reused; only the query prompt changes, with relative alpha=.03. Preserve 'class-routed pooled' in the external caption. This variant selects a correction using a class prediction and is not the continuous acoustic-to-gradient regressor of the main method.

## 6. Typography and restrained emphasis
Use one regular sans-serif family, tabular count numerals, and short row labels 'Paraphrase' and 'Reversed order'. Labels should remain readable at a 5.5-inch composite width. Use small directional descriptors 'Damaged' and 'Corrected' near the zero-centered chart, not a sentence inside each bar. Keep the neutral zero reference slightly stronger than the sparse vertical grid. The visual message should emerge from the relative lengths: reversal both corrects fewer errors and damages more correct answers. Do not conceal damage in a footnote or enlarge corrected bars nonlinearly. Leave both-correct and both-wrong counts to the caption or companion table instead of overcrowding the bars.

## 7. Evidence and interpretation
The chart measures paired decision changes; it does not prove that the model has acquired new animal reasoning skills. The same recordings and router appear under both prompts, so neither the two rows nor their count ratios establish universal prompt robustness. Existing paired gain intervals may be reported in the manuscript, but do not draw error bars around integer transition counts without a separately specified estimator. Do not reuse this design with continuous-AKR numbers unless the underlying paired predictions actually belong to that method. Keep invalid outputs counted as wrong, consistent with the source evaluation. Do not remove any such case to improve the plotted ratio.

## 8. Export and final checks
Export figures/analysis/04_cross_prompt_repairs_and_harms.pdf, an editable SVG, and a 16:9 PNG preview. Retain the underlying count CSV. Verify both four-cell partitions sum to 139, gains equal repaired minus damaged over N, and the reference is native generation rather than fixed KV. Inspect labels at print size, making sure negative-axis display is not mistaken for negative sample counts. Keep external caption text separate from the image. Render from saved results only, with no new inference, invented uncertainty, interpolated prompt, or method-name substitution. The finished chart should make both benefits and collateral errors immediately visible.
```


---

## 11｜完整首划分类别混淆变化

**画幅：1:1｜完整 prompt：4,960 字符｜已有图示／数值已给出**

随附数据：data/first_draw_predictions.csv、data/first_draw_provenance.json

```text
## 1. Figure goal
Draw a class-resolved difference matrix showing how continuous AKR changes predictions relative to a fixed-mean K/V correction in the first indexed historical MarmAudio draw. Use every one of its 39 query events. The goal is to identify selective corrections and newly introduced errors, rather than imply that every call type improves or that this single draw represents the five-draw mean.

## 2. Canvas and composition
Use a 1:1 square white canvas. Fit a six-row by seven-column matrix into its central area, with sufficient left margin for full class labels and counts. Keep a narrow vertical color scale on the right. The cells may be slightly rectangular to preserve legibility within the square outer canvas. Use no overall title, animal portraits, decorative arrows, successful-class badges, or large text paragraphs. The displayed data should dominate, with a thin diagonal-cell outline helping readers locate class recall changes.

## 3. Exact data and identity
Read data/first_draw_predictions.csv and data/first_draw_provenance.json. The seed is 20250813, N=39, support is twenty total error-enriched recordings, rank=4, feature index=0, and raw eta=300. The comparison is conditional/continuous AKR minus fixed_mean, not AKR minus native. True-label row order is [Infant Cry,Phee,Seep,Trill,Tsik,Twitter], with counts [9,5,7,4,4,10]. Prediction columns use the same six labels plus Invalid. Keep all rows, including classes with no correct prediction from either method, and retain the Invalid column even if all its counts are zero.

## 4. Computation and validation anchors
For each method, count predictions into a confusion matrix C. Normalize every row by that true class's query count, then plot Delta C=100*(C_AKR-C_fixed), where both matrices are row-normalized proportions. Each row of Delta C sums to zero up to floating-point tolerance. Recorded fixed versus AKR correct counts are: Infant Cry 0 versus 0; Phee 0 versus 5; Seep 0 versus 1; Trill 0 versus 0; Tsik 2 versus 1; Twitter 0 versus 5. Consequently diagonal changes are [0,+100,+14.29,0,-25,+50] pp. Compute every off-diagonal cell from the full saved predictions; never infer it from the diagonal totals.

## 5. Visual encoding
Use a symmetric diverging scale from -100 to +100 pp with a light neutral midpoint. Positive diagonal changes indicate higher recall; a positive off-diagonal change indicates a more frequent error, so the positive color must not be named 'good'. Annotate each cell to one decimal, using a plus sign for positive changes and a clear zero for unchanged cells. Change annotation tone only for contrast. Use very thin cell dividers and a quiet border around the six diagonal cells. Keep the color scale labelled 'Change in row-normalized predictions (pp)'. No per-row color normalization is allowed; a +25 cell must have the same intensity anywhere in the matrix.

## 6. Labels and layout polish
Use 'True call type' vertically and 'Predicted label' horizontally. Put n values beside the row names, not in tiny superscripts. Wrap 'Infant Cry' over two lines on the x axis, and rotate labels only as much as necessary to avoid collision. Use regular sans-serif text, with cell numbers at a legible size for a roughly 4.5-inch square. Maintain balanced whitespace around the axis names and color bar. Place the protocol identity and comparator in the external caption. Do not clutter the image with the entire method equation or another bar chart. The heatmap and class counts already provide the analysis.

## 7. Scientific interpretation
All 39 events belong to the first indexed draw, selected by index rather than by visual attractiveness. Relative to fixed mean, twelve previously wrong predictions become correct and two previously correct predictions become wrong. Total correct counts change from 2 to 12; this does not mean only ten predictions changed. Phee and Twitter account for ten of the twelve AKR-correct predictions. Infant Cry and Trill remain unresolved, while Tsik recall decreases. Preserve this uneven pattern. Do not relabel the reference as native, pool duplicated events from other draws without a declared rule, or claim complete recognition of the repertoire. Output concentration and correctness are separate measurements.

## 8. Delivery and checks
Export figures/analysis/07_first_draw_confusion_change.pdf, an editable SVG, and a square PNG. Save the raw and normalized matrices beside the plot. Verify event IDs are paired, targets agree between methods, N equals 39, every row total is correct, and the diagonal matches the anchors above. Preserve Invalid in the denominator according to the supplied parser. Inspect all zero rows and negative diagonal cells, not just the attractive regions. Do not synthesize predictions or add smooth density backgrounds. The external caption should state that this is a complete single-draw case study and that the five-draw aggregate remains the main experiment.
```


---

## 12｜输出覆盖与识别质量

**画幅：1:1｜完整 prompt：4,925 字符｜已有图示／数值已给出**

随附数据：data/first_draw_predictions.csv、data/first_draw_provenance.json

```text
## 1. Figure goal
Make a small, precise diagnostic that separates a less concentrated output-label distribution from better recognition. Compare fixed-mean correction and continuous AKR on the same complete first MarmAudio draw. The figure should show that distributional change and macro-F1 improve together in this case, while avoiding the false claim that simply emitting more diverse labels proves correct acoustic reasoning.

## 2. Canvas and composition
Use a 1:1 square white canvas with one Cartesian plot and exactly two observations. Leave ample white space around the marks rather than adding decorative density, clusters, or a fictitious frontier. A compact legend or direct two-line labels should identify the methods. Use no global title, animal icon, gradient background, explanatory quadrant banner, or oversized numerical badge. Keep the plot readable at about 3.5 inches in print; a square single-panel design is preferable to compressing several unrelated metrics into a wide dashboard.

## 3. Locked source and counts
Use data/first_draw_predictions.csv and data/first_draw_provenance.json. This is seed 20250813, all 39 queries, with twenty total error-enriched support recordings, rank=4, feature index=0, raw eta=300. Prediction-label order is [Infant Cry,Phee,Seep,Trill,Tsik,Twitter]. Fixed mean emits [0,11,1,0,24,3]; continuous AKR emits [0,10,5,0,13,11]. Both invalid counts are zero and both methods emit four distinct labels. Their correct counts are 2 and 12. The reference is fixed-mean correction, not native generation; this is not the mean of all five recording-group draws.

## 4. Exact plotted coordinates
For valid predictions, form p_c=count_c/sum_c count_c and calculate N_eff=exp(-sum_{c:p_c>0} p_c*log(p_c)) using natural logarithms. Plot fixed mean at x=2.577959, y=2.380952 and continuous AKR at x=3.801913, y=23.786181, where y is macro-F1 in percent. Compute from source precision when available; these values are display anchors. Accuracy annotations are 5.128205% and 30.769231%, respectively, rounded to 5.13% and 30.77% only in text. Do not place accuracy on the y axis or replace N_eff with the count of distinct labels, which stays at four for both methods.

## 5. Axes and marks
Set the horizontal axis from 1 to 6, the six-label maximum, and label it 'Effective emitted labels'. Use a 0-35% vertical axis labelled 'Macro-F1 (%)'. Give Fixed mean a circle and Continuous AKR a square, using a neutral tone for the former and one muted accent for the latter. A thin dashed segment may connect the two measured points. It is a comparison connector, not a trajectory over alpha, an optimization path, or a fitted trade-off. Keep marker sizes equal. Light horizontal guides are sufficient. Do not add extra intermediate points, spline interpolation, or arrows suggesting guaranteed improvement.

## 6. Text and visual emphasis
Direct labels should read 'Fixed mean' and 'Continuous AKR', with one smaller line containing 'Acc. 5.13%' and 'Acc. 30.77%'. Place these labels with generous offsets so they cannot be mistaken for axis coordinates. A compact note '4 distinct labels; 0 invalid, both methods' can appear below the plot or in the caption, not as a large callout. Use one sans-serif family, consistent number formatting, and a typography hierarchy that survives reduction to paper size. The whitespace is intentional: two measured outcomes should not be disguised as a dense empirical relationship.

## 7. Interpretation boundary
A larger effective label count means predictions are distributed less unevenly, not necessarily that they match ground truth better. Macro-F1 provides the separate correctness measurement. In this draw, two true call types still have zero recall despite increased entropy. The finding is a selective redistribution of decisions, not full repertoire recognition or general downstream reasoning recovery. Keep the single-draw scope explicit and do not combine this count distribution with Dogs arbitrary-letter predictions. If a future dataset has invalid outputs, record their rate separately rather than silently discarding them from accuracy while computing entropy on valid predictions.

## 8. Delivery and validation
Export figures/analysis/08_first_draw_coverage.pdf, an editable SVG, and a square PNG. Save the two coordinate rows and the underlying label counts. Verify each count vector sums to 39, both distinct-label counts equal four, both invalid rates are zero, and exponentiating entropy yields the displayed N_eff values. Recompute accuracy from 2/39 and 12/39. Preserve method labels and do not rename this a native-versus-AKR comparison. No error bars or confidence ellipses are justified by the supplied two summary points. Use only saved predictions; do not generate samples, refit a model, or infer a continuous strength curve. Inspect the final-size output for label collisions and keep its external caption separate.
```


---

## 13｜模型规模与频谱交叉

**画幅：1:1｜完整 prompt：4,864 字符｜已有图示／数值已给出**

随附数据：additional_analysis/source_data.json

```text
## 1. Figure goal
Show that the native recognition advantage of Qwen2.5-Omni-7B over 3B depends on input bandwidth. The measured advantage is positive on full observable input but negative at the strongest 1-kHz filtering condition. This is a model-size reference in the AKR paper, not a comparison of AKR-adapted checkpoints or a cross-family generalization result.

## 2. Canvas and visual hierarchy
Use a 1:1 square white canvas with one uncluttered point-and-line chart. Reserve room beneath the x axis for short condition labels and use a balanced left margin for a readable difference-axis title. No overall title, large up/down arrows, brand logos, animal portraits, shaded success region, or dramatic red/green background. Choose one restrained line tone and equal-size markers. The zero reference should be visible but less dominant than the data. The figure must remain understandable in grayscale and at roughly 3.5 inches in print.

## 3. Locked observations
Read additional_analysis/source_data.json, section size_crossover. In fixed condition order [LP1,LP2,LP4,LP6,LP8,Full], the paired accuracy differences 7B minus 3B are [-4.58,+0.55,+3.48,+7.88,+8.42,+9.16] percentage points. There are six condition summaries, not six independent datasets. LP1 native accuracies are 17.22% for 3B and 12.64% for 7B. Full native accuracies are 20.33% and 29.49%, respectively. The differences follow the saved sweep's displayed precision. Preserve its -4.58 pp LP1 difference rather than silently recomputing a conflicting rounded subtraction from the two rounded accuracies.

## 4. Axes and condition identity
Use a categorical x axis for LP1,LP2,LP4,LP6,LP8,Full. The Full endpoint is a distinct unfiltered condition, not a numerical 8- or 9-kHz cutoff. Define LP1/2/4/6/8 as low-pass cutoffs in kHz in the caption. Label x 'Input condition' and y 'Native accuracy: 7B - 3B (pp)'. Choose a vertical range approximately -7 to 12, with a clear horizontal zero line and ticks that include negative and positive values. Do not truncate the plot at zero, replace the negative point by its magnitude, or assign Full the same position as LP8.

## 5. Marks and numerical emphasis
Plot one marker per measured condition and connect neighboring conditions with thin straight line segments. The line organizes observations; it is not an interpolated scaling law. Label every point with its signed difference to two decimals. Offset the small +0.55 label above the zero reference so it remains legible. Use equal marker area for all conditions; do not enlarge the negative result or the full-input gain. A slight extra gap before Full may distinguish the unfiltered reference, but retain explicit categorical tick labels. Avoid confidence bands, fitted monotonic curves, or separate accuracy bars that would crowd the figure and mix two y-axis meanings.

## 6. Typography and layout detail
Use a clean sans-serif family, regular-weight axis text, and tabular numerals. Keep the y-axis title on two lines if needed rather than shrinking it. Place the six signed annotations close to their marks without touching the line. Sparse horizontal guides can help compare magnitudes; avoid full dark grids. Do not fill the empty lower-right area with a methods paragraph. A compact external caption can state the two endpoint accuracies and describe paired evaluation. Small panel lettering is optional when this chart is assembled with another figure, but no new 'bigger is worse' title should be introduced.

## 7. Scope and interpretation
These are native Qwen2.5-Omni 3B and 7B evaluations on paired MarmAudio recordings, not the new balanced-support F3 experiment. No correction regressor is fitted for this plot. The reversal supports a condition-dependent model-size advantage, not a claim that smaller models are better in general. It also does not show that AKR outperforms increasing model size. The use of six bandwidth conditions on the same source recordings does not create independent seeds. Full means the model-observable baseband after the declared processing path, not access to all frequencies in the original high-rate recordings. Preserve these distinctions in the caption.

## 8. Export and final verification
Export figures/analysis/05_model_size_bandwidth_interaction.pdf, an editable SVG, and a square PNG preview. Use the saved difference array and retain the source precision note. Verify order, signs, units, and the distinct Full/LP8 labels. Make sure the line crosses zero between LP1 and LP2 without adding a measured crossing point or claiming its exact cutoff. Check final-size legibility of the negative sign and decimal values. Do not draw uncertainty unless a matching source estimator is explicitly supplied; do not replace source values with a new model run. Keep the 1:1 outer frame in all exports and the figure caption outside the image.
```


---

## 14｜回归秩与特征层交互

**画幅：1:1｜完整 prompt：4,898 字符｜已有图示／数值已给出**

随附数据：additional_analysis/source_data.json

```text
## 1. Figure goal
Visualize the interaction between the acoustic feature supplied to the correction regressor and the number of retained correction coordinates. Use the existing Dogs development grid. Feature depth and correction rank are different design variables; the figure must not mislabel the feature index as the K/V intervention layer. This is sensitivity analysis of recorded configurations, not a new test-set hyperparameter search.

## 2. Canvas and composition
Use one 1:1 square white canvas with a single rank-versus-macro-F1 chart. Keep margins large enough for the logarithmic rank labels and a compact four-series legend. Use the plot area as the main visual object. No overall title, separate architecture drawing, colorful grid background, or large best-model badge. Give all four feature curves equal visual treatment, preserving grayscale readability through distinct marker shapes and line styles. Use restrained accents rather than four highly saturated competing colors.

## 3. Locked grid
Read additional_analysis/source_data.json, section rank_layer. There are N=64 development queries and twenty total support examples. Ranks are [1,2,4,8,16], and feature indices are [0,8,16,28]. The source macro-F1 arrays are proportions; convert to percent once. Rounded percentages in rank order are: feature 0 [1.18,1.33,4.19,7.80,7.94]; feature 8 [1.25,1.30,1.25,10.22,9.84]; feature 16 [0.71,0.73,3.95,8.36,9.90]; feature 28 [0.71,0.74,3.58,10.32,10.09]. Plot the higher-precision JSON values and preserve every small increase and decrease.

## 4. Axis and mark grammar
Use a log2 horizontal axis with ticks only at 1,2,4,8,16, labelled 'Correction rank'. Use a 0-12% vertical axis labelled 'Macro-F1 (%)'. Assign feature 0 circles/solid, feature 8 squares/dashed, feature 16 triangles/dash-dot, and feature 28 diamonds/dotted. Keep marker size and stroke weight consistent. Connect only observed settings using straight segments. Do not smooth the abrupt rise between ranks four and eight or force curves to plateau. Rank one values near zero remain visible; do not use a truncated y axis to magnify only the best scores. No dual axis for accuracy is needed.

## 5. Feature identity and selection
The legend labels are 'Feature 0', 'Feature 8', 'Feature 16', and 'Feature 28'. These indices identify stored acoustic features, not four different backbones or intervention locations. Source selection prioritizes macro-F1, then accuracy, then smaller rank and earlier feature layer. Feature 28/rank 8 is the recorded selected point, with macro-F1 about 10.32%; feature 8/rank 8 has about 10.22%, even though its accuracy is higher. A small open ring may identify the recorded selection, provided it is explicitly labelled as historical selection and not as a new optimum derived for this graphic.

## 6. Typography and visual emphasis
Use one clean sans-serif family, regular-weight ticks, and tabular decimal annotations. Label at most the recorded selected point and the rightmost endpoints when space permits. A table already contains all values; the chart should show interactions rather than repeat every number. Keep a small n=64 note and total support=20 in the external caption. Put the legend in a region that does not cover the low-rank lines or rank-eight peak cluster, using two short columns if necessary. Design for a square width of about 4 inches. Reduce decorative elements before reducing type below a readable paper-scale size.

## 7. Interpretation boundary
The grid shows recognition performance after choosing a correction-feature pair and rank. It does not measure the SVD energy captured, a held-out gradient regression error, or the number of biologically meaningful call dimensions. It cannot establish that rank eight is universally optimal or that late features always work better. Increasing rank to sixteen slightly reduces macro-F1 for some features and improves it for others; preserve that heterogeneity. This 64-query development grid is distinct from the 139-query Dogs relative-norm study and the MarmAudio five-draw result. Do not combine their values or relabel these rows as locked confirmation results.

## 8. Production and checks
Export figures/analysis/06_rank_feature_depth.pdf, an editable SVG, and a square PNG. Keep all twenty source observations with their rank and feature labels in a companion data table. Verify the percent conversion, rank order, and exact selection identity before annotation. Check that near-overlapping rank-one points remain distinguishable by marker edge and line style without artificial jitter. Do not invent error bars, add unseen ranks, reselect the best layer on the test set, or run new model inference. Preserve the complete 1:1 frame and keep the scientific caption outside the artwork. The result should make both target dimension and input-feature dependence clear from the four observed curves.
```


---

## 15｜修正强度与输出稳定性

**画幅：1:1｜完整 prompt：4,900 字符｜已有图示／数值已给出**

随附数据：data/strength_stability.json

```text
## 1. Figure goal
Display the accuracy-versus-strength behavior of the recorded Dogs A-J class-routed correction variants. More detailed tokenwise updates do not necessarily remain useful as intervention strength increases. The figure should show the pooled, ordered, and token-permuted curves together, with one clearly identified invalid-output endpoint. Do not describe this as the main continuous-AKR strength sweep.

## 2. Canvas and hierarchy
Use a 1:1 square white canvas with one large accuracy plot. Reserve space below for a compact three-method legend and above the highest curve for breathing room, not a title banner. Use flat vector lines, small solid markers, and muted accents. No overall title, unstable-model caricature, warning triangle, colored background zone, or decorative waveform. At final paper size, the alpha ticks and legend must be comfortably readable. This is a scientific result figure, not a dashboard with many competing metric cards.

## 3. Locked observations
Read data/strength_stability.json. The three relative-alpha values are [.003,.01,.03]. In that order, class-routed pooled accuracy is [14.39,18.71,20.14]%; class-routed ordered tokenwise accuracy is [15.11,6.47,0.00]%; class-routed token-permuted accuracy is [7.91,4.32,0.72]%. Each setting evaluates all 139 validation queries with the registered A-J support/router. The ordered invalid rate is recorded as 24.46% at .003 and 99.28% at .03; the middle invalid-rate entry is missing in this snapshot. Never fill the missing entry with zero or interpolate a validity curve.

## 4. Axes and curves
Set a logarithmic x axis with exact ticks .003,.01,.03 and label it 'Relative correction strength (alpha)'. Set a 0-25% y axis labelled 'Accuracy (%)'. Pooled uses circles and a solid line, ordered squares and a dashed line, permuted triangles and a dotted line. Use straight connectors between the three observed values. Preserve the ordered endpoint exactly at zero and the permuted endpoint at 0.72%. Avoid logarithmic accuracy scales or a lower y limit above zero. The alpha values are a relative active-block norm parameter, not raw eta and not a universal total perturbation energy across different intervention scopes.

## 5. Invalid-output annotation
At the ordered .03 endpoint, place a short label '99.28% invalid' with a fine leader to the square at zero accuracy. Use a neutral outlined note rather than a dramatic red callout. The label reports a different metric, so do not place a marker at y=99.28 on the accuracy axis or introduce a second axis merely to display it. The other known validity endpoint, 24.46% at .003, can remain in the caption or accompanying table. State there that the middle value is unavailable. The observed zero accuracy must not be replaced by an assumed chance-level value.

## 6. Typography and visual emphasis
Use a single sans-serif family, regular-weight axis text, and short legend names 'Pooled', 'Ordered tokenwise', 'Token-permuted'. Keep the prefix 'Class-routed' in the external caption or a small legend header, never silently omit method identity. Place the legend away from the low-valued descending curves. Direct accuracy labels may be limited to the rightmost three endpoints, since all points are listed in the data. Line styles and shapes must distinguish the methods in grayscale. Keep labels from colliding at alpha=.003, where pooled and ordered scores are close; adjust text, not data positions.

## 7. Scientific interpretation
This figure describes stability within a fixed recorded sweep. It does not prove that all ordered or tokenwise corrections fail, nor that pooled correction is best at every possible strength. The rise of the pooled curve over three observed settings is not an extrapolation to larger alpha. Invalid outputs remain incorrect in the accuracy denominator. The curves are class-routed A-J validation results, distinct from continuous-field regression and from semantic-label Dogs scores. Preserve that identity in the caption. Do not reinterpret a single local ordering advantage as a stable method improvement or equate an invalid-output surge with lost acoustic information in the frozen representation.

## 8. Production and final checks
Export figures/analysis/09_strength_stability.pdf, an editable SVG, and a square PNG. Verify all nine accuracy entries and the two available validity endpoints against the source file; keep any null validity field missing. Ensure the alpha axis is logarithmic, the zero-accuracy point is visible, and the invalid note points to the correct method and strength. Preserve the square outer frame on export and inspect at final print width for legend collisions. Do not add fitted uncertainty bands, extra strength values, nonexistent model families, or new measurements. Keep the full explanation in the external caption rather than placing several paragraphs inside the artwork.
```


---

## F1｜可压缩性与可预测性

**画幅：16:9｜完整 prompt：4,797 字符｜待完整结果模板；不代表已跑完**

随附数据：templates/F1_predictability.json

```text
## 1. Figure goal and status
Design the F1 analysis of whether frozen acoustic features predict corrective gradients, beyond the fact that those gradients can be compressed. This is a pending-result template for the AKR paper. Use only complete exported F1 metrics; templates/F1_predictability.json currently contains null values. No numerical result, favorable slope, or significance claim may be invented to finish the design.

## 2. Canvas and composition
Use a 16:9 landscape white canvas containing two aligned charts, Full on the left and LP1 on the right. Keep a shared legend below, generous margins, and matched chart heights. Do not add a global title, training-loop illustration, model logo, decorative acoustic traces, or shaded success region. Use restrained accents and distinct marker/line styles. Panel tags (a) Full and (b) LP1 are sufficient.

## 3. Required inputs
Read the actual F1 RESULT.json rows and folds, using the template only as a schema. Required row fields are rank, predicted_nmse, projection_nmse, coefficient_nmse, shuffled_nmse, fixed_mean_nmse, mean_cosine, nonzero_cosine_n, and decomposition_residual. Retain actual support, recording-group, and seed counts from the export. The bounded MA-CT setting uses six call types and k=8, or 48 support recordings before cross-validation; do not substitute the old twenty-total error-enriched experiment. A one-seed result is one seed even if the procedure contains several grouped folds.

## 4. Measurement definitions
Each recording-group fold refits the feature normalizer, SVD basis, and ridge regressor using training support only. Evaluation uses held-out SUPPORT gradients, not gradients of formal confirmation/test queries. The denominator is the sum of fold-wise squared errors of the fold-trained fixed-mean predictor. Predicted NMSE uses the summed prediction squared error divided by this same denominator; projection and coefficient NMSE use their respective summed errors. Use exported ratio-of-sums values, not the average of per-fold ratios. Projection NMSE measures truncation error, coefficient NMSE measures acoustic-coordinate prediction error, and they sum to total predicted NMSE up to tolerance.

## 5. Plot design
Use a log2 x axis labelled 'Correction rank' with recorded ticks, expected to be 1,2,4,8 in this bounded design. Label y 'Held-out gradient NMSE'. In both panels show real-feature prediction with squares/solid, shuffled-feature prediction with triangles/dashed, and projection residual with circles/dotted. Add a thin horizontal line at fixed mean=1 only when the denominator is valid. Use one common y range determined by the complete observations, preserving values above one. Include coefficient error in the data table instead of overcrowding the chart with a fourth nearly redundant component curve. No smoothing, extrapolation, or forced decline with rank is allowed.

## 6. Typography and visual emphasis
Keep line widths consistent and legends short: 'Acoustic prediction', 'Shuffled pairing', 'Projection residual', 'Fixed mean'. Explain the normalized denominator and fold protocol in the external caption. Do not label an error curve as classification accuracy or compare its height with a percent-correct score. Retain the full observed rank range, including bad settings. When several seeds are provided, use the exported aggregation rule and actual per-seed data; do not turn grouped folds into independent seeds or fabricate across-seed whiskers from one run.

## 7. Interpretation and missing data
A low projection residual establishes compressibility, not successful prediction from sound. A low prediction error does not guarantee an improvement in discrete recognition. The real-versus-shuffled comparison tests the acoustic-to-target pairing under this protocol, not every possible semantic mechanism. If true-label centroid scores are supplied, keep them in an explicitly labelled diagnostic table, not as a deployable method curve; preserve any fallback count for absent training classes. If results or denominators are missing, return empty labelled axes marked 'Awaiting complete results' and a missing-field list. Empty space means unavailable data, never a score of zero.

## 8. Production and validation
After complete results are supplied, export figures/new/f1_predictability.pdf, editable SVG components, and a full 16:9 PNG. Check expected ranks, group separation, aggregation denominators, support identity, and the numerical error decomposition before plotting. Keep this task limited to rendering already exported metrics: do not train a model, generate gradients, select a more favorable seed, or fit the normalizer on all support before claiming held-out evaluation. Maintain label readability and place the detailed caption outside the image.
```


---

## F2｜连续 AKR 跨频谱迁移

**画幅：1:1｜完整 prompt：4,757 字符｜待完整结果模板；不代表已跑完**

随附数据：templates/F2_transfer.csv

```text
## 1. Figure goal and status
Prepare the F2 source-bandwidth transfer figure for continuous AKR. It asks whether a correction map fitted from one support bandwidth remains useful when applied to another query bandwidth. This is a pending-result design, not a completed experiment. templates/F2_transfer.csv contains a header only. Do not populate the figure using frozen-probe transfer scores or assume that cross-condition repair must succeed.

## 2. Canvas and layout
Use a 1:1 square white canvas centered on a 2 by 2 matrix. Put source support condition on the vertical axis and target query condition on the horizontal axis, each ordered Full then LP1. Leave room beside the matrix for a narrow signed color scale, and below it for a compact two-item numerical key. Use generous white margins and large readable cell annotations rather than a crowded multi-panel dashboard. No global title, diagonal success slogan, large model icon, gradient backdrop, or decorative waveform.

## 3. Required result identity
Use complete F2 per-cell metrics or aligned predictions. Within each seed, all four cells must use identical support identities and query identities. The bounded setting is MA-CT, semantic labels, eight support recordings per call type, or 48 total, with Full and LP1 audio. Each source condition fits its own normalizer, correction mean, basis, and ridge map. Those objects are transferred unchanged to target-condition features. All four cells share the recorded numeric rank, ridge penalty, and relative alpha fixed for that comparison. Preserve the actual run identity; do not call a newly declared setting an inherited historical lock unless the export supports that description.

## 4. Matrix values
The primary displayed quantity is continuous-AKR accuracy minus native accuracy on the SAME target queries, expressed in percentage points. Derive it from aligned results or use an equivalent complete exported gain_pp. Use rows/columns Full->Full, Full->LP1, LP1->Full, LP1->LP1 with source and target orientation maintained. Native accuracy is target dependent, not source dependent, so repeated native references are not independent observations. Each cell may show a large signed gain and a smaller line 'AKR: xx.xx%' if the raw accuracy is available. Macro-F1 and invalid rate should remain in an accompanying table rather than sharing the color scale.

## 5. Visual encoding
Use a restrained diverging scale centered at zero. Select symmetric limits from the complete observed gain range, state the limits, and do not saturate unfavorable cells to make the diagonal look stronger. Keep negative gains visible and annotate all four cells to the same precision. Use fine cell separators and a color scale labelled 'AKR - native (pp)'. The zero-centered palette represents signed gain, not a probability or mutual information. Do not draw a diagonal line implying perfect transfer. Full is unfiltered model-observable input; LP1 is a 1-kHz low-pass version, not a different dataset or a rescaled pitch.

## 6. Controls and aggregation
The accompanying numerical table should retain native, fixed_mean, continuous_pooled, shuffled_query_field, and ridge_direct if exported. The shuffled field must refer to the actual recorded control, not a newly synthesized permutation for the graphic. Do not silently discard a weak control or select only seeds that passed an old gate. With one seed, annotate point values and no across-seed standard deviations. With several compatible seeds, show the declared mean and retain per-seed values in the table; use actual counts and do not treat reused recordings as independent. If a method or cell is missing, report it as missing, not zero or native-equivalent.

## 7. Typography and interpretation
Use one clean sans-serif family, tabular numerals, and large enough type for a 3.5-4-inch square in print. Axis titles should read 'Support bandwidth' and 'Query bandwidth'. Keep just two annotation sizes inside cells: the gain and optional raw accuracy. Protocol details belong in an external caption. This compares correction-map applicability, unlike the six-by-six probe matrices.

## 8. Delivery and completion checks
With complete result exports, create figures/new/f2_correction_transfer.pdf, editable SVG, and a square PNG. Save the four plotted gains and their target-native references. Verify aligned IDs, identical per-seed numeric settings, full cell coverage, source/target labels, metric units, and invalid-output accounting. If results are unavailable, return a blank matrix explicitly marked 'Awaiting complete results' plus missing fields; never sketch an imagined positive diagonal or negative off-diagonal. This prompt is optional scope and does not authorize another experiment.
```


---

## F3｜7B 与 3B 规模比较

**画幅：16:9｜完整 prompt：4,826 字符｜待完整结果模板；不代表已跑完**

随附数据：templates/F3_scale.csv

```text
## 1. Figure goal and status
Prepare the F3 comparison of continuous AKR on Qwen2.5-Omni-3B and Qwen2.5-Omni-7B under matched support and query identities. This is within-family scale replication, not cross-family generalization. The current prompt package supplies no completed F3 values: templates/F3_scale.csv is an input schema. Do not fill it using the historical native-only bandwidth sweep or the old twenty-total-support correction result.

## 2. Canvas and composition
Use a 16:9 landscape white canvas with four horizontal result rows. Reserve a left label column and a right gain-annotation column around one shared accuracy axis. Rows, in order, are 3B/Full, 3B/LP1, 7B/Full, 7B/LP1. Add a slightly larger gap between model groups, not a shaded background block. Use no global title, checkpoint logo, decorative transformer stack, bar-chart perspective, or claim banner. The paired native-to-AKR comparison should be visually dominant, while two smaller reference markers retain fixed and ridge results.

## 3. Required data and protocol
Use complete matched-protocol F3 or compatible exported E9 predictions. The bounded MA-CT setup uses semantic labels, six call types, k=8 per class, 48 support recordings, Full and LP1, and the declared confirmation query IDs. Both model sizes must share those IDs and the intended numerical settings for each condition. In a source-locked run, 3B inherits the corresponding 7B settings; in an explicitly standalone run, state the common predeclared settings instead. Do not pretend a missing historical lock has been recovered. Preserve actual seed count, query count, model revision, and intervention scope from the result exports.

## 4. Model-specific fitting
The 3B model uses its own acoustic representations and support gradients to fit its own feature standardizer, correction mean, low-rank basis, and ridge regressor. It does not receive copied 7B K/V tensors or incompatible coefficient matrices. Relative feature/intervention depth follows the declared model-mapping policy, not a forced identical layer number. All backbone weights remain frozen.

## 5. Marks and axes
Use a common 0-100% horizontal accuracy axis. In each row, plot Native with a triangle, Fixed mean with a circle, Frozen ridge with an open diamond, and Continuous AKR with a square. Connect Native to AKR with a thin solid segment to show their paired difference; keep fixed and ridge as unconnected smaller reference markers. Place the signed AKR-minus-native gain in the right annotation column, computed on the same target queries. Do not shrink an unfavorable method mark or hide a negative connector direction. Use one compact four-method legend below the plot, with muted consistent colors and grayscale-distinct shapes.

## 6. Typography and statistical display
Use regular sans-serif labels and tabular numbers, aiming for readable 8-9 pt type at a 5.5-inch composite width. Keep row labels short; spell out the model family and conditions in the caption. Full means unfiltered observable baseband and LP1 means a 1-kHz low-pass version. For a single seed, show point estimates without invented across-seed whiskers. For multiple compatible completed seeds, preserve individual seed values in a sidecar table and use only an explicitly defined summary. Macro-F1, invalid rate, and paired corrected/damaged counts remain in that table, not on a second incompatible accuracy axis.

## 7. Missingness and interpretation
If either size lacks complete matching data, leave the corresponding row blank and identify the comparison as incomplete. Blank is not zero, no effect, or chance accuracy. Do not substitute an old 3B baseline measured on different queries, populate GLM or Qwen3-Omni positions from model specifications, or copy the main 32.47% historical result into a new cell. The new comparison tests whether the correction rule transfers across size after model-specific fitting. It need not reproduce the same magnitude or sign at every condition. A stronger ridge reference does not invalidate the numerical state-correction comparison but must remain visible.

## 8. Export and final checks
When complete results exist, export figures/new/f3_locked_scale_comparison.pdf, editable SVG, and a 16:9 PNG. Save all four model/condition rows with method scores, gains, counts, and protocol identifiers. Confirm common support/query IDs, the correct Full/LP1 ordering, per-model regressor fitting, and no confusion with native-only size analysis. If the new backbone table already presents exactly these values, retain this chart as an appendix visualization rather than adding an invented second conclusion. Without results, return an explicitly empty layout and missing-field list. Do not run inference, generate gradients, tune alpha, or claim completion from this design template.
```
