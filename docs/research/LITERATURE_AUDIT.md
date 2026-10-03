# Bounded closest-work check — 2026-10-03

This is a single-assistant retrospective comparison, not a full adversarial collision audit. The search/open records below distinguish full-method text from abstract/model-card inspection. No statement that no prior work exists is licensed by this file. Existing manuscript references remain source-authored; this check does not certify every old bibliographic field.

## Retrieval performed

Queries: `site.arxiv.org 2507.08799 KV Cache Steering`; `site.arxiv.org NatureLM audio 2025 bioacoustic BEANS Zero`; `site.huggingface.co Qwen2.5-Omni-7B model card Thinker`; `site.arxiv.org 2601.18904 2606.02615 speech in context`.

Primary records opened on 2026-10-03:

| Work | Primary source/version | Read depth | Consequence for AKR |
|---|---|---|---|
| KV Cache Steering | https://arxiv.org/html/2507.08799v2 , Sections 3.2--3.5 | full method passages | Uses contrastive prompt states and mean differences to steer stored cache. AKR must not claim first KV steering. Its implemented target is support answer-loss gradient coefficients, predicted from acoustics and added at projection outputs. Fixed mean AKR is not a faithful reproduction of their contrastive procedure. |
| NatureLM-audio | https://arxiv.org/abs/2411.07186 | abstract; attempted v3 HTML returned internal error | Specialized bioacoustic audio-language training and BEANS-Zero already exist. AKR does not introduce animal-audio understanding or the benchmark. Full-text comparative coverage remains incomplete. |
| Qwen2.5-Omni-7B | https://huggingface.co/Qwen/Qwen2.5-Omni-7B ; report https://arxiv.org/abs/2503.20215 | model card | General multimodal model; our input/output scope is audio-to-text with Thinker only. The card is not a run on our animal data. Experiment revisions are pinned in the registry, not silently replaced by current hub main. |
| MetaSICL | https://arxiv.org/abs/2601.18904 | abstract page | Audio in-context learning can be explicitly strengthened; not an AKR baseline run here. |
| FSA-GRPO | https://arxiv.org/abs/2606.02615 | abstract page | Rewarding effective demonstration use is adjacent to support utilization; not evidence that any particular KV method is necessary. |
| Anatomy of the Modality Gap | https://arxiv.org/abs/2603.01502 | abstract page | Internal speech/text disparities are an existing research direction; avoid broad originality claims about encoded-but-unused information. |
| Beyond the Baseband | https://arxiv.org/abs/2604.27936 | abstract page | Higher-frequency bioacoustic evidence is studied elsewhere. Our LP manipulation is within the model-observable input band. |

The validated distinction is an implementation description, not a certified novelty verdict. No exhaustive alias/forward-citation search, independent prosecution/defense contexts, or complete primary text for all neighbors was obtained. Formal collision clearance remains provisional. The strongest immediate competing solution is the same-support ridge classifier, not merely another steering paper; S1 tests simple text-output/evidence alternatives on identical recordings.
