# CT-KV baseline assessment

## Why the current cache baselines are not CT-KV

The official [Context Tuning paper](https://arxiv.org/abs/2507.04221) and
[implementation](https://github.com/agentic-learning-ai-lab/context-tuning)
define CT-KV as an optimized **prefix cache initialized from the concatenated
demonstration pairs**. It is not a fixed steering vector and it is not the
query-token correction used in this repository.

A faithful audio adaptation must preserve all of the following:

1. Run the complete labeled multi-audio demonstration context once and use its
   layerwise K/V cache as initialization.
2. Make that initialized K/V prefix trainable while every Qwen parameter stays
   frozen.
3. Optimize each support answer from the other supports, masking the K/V span of
   the held-out demonstration (leave-one-out masking).
4. Apply token dropout to the visible prefix during optimization.
5. At inference, expose the complete optimized prefix to an unlabeled query.

The released reference uses AdamW, cosine decay, learning rate 1e-3, token
dropout 0.05, and 200 epochs in its documented main command. Its released code
also evaluates labels by candidate loss, so the primary comparison here should
use the same constrained candidate readout in addition to free generation.

Our fixed mean KV, conditional pooled KV, and query-token class dictionary do
none of steps 1--5 jointly. They remain useful ablations but must not be labeled
CT-KV.

## Qwen2.5-Omni implementation boundary

Qwen-7B Thinker has 28 layers, four KV heads, and 128 dimensions per head. A
BF16 demonstration prefix therefore stores about 56 KiB per token before
gradients and Adam states. A trainable prefix costs roughly six times that once
BF16 gradients and FP32 Adam moments are included. Multi-audio Dogs K=2/class
can contain thousands of prefix tokens, leaving little room beside the roughly
21.7 GiB loaded model on a 24 GiB RTX 3090.

Memory alone is not the only issue. Omni uses multimodal rotary positions. A
faithful port must carry the demonstration cache positions into every held-out
audio pair and query, while applying leave-one-out attention masks to exact
demonstration spans. Treating the prefix as ordinary text positions or applying
post-RoPE additive hooks would change the method.

## Registered implementation path

1. Add a runner path that returns the K/V cache, multimodal position state, and
   exact token span of each labeled audio demonstration.
2. Reproduce reference CT-KV loss on a two-example text-only unit test first;
   iteration zero must exactly match ordinary ICL candidate scores.
3. Run a one-query K=1/class MarmAudio BF16 memory/numerical gate. MarmAudio is
   the smallest complete equal-support prefix and is the only 24 GiB starting
   point that is likely to fit full K+V.
4. If full CT-KV does not fit, evaluate the paper's CT-V parameter-efficient
   variant, explicitly labeled CT-V rather than CT-KV.
5. Use the frozen recording-disjoint 75-query MarmAudio split and exactly the
   same six supports as audio ICL, ridge, and our cache repair. Report both
   candidate scoring and free generation.
6. Only after K=1 succeeds, attempt K=2. Dogs and Watkins are memory-gated rather
   than silently shortened or downsampled.

## Current status

Not run. The official source has been audited, and the required adaptation is
now specified, but a faithful multimodal/mRoPE implementation is not treated as
a quick 12-hour baseline. Until it is complete, a method-forward submission must
list CT-KV as a blocking comparison. The current evidence supports a
diagnostic/causal paper and a partial conditional-pooled repair result; it does
not establish superiority over Context Tuning.
