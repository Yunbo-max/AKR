"""Optional Qwen3-Omni text Thinker backend for the existing AKR protocol.

Use requirements-qwen3.txt in a separate environment: Qwen2.5 stays pinned to
its existing Transformers version. Only the Thinker is loaded, never Talker or
Code2Wav. Audio placeholders and features come from Qwen3's own processor.

The inherited support loss, feature extraction, generation and K/V intervention
paths run the real Thinker. Qwen3 normalizes keys AFTER k_proj and BEFORE RoPE;
the intervention is deliberately at k_proj/v_proj output, as in Qwen2.5, and
its measured norms describe that projection space, not the normalized cache.
"""
from __future__ import annotations

import inspect
import math

from .backend import OmniBackend


class Qwen3OmniBackend(OmniBackend):
    """Frozen Qwen3 Thinker with answer-only support gradients and prefill repair."""

    def __init__(self, model_id, revision, *, profile='24gb',
                 layer_fractions=(.25, .5, .75, .875), feature_fraction=.875,
                 max_context_tokens=32768, icl_audio_attention=None):
        import torch
        import transformers
        if icl_audio_attention not in (None, False, '', 'sdpa'):
            # C04 runs on the Qwen2.5 family; never silently ignore the switch.
            raise ValueError('icl_audio_attention is only implemented for the Qwen2.5-Omni backend')
        try:
            from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                'Qwen3 requires its separate environment; install requirements-qwen3.txt '
                '(tested with transformers==4.57.3). Do not upgrade the Qwen2.5 environment.'
            ) from exc
        budget = {'16gb': '13GiB', '24gb': '20GiB', '48gb': '43GiB', '80gb': '72GiB'}
        if profile not in budget:
            raise ValueError('profile must be 16gb/24gb/48gb/80gb')
        if not isinstance(max_context_tokens, int) or max_context_tokens < 1:
            raise ValueError('max_context_tokens must be a positive integer')
        self._validate_fractions(layer_fractions, feature_fraction)
        if not torch.cuda.is_available():
            raise RuntimeError('Real experiments require CUDA; CPU tests use a tiny synthetic Thinker.')
        self.torch = torch
        self.max_context_tokens = max_context_tokens
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            model_id, revision=revision, use_fast=False)
        self.thinker = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_id, revision=revision, dtype=torch.bfloat16,
            device_map='auto', max_memory={0: budget[profile], 'cpu': '64GiB'},
            attn_implementation='sdpa')
        self._bind_thinker(layer_fractions, feature_fraction)
        processor_audio_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.audio_token)
        if processor_audio_id != self.audio_token_id:
            raise RuntimeError('Qwen3 processor and Thinker audio placeholder IDs disagree')
        self.provenance = {
            'model_id': model_id, 'revision': revision, 'dtype': 'bfloat16', 'batch_size': 1,
            'backend': 'Qwen3-Omni direct text Thinker, SDPA, frozen MoE backbone',
            'transformers_version': transformers.__version__,
            'layers': self.layer_ids, 'feature_layer': self.feature_layer, 'profile': profile,
            'audio_token_id': self.audio_token_id,
            'intervention_site': 'pre-RoPE k_proj/v_proj output; K is before QK normalization',
            'audio_output_loaded': False,
        }

    @staticmethod
    def _validate_fractions(layer_fractions, feature_fraction):
        if not layer_fractions or any(not math.isfinite(f) or not 0 <= f <= 1 for f in layer_fractions):
            raise ValueError('layer_fractions must be nonempty and within [0, 1]')
        if not math.isfinite(feature_fraction) or not 0 <= feature_fraction <= 1:
            raise ValueError('feature_fraction must be within [0, 1]')

    def _bind_thinker(self, layer_fractions, feature_fraction):
        """Bind actual Qwen3 decoder projections; also usable for tiny CPU contracts."""
        self._validate_fractions(layer_fractions, feature_fraction)
        self.thinker.eval()
        for parameter in self.thinker.parameters():
            parameter.requires_grad_(False)
        self.device = self.thinker.get_input_embeddings().weight.device
        layers = self.thinker.model.layers
        n = len(layers)
        if not n:
            raise RuntimeError('Qwen3 Thinker has no decoder layers')
        self.layer_ids = sorted({min(n - 1, int(f * n)) for f in layer_fractions})
        self.feature_layer = min(n, round(feature_fraction * n))
        self.modules = {(i, kind): getattr(layers[i].self_attn, kind + '_proj')
                        for i in self.layer_ids for kind in ('k', 'v')}
        text_config = self.thinker.config.text_config
        expected_width = text_config.num_key_value_heads * text_config.head_dim
        if any(module.out_features != expected_width for module in self.modules.values()):
            raise RuntimeError('Unexpected Qwen3 grouped-query projection shape')
        if any(not hasattr(layers[i].self_attn, 'k_norm') for i in self.layer_ids):
            raise RuntimeError('Expected Qwen3 key normalization before RoPE')
        self.keys = sorted(self.modules)
        self.layout = [(key, int(self.modules[key].out_features)) for key in self.keys]
        self.audio_token_id = self.thinker.config.audio_token_id
        self._forward_parameters = inspect.signature(self.thinker.forward).parameters
        self.forward_count = 0
        self._count_handle = self.thinker.register_forward_hook(
            lambda *_: setattr(self, 'forward_count', self.forward_count + 1))

    def _prepare(self, q, prompt, support=()):
        # Qwen3 keeps the same process_mm_info -> processor(audio=...) contract.
        # Its processor performs Qwen3-specific audio downsampling expansion.
        inputs = super()._prepare(q, prompt, support)
        if not self._mask(inputs, 'audio').any():
            raise RuntimeError('Qwen3 input contains no active audio placeholders')
        if 'input_features' not in inputs:
            raise RuntimeError('Qwen3 input contains no audio features')
        return inputs

    def _mask(self, x, scope, prefix_length=None):
        ids = x['input_ids']
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError('Qwen3 AKR requires batch-one inputs')
        n = ids.shape[1] if prefix_length is None else prefix_length
        if not 0 <= n <= ids.shape[1]:
            raise ValueError('prefix length is outside the input sequence')
        prefix = self.torch.arange(ids.shape[1], device=ids.device)[None, :] < n
        if 'attention_mask' in x:
            prefix = prefix & x['attention_mask'].to(ids.device).bool()
        audio = ids.eq(self.audio_token_id) & prefix
        if scope == 'audio':
            return audio
        if scope == 'text':
            return prefix & ~audio
        if scope == 'full_prefill':
            return prefix
        raise ValueError('scope must be audio/text/full_prefill')
