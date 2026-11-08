"""GLM-4-Voice AKR adapter; optional escalation backend, not benchmark evidence.

The pinned checkpoint is batch-major despite an outdated time-major docstring.
Its MQA projection stores contiguous Q, K, V slices. Parameter-free taps expose
the K/V slices before the checkpoint applies RoPE or inserts them into a cache.
Audio uses the official Whisper-VQ encoder, first-channel/16 kHz/30 s protocol.
Only text is generated; the speech decoder is deliberately unnecessary.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.metadata
import math
import numbers
import os
import subprocess
import sys

import numpy as np

from .backend import ContextLimit, OmniBackend, ProjectionIntervention


MODEL_REVISION = '352e5a64063448d5e46394753fb65bf1dbab975b'
TOKENIZER_ID = 'THUDM/glm-4-voice-tokenizer'
TOKENIZER_REVISION = 'a5f2404e63c84e92f5238908e1706316324ebafa'
SOURCE_URL = 'https://github.com/zai-org/GLM-4-Voice'
SOURCE_REVISION = 'eb00ce9142e8d98b0ed7c57cd47e0d6d5dce9a1a'


def check_dependencies(*, source_path=None):
    """Import-only preflight: no checkpoint weights or CUDA initialization."""
    import torch
    if importlib.metadata.version('transformers') != '4.44.1':
        raise RuntimeError('GLM requires its isolated transformers==4.44.1 environment; run scripts/setup_glm_env.sh')
    # Import each runtime dependency now, including the native tiktoken module.
    import soundfile
    import scipy.signal
    import tiktoken
    from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor
    source = Path(source_path or os.environ.get('AKR_GLM_SOURCE',
                  Path(__file__).resolve().parents[1] / 'external' / 'GLM-4-Voice')).resolve()
    git = ['git', '-c', f'safe.directory={source}', '-C', str(source)]
    try:
        actual_revision = subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True,
                                                  stderr=subprocess.PIPE).strip()
        dirty = subprocess.check_output(git + ['status', '--porcelain', '--',
                                               'speech_tokenizer', ':(exclude)**/__pycache__/**'], text=True,
                                        stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError('install pinned GLM source with scripts/setup_glm_env.sh') from exc
    if actual_revision != SOURCE_REVISION or dirty:
        raise RuntimeError('pinned GLM source revision or worktree differs from the pin')
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from speech_tokenizer.modeling_whisper import WhisperVQEncoder
    imported_source = Path(sys.modules[WhisperVQEncoder.__module__].__file__).resolve()
    if not imported_source.is_relative_to(source):
        raise RuntimeError('another speech_tokenizer package shadows the pinned GLM source')
    return {'source_path': str(source), 'source_revision': actual_revision,
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (source / 'speech_tokenizer').glob('*.py')},
            'transformers': importlib.metadata.version('transformers'),
            'speech_encoder_imported': True, 'cuda_initialized': torch.cuda.is_initialized()}


def speech_token_ids(codes, offset, vocabulary_size):
    """Map VQ indices to the LM's contiguous audio-token vocabulary."""
    codes = list(codes)
    if not codes or any(isinstance(c, bool) or not isinstance(c, numbers.Integral)
                        or not 0 <= c < vocabulary_size for c in codes):
        raise ValueError('nonempty, in-range integer speech codes required')
    return [offset + int(c) for c in codes]


def resolve_audio_token_ids(tokenizer, vocabulary_size):
    """Resolve audio-token ids, filling tokenizer gaps with contiguous ids.

    The pinned THUDM/glm-4-voice-tokenizer does not expose every
    ``<|audio_i|>`` entry that ``quantize_vocab_size`` requires; missing
    entries resolve to ``None`` or ``unk_token_id``. Tokenizer probing
    verified that the LM audio embeddings map to the contiguous id block
    starting at ``<|audio_0|>`` (152353 in the pinned tokenizer), so the one
    missing final entry falls back to ``audio_offset + i`` = 168736. Any
    real, non-contiguous mapping still fails.
    """
    if (isinstance(vocabulary_size, bool) or not isinstance(vocabulary_size, numbers.Integral)
            or vocabulary_size < 1):
        raise ValueError('positive integer speech vocabulary size required')
    size = int(vocabulary_size)
    unk = getattr(tokenizer, 'unk_token_id', None)
    missing_types = (KeyError, IndexError, ValueError, TypeError)
    try:
        offset = tokenizer.convert_tokens_to_ids('<|audio_0|>')
    except missing_types as error:
        raise RuntimeError('missing <|audio_0|>; cannot establish the audio-token offset') from error
    if (isinstance(offset, bool) or not isinstance(offset, numbers.Integral) or offset < 0
            or (unk is not None and offset == unk)):
        raise RuntimeError('missing <|audio_0|>; cannot establish the audio-token offset')
    offset = int(offset)
    tokens = [f'<|audio_{i}|>' for i in range(size)]
    try:
        raw = tokenizer.convert_tokens_to_ids(tokens)
    except missing_types:
        raw = None
    if not isinstance(raw, list) or len(raw) != size:
        # Some tokenizers raise on an unknown token instead of returning unk;
        # resolve one at a time so only the genuinely missing entries fall back.
        raw = []
        for token in tokens:
            try:
                raw.append(tokenizer.convert_tokens_to_ids(token))
            except missing_types:
                raw.append(None)
    resolved = []
    for value in raw:
        valid = (isinstance(value, numbers.Integral) and not isinstance(value, bool)
                 and value >= 0 and (unk is None or value != unk))
        resolved.append(int(value) if valid else None)
    missing = [tokens[i] for i, value in enumerate(resolved) if value is None]
    fallback_ids = {tokens[i]: offset + i for i, value in enumerate(resolved) if value is None}
    ids = [offset + i if value is None else value for i, value in enumerate(resolved)]
    if ids != list(range(offset, offset + size)):
        bad = [(tokens[i], raw[i], ids[i]) for i in range(size) if ids[i] != offset + i]
        raise RuntimeError(f'non-contiguous or incomplete LM speech-token vocabulary: {bad[:8]}')
    engine_note = {
        'kind': 'tokenizer_audio_vocab_gap',
        'audio_vocab_size': size,
        'audio_offset': offset,
        'missing_tokens': missing,
        'fallback_ids': fallback_ids,
        'fallback_applied': bool(missing),
        'reason': ('THUDM/glm-4-voice-tokenizer omits audio tokens; tokenizer probing verified '
                   'the LM audio vocabulary is the contiguous block starting at <|audio_0|> '
                   '(152353), so missing entries use audio_offset+i instead of unk '
                   '(<|audio_16383|> -> 168736).'),
    }
    return offset, ids, engine_note


class FusedKVBridge:
    """Expose independent batch-major K/V hook sites on fused MQA outputs.

    The original projection and its Accelerate hooks remain in place. Identity
    taps have no weights and never transform Q. ``close`` removes our hooks.
    """
    def __init__(self, fused_modules, *, query_width, kv_width, layout='batch_first'):
        import torch
        if layout not in {'batch_first', 'sequence_first'}:
            raise ValueError('unknown fused projection layout')
        if query_width < 1 or kv_width < 1:
            raise ValueError('positive Q/K/V widths required')
        self.modules = {(layer, kind): torch.nn.Identity()
                        for layer in fused_modules for kind in ('k', 'v')}
        self.handles = []
        for layer, module in fused_modules.items():
            def bridge(_module, _args, out, layer=layer):
                if out.ndim != 3 or out.shape[-1] != query_width + 2 * kv_width:
                    raise ValueError('unexpected fused MQA projection dimensions')
                canonical = out if layout == 'batch_first' else out.transpose(0, 1)
                if canonical.shape[0] != 1:
                    raise ValueError('GLM AKR requires batch one and the declared axis layout')
                q, k, v = canonical.split([query_width, kv_width, kv_width], dim=-1)
                k = self.modules[(layer, 'k')](k)
                v = self.modules[(layer, 'v')](v)
                result = torch.cat((q, k, v), dim=-1)
                return result if layout == 'batch_first' else result.transpose(0, 1)
            self.handles.append(module.register_forward_hook(bridge))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def extract_audio_codes(path, encoder, extractor, *, device):
    """Official VQ framing with CPU waveform I/O and scipy polyphase resampling.

    Unlike upstream's CUDA-only torchaudio helper, this supports CPU tokenizing
    and processes one 30-second chunk at a time to bound memory. The resampler
    implementation difference is recorded in backend provenance.
    """
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    audio, sample_rate = sf.read(path, dtype='float32', always_2d=True)
    audio = audio[:, 0]
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError('audio must contain finite, nonempty samples')
    if sample_rate != 16000:
        divisor = math.gcd(int(sample_rate), 16000)
        audio = resample_poly(audio, 16000 // divisor, sample_rate // divisor)
    conv_stride = encoder.conv1.stride[0] * encoder.conv2.stride[0]
    pooling = encoder.config.pooling_kernel_size or 1
    stride = conv_stride * pooling * extractor.hop_length
    codes = []
    with torch.inference_mode():
        for start in range(0, len(audio), 30 * 16000):
            features = extractor([audio[start:start + 30 * 16000]], sampling_rate=16000,
                                 return_attention_mask=True, return_tensors='pt',
                                 padding='longest', pad_to_multiple_of=stride)
            features = features.to(device=device)
            features['input_features'] = features['input_features'].to(encoder.dtype)
            output = encoder(**features)
            mask = features.attention_mask[:, ::conv_stride][:, ::pooling].bool()
            if mask.shape != output.quantized_token_ids.shape:
                raise RuntimeError('Whisper-VQ token/attention-mask alignment mismatch')
            codes.extend(output.quantized_token_ids[mask].cpu().tolist())
    return codes


class GLMVoiceBackend(OmniBackend):
    """Full feature/gradient/injection/calibration API for the pinned GLM model.

    Real model execution is an optional escalation after Qwen2.5 evaluation.
    Construction loads weights and therefore must not be used for CPU tests.
    """
    def __init__(self, model_id, revision, *, profile='24gb',
                 layer_fractions=(.25, .5, .75, .875), feature_fraction=.875,
                 max_context_tokens=8192, source_path=None, audio_device='cpu'):
        import torch
        from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor
        if revision != MODEL_REVISION:
            raise ValueError('this adapter requires the audited GLM checkpoint revision')
        dependencies = check_dependencies(source_path=source_path)
        if not torch.cuda.is_available():
            raise RuntimeError('real GLM experiments require CUDA; use CPU contract tests instead')
        budgets = {'16gb': '11GiB', '24gb': '18GiB', '48gb': '40GiB', '80gb': '69GiB'}
        if profile not in budgets:
            raise ValueError('profile must be 16gb/24gb/48gb/80gb')
        from speech_tokenizer.modeling_whisper import WhisperVQEncoder
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
        self.processor = SimpleNamespace(tokenizer=self.tokenizer)
        self.thinker = AutoModel.from_pretrained(
            model_id, revision=revision, trust_remote_code=True, torch_dtype=torch.bfloat16,
            device_map='auto', max_memory={0: budgets[profile], 'cpu': '64GiB'},
            attn_implementation='sdpa')
        self.thinker.eval()
        for parameter in self.thinker.parameters():
            parameter.requires_grad_(False)
        config = self.thinker.config
        if not config.multi_query_attention:
            raise ValueError('only contiguous-QKV multi-query GLM attention is supported')
        self.device = self.thinker.get_input_embeddings().weight.device
        self.max_context_tokens = min(int(max_context_tokens), int(config.seq_length))
        if self.max_context_tokens < 1:
            raise ValueError('positive context limit required')
        layers = self.thinker.transformer.encoder.layers
        n = len(layers)
        self.layer_ids = sorted({min(n - 1, max(0, int(f * n))) for f in layer_fractions})
        if not self.layer_ids:
            raise ValueError('at least one intervention layer required')
        self.feature_layer = min(n, max(0, round(feature_fraction * n)))
        self._bridge = FusedKVBridge(
            {i: layers[i].self_attention.query_key_value for i in self.layer_ids},
            query_width=config.num_attention_heads * config.kv_channels,
            kv_width=config.multi_query_group_num * config.kv_channels)
        self.modules = self._bridge.modules
        self.keys = sorted(self.modules)
        self.layout = [(key, config.multi_query_group_num * config.kv_channels) for key in self.keys]
        self.forward_count = 0
        self._count_handle = self.thinker.register_forward_hook(
            lambda *_: setattr(self, 'forward_count', self.forward_count + 1))
        self.audio_encoder = WhisperVQEncoder.from_pretrained(
            TOKENIZER_ID, revision=TOKENIZER_REVISION, torch_dtype=torch.float32).eval().to(audio_device)
        for parameter in self.audio_encoder.parameters():
            parameter.requires_grad_(False)
        self.audio_extractor = WhisperFeatureExtractor.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)
        self.audio_device = audio_device
        self.audio_vocab_size = self.audio_encoder.config.quantize_vocab_size
        self.audio_offset, _audio_ids, self._engine_note = resolve_audio_token_ids(
            self.tokenizer, self.audio_vocab_size)
        self._audio_cache = OrderedDict()
        self._stop_ids = sorted(set(config.eos_token_id if isinstance(config.eos_token_id, list)
                                    else [config.eos_token_id]) | {self._special('<|user|>')})
        self._suppressed_ids = list(range(self.audio_offset, self.audio_offset + self.audio_vocab_size))
        for token in ('<|begin_of_audio|>', '<|end_of_audio|>'):
            self._suppressed_ids.append(self._special(token))
        self.provenance = {
            'model_id': model_id, 'revision': revision, 'dtype': 'bfloat16', 'batch_size': 1,
            'backend': 'GLM-4-Voice fused MQA pre-RoPE adapter', 'profile': profile,
            'layers': self.layer_ids, 'feature_layer': self.feature_layer,
            'max_context_tokens': self.max_context_tokens, 'projection_layout': 'batch_first',
            'intervention_site': 'fused query_key_value K/V slices before RoPE; prefill only',
            'attention': 'sdpa', 'tokenizer_id': TOKENIZER_ID, 'tokenizer_revision': TOKENIZER_REVISION,
            'audio_vocab_size': self.audio_vocab_size, 'engine_note': self._engine_note,
            'source_url': SOURCE_URL, 'source_revision': SOURCE_REVISION, 'source_sha256': dependencies['source_sha256'],
            'audio_protocol': 'first channel; 16 kHz; 30 s chunks; Whisper VQ attention-mask filtering',
            'audio_resampler': 'scipy.signal.resample_poly (upstream uses torchaudio sinc)',
            'audio_encoder_device': str(audio_device), 'audio_encoder_dtype': 'float32',
            'audio_generation': False, 'decoder_required': False,
            'prompt_protocol': 'GLM role/audio markers, streaming_transcription assistant prefix; text-only system instruction',
            'generation': 'greedy; suppress audio codebook and audio boundaries',
            'validation_status': 'implementation only; real-checkpoint AKR validation required',
            'versions': {name: importlib.metadata.version(name) for name in
                         ('torch', 'transformers', 'tokenizers', 'accelerate', 'numpy', 'scipy', 'soundfile')},
        }

    def _special(self, token):
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f'missing GLM special token {token}')
        return ids[0]

    def _audio_ids(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self._audio_cache:
            codes = extract_audio_codes(path, self.audio_encoder, self.audio_extractor, device=self.audio_device)
            self._audio_cache[key] = speech_token_ids(codes, self.audio_offset, self.audio_vocab_size)
            if len(self._audio_cache) > 256:
                self._audio_cache.popitem(last=False)
        self._audio_cache.move_to_end(key)
        return self._audio_cache[key]

    def _prepare(self, q, prompt, support=()):
        # Construct audio IDs directly: no lossy decode/re-encode or transcript substitution.
        encode = lambda text: self.tokenizer.encode(text, add_special_tokens=False)
        ids = self.tokenizer.encode(
            '<|system|>\nListen to the audio and follow the user instruction. '
            'Respond with text only, using the requested label and no explanation.',
            add_special_tokens=True)
        for item, answer in list(support) + [(q, None)]:
            ids += encode('<|user|>\n')
            ids += [self._special('<|begin_of_audio|>')] + self._audio_ids(item.audio_path)
            ids += [self._special('<|end_of_audio|>')] + encode('\n' + prompt)
            ids += encode('<|assistant|>streaming_transcription\n')
            if answer is not None:
                ids += encode(answer)
        if len(ids) > self.max_context_tokens:
            raise ContextLimit(f'{len(ids)} tokens exceeds budget {self.max_context_tokens}; no truncation')
        input_ids = self.torch.tensor([ids], dtype=self.torch.long, device=self.device)
        return {'input_ids': input_ids, 'attention_mask': self.torch.ones_like(input_ids)}

    def _mask(self, x, scope, prefix_length=None):
        ids = x['input_ids']
        n = ids.shape[1] if prefix_length is None else prefix_length
        prefix = (self.torch.arange(ids.shape[1], device=ids.device)[None, :] < n)
        prefix = prefix & x.get('attention_mask', self.torch.ones_like(ids)).bool()
        audio = (ids >= self.audio_offset) & (ids < self.audio_offset + self.audio_vocab_size)
        if scope == 'audio': return prefix & audio
        if scope == 'text': return prefix & ~audio
        if scope == 'full_prefill': return prefix
        raise ValueError('scope must be audio/text/full_prefill')

    def feature(self, q, prompt):
        x = self._prepare(q, prompt)
        mask = self._mask(x, 'audio')
        if not mask.any():
            raise RuntimeError('no audio tokens')
        with self.torch.inference_mode():
            output = self.thinker(**x, use_cache=False, output_hidden_states=True,
                                  return_dict=True, return_last_logit=True)
            h = output.hidden_states[self.feature_layer]
            if h.shape[:2] != mask.shape:
                raise RuntimeError('unexpected GLM hidden-state axis layout')
            return h[mask.to(h.device)].float().mean(0).cpu().numpy()

    def predict(self, q, prompt, labels, *, support=(), direction=None, alpha=0., scope='audio',
                kind='kv', layer_group='all', max_new_tokens=12, scale_mode='relative_norm', raw_eta=1.):
        if direction is not None and support:
            raise ValueError('repair uses no audio ICL prefix in this protocol')
        x = self._prepare(q, prompt, support)
        n = x['input_ids'].shape[1]
        if max_new_tokens < 1 or n + max_new_tokens > self.max_context_tokens:
            raise ContextLimit('prefix plus generation exceeds GLM context budget')
        directions = {} if direction is None else self.unflatten(direction, kind, layer_group)
        with self.torch.inference_mode(), ProjectionIntervention(
                self.modules, directions, self._mask(x, scope), alpha,
                scale_mode=scale_mode, raw_eta=raw_eta) as hooks:
            output = self.thinker.generate(**x, do_sample=False, max_new_tokens=max_new_tokens,
                                           use_cache=True, eos_token_id=self._stop_ids,
                                           pad_token_id=self.tokenizer.pad_token_id,
                                           suppress_tokens=self._suppressed_ids)
            hooks.assert_applied()
        raw = self.tokenizer.decode(output[0, n:], skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False).strip()
        from animal_omni.metrics import normalize_label
        return {'prediction': normalize_label(raw, labels) or '', 'raw_prediction': raw,
                'applied_relative_norms': hooks.ratios, 'applied_norms': hooks.norms,
                'effective_relative_norms': hooks.effective_ratios, 'scale_mode': scale_mode,
                'raw_eta': raw_eta, 'generated_tokens': int(output.shape[1] - n), 'input_tokens': n}
