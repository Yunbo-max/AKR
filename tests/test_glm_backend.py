"""CPU contract tests: no checkpoint loading and no CUDA initialization."""
from types import SimpleNamespace
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pytest
import torch

from akr_closing.backend import ProjectionIntervention
from akr_closing.glm_backend import FusedKVBridge, GLMVoiceBackend, speech_token_ids


@pytest.mark.parametrize('layout', ['batch_first', 'sequence_first'])
def test_fused_bridge_preserves_query_and_only_changes_audio_k(layout):
    fused = torch.nn.Identity()
    bridge = FusedKVBridge({3: fused}, query_width=4, kv_width=2, layout=layout)
    original = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8) + 1
    source = original if layout == 'batch_first' else original.transpose(0, 1)
    mask = torch.tensor([[False, True, True, False]])
    try:
        with ProjectionIntervention(bridge.modules, {(3, 'k'): [2., -1.]}, mask, 1.,
                                    scale_mode='raw') as intervention:
            actual = fused(source)
            decode = fused(source[:, :1] if layout == 'batch_first' else source[:1])
            # A second same-length call must also remain untouched.
            again = fused(source)
            intervention.assert_applied()
        actual = actual if layout == 'batch_first' else actual.transpose(0, 1)
        expected = original.clone()
        expected[0, 1:3, 4:6] += torch.tensor([2., -1.])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(again, source)
        torch.testing.assert_close(decode, source[:, :1] if layout == 'batch_first' else source[:1])
    finally:
        bridge.close()
    torch.testing.assert_close(fused(source), source)


def test_fused_bridge_gradients_have_independent_k_v_coordinates():
    fused = torch.nn.Identity()
    bridge = FusedKVBridge({0: fused}, query_width=4, kv_width=2, layout='sequence_first')
    leaves = {}
    handles = []
    def hook(key):
        def add(_module, _inputs, out):
            leaves[key] = torch.zeros_like(out, requires_grad=True)
            return out + leaves[key]
        return add
    try:
        for key, module in bridge.modules.items():
            handles.append(module.register_forward_hook(hook(key)))
        result = fused(torch.ones(3, 1, 8))
        loss = (result[1, 0, 4:6] * torch.tensor([3., 5.])).sum() + result[2, 0, 6:].sum() * 7
        k, v = torch.autograd.grad(loss, [leaves[(0, 'k')], leaves[(0, 'v')]])
        torch.testing.assert_close(k, torch.tensor([[[0., 0.], [3., 5.], [0., 0.]]]))
        torch.testing.assert_close(v, torch.tensor([[[0., 0.], [0., 0.], [7., 7.]]]))
    finally:
        for handle in handles: handle.remove()
        bridge.close()


def test_audio_mask_excludes_boundaries_padding_and_answer():
    backend = GLMVoiceBackend.__new__(GLMVoiceBackend)
    backend.torch = torch
    backend.audio_offset = 100
    backend.audio_vocab_size = 4
    x = {'input_ids': torch.tensor([[99, 100, 103, 104, 8, 101, 100]]),
         'attention_mask': torch.tensor([[1, 1, 1, 1, 0, 1, 1]])}
    assert backend._mask(x, 'audio', 5).tolist() == [[False, True, True, False, False, False, False]]
    assert backend._mask(x, 'text', 5).tolist() == [[True, False, False, True, False, False, False]]
    assert backend._mask(x, 'full_prefill', 5).tolist() == [[True, True, True, True, False, False, False]]


def test_speech_codebook_mapping_is_exact_and_rejects_invalid_codes():
    assert speech_token_ids([0, 4, 16383], 151552, 16384) == [151552, 151556, 167935]
    for invalid in ([], [-1], [16384], [1.5], [True]):
        with pytest.raises(ValueError): speech_token_ids(invalid, 151552, 16384)


def test_fused_bridge_rejects_wrong_width_and_batch_axis():
    fused = torch.nn.Identity()
    bridge = FusedKVBridge({0: fused}, query_width=4, kv_width=2)
    try:
        with pytest.raises(ValueError): fused(torch.ones(1, 3, 9))
        with pytest.raises(ValueError): fused(torch.ones(3, 1, 8))
    finally:
        bridge.close()


@pytest.fixture
def tiny_real_glm():
    """Actual pinned remote class, random tiny CPU weights, real LM tokenizer."""
    import transformers
    if transformers.__version__ != '4.44.1':
        pytest.skip('run with /root/.venvs/akr-glm/bin/python for pinned-class CPU integration')
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from akr_closing.glm_backend import MODEL_REVISION
    snapshot = Path.home() / '.cache/huggingface/hub/models--zai-org--glm-4-voice-9b/snapshots' / MODEL_REVISION
    if not snapshot.exists():
        pytest.skip('pinned GLM remote code/tokenizer cache unavailable')
    config = AutoConfig.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    config.num_layers = config.num_hidden_layers = 2
    config.hidden_size = 16
    config.ffn_hidden_size = 32
    config.num_attention_heads = 2
    config.kv_channels = 8
    config.multi_query_group_num = 1
    config.seq_length = 256
    config.torch_dtype = torch.float32
    config._attn_implementation = 'sdpa'
    torch.manual_seed(11)
    model = AutoModel.from_config(config, trust_remote_code=True, empty_init=False).eval()
    # Upstream RMSNorm intentionally allocates empty weights for checkpoint
    # loading and has no initializer. Tiny random CPU models must fill them.
    with torch.no_grad():
        for module in model.modules():
            if module.__class__.__name__ == 'RMSNorm': module.weight.fill_(1.)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    backend = GLMVoiceBackend.__new__(GLMVoiceBackend)
    backend.torch = torch
    backend.device = torch.device('cpu')
    backend.thinker = model
    backend.tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    backend.processor = SimpleNamespace(tokenizer=backend.tokenizer)
    backend.max_context_tokens = 256
    backend.feature_layer = 1
    backend.layer_ids = [0, 1]
    backend._bridge = FusedKVBridge(
        {i: layer.self_attention.query_key_value for i, layer in enumerate(model.transformer.encoder.layers)},
        query_width=16, kv_width=8)
    backend.modules = backend._bridge.modules
    backend.keys = sorted(backend.modules)
    backend.layout = [(key, 8) for key in backend.keys]
    backend.audio_offset = backend.tokenizer.convert_tokens_to_ids('<|audio_0|>')
    backend.audio_vocab_size = 16384
    # The fixture substitutes only waveform encoding; LM and tokenizer are real.
    backend._audio_ids = lambda path: [backend.audio_offset, backend.audio_offset + 1]
    backend._stop_ids = config.eos_token_id
    backend._suppressed_ids = list(range(backend.audio_offset, backend.audio_offset + 16384))
    yield backend
    backend._bridge.close()


def test_real_glm_feature_gradient_calibration_and_text_generation(tiny_real_glm):
    from akr_closing.core import Query
    backend = tiny_real_glm
    query = Query('cpu-fixture', 'unused.wav', 'fixture')
    prompt = 'Classify the audio: cat or dog. Reply with one label.'
    prepared = backend._prepare(query, prompt)
    assert backend._mask(prepared, 'audio').sum().item() == 2
    assert prepared['input_ids'].dtype == torch.long
    feature = backend.feature(query, prompt)
    assert feature.shape == (16,) and np.isfinite(feature).all()
    gradients = backend.support_gradients(query, 'cat', prompt)
    assert set(gradients) == {'audio', 'text', 'full_prefill'}
    for vector in gradients.values():
        assert vector.shape == (32,) and np.isfinite(vector).all()
    assert np.linalg.norm(gradients['audio']) > 0
    baseline = backend.support_answer_loss(query, 'cat', prompt)
    zero = backend.support_answer_loss(query, 'cat', prompt, direction=gradients['audio'], alpha=0.)
    assert baseline['answer_loss'] == zero['answer_loss']
    repaired = backend.support_answer_loss(query, 'cat', prompt, direction=gradients['audio'], alpha=.01)
    assert repaired['answer_loss'] < baseline['answer_loss']
    result = backend.predict(query, prompt, ['cat', 'dog'], direction=gradients['audio'], alpha=.01,
                             max_new_tokens=2)
    assert 1 <= result['generated_tokens'] <= 2
    assert len(result['applied_norms']) == 4
    assert all(x['active_tokens'] == 2 for x in result['applied_norms'].values())
    assert '<|audio_' not in result['raw_prediction']
    assert all(parameter.grad is None for parameter in backend.thinker.parameters())


def test_real_glm_support_prompt_keeps_each_audio_and_context_limit(tiny_real_glm):
    from akr_closing.core import Query
    from akr_closing.backend import ContextLimit
    backend = tiny_real_glm
    q = Query('q', 'q.wav', 'q')
    s = Query('s', 's.wav', 's')
    prepared = backend._prepare(q, 'cat or dog?', [(s, 'cat')])
    assert backend._mask(prepared, 'audio').sum().item() == 4
    text = backend.tokenizer.decode(prepared['input_ids'][0], spaces_between_special_tokens=False)
    assert text.count('<|begin_of_audio|>') == 2
    assert text.count('<|end_of_audio|>') == 2
    assert 'cat<|user|>' in text
    backend.max_context_tokens = 1
    with pytest.raises(ContextLimit): backend._prepare(q, 'cat or dog?')


def test_vq_extraction_preserves_chunk_order_and_masks_padding(tmp_path):
    """Boundary double makes padding codes different from real audio codes."""
    import soundfile as sf
    from akr_closing.glm_backend import extract_audio_codes
    path = tmp_path / 'stereo.wav'
    waveform = np.zeros((16000 * 31, 2), dtype=np.float32)
    waveform[:16000 * 30, 0] = .25
    waveform[16000 * 30:, 0] = .5
    waveform[:, 1] = .75
    sf.write(path, waveform, 16000, subtype='FLOAT')
    seen = []
    class Features(dict):
        def to(self, **kwargs): return self
        @property
        def attention_mask(self): return self['attention_mask']
    class Extractor:
        hop_length = 160
        def __call__(self, audios, **kwargs):
            seen.append(audios[0].copy())
            return Features(input_features=torch.zeros(1, 2, 4),
                            attention_mask=torch.tensor([[1, 1, 0, 0]]))
    class Encoder:
        conv1 = conv2 = SimpleNamespace(stride=(1,))
        config = SimpleNamespace(pooling_kernel_size=1)
        dtype = torch.float32
        def __call__(self, **kwargs):
            return SimpleNamespace(quantized_token_ids=torch.tensor([[1, len(seen) + 1, 99, 99]]))
    codes = extract_audio_codes(path, Encoder(), Extractor(), device='cpu')
    assert codes == [1, 2, 1, 3]
    assert [len(x) for x in seen] == [480000, 16000]
    assert np.all(seen[0] == .25) and np.all(seen[1] == .5)


def test_dependency_preflight_checks_source_without_loading_weights(tmp_path):
    import transformers
    from akr_closing.glm_backend import check_dependencies
    if transformers.__version__ != '4.44.1':
        with pytest.raises(RuntimeError, match='4.44.1'): check_dependencies()
        return
    with pytest.raises(RuntimeError, match='pinned GLM source'):
        check_dependencies(source_path=tmp_path / 'missing')
    result = check_dependencies()
    assert result['transformers'] == '4.44.1'
    assert result['source_revision'] == 'eb00ce9142e8d98b0ed7c57cd47e0d6d5dce9a1a'
    assert result['speech_encoder_imported'] is True
    assert result['cuda_initialized'] is False


def test_actual_whisper_vq_class_encodes_waveform_on_cpu(tmp_path):
    import transformers
    if transformers.__version__ != '4.44.1':
        pytest.skip('requires isolated GLM environment')
    import soundfile as sf
    from akr_closing.glm_backend import check_dependencies, extract_audio_codes
    check_dependencies()
    from speech_tokenizer.configuration_whisper import WhisperVQConfig
    from speech_tokenizer.modeling_whisper import WhisperVQEncoder
    from transformers import WhisperFeatureExtractor
    config = WhisperVQConfig(d_model=16, encoder_attention_heads=2, encoder_ffn_dim=32,
                             encoder_layers=1, quantize_encoder_only=True, quantize_position=1,
                             quantize_vocab_size=8, pooling_position=1, pooling_kernel_size=4,
                             pooling_type='avg', num_mel_bins=128, max_source_positions=1500,
                             encoder_causal_convolution=True, quantize_causal_block_size=200)
    encoder = WhisperVQEncoder(config).eval()
    extractor = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000)
    path = tmp_path / 'tone.wav'
    # 24 kHz forces the production resampler; 0.8 seconds produces 10 VQ tokens.
    sf.write(path, np.sin(np.arange(19200) * (2 * np.pi * 440 / 24000)).astype(np.float32), 24000)
    codes = extract_audio_codes(path, encoder, extractor, device='cpu')
    assert len(codes) == 10
    assert all(isinstance(code, int) and 0 <= code < 8 for code in codes)
    assert torch.cuda.is_initialized() is False
