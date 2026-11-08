"""CPU contracts for the C04 audio-ICL attention switch.

The 50-audio equivalence test uses a tiny random Qwen2.5-Omni audio encoder in
float32; the real 7B weights are exercised only by the GPU probe.  Chunking is
exact because the encoder's attention is block-diagonal over the concatenated
per-audio frame ranges, which is the same ``cu_seqlens`` partition the whole
forward masks.
"""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoderConfig)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoder)

from akr_closing.backend import (ICL_AUDIO_ATTENTION_MODES, audio_flash_attention_modules,
                                 chunked_audio_forward, normalize_icl_audio_attention)


def tiny_config():
    return Qwen2_5OmniAudioEncoderConfig(
        num_mel_bins=8, encoder_layers=2, encoder_attention_heads=2,
        encoder_ffn_dim=16, d_model=8, output_dim=8, max_source_positions=500)


def test_normalize_switch_rejects_unknown_and_accepts_off():
    assert normalize_icl_audio_attention(None) is None
    assert normalize_icl_audio_attention('sdpa') is None
    assert normalize_icl_audio_attention(False) is None
    for mode in ('flash_attention_2', 'chunked'):
        assert mode in ICL_AUDIO_ATTENTION_MODES
        assert normalize_icl_audio_attention(mode) == mode
    with pytest.raises(ValueError):
        normalize_icl_audio_attention('eager')


def test_chunked_forward_matches_whole_fifty_audio_context():
    torch.manual_seed(0)
    tower = Qwen2_5OmniAudioEncoder(tiny_config()).eval().float()
    # Fifty independent audio rows, each spanning multiple 2 s chunks.
    lengths = torch.tensor([100 + (i % 7) * 13 for i in range(50)], dtype=torch.long)
    features = torch.randn(tower.num_mel_bins, int(lengths.sum()))
    aftercnn, _ = tower._get_feat_extract_output_lengths(lengths)
    with torch.inference_mode():
        whole = tower(features, feature_lens=lengths, aftercnn_lens=aftercnn).last_hidden_state
        pieces = chunked_audio_forward(tower.forward, features, lengths, aftercnn).last_hidden_state
    assert pieces.shape == whole.shape
    diff = (pieces - whole).abs().max().item()
    assert diff <= 1e-5, f'chunked audio tower differs from whole forward: {diff}'


def test_chunked_forward_passes_through_single_audio():
    torch.manual_seed(1)
    tower = Qwen2_5OmniAudioEncoder(tiny_config()).eval().float()
    lengths = torch.tensor([321], dtype=torch.long)
    features = torch.randn(tower.num_mel_bins, 321)
    aftercnn, _ = tower._get_feat_extract_output_lengths(lengths)
    with torch.inference_mode():
        whole = tower(features, feature_lens=lengths, aftercnn_lens=aftercnn).last_hidden_state
        wrapped = chunked_audio_forward(tower.forward, features, lengths, aftercnn).last_hidden_state
    assert torch.equal(wrapped, whole)


def test_flash_modules_share_original_parameter_storage():
    pytest.importorskip('flash_attn')
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniAudioFlashAttention2)
    tower = Qwen2_5OmniAudioEncoder(tiny_config()).eval()
    modules, version = audio_flash_attention_modules(tower)
    assert len(modules) == len(tower.layers) and version
    for layer, new in zip(tower.layers, modules):
        old = layer.self_attn
        assert isinstance(new, Qwen2_5OmniAudioFlashAttention2)
        for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
            assert (getattr(new, name).weight.data_ptr()
                    == getattr(old, name).weight.data_ptr())


def test_switch_only_swaps_audio_forward_for_support_carrying_predicts():
    from akr_closing.backend import OmniBackend
    tower = Qwen2_5OmniAudioEncoder(tiny_config()).eval()
    original = tower.forward
    backend = SimpleNamespace(
        icl_audio_attention='chunked', _icl_attention_active=False,
        thinker=SimpleNamespace(audio_tower=tower),
        _icl_chunked_forward=lambda *a, **k: 'chunked')
    context = OmniBackend._icl_audio_attention_context.__get__(backend, SimpleNamespace)
    with context(()):
        assert tower.forward.__func__ is original.__func__
    with context(((SimpleNamespace(event_id='s1'), 'a'),)):
        assert tower.forward is backend._icl_chunked_forward
    assert tower.forward.__func__ is original.__func__
