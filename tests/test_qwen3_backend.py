"""CPU Qwen3 contracts; optional tiny real architecture needs requirements-qwen3.txt."""
from types import SimpleNamespace
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from akr_closing.qwen3_backend import Qwen3OmniBackend
from akr_closing.backend import ProjectionIntervention


def bare_backend():
    backend = object.__new__(Qwen3OmniBackend)
    backend.torch = torch
    backend.audio_token_id = 10
    backend.max_context_tokens = 64
    return backend


def test_masks_exclude_padding_answer_and_audio_boundaries():
    backend = bare_backend()
    x = {'input_ids': torch.tensor([[0, 11, 10, 10, 12, 3, 10]]),
         'attention_mask': torch.tensor([[0, 1, 1, 1, 1, 1, 1]])}
    assert backend._mask(x, 'audio', 6).tolist() == [[False, False, True, True, False, False, False]]
    assert backend._mask(x, 'text', 6).tolist() == [[False, True, False, False, True, True, False]]
    assert backend._mask(x, 'full_prefill', 6).sum() == 5
    assert not backend._mask(x, 'audio', 0).any()


@pytest.mark.parametrize('fractions,feature', [([], .5), ([float('nan')], .5),
                                               ([1.1], .5), ([.5], -1)])
def test_invalid_layer_selection_rejected(fractions, feature):
    with pytest.raises(ValueError):
        Qwen3OmniBackend._validate_fractions(fractions, feature)


def test_prepare_preserves_integer_masks_and_uses_qwen3_processor(monkeypatch):
    import sys
    from transformers.feature_extraction_utils import BatchFeature
    from akr_closing.backend import ContextLimit
    backend = bare_backend()
    backend.device = torch.device('cpu')
    seen = {}
    class Processor:
        def apply_chat_template(self, conversations, **kwargs):
            seen['conversations'] = conversations
            return 'templated audio'
        def __call__(self, **kwargs):
            seen['kwargs'] = kwargs
            return BatchFeature({'input_ids': torch.tensor([[11, 10, 12]]),
                                 'attention_mask': torch.ones(1, 3, dtype=torch.long),
                                 'feature_attention_mask': torch.ones(1, 8, dtype=torch.long),
                                 'input_features': torch.ones(1, 8, 8)})
    backend.processor = Processor()
    monkeypatch.setitem(sys.modules, 'qwen_omni_utils', SimpleNamespace(
        process_mm_info=lambda *args, **kwargs: ([np.zeros(160)], None, None)))
    q = SimpleNamespace(audio_path='query.wav')
    support = SimpleNamespace(audio_path='support.wav')
    result = backend._prepare(q, 'classify', [(support, 'cat')])
    assert [c['role'] for c in seen['conversations']] == ['user', 'assistant', 'user']
    assert seen['kwargs']['audio'][0].shape == (160,)
    assert result['input_features'].dtype == torch.bfloat16
    assert all(result[k].dtype == torch.long for k in ('input_ids', 'attention_mask', 'feature_attention_mask'))
    backend.max_context_tokens = 2
    with pytest.raises(ContextLimit):
        backend._prepare(q, 'classify')


@pytest.fixture
def tiny_backend():
    transformers = pytest.importorskip('transformers')
    if not hasattr(transformers, 'Qwen3OmniMoeThinkerForConditionalGeneration'):
        pytest.skip('Qwen3 optional environment required')
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeThinkerConfig
    config = Qwen3OmniMoeThinkerConfig(
        audio_token_id=10, audio_start_token_id=11, audio_end_token_id=12,
        image_token_id=14, video_token_id=15, vision_start_token_id=16,
        audio_config=dict(num_mel_bins=8, encoder_layers=1, encoder_attention_heads=2,
                          encoder_ffn_dim=32, d_model=16, output_dim=16,
                          downsample_hidden_size=4, n_window=50, n_window_infer=100),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                           out_hidden_size=16, deepstack_visual_indexes=[]),
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=8, moe_intermediate_size=16, num_experts=2,
                         num_experts_per_tok=1, pad_token_id=0, eos_token_id=2,
                         rope_scaling={'rope_type': 'default', 'mrope_section': [1, 1, 2]}))
    config._attn_implementation = 'sdpa'
    torch.manual_seed(7)
    backend = bare_backend()
    backend.thinker = transformers.Qwen3OmniMoeThinkerForConditionalGeneration(config)
    backend._bind_thinker((0., .5), .5)
    def tokenize(text, **kwargs):
        ids = [4, 5]
        return {'input_ids': torch.tensor([ids]) if kwargs.get('return_tensors') else ids}
    backend.processor = SimpleNamespace(tokenizer=tokenize,
        batch_decode=lambda *args, **kwargs: ['cat'])
    prefix = {'input_ids': torch.tensor([[1, 11, 10, 10, 12, 3]]),
              'attention_mask': torch.ones(1, 6, dtype=torch.long),
              'input_features': torch.randn(1, 8, 16),
              'feature_attention_mask': torch.ones(1, 16, dtype=torch.long)}
    backend._prepare = lambda *args: {k: v.clone() for k, v in prefix.items()}
    return backend


def test_real_tiny_audio_features_gradients_and_answer_loss(tiny_backend):
    backend = tiny_backend
    feature = backend.feature(None, 'classify')
    assert feature.shape == (16,) and np.isfinite(feature).all()
    gradients = backend.support_gradients(None, 'cat', 'classify')
    assert set(gradients) == {'audio', 'text', 'full_prefill'}
    assert all(v.shape == (32,) and np.isfinite(v).all() for v in gradients.values())
    assert np.linalg.norm(gradients['audio']) > 0
    baseline = backend.support_answer_loss(None, 'cat', 'classify')
    changed = backend.support_answer_loss(None, 'cat', 'classify',
                                          direction=gradients['audio'], alpha=.01)
    assert baseline['answer_tokens'] == changed['answer_tokens'] == 2
    assert len(changed['applied_norms']) == 4
    assert all(v['active_tokens'] == 2 for v in changed['applied_norms'].values())
    assert changed['answer_loss'] != baseline['answer_loss']
    assert all(p.grad is None and not p.requires_grad for p in backend.thinker.parameters())
    assert all(not module._forward_hooks for module in backend.modules.values())


def test_real_tiny_generate_applies_only_prefill(tiny_backend):
    backend = tiny_backend
    result = backend.predict(None, 'classify', ['cat', 'dog'],
                             direction=np.ones(32), alpha=.01, max_new_tokens=2)
    assert result['prediction'] == 'cat'
    assert 1 <= result['generated_tokens'] <= 2
    assert len(result['applied_norms']) == 4
    assert all(v['active_tokens'] == 2 for v in result['applied_norms'].values())


def test_actual_key_hook_is_before_normalization_and_rope(tiny_backend):
    backend = tiny_backend
    attn = backend.thinker.model.layers[0].self_attn
    hidden = torch.randn(1, 4, 16)
    observed = []
    handle = attn.k_norm.register_forward_pre_hook(lambda _m, args: observed.append(args[0].detach().clone()))
    original = attn.k_proj(hidden).view(1, 4, 1, 8).detach()
    mask = torch.tensor([[False, True, True, False]])
    try:
        with ProjectionIntervention({(0, 'k'): attn.k_proj}, {(0, 'k'): np.ones(8)},
                                    mask, 1., scale_mode='raw', raw_eta=.1) as hooks:
            attn(hidden, (torch.ones(1, 4, 8), torch.zeros(1, 4, 8)), None)
            hooks.assert_applied()
        expected = original.clone()
        expected[:, 1:3] += .1
        torch.testing.assert_close(observed[0], expected)
    finally:
        handle.remove()
