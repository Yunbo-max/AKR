import torch
from torch import nn

from animal_omni.kv_hooks import KVDeltaHooks, pooled_audio_gradient


class Attention(nn.Module):
    def __init__(self):
        super().__init__(); self.k_proj = nn.Linear(3, 2); self.v_proj = nn.Linear(3, 2)
    def forward(self, x): return self.k_proj(x) + self.v_proj(x)


class Layer(nn.Module):
    def __init__(self): super().__init__(); self.self_attn = Attention()


class Model(nn.Module):
    def __init__(self): super().__init__(); self.layers = nn.ModuleList([Layer()])


class Thinker(nn.Module):
    def __init__(self): super().__init__(); self.model = Model()
    def forward(self, x): return self.model.layers[0].self_attn(x)


def test_learned_and_fixed_projection_deltas():
    thinker = Thinker(); x = torch.randn(1, 4, 3)
    baseline = thinker(x)
    with KVDeltaHooks(thinker, learn=True) as hooks:
        output = thinker(x); output.sum().backward()
        assert hooks.learned[(0, "k")].grad.shape == (1, 4, 2)
    delta = torch.ones(1, 4, 2)
    with KVDeltaHooks(thinker, {(0, "k"): delta}):
        assert torch.allclose(thinker(x), baseline + 1)
    # A prefill-shaped delta is ignored for a length-one decoding step.
    with KVDeltaHooks(thinker, {(0, "k"): delta}):
        assert torch.allclose(thinker(x[:, :1]), baseline[:, :1])


def test_pool_gradient_only_over_audio_positions():
    gradient = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    mask = torch.tensor([[False, True, True, False]])
    pooled = pooled_audio_gradient({(0, "k"): gradient}, mask)
    assert torch.equal(pooled[(0, "k")], torch.tensor([4.5, 5.5, 6.5]))


def test_relative_delta_has_requested_active_state_norm():
    thinker = Thinker(); x = torch.randn(1, 4, 3)
    delta = torch.zeros(1, 4, 2); delta[:, 1:3] = torch.randn(1, 2, 2)
    baseline_k = thinker.model.layers[0].self_attn.k_proj(x)
    with KVDeltaHooks(thinker, {(0, "k"): delta}, relative_alpha=.03) as hooks:
        thinker(x)
    active = delta.square().sum(-1) > 0
    applied = delta * hooks.applied_scales[(0, "k")]
    ratio = applied[active].norm() / baseline_k[active].norm()
    assert torch.allclose(ratio, torch.tensor(.03), atol=1e-5)
    assert abs(hooks.applied_relative_norms[(0, "k")] - .03) < 1e-5


def test_relative_delta_normalizes_each_batch_example_independently():
    thinker = Thinker(); x = torch.randn(2, 4, 3)
    delta = torch.zeros(2, 4, 2)
    delta[0, 1:3] = 1.; delta[1, 1:3] = 10.
    baseline_k = thinker.model.layers[0].self_attn.k_proj(x)
    with KVDeltaHooks(thinker, {(0, "k"): delta}, relative_alpha=.01) as hooks:
        thinker(x)
    scales = torch.tensor(hooks.applied_scales_by_example[(0, "k")])[:, None, None]
    applied = delta * scales
    active = delta.square().sum(-1) > 0
    for index in range(2):
        ratio = applied[index][active[index]].norm() / baseline_k[index][active[index]].norm()
        assert torch.allclose(ratio, torch.tensor(.01), atol=1e-5)
