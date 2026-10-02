import numpy as np
import torch

from animal_omni.conditional_kv import (
    ConditionalGradientRouter, ConditionalLocalTokenRouter, ConditionalTokenGradientRouter,
    broadcast_audio_delta, flatten_gradient, flatten_token_gradient,
    place_audio_token_delta, unflatten_gradient, unflatten_token_gradient,
)


def test_flatten_roundtrip_and_broadcast():
    gradient = {(1, "v"): torch.tensor([3., 4.]), (0, "k"): torch.tensor([1., 2.])}
    vector, keys = flatten_gradient(gradient)
    restored = unflatten_gradient(vector, keys, 2)
    assert keys == [(0, "k"), (1, "v")]
    assert torch.equal(restored[(1, "v")], gradient[(1, "v")])
    delta = broadcast_audio_delta(restored, torch.tensor([[False, True, False]]), .5)
    assert tuple(delta[(0, "k")].shape) == (1, 3, 2)
    assert torch.equal(delta[(0, "k")][0, 1], torch.tensor([.5, 1.]))
    assert not delta[(0, "k")][0, 0].any()


def test_router_learns_support_mapping_and_k1_is_mean():
    x = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=np.float32)
    g = np.c_[x, x[:, :1] + x[:, 1:]].astype(np.float32)
    router = ConditionalGradientRouter.fit(x, g, rank=3, alpha=1e-6)
    assert np.allclose(router.predict(x[3]), g[3], atol=1e-4)
    one = ConditionalGradientRouter.fit(x[:1], g[:1], rank=4, alpha=1.)
    assert one.rank == 0
    assert np.array_equal(one.predict(np.array([9., 9.])), g[0])


def test_token_gradient_roundtrip_and_placement():
    gradient = {
        (0, "k"): torch.tensor([[1., 2.], [3., 4.]]),
        (1, "v"): torch.tensor([[5., 6.], [7., 8.]]),
    }
    vector, keys, token_count, width = flatten_token_gradient(gradient)
    restored = unflatten_token_gradient(vector, keys, token_count, width)
    assert torch.equal(restored[(1, "v")], gradient[(1, "v")])
    mask = torch.tensor([[False, True, False, True, False]])
    delta = place_audio_token_delta(restored, mask, .5)
    assert tuple(delta[(0, "k")].shape) == (1, 5, 2)
    assert torch.equal(delta[(0, "k")][0, 1], torch.tensor([.5, 1.]))
    assert torch.equal(delta[(0, "k")][0, 3], torch.tensor([1.5, 2.]))


def test_token_router_uses_wide_gram_svd():
    x = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=np.float32)
    base = np.c_[x, x.sum(1)].astype(np.float32)
    gradients = np.tile(base, (1, 100))
    router = ConditionalTokenGradientRouter.fit(x, gradients, rank=3, alpha=1e-6)
    assert router.rank <= 3
    assert np.allclose(router.predict(x[3]), gradients[3], atol=2e-3)


def test_local_token_router_predicts_aligned_fields():
    rng = np.random.default_rng(7)
    tokens = rng.normal(size=(4, 3, 5)).astype(np.float32)
    global_features = tokens.mean(1)
    gradients = []
    for event in range(4):
        base = tokens[event, :, :2] + global_features[event, :2]
        gradients.append({
            (0, "k"): torch.from_numpy(base.astype(np.float32)),
            (0, "v"): torch.from_numpy((2 * base).astype(np.float32)),
        })
    router = ConditionalLocalTokenRouter.fit(
        tokens, global_features, gradients, local_rank=4, global_rank=3,
        gradient_rank=2, alpha=1e-5, seed=3,
    )
    prediction = router.predict(tokens[2], global_features[2])
    assert prediction[(0, "k")].shape == (3, 2)
    assert np.isfinite(prediction[(0, "v")]).all()
    fixed = router.fixed_mean(3)
    assert fixed[(0, "k")].shape == (3, 2)
    assert np.allclose(fixed[(0, "k")][0], fixed[(0, "k")][2])
