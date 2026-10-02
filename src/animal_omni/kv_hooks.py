from __future__ import annotations

from contextlib import AbstractContextManager


class KVDeltaHooks(AbstractContextManager):
    """Inject or differentiate additive pre-RoPE K/V projection states.

    The hooked tensors are the outputs of each Thinker layer's k_proj/v_proj,
    before head reshaping and rotary position encoding. This definition is kept
    explicit because it is not identical to Transformers' post-RoPE cache.
    """

    def __init__(
        self, thinker, deltas: dict[tuple[int, str], object] | None = None,
        learn: bool = False, relative_alpha: float | None = None,
    ):
        self.thinker = thinker
        self.fixed_deltas = deltas or {}
        self.learn = learn
        if relative_alpha is not None and relative_alpha < 0:
            raise ValueError("relative_alpha must be non-negative")
        self.relative_alpha = relative_alpha
        self.learned = {}
        self.applied_scales = {}
        self.applied_relative_norms = {}
        self.applied_scales_by_example = {}
        self.applied_relative_norms_by_example = {}
        self.handles = []

    def _hook(self, layer_index: int, kind: str):
        def apply(_module, _inputs, output):
            key = (layer_index, kind)
            if self.learn:
                delta = output.new_zeros(output.shape, requires_grad=True)
                self.learned[key] = delta
                return output + delta
            delta = self.fixed_deltas.get(key)
            # During autoregressive decode the sequence dimension becomes 1;
            # fixed Oracle deltas apply only to the matching prefill state.
            if delta is not None and tuple(delta.shape) == tuple(output.shape):
                delta = delta.to(device=output.device, dtype=output.dtype)
                if self.relative_alpha is not None:
                    import torch

                    # Normalize on the positions actually modified. Scaling K
                    # and V separately to the same ratio also gives that ratio
                    # for their joint Frobenius norm within each layer.
                    active = delta.float().square().sum(dim=-1) > 0
                    mask = active.unsqueeze(-1).float()
                    delta_norm = torch.sqrt(
                        (delta.float().square() * mask).sum(dim=(1, 2))
                    )
                    base_norm = torch.sqrt(
                        (output.float().square() * mask).sum(dim=(1, 2))
                    )
                    scale = torch.where(
                        delta_norm > 0,
                        self.relative_alpha * base_norm / delta_norm.clamp_min(1e-12),
                        torch.zeros_like(delta_norm),
                    )
                    delta = delta * scale[:, None, None].to(dtype=delta.dtype)
                    achieved = torch.sqrt(
                        (delta.float().square() * mask).sum(dim=(1, 2))
                    ) / base_norm.clamp_min(1e-12)
                    scales_cpu = [float(value) for value in scale.detach().cpu()]
                    ratios_cpu = [float(value) for value in achieved.detach().cpu()]
                    self.applied_scales_by_example[key] = scales_cpu
                    self.applied_relative_norms_by_example[key] = ratios_cpu
                    self.applied_scales[key] = float(sum(scales_cpu) / len(scales_cpu))
                    self.applied_relative_norms[key] = float(sum(ratios_cpu) / len(ratios_cpu))
                return output + delta
            return output
        return apply

    def __enter__(self):
        layers = self.thinker.model.layers
        for index, layer in enumerate(layers):
            self.handles.append(layer.self_attn.k_proj.register_forward_hook(self._hook(index, "k")))
            self.handles.append(layer.self_attn.v_proj.register_forward_hook(self._hook(index, "v")))
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


def label_kv_gradients(thinker, inputs) -> tuple[float, dict]:
    """Compute -dL/d(K,V) for one labeled support example with frozen weights."""
    import torch

    thinker.zero_grad(set_to_none=True)
    with torch.enable_grad(), KVDeltaHooks(thinker, learn=True) as hooks:
        outputs = thinker(**inputs, use_cache=False, return_dict=True)
        outputs.loss.backward()
        directions = {
            key: -delta.grad.detach().cpu() for key, delta in hooks.learned.items()
        }
    return float(outputs.loss.detach()), directions


def loss_with_kv_delta(thinker, inputs, directions: dict, eta: float) -> float:
    """Evaluate teacher-forced loss after adding eta * a precomputed direction."""
    import torch

    deltas = {key: value.mul(eta) for key, value in directions.items()}
    with torch.inference_mode(), KVDeltaHooks(thinker, deltas=deltas):
        outputs = thinker(**inputs, use_cache=False, return_dict=True)
    return float(outputs.loss.detach())


def pooled_audio_gradient(directions: dict, audio_mask) -> dict:
    """Mean-pool comparable K/V gradients over expanded audio-token positions."""
    mask = audio_mask.detach().cpu().bool()
    if mask.ndim != 2 or mask.shape[0] != 1 or not mask.any():
        raise ValueError("audio_mask must be [1, sequence] and contain audio tokens")
    pooled = {}
    for key, gradient in directions.items():
        # Teacher-forced gradients include answer tokens; mask selects only the
        # prefill audio positions and yields a fixed projection-width vector.
        if gradient.shape[:2] != mask.shape:
            raise ValueError(f"gradient/mask mismatch for {key}: {gradient.shape} vs {mask.shape}")
        pooled[key] = gradient[mask].float().mean(dim=0)
    return pooled


def tokenwise_audio_gradient(directions: dict, audio_mask) -> dict:
    """Keep the ordered gradient field over expanded audio-token positions."""
    mask = audio_mask.detach().cpu().bool()
    if mask.ndim != 2 or mask.shape[0] != 1 or not mask.any():
        raise ValueError("audio_mask must be [1, sequence] and contain audio tokens")
    tokenwise = {}
    for key, gradient in directions.items():
        if gradient.shape[:2] != mask.shape:
            raise ValueError(f"gradient/mask mismatch for {key}: {gradient.shape} vs {mask.shape}")
        tokenwise[key] = gradient[mask].float()
    return tokenwise
