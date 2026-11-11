"""Native Qwen3.5 MTP backend for the migrated OPSD baseline.

This is a text-only, batch-one research backend.  Its MTP policy emits a main-head
token followed by an *unverified* pretrained MTP-head token.  It deliberately
does not call speculative decoding: verifying every draft against the main head
would make the rollout policy NTP again.

Source contracts inspected for this implementation:
* transformers v5.16.1, commit 93c8b7b485963a10800c91f55304db6be211c2bd:
  modeling_qwen3_5.py and conversion_mapping.py.
* VeOmni 95f93e35f6ca5fae0b2994203a86b15b73e6a66b:
  qwen3_5_gpu_patch_gen_config.py, Qwen3_5MTP and MTP context.
* vLLM 92f3ee5ae4abb86f2b175b13e623dab4a21b0986:
  vllm/model_executor/models/qwen3_5_mtp.py.

The native decoder/attention implementations are imported from Transformers
(Apache-2.0).  No pretrained MTP tensor is initialized from a backbone layer or
silently discarded.  Local GPU qualification is required before scientific use.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn


MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TRANSFORMERS_VERSION = "5.16.1"
DEFAULT_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "fc",
)


@dataclass
class Generation:
    text: str
    token_ids: list[int]
    finish_reason: str
    head_counts: dict[str, int]
    prompt_tokens: int
    elapsed_seconds: float
    raw_logprobs: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LoRALinear(nn.Module):
    """W x + (alpha / rank) B A x; base weights and output head stay frozen.

    The small adapters use FP32 optimizer state.  This adapter format is owned by
    this baseline; it is not advertised as a PEFT or merged HF checkpoint.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scale = self.alpha / rank
        self.lora_A = nn.Parameter(torch.empty(
            rank, base.in_features, device=base.weight.device, dtype=torch.float32,
        ))
        self.lora_B = nn.Parameter(torch.zeros(
            base.out_features, rank, device=base.weight.device, dtype=torch.float32,
        ))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        update = F.linear(F.linear(x.float(), self.lora_A), self.lora_B)
        return base + (self.scale * update).to(base.dtype)


class NativeMTPHead(nn.Module):
    """The released conditional next-next-token module, with its own KV cache.

    Row t combines normalized main hidden h[t] and embedding x[t+1].  Its output
    predicts x[t+2].  The causal attention sees only those rows <= t.  RoPE uses
    origin positions t; shifting every MTP position by one has the same relative
    positions and follows VeOmni's full-sequence objective convention.
    """

    def __init__(self, text_config):
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding,
        )

        if int(getattr(text_config, "mtp_num_hidden_layers", 0)) != 1:
            raise ValueError("This pinned backend supports exactly one released MTP layer (two-token policy).")
        if getattr(text_config, "mtp_use_dedicated_embeddings", False):
            raise ValueError("Dedicated MTP embeddings are outside this checkpoint's contract.")
        self.config = copy.deepcopy(text_config)
        # Cache indexing is local to the MTP stack, rather than backbone layer 3.
        self.config.num_hidden_layers = 1
        self.config.layer_types = ["full_attention"]
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(text_config.hidden_size, text_config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(text_config.hidden_size, text_config.rms_norm_eps)
        self.fc = nn.Linear(text_config.hidden_size * 2, text_config.hidden_size, bias=False)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(self.config, 0)])
        self.norm = Qwen3_5RMSNorm(text_config.hidden_size, text_config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(self.config)

    def forward(
        self,
        main_hidden: torch.Tensor,
        next_embeddings: torch.Tensor,
        *,
        past_key_values=None,
        use_cache: bool = False,
        start_position: int = 0,
    ):
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask

        if main_hidden.shape != next_embeddings.shape:
            raise ValueError("MTP hidden states and shifted embeddings must have identical shapes.")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        hidden = self.fc(torch.cat((
            self.pre_fc_norm_embedding(next_embeddings),
            self.pre_fc_norm_hidden(main_hidden),
        ), dim=-1))
        positions = torch.arange(
            start_position, start_position + hidden.shape[1], device=hidden.device,
        )[None, :].expand(hidden.shape[0], -1)
        mask = create_causal_mask(
            config=self.config, inputs_embeds=hidden, attention_mask=None,
            past_key_values=past_key_values, position_ids=positions,
        )
        hidden = self.layers[0](
            hidden, position_embeddings=self.rotary_emb(hidden, positions),
            attention_mask=mask, position_ids=positions,
            past_key_values=past_key_values, use_cache=use_cache,
        )
        return self.norm(hidden), past_key_values


class _NativeModel(nn.Module):
    def __init__(self, base, mtp: NativeMTPHead):
        super().__init__()
        self.base = base
        self.mtp = mtp


def _load_native_mtp(path: Path, text_config, *, device, dtype):
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing official sharded checkpoint index: {index_path}")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    native = {name: shard for name, shard in weight_map.items() if name.startswith("mtp.")}
    head = NativeMTPHead(text_config)
    expected = {f"mtp.{name}" for name in head.state_dict()}
    if set(native) != expected:
        raise RuntimeError(
            "Native MTP checkpoint mismatch; no random-head fallback is permitted. "
            f"missing={sorted(expected - set(native))}, unexpected={sorted(set(native) - expected)}"
        )
    state = {}
    for shard in sorted(set(native.values())):
        shard_path = (path / shard).resolve()
        if not shard_path.is_relative_to(path.resolve()) or not shard_path.is_file():
            raise FileNotFoundError(f"Missing or invalid MTP shard: {shard}")
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for name in sorted(name for name, value in native.items() if value == shard):
                state[name.removeprefix("mtp.")] = handle.get_tensor(name)
    head.load_state_dict(state, strict=True)
    del state
    head.to(device=device, dtype=dtype)
    identity = {
        "native_mtp_loaded": True,
        "native_mtp_tensor_count": len(expected),
        "native_mtp_tensor_names": sorted(expected),
        "checkpoint_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    }
    return head, identity


def _inject_lora(model: nn.Module, spec: Mapping[str, Any]) -> list[str]:
    rank, alpha = int(spec.get("rank", 16)), float(spec.get("alpha", 32))
    if rank <= 0 or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive.")
    if float(spec.get("dropout", 0.0)) != 0.0:
        raise ValueError("This self-teacher baseline fixes LoRA dropout=0 to avoid teacher/student dropout confounds.")
    targets = set(spec.get("target_modules", DEFAULT_LORA_TARGETS))
    selected = [
        (name, layer) for name, layer in model.named_modules()
        if isinstance(layer, nn.Linear) and name.rsplit(".", 1)[-1] in targets
        and not name.endswith("lm_head")
    ]
    if not selected:
        raise ValueError("LoRA targets did not match any linear modules.")
    for name, layer in selected:
        parent_name, leaf = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), leaf, LoRALinear(layer, rank, alpha))
    if not any(name.startswith("mtp.") for name, _ in selected):
        raise ValueError("LoRA targets must include native MTP parameters for the MTP training arm.")
    return [name for name, _ in selected]


class NativeMTPBackend:
    def __init__(self, model, tokenizer, config, device, provenance):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(device)
        self.provenance = provenance
        self.max_context_length = min(
            int(config.get("max_context_tokens", 8192)),
            int(model.base.config.max_position_embeddings),
        )
        self.counters: Counter[str] = Counter()
        eos = model.base.config.eos_token_id
        self.eos_token_ids = set(eos if isinstance(eos, list) else [eos])
        generation_eos = getattr(model.base.generation_config, "eos_token_id", None)
        if generation_eos is not None:
            self.eos_token_ids.update(generation_eos if isinstance(generation_eos, list) else [generation_eos])
        if tokenizer.eos_token_id is not None:
            self.eos_token_ids.add(tokenizer.eos_token_id)
        self.eos_token_ids.discard(None)

    @classmethod
    def from_pretrained(
        cls, config: Mapping[str, Any], *, adapter_dir=None, trainable: bool = True, device=None,
    ) -> "NativeMTPBackend":
        import transformers
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise RuntimeError(
                f"Backend source contract requires transformers=={TRANSFORMERS_VERSION}; "
                f"found {transformers.__version__}. Use this branch's isolated environment."
            )
        spec = copy.deepcopy(dict(config.get("model", config)))
        if spec.get("id", MODEL_ID) != MODEL_ID or spec.get("revision", MODEL_REVISION) != MODEL_REVISION:
            raise ValueError("Only the pinned official Qwen3.5-4B snapshot is qualified for this baseline backend.")
        path = Path(spec["path"]).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Acquire and verify the pinned complete model snapshot first: {path}")
        manifest_path = path / "model_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Missing pinned model_manifest.json; use this branch's asset acquisition command.")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("repo_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
            raise ValueError("Local model manifest identity does not match the required official revision.")
        for filename in ("config.json", "model.safetensors.index.json"):
            expected_hash = manifest.get("files", {}).get(filename, {}).get("sha256")
            actual_hash = hashlib.sha256((path / filename).read_bytes()).hexdigest()
            if not expected_hash or actual_hash != expected_hash:
                raise ValueError(f"Local model manifest hash mismatch: {filename}")
        dtype_name = spec.get("dtype", "bfloat16")
        if dtype_name not in {"bfloat16", "float32"}:
            raise ValueError("Use bfloat16 on supported GPUs or float32 for qualification; no implicit quantization.")
        dtype = getattr(torch, dtype_name)
        device = torch.device(device or spec.get("device", "cuda"))
        config_path = path / "config.json"
        raw_config = json.loads(config_path.read_text())
        if raw_config.get("model_type") != "qwen3_5":
            raise ValueError("The local snapshot is not the expected Qwen3.5 checkpoint.")
        text_config = Qwen3_5TextConfig.from_pretrained(str(path), local_files_only=True)
        if int(text_config.hidden_size) != 2560 or int(text_config.num_hidden_layers) != 32:
            raise ValueError("Checkpoint dimensions do not match the pinned Qwen3.5-4B model.")
        attention = spec.get("attn_implementation", "sdpa")
        if attention not in {"sdpa", "eager"}:
            raise ValueError("This backend source contract supports sdpa/eager only.")
        base, loading = Qwen3_5ForCausalLM.from_pretrained(
            str(path), config=text_config, dtype=dtype,
            device_map={"": str(device)}, attn_implementation=attention,
            local_files_only=True, output_loading_info=True,
        )
        unexpected = [
            key for key in loading.get("unexpected_keys", [])
            if not key.startswith(("mtp.", "model.visual."))
        ]
        if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs") or unexpected:
            raise RuntimeError(f"Text-backbone checkpoint load was not exact: {loading}")
        # Upstream knows the multimodal -> text prefix conversion, but ignores MTP.
        # Load every native MTP tensor independently and strictly before training.
        mtp, mtp_identity = _load_native_mtp(path, base.config, device=device, dtype=dtype)
        model = _NativeModel(base, mtp)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        adapter_spec = None
        adapter_config_sha256 = None
        adapter_weights_sha256 = None
        if adapter_dir is not None:
            adapter_config_bytes = (Path(adapter_dir) / "adapter_config.json").read_bytes()
            adapter_config_sha256 = hashlib.sha256(adapter_config_bytes).hexdigest()
            adapter_spec = json.loads(adapter_config_bytes)
            if adapter_spec.get("format") != "akr_native_mtp_lora_v1":
                raise ValueError("Unsupported adapter format.")
            if adapter_spec.get("base_model_id") != MODEL_ID or adapter_spec.get("base_revision") != MODEL_REVISION:
                raise ValueError("Adapter base model identity does not match this pinned backend.")
            lora = adapter_spec["lora"]
        else:
            lora = spec.get("lora", {"enabled": False})
        lora_names = _inject_lora(model, lora) if lora.get("enabled", True) else []
        if adapter_dir is not None:
            adapter_weights = Path(adapter_dir) / "adapter.safetensors"
            adapter_weights_sha256 = hashlib.sha256(adapter_weights.read_bytes()).hexdigest()
            if adapter_weights_sha256 != adapter_spec.get("weights_sha256"):
                raise ValueError("Adapter weights hash mismatch.")
            tensors = load_file(str(adapter_weights), device="cpu")
            expected = {name for name, _ in model.named_parameters() if name.endswith(("lora_A", "lora_B"))}
            if set(tensors) != expected or sorted(lora_names) != sorted(adapter_spec["target_modules_resolved"]):
                raise ValueError("Adapter keys or resolved modules mismatch; refusing partial adapter restore.")
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in tensors:
                        if tensors[name].shape != parameter.shape:
                            raise ValueError(f"Adapter shape mismatch: {name}")
                        parameter.copy_(tensors[name].to(device=parameter.device, dtype=parameter.dtype))
        if not trainable:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        elif not lora_names:
            raise ValueError("Training requires enabled LoRA adapters; native evaluation uses trainable=False.")
        if trainable and bool(spec.get("gradient_checkpointing", True)):
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        spec["lora"] = dict(lora)
        provenance = {
            "backend": "native_qwen35_two_token_v1", "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION, "transformers_version": transformers.__version__,
            "model_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "model_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "adapter_config_sha256": adapter_config_sha256,
            "adapter_weights_sha256": adapter_weights_sha256,
            "rollout_mtp_policy": "main_then_unverified_native_mtp",
            "mtp_horizon": 2, "lora_modules": lora_names,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            **mtp_identity,
        }
        return cls(model, tokenizer, spec, device, provenance)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.model.parameters() if parameter.requires_grad)

    def reset_counters(self):
        self.counters.clear()

    def make_prompt(self, question: str, reference: str | None = None) -> list[int]:
        system = "Solve the mathematics problem. Explain your reasoning and put the final answer in \\boxed{...}."
        if reference is not None:
            system += (
                "\nA candidate solution sampled by this same model is available as a reference. "
                "It is a majority-selected candidate, not a verified ground truth. "
                "Use it to guide continuation of the assistant's existing response.\n"
                "<reference_solution>\n" + reference + "\n</reference_solution>"
            )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        tokens = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=bool(self.config.get("enable_thinking", False)),
        )
        return list(tokens)

    def _ids(self, input_ids) -> torch.Tensor:
        if isinstance(input_ids, torch.Tensor):
            tokens = input_ids.to(device=self.device, dtype=torch.long)
        else:
            tokens = torch.tensor(input_ids, device=self.device, dtype=torch.long)
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] == 0:
            raise ValueError("Native baseline operates on one nonempty, unpadded sequence at a time.")
        if tokens.shape[1] > self.max_context_length:
            raise ValueError("Sequence exceeds configured context; it must not be silently truncated.")
        return tokens

    def prepare_hidden(self, input_ids, include_mtp: bool = False) -> dict[str, torch.Tensor]:
        """Return all-prefix states without allocating sequence-by-vocabulary logits.

        head0[:, t] predicts input_ids[:, t+1].
        head1[:, t] conditions on input_ids[:, t+1] and predicts input_ids[:, t+2].
        Teacher and student append *identical sampled response IDs* to their own
        prompts, then index these states using their respective prompt lengths.
        """
        tokens = self._ids(input_ids)
        self.counters["backbone_forward"] += 1
        hidden = self.model.base.model(input_ids=tokens, use_cache=False).last_hidden_state
        result = {"head0": hidden}
        if include_mtp:
            if tokens.shape[1] < 2:
                raise ValueError("MTP hidden-state computation needs a shifted input token.")
            self.counters["mtp_forward"] += 1
            mtp_hidden, _ = self.model.mtp(
                hidden[:, :-1, :], self.model.base.model.embed_tokens(tokens[:, 1:]),
            )
            result["head1"] = mtp_hidden
        return result

    def project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.base.lm_head(hidden)

    def _sample(self, logits, *, temperature: float, top_p: float, generator) -> int:
        if temperature < 0 or not 0 < top_p <= 1:
            raise ValueError("Require temperature >= 0 and 0 < top_p <= 1.")
        # Clone so top-p filtering cannot mutate the raw logits used by the
        # optional cached-vs-full-sequence qualification trace.
        logits = logits.float().reshape(-1).clone()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite generation logits.")
        if temperature == 0:
            return int(logits.argmax().item())
        logits = logits / temperature
        if top_p < 1:
            ordered, indices = logits.sort(descending=True)
            remove = ordered.softmax(-1).cumsum(-1) > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            logits[indices[remove]] = -torch.inf
        return int(torch.multinomial(logits.softmax(-1), 1, generator=generator).item())

    @torch.no_grad()
    def generate(
        self,
        prompt: str | Sequence[int],
        *,
        mode: str = "mtp",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        seed: int | None = None,
        stop: Sequence[str] = (),
        record_logprobs: bool = False,
    ) -> Generation:
        if mode not in {"ntp", "mtp"}:
            raise ValueError("Generation mode must be ntp or mtp.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt, add_special_tokens=False)
        tokens = self._ids(prompt)
        if tokens.shape[1] + max_new_tokens > self.max_context_length:
            raise ValueError("Prompt plus generation budget exceeds configured context; lower the explicit budget.")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
        training = self.model.training
        self.model.eval()
        started = time.monotonic()
        generated: list[int] = []
        raw_logprobs = [] if record_logprobs else None
        counts = {"head0": 0, "head1": 0}
        finish = "length"
        stops = tuple(item for item in stop if item)

        def should_stop(token):
            nonlocal finish
            if token in self.eos_token_ids:
                finish = "eos"
                return True
            if stops:
                current_text = self.tokenizer.decode(generated, skip_special_tokens=True)
                if any(marker in current_text for marker in stops):
                    finish = "stop"
                    return True
            return False

        try:
            self.counters["backbone_forward"] += 1
            base_output = self.model.base.model(input_ids=tokens, use_cache=True)
            base_cache = base_output.past_key_values
            pending_hidden = base_output.last_hidden_state
            mtp_cache = None
            mtp_processed = 0
            previous_second = None
            while len(generated) < max_new_tokens:
                first_logits = self.project_logits(pending_hidden[:, -1, :])
                first = self._sample(
                    first_logits,
                    temperature=temperature, top_p=top_p, generator=generator,
                )
                generated.append(first)
                counts["head0"] += 1
                if raw_logprobs is not None:
                    raw_logprobs.append(float(first_logits.float().log_softmax(-1)[0, first].item()))
                if should_stop(first) or len(generated) == max_new_tokens:
                    break
                consumed = [first]
                if mode == "mtp":
                    first_tensor = torch.tensor([[first]], device=self.device)
                    if mtp_cache is None:
                        shifted_ids = torch.cat((tokens[:, 1:], first_tensor), dim=1)
                    else:
                        shifted_ids = torch.tensor([[previous_second, first]], device=self.device)
                    self.counters["mtp_forward"] += 1
                    mtp_hidden, mtp_cache = self.model.mtp(
                        pending_hidden, self.model.base.model.embed_tokens(shifted_ids),
                        past_key_values=mtp_cache, use_cache=True, start_position=mtp_processed,
                    )
                    mtp_processed += shifted_ids.shape[1]
                    second_logits = self.project_logits(mtp_hidden[:, -1, :])
                    second = self._sample(
                        second_logits,
                        temperature=temperature, top_p=top_p, generator=generator,
                    )
                    generated.append(second)
                    counts["head1"] += 1
                    if raw_logprobs is not None:
                        raw_logprobs.append(float(second_logits.float().log_softmax(-1)[0, second].item()))
                    consumed.append(second)
                    previous_second = second
                    if should_stop(second) or len(generated) == max_new_tokens:
                        break
                self.counters["backbone_forward"] += 1
                base_output = self.model.base.model(
                    input_ids=torch.tensor([consumed], device=self.device),
                    past_key_values=base_cache, use_cache=True,
                )
                base_cache = base_output.past_key_values
                pending_hidden = base_output.last_hidden_state
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if finish == "stop":
                text = text[:min(text.index(marker) for marker in stops if marker in text)]
            self.counters["generated_head0_tokens"] += counts["head0"]
            self.counters["generated_head1_tokens"] += counts["head1"]
            return Generation(
                text=text, token_ids=generated, finish_reason=finish, head_counts=counts,
                prompt_tokens=tokens.shape[1], elapsed_seconds=time.monotonic() - started,
                raw_logprobs=raw_logprobs,
            )
        finally:
            self.model.train(training)

    def save_adapter(self, directory) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tensors = {
            name: value.detach().float().cpu().contiguous()
            for name, value in self.model.named_parameters()
            if name.endswith(("lora_A", "lora_B"))
        }
        if not tensors:
            raise ValueError("There are no adapters to save.")
        weights_path = directory / "adapter.safetensors"
        if weights_path.exists() or (directory / "adapter_config.json").exists():
            raise FileExistsError("Adapter checkpoint directory is already populated; do not overwrite it.")
        save_file(tensors, str(weights_path))
        metadata = {
            "format": "akr_native_mtp_lora_v1", "base_model_id": MODEL_ID,
            "base_revision": MODEL_REVISION, "backend": self.provenance["backend"],
            "lora": self.config["lora"],
            "target_modules_resolved": self.provenance["lora_modules"],
            "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
            "native_mtp": "All base MTP tensors are restored strictly from the pinned official checkpoint.",
        }
        (directory / "adapter_config.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_backend(model_config: Mapping[str, Any], *, adapter_dir=None, trainable=True, device=None):
    return NativeMTPBackend.from_pretrained(
        model_config, adapter_dir=adapter_dir, trainable=trainable, device=device,
    )
