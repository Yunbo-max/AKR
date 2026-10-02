from __future__ import annotations

from pathlib import Path

import numpy as np


class QwenThinkerRunner:
    """Thin, lazy-loading wrapper around the official Transformers interface."""

    def __init__(self, model_id: str, *, device_map: str = "auto", dtype: str = "bfloat16"):
        try:
            import torch
            from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        except ImportError as exc:
            raise RuntimeError("install the qwen extra: pip install -e '.[qwen]'") from exc
        torch_dtype = getattr(torch, dtype)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device_map
        )
        self.model.disable_talker()
        self.model.eval()

    def prepare_inputs(self, audio_path: str | Path, prompt: str):
        from qwen_omni_utils import process_mm_info

        audio_path = str(Path(audio_path).resolve())
        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio": audio_path}, {"type": "text", "text": prompt}
        ]}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False,
        )
        return inputs.to(self.model.device).to(self.model.dtype)

    def prepare_batch(self, audio_paths: list[str | Path], prompt: str):
        from qwen_omni_utils import process_mm_info

        conversations = [[{"role": "user", "content": [
            {"type": "audio", "audio": str(Path(path).resolve())},
            {"type": "text", "text": prompt},
        ]}] for path in audio_paths]
        texts = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
        inputs = self.processor(
            text=texts, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False,
        )
        return inputs.to(self.model.device).to(self.model.dtype)

    def prepare_icl_inputs(
        self,
        support_examples: list[tuple[str | Path, str]],
        query_audio_path: str | Path,
        prompt: str,
        *,
        support_prompt: str | None = None,
    ):
        """Build a multi-turn audio ICL conversation with labeled support."""
        from qwen_omni_utils import process_mm_info

        example_prompt = support_prompt or prompt
        conversation = []
        for audio_path, label in support_examples:
            conversation.extend([
                {"role": "user", "content": [
                    {"type": "audio", "audio": str(Path(audio_path).resolve())},
                    {"type": "text", "text": example_prompt},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": label},
                ]},
            ])
        conversation.append({"role": "user", "content": [
            {"type": "audio", "audio": str(Path(query_audio_path).resolve())},
            {"type": "text", "text": prompt},
        ]})
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False,
        )
        return inputs.to(self.model.device).to(self.model.dtype)

    def predict_icl(
        self,
        support_examples: list[tuple[str | Path, str]],
        query_audio_path: str | Path,
        prompt: str,
        *,
        support_prompt: str | None = None,
        max_new_tokens: int = 8,
    ) -> str:
        inputs = self.prepare_icl_inputs(
            support_examples, query_audio_path, prompt, support_prompt=support_prompt
        )
        input_length = inputs["input_ids"].shape[1]
        output = self.model.generate(
            **inputs, return_audio=False, do_sample=False, max_new_tokens=max_new_tokens,
            use_audio_in_video=False,
        )
        return self.processor.batch_decode(
            output[:, input_length:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def score_candidates_icl(
        self,
        support_examples: list[tuple[str | Path, str]],
        query_audio_path: str | Path,
        prompt: str,
        candidates: list[str],
        *,
        support_prompt: str | None = None,
    ) -> list[dict[str, object]]:
        """Score candidates after a multi-audio ICL prefix.

        The processor/audio prefix is built once. Candidates are evaluated one
        at a time because multimodal feature tensors encode all support clips
        inside a single conversation rather than a conventional batch axis.
        """
        import torch

        base = self.prepare_icl_inputs(
            support_examples, query_audio_path, prompt, support_prompt=support_prompt
        )
        tokenizer = self.processor.tokenizer
        prompt_width = int(base["input_ids"].shape[1])
        records = []
        for candidate in candidates:
            ids = tokenizer(
                candidate, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(base["input_ids"].device)
            inputs = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in base.items()
            }
            inputs["input_ids"] = torch.cat([inputs["input_ids"], ids], dim=1)
            inputs["attention_mask"] = torch.cat([
                inputs["attention_mask"], torch.ones_like(ids)
            ], dim=1)
            with torch.inference_mode():
                outputs = self.model.thinker(**inputs, use_cache=False, return_dict=True)
            positions = slice(prompt_width - 1, prompt_width + ids.shape[1] - 1)
            logits = outputs.logits[0, positions].float()
            token_logprobs = torch.log_softmax(logits, dim=-1).gather(
                1, ids[0, :, None]
            )[:, 0]
            sequence = float(token_logprobs.sum().cpu())
            records.append({
                "candidate": candidate,
                "token_ids": [int(value) for value in ids[0].cpu()],
                "token_count": int(ids.shape[1]),
                "sequence_logprob": sequence,
                "mean_token_logprob": sequence / ids.shape[1],
                "token_logprobs": [float(value) for value in token_logprobs.cpu()],
            })
            del outputs, inputs
        return records

    def predict(self, audio_path: str | Path, prompt: str, *, max_new_tokens: int = 8) -> str:
        inputs = self.prepare_inputs(audio_path, prompt)
        input_length = inputs["input_ids"].shape[1]
        output = self.model.generate(
            **inputs, return_audio=False, do_sample=False, max_new_tokens=max_new_tokens,
            use_audio_in_video=False,
        )
        # Decode generated continuation only; prompt labels would otherwise cause
        # ambiguous multi-label parsing.
        generated = output[:, input_length:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def predict_batch(self, audio_paths: list[str | Path], prompt: str, *, max_new_tokens: int = 8) -> list[str]:
        inputs = self.prepare_batch(audio_paths, prompt)
        input_length = inputs["input_ids"].shape[1]
        output = self.model.generate(
            **inputs, return_audio=False, do_sample=False, max_new_tokens=max_new_tokens,
            use_audio_in_video=False,
        )
        decoded = self.processor.batch_decode(
            output[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return [text.strip() for text in decoded]

    def score_candidates(
        self,
        audio_path: str | Path,
        prompt: str,
        candidates: list[str],
        *,
        candidate_batch_size: int = 6,
    ) -> list[dict[str, object]]:
        """Score exact answer strings without free-form decoding.

        Each returned record contains the causal sequence log likelihood and
        its token-length-normalized counterpart.  Candidate batches repeat the
        same audio/prompt so that the expensive audio forward pass is shared;
        no candidate label is exposed in the prompt beyond the task's fixed
        candidate list.
        """
        import torch

        if not candidates:
            raise ValueError("candidates must not be empty")
        if candidate_batch_size <= 0:
            raise ValueError("candidate_batch_size must be positive")
        tokenizer = self.processor.tokenizer
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        records: list[dict[str, object]] = []
        for start in range(0, len(candidates), candidate_batch_size):
            batch = candidates[start:start + candidate_batch_size]
            inputs = self.prepare_batch([audio_path] * len(batch), prompt)
            prompt_width = int(inputs["input_ids"].shape[1])
            answer_ids = [
                tokenizer(candidate, add_special_tokens=False, return_tensors="pt")
                ["input_ids"][0].to(inputs["input_ids"].device)
                for candidate in batch
            ]
            if any(len(ids) == 0 for ids in answer_ids):
                raise ValueError("candidate tokenization produced an empty answer")
            answer_width = max(len(ids) for ids in answer_ids)
            suffix_ids = torch.full(
                (len(batch), answer_width), pad_token_id,
                dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device,
            )
            suffix_mask = torch.zeros(
                (len(batch), answer_width), dtype=inputs["attention_mask"].dtype,
                device=inputs["attention_mask"].device,
            )
            for index, ids in enumerate(answer_ids):
                suffix_ids[index, :len(ids)] = ids
                suffix_mask[index, :len(ids)] = 1
            inputs["input_ids"] = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
            inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"], suffix_mask], dim=1
            )
            with torch.inference_mode():
                outputs = self.model.thinker(
                    **inputs, use_cache=False, return_dict=True
                )
            for index, (candidate, ids) in enumerate(zip(batch, answer_ids)):
                # Causal logits at prompt_width-1 predict the first answer token.
                positions = slice(prompt_width - 1, prompt_width + len(ids) - 1)
                logits = outputs.logits[index, positions].float()
                token_logprobs = torch.log_softmax(logits, dim=-1).gather(
                    1, ids[:, None]
                )[:, 0]
                sequence = float(token_logprobs.sum().cpu())
                records.append({
                    "candidate": candidate,
                    "token_ids": [int(value) for value in ids.cpu()],
                    "token_count": int(len(ids)),
                    "sequence_logprob": sequence,
                    "mean_token_logprob": sequence / len(ids),
                    "token_logprobs": [float(value) for value in token_logprobs.cpu()],
                })
            del outputs, inputs
        return records

    def extract_audio_representations(self, audio_path: str | Path, prompt: str) -> np.ndarray:
        """Return layerwise mean-pooled hidden states at audio placeholder tokens."""
        import torch

        inputs = self.prepare_inputs(audio_path, prompt)
        audio_token_id = self.model.thinker.config.audio_token_id
        mask = inputs["input_ids"].eq(audio_token_id)
        if not mask.any():
            raise RuntimeError("processor produced no audio placeholder tokens")
        with torch.inference_mode():
            outputs = self.model.thinker(**inputs, use_cache=False, output_hidden_states=True, return_dict=True)
        pooled = [state[mask].mean(dim=0).float().cpu().numpy() for state in outputs.hidden_states]
        return np.stack(pooled)

    def extract_audio_representations_with_tokens(
        self, audio_path: str | Path, prompt: str, tokenwise_layer: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return all pooled levels plus ordered audio tokens at one level."""
        import torch

        inputs = self.prepare_inputs(audio_path, prompt)
        audio_token_id = self.model.thinker.config.audio_token_id
        mask = inputs["input_ids"].eq(audio_token_id)
        if not mask.any():
            raise RuntimeError("processor produced no audio placeholder tokens")
        with torch.inference_mode():
            outputs = self.model.thinker(
                **inputs, use_cache=False, output_hidden_states=True, return_dict=True
            )
        levels = outputs.hidden_states
        if not -len(levels) <= tokenwise_layer < len(levels):
            raise IndexError(
                f"tokenwise layer {tokenwise_layer} outside {len(levels)} representation levels"
            )
        pooled = np.stack([
            state[mask].mean(dim=0).float().cpu().numpy() for state in levels
        ])
        tokens = levels[tokenwise_layer][mask].float().cpu().numpy()
        return pooled, tokens

    def teacher_forced_inputs(self, audio_path: str | Path, prompt: str, label: str):
        """Append label tokens and mask prompt positions for causal loss."""
        import torch

        inputs = self.prepare_inputs(audio_path, prompt)
        answer_ids = self.processor.tokenizer(
            label, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(inputs["input_ids"].device)
        prompt_len = inputs["input_ids"].shape[1]
        inputs["input_ids"] = torch.cat([inputs["input_ids"], answer_ids], dim=1)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones_like(answer_ids)], dim=1
        )
        labels = torch.full_like(inputs["input_ids"], -100)
        labels[:, prompt_len:] = answer_ids
        inputs["labels"] = labels
        return inputs
