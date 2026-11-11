"""Distributional objectives for the migrated, positional MTP OPSD baseline.

There is no block-consensus objective in this module.  Both head targets come
from the same model conditioned on a self-generated reference.  A head-1 target
is conditional on the actual student token immediately before its target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass(frozen=True)
class PositionSlice:
    """Aligned rows in student/teacher hidden states predicting the same tokens."""

    head: str
    student_start: int
    teacher_start: int
    length: int
    response_target_start: int


def response_slices(
    student_prompt_length: int,
    teacher_prompt_length: int,
    response_length: int,
    *,
    include_mtp: bool,
) -> list[PositionSlice]:
    """Return the causal alignment for an unpadded, single-example response.

    Main row ``i`` predicts input token ``i+1``.  Native MTP row ``i`` consumes
    main hidden row ``i`` and the embedding of input token ``i+1``, predicting
    input token ``i+2``.  Consequently the first auxiliary response target is
    response[1], not response[0].  No reference/gold response token is supplied
    to the student, and no row past the generated response is supervised.
    """
    if student_prompt_length < 1 or teacher_prompt_length < 1:
        raise ValueError("Both chat prompts must contain at least one token")
    if response_length < 1:
        raise ValueError("Cannot distil an empty response")
    out = [
        PositionSlice(
            "head0", student_prompt_length - 1, teacher_prompt_length - 1,
            response_length, 0,
        )
    ]
    if include_mtp and response_length > 1:
        out.append(
            PositionSlice(
                "head1", student_prompt_length - 1, teacher_prompt_length - 1,
                response_length - 1, 1,
            )
        )
    return out


def chunk_ranges(length: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    if length < 0 or chunk_size < 1:
        raise ValueError("length must be nonnegative and chunk_size positive")
    for start in range(0, length, chunk_size):
        yield start, min(start + chunk_size, length)


def full_vocab_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sum of full-vocabulary KL(teacher || student), in FP32.

    The teacher is detached even if the caller inadvertently supplied a tensor
    with a graph.  No top-k target approximation, hard argmax target, entropy
    reward, sampled-token RKL estimator, or clipping is substituted for FKL.
    The usual T^2 factor is explicit; the delivered cases use T=1.
    """
    if student_logits.ndim != 2 or student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "Expected matching [positions, vocabulary] logits; got "
            f"{tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
        )
    if temperature <= 0:
        raise ValueError("KD temperature must be positive")
    teacher_logp = torch.log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
    student_logp = torch.log_softmax(student_logits.float() / temperature, dim=-1)
    # Teacher tensors have no grad path; retaining the entropy term makes the
    # logged number a KL rather than a cross entropy with an unstated offset.
    loss = (teacher_logp.exp() * (teacher_logp - student_logp)).sum()
    loss = loss * (temperature * temperature)
    if not bool(torch.isfinite(loss.detach()).item()):
        raise FloatingPointError("Nonfinite full-vocabulary distillation KL")
    return loss
