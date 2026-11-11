"""Question-only self-consistency selection, following the U-OPSD protocol.

This is the training selector, not a replacement benchmark scorer.  It never
accepts a gold answer.  Math-Verify parses the final boxed expression and checks
equivalence; truncated generations remain invalid even if they contain a box.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Selection:
    reference_index: int | None
    target_indices: list[int]
    groups: list[list[int]]
    parsed_answers: list[str | None]
    invalid_indices: list[int]
    invalid_reasons: dict[int, str]
    confidence: float
    status: str


def _field(rollout: Any, name: str, default: Any = None) -> Any:
    return rollout.get(name, default) if isinstance(rollout, Mapping) else getattr(rollout, name, default)


def _last_complete_box(text: str) -> str | None:
    """Select only the final explicit box; leave mathematical parsing upstream."""
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if not matches:
        return None
    start = matches[-1].start()
    opening = text.find("{", start)
    depth = 0
    for index in range(opening, len(text)):
        character = text[index]
        # Escaped literal braces do not delimit the outer answer box.
        backslashes = 0
        before = index - 1
        while before >= 0 and text[before] == "\\":
            backslashes += 1
            before -= 1
        if backslashes % 2:
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1].replace("\\fbox", "\\boxed", 1)
    return None


def parse_final_answer(text: str) -> list[Any]:
    """Parse the final boxed answer with Math-Verify, without string fallbacks."""
    from math_verify import LatexExtractionConfig, parse

    # Internal reasoning may contain provisional boxes.  An unclosed thinking
    # block is not a final answer, even if a generation stopped unexpectedly.
    if "<think>" in text and "</think>" not in text:
        return []
    answer_text = text.rsplit("</think>", 1)[-1]
    boxed = _last_complete_box(answer_text)
    if boxed is None:
        return []
    return parse(
        boxed,
        extraction_config=[LatexExtractionConfig(boxed_match_priority=0,
                                                 try_extract_without_anchor=False)],
        fallback_mode="no_fallback", extraction_mode="first_match",
        parsing_timeout=5, raise_on_error=False,
    )


def equivalent_answers(first: list[Any], second: list[Any]) -> bool:
    """Symmetric matching avoids treating a one-sided implication as equality."""
    from math_verify import verify

    if not first or not second:
        return False
    return bool(
        verify(first, second, strict=True, timeout_seconds=5)
        and verify(second, first, strict=True, timeout_seconds=5)
    )


def select_reference(rollouts: Sequence[Any], *, threshold: float = 0.5,
                     rng: random.Random | None = None) -> Selection:
    """Choose the longest majority-agreeing reference and valid disagreeing targets.

    The denominator includes all sampled rollouts, including invalid ones.
    Equal-size modal groups are resolved by the provided seeded RNG.  The
    0.5 threshold follows the paper's non-strict threshold convention; a 2/2
    tie at G=4 may qualify after seeded tie-breaking, and this is logged.
    The caller may bound the number of targets, retaining the full selection.
    """
    if len(rollouts) < 2:
        raise ValueError("Disagreeing-trajectory majority OPSD requires at least two rollouts")
    if not 0.5 <= threshold <= 1.0:
        raise ValueError("The majority confidence threshold must be in [0.5, 1.0]")
    rng = rng if rng is not None else random.Random(0)
    parsed: list[list[Any]] = []
    labels: list[str | None] = []
    invalid_reasons: dict[int, str] = {}
    groups: list[list[int]] = []

    for index, rollout in enumerate(rollouts):
        reason = _field(rollout, "finish_reason")
        text = _field(rollout, "text", "")
        answer: list[Any] = []
        if reason not in {"eos", "stop"}:
            invalid_reasons[index] = f"unfinished_or_failed:{reason}"
        elif not isinstance(text, str) or not text:
            invalid_reasons[index] = "empty_text"
        else:
            answer = parse_final_answer(text)
            if not answer:
                invalid_reasons[index] = "no_parseable_final_box"
        parsed.append(answer)
        labels.append(str(answer[0]) if answer else None)
        if not answer:
            continue
        # Complete-link grouping avoids a non-transitive tolerance chain.
        for group in groups:
            if all(equivalent_answers(answer, parsed[other]) for other in group):
                group.append(index)
                break
        else:
            groups.append([index])

    invalid = sorted(invalid_reasons)
    if not groups:
        return Selection(None, [], groups, labels, invalid, invalid_reasons, 0.0, "no_valid_answers")
    largest = max(map(len, groups))
    majority = rng.choice([group for group in groups if len(group) == largest])
    confidence = largest / len(rollouts)
    if confidence < threshold:
        return Selection(None, [], groups, labels, invalid, invalid_reasons, confidence, "no_majority")
    targets = [index for group in groups if group is not majority for index in group]
    if not targets:
        return Selection(None, [], groups, labels, invalid, invalid_reasons, confidence, "all_valid_agree")

    # Stable tie rule: earlier rollout wins if token lengths are equal.
    reference = max(majority, key=lambda index: (len(_field(rollouts[index], "token_ids", [])), -index))
    return Selection(reference, sorted(targets), groups, labels, invalid,
                     invalid_reasons, confidence, "selected")
