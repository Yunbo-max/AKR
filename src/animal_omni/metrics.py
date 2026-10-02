from __future__ import annotations

import re
from collections import Counter


def normalize_label(text: str, labels: list[str]) -> str | None:
    def lexical(value: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9 ]", " ", value.lower()).split())

    cleaned = lexical(text)
    matches = [label for label in labels
               if re.search(rf"\b{re.escape(lexical(label))}\b", cleaned)]
    if len(matches) <= 1:
        return matches[0] if matches else None
    # Taxonomies can contain nested names (e.g. Killer Whale and False Killer
    # Whale). A single longest mention should not be rejected merely because its
    # suffix is also a class. Distinct non-nested mentions remain invalid.
    longest = max(matches, key=lambda label: len(lexical(label)))
    if all(re.search(rf"\b{re.escape(lexical(label))}\b", lexical(longest))
           for label in matches):
        return longest
    return None


def classification_metrics(targets: list[str], predictions: list[str | None], labels: list[str]) -> dict[str, float]:
    if len(targets) != len(predictions) or not targets:
        raise ValueError("targets and predictions must be non-empty and equal length")
    accuracy = sum(y == p for y, p in zip(targets, predictions)) / len(targets)
    f1s = []
    for label in labels:
        counts = Counter((y == label, p == label) for y, p in zip(targets, predictions))
        tp, fp, fn = counts[(True, True)], counts[(False, True)], counts[(True, False)]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": accuracy, "macro_f1": sum(f1s) / len(labels)}
