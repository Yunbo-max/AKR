from __future__ import annotations

import re
from collections import Counter


def normalize_label(text: str, labels: list[str]) -> str | None:
    def lexical(value: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9 ]", " ", value.lower()).split())

    cleaned = lexical(text)
    mentions = [(m.start(), m.end(), label) for label in labels if lexical(label)
                for m in re.finditer(rf"\b{re.escape(lexical(label))}\b", cleaned)]
    # Ignore a shorter label only when THAT occurrence sits inside a longer
    # mention. A separate "Killer Whale" remains ambiguous with "False Killer Whale".
    matches = {label for start, end, label in mentions
               if not any(a <= start and end <= b and b - a > end - start
                          for a, b, _ in mentions)}
    return next(iter(matches)) if len(matches) == 1 else None


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
