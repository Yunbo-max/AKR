"""Full class-routing evidence for a frozen-LM text-channel comparison.

The evidence carries the router's entire query class-weight vector and support
class priors, in the declared vocabulary order. It carries no query truth and no
gradient prototypes. Matching routing information across channels does not make
their gradient information identical or establish decoder necessity.
"""
from __future__ import annotations

import json
import numpy as np


def _finite_array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f'{name} must contain finite real numbers')
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must contain finite real numbers') from error
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must contain finite real numbers')
    return array


def class_probabilities(scores, temperature=1):
    """Stable row-wise softmax of finite [query, class] ridge scores.

    Scores are shifted before temperature scaling, preserving the correction
    router's behavior even for very small positive temperatures. Empty query
    batches are accepted, but the class dimension must be nonempty. The result
    uses float64 and never modifies the caller's score array.
    """
    scores = _finite_array(scores, 'class scores')
    if scores.ndim != 2 or not scores.shape[1]:
        raise ValueError('class scores must be 2-D with at least one class')
    if isinstance(temperature, (bool, np.bool_)):
        raise ValueError('temperature must be finite and positive')
    try:
        temperature = float(temperature)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('temperature must be finite and positive') from error
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if not len(scores):
        return np.empty(scores.shape, dtype=np.float64)
    with np.errstate(over='ignore', under='ignore'):
        logits = (scores - scores.max(axis=1, keepdims=True)) / temperature
        weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def _distribution(values, n_classes, name):
    values = _finite_array(values, name)
    if values.shape != (n_classes,) or (values < 0).any() or (values > 1).any():
        raise ValueError(f'{name} must be a 1-D probability vector in declared class order')
    # Permit ordinary float32 summation error; do not renormalize or round values.
    if not np.isclose(values.sum(), 1., rtol=1e-7, atol=1e-8):
        raise ValueError(f'{name} must sum to one')
    return values


def probability_evidence_prompt(task_prompt, labels, probabilities, priors):
    """Append full query class weights and support priors as lossless JSON.

    Vectors must be one-dimensional distributions aligned with distinct nonempty
    string labels. Their sums may differ from one by normal float32 summation
    error (``np.isclose(rtol=1e-7, atol=1e-8)``). All classes and supplied float
    values are serialized; no top-k selection, decimal truncation or implicit
    renormalization occurs. These uncalibrated external classifier weights are
    explicitly described as fallible evidence, never ground truth.
    """
    if not isinstance(task_prompt, str):
        raise ValueError('task_prompt must be a string')
    if isinstance(labels, (str, bytes, dict, set)):
        raise ValueError('labels must be an ordered sequence of distinct nonempty strings')
    try:
        labels = list(labels)
    except TypeError as error:
        raise ValueError('labels must be an ordered sequence of distinct nonempty strings') from error
    if (not labels or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError('labels must be an ordered sequence of distinct nonempty strings')
    probabilities = _distribution(probabilities, len(labels), 'query probabilities')
    priors = _distribution(priors, len(labels), 'support priors')
    evidence = {'classes': [
        {'label': label, 'query_probability': float(probability), 'support_prior': float(prior)}
        for label, probability, prior in zip(labels, probabilities, priors)
    ]}
    explanation = (
        '\nThe following external classifier evidence lists every allowed class in the fixed task label order. '
        'query_probability is a normalized external ridge-softmax class weight for this recording; '
        'these weights are not calibrated probabilities of correctness. '
        'support_prior is the corresponding class frequency in the labeled support set. '
        'This evidence may be incorrect and is not ground truth. '
        'Listen to the recording, consider the full evidence, and return exactly one allowed label.'
        '\nExternal classifier evidence JSON:\n'
    )
    return task_prompt + explanation + json.dumps(evidence, ensure_ascii=False, allow_nan=False)
