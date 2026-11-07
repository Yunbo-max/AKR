import json

import numpy as np
import pytest

from akr_final.train50_evidence import class_probabilities, probability_evidence_prompt
from akr_final.train50_methods import fit_correction


def payload(prompt):
    return json.loads(prompt.split('\nExternal classifier evidence JSON:\n', 1)[1])


def test_complete_probability_and_prior_vectors_are_lossless_and_ordered():
    labels = ['蝙蝠', 'dog', 'whale', 'insect']
    probabilities = np.array([0.12345678901234567, 0.003456789012345678,
                              0.2730864219753087, 0.6])
    priors = np.array([.1, .2, .3, .4])
    prompt = probability_evidence_prompt('Listen and classify.', labels, probabilities, priors)
    data = payload(prompt)
    assert prompt.startswith('Listen and classify.\n')
    assert 'not ground truth' in prompt and 'not calibrated' in prompt
    assert [r['label'] for r in data['classes']] == labels
    np.testing.assert_array_equal([r['query_probability'] for r in data['classes']], probabilities)
    np.testing.assert_array_equal([r['support_prior'] for r in data['classes']], priors)
    assert len(data['classes']) == 4
    assert probability_evidence_prompt('Listen and classify.', labels, probabilities, priors) == prompt


def test_same_full_routing_information_as_correction():
    rng = np.random.default_rng(12)
    x, g = rng.normal(size=(12, 4)), rng.normal(size=(12, 8))
    labels = np.array([0] * 7 + [1] * 3 + [2] * 2)
    correction = fit_correction(x, g, labels, 3, {'kind': 'soft', 'temperature': .37})
    q = rng.normal(size=(5, 4))
    scores = correction.readout.mapping.predict(q)
    expected = correction.probabilities(q)
    np.testing.assert_array_equal(class_probabilities(scores, .37), expected)
    evidence = payload(probability_evidence_prompt('Task', ['c', 'b', 'a'], expected[0], correction.priors))
    np.testing.assert_array_equal([r['query_probability'] for r in evidence['classes']], expected[0])
    np.testing.assert_array_equal([r['support_prior'] for r in evidence['classes']], [7 / 12, 3 / 12, 2 / 12])


def test_softmax_is_stable_and_does_not_mutate_scores():
    scores = np.array([[1e308, -1e308], [500., 500.]])
    before = scores.copy()
    result = class_probabilities(scores, np.nextafter(0., 1.))
    np.testing.assert_array_equal(scores, before)
    np.testing.assert_array_equal(result, [[1., 0.], [.5, .5]])
    assert class_probabilities(np.empty((0, 3))).shape == (0, 3)
    np.testing.assert_array_equal(class_probabilities([[3.], [-7.]]), [[1.], [1.]])


@pytest.mark.parametrize('scores', [[1., 2.], np.zeros((2, 1, 1)), np.empty((2, 0)),
                                  [[np.nan, 0]], [[np.inf, 0]], [[1j, 0]], [['invalid', 0]]])
def test_invalid_score_arrays(scores):
    with pytest.raises(ValueError):
        class_probabilities(scores)


@pytest.mark.parametrize('temperature', [0, -1, np.nan, np.inf, True, None, [1, 2]])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        class_probabilities([[0., 1.]], temperature)


@pytest.mark.parametrize('labels,probabilities,priors', [
    ([], [], []), (['a', 'a'], [.5, .5], [.5, .5]),
    (['a', ''], [.5, .5], [.5, .5]), ('ab', [.5, .5], [.5, .5]),
    (['a', 2], [.5, .5], [.5, .5]),
    (['a', 'b'], [.5], [.5, .5]), (['a', 'b'], [[.5, .5]], [.5, .5]),
    (['a', 'b'], [.5, .5], [.5]), (['a', 'b'], [.2, .3], [.5, .5]),
    (['a', 'b'], [.5, .5], [.2, .3]), (['a', 'b'], [-.1, 1.1], [.5, .5]),
    (['a', 'b'], [.5, .5], [-.1, 1.1]), (['a', 'b'], [np.nan, .5], [.5, .5]),
    (['a', 'b'], [.5, .5], [np.inf, .5]),
    (['a', 'b'], [.5 + 1j, .5], [.5, .5]),
])
def test_invalid_evidence_vectors(labels, probabilities, priors):
    with pytest.raises(ValueError):
        probability_evidence_prompt('Task', labels, probabilities, priors)


def test_prompt_requires_task_string_and_accepts_degenerate_distribution():
    assert payload(probability_evidence_prompt('Task', ['a', 'b'], [1., 0.], [0., 1.]))['classes'][1]['query_probability'] == 0.
    with pytest.raises(ValueError):
        probability_evidence_prompt(None, ['a'], [1.], [1.])
