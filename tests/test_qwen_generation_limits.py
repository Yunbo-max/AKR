"""Text limits must survive the installed Omni wrapper's submodel routing."""
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from animal_omni.qwen_runner import QwenThinkerRunner


@pytest.fixture
def runner():
    # Execute the installed wrapper routing rather than reimplementing it in a
    # mock. Only its expensive Thinker generation and audio processor are fakes.
    from transformers import Qwen2_5OmniForConditionalGeneration

    class Thinker:
        def generate(self, input_ids, *, max_new_tokens, do_sample, **kwargs):
            assert do_sample is False
            continuation = np.full((len(input_ids), max_new_tokens), 7, dtype=np.int64)
            return np.concatenate([input_ids, continuation], axis=1)

    model = SimpleNamespace(thinker=Thinker(), has_talker=False, speaker_map={'Chelsie': {}})
    model.generate = MethodType(Qwen2_5OmniForConditionalGeneration.generate, model)
    instance = QwenThinkerRunner.__new__(QwenThinkerRunner)
    instance.model = model
    instance.processor = SimpleNamespace(batch_decode=lambda rows, **kwargs: [
        ' '.join('token' if token == 7 else 'PROMPT' for token in row) for row in rows])
    instance.prepare_inputs = lambda *args, **kwargs: {'input_ids': np.full((1, 5), 99, dtype=np.int64)}
    instance.prepare_batch = lambda paths, *args, **kwargs: {
        'input_ids': np.full((len(paths), 5), 99, dtype=np.int64)}
    instance.prepare_icl_inputs = lambda *args, **kwargs: {
        'input_ids': np.full((1, 13), 99, dtype=np.int64)}
    return instance


@pytest.mark.parametrize('method', ['predict', 'predict_batch', 'predict_icl'])
@pytest.mark.parametrize('limit', [1, 2, 7, None])
def test_requested_text_limit_reaches_thinker_for_every_generation_path(runner, method, limit):
    kwargs = {} if limit is None else {'max_new_tokens': limit}
    expected = ['token'] * (8 if limit is None else limit)
    if method == 'predict':
        outputs = [runner.predict('query.wav', 'Label only.', **kwargs)]
    elif method == 'predict_batch':
        outputs = runner.predict_batch(['query1.wav', 'query2.wav'], 'Label only.', **kwargs)
        assert len(outputs) == 2
    else:
        outputs = [runner.predict_icl([('support.wav', 'label')], 'query.wav',
                                      'Label only.', support_prompt='Support prompt.', **kwargs)]
    assert all(output.split() == expected for output in outputs)
