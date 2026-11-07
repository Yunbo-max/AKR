import copy
import numpy as np
import pytest

from akr_final.calibration_pilot import choose_alpha, allocate, make_entry, PilotRunner


def test_selection_native_ties_nonfinite_and_improvement():
    assert choose_alpha({0.: [2, 2], .01: [2, 2]})['alpha'] == 0
    assert choose_alpha({0.: [2, 2], .01: [1, 2]})['alpha'] == .01
    with pytest.raises(ValueError, match='finite'):
        choose_alpha({0.: [2], .01: [float('nan')]})
    with pytest.raises(ValueError):
        choose_alpha({0.: [2], .01: [1, 2]})


def rows():
    return [dict(event_id=f'{c}-{s}-{i}', label=c, split=s,
                 source_filename=f'{c}-{s}-{i}', pcm_sha256=f'{c}-{s}-{i}')
            for c in ['A', 'B'] for s in ['train', 'valid', 'test'] for i in range(6)]


def test_allocation_stratified_disjoint_reproducible_no_test():
    r = rows(); a = allocate(r, ['A', 'B'], 7)
    assert a == allocate(list(reversed(r)), ['A', 'B'], 7)
    assert len(a['support_ids']) == len(a['query_ids']) == 8
    assert not set(a['support_ids']) & set(a['query_ids'])
    assert all('-test-' not in i for i in a['support_ids'] + a['query_ids'])
    assert set(a['folds'][0]) | set(a['folds'][1]) == set(a['support_ids'])
    assert not set(a['folds'][0]) & set(a['folds'][1])
    assert len(a['fit_ids']) == 4 and len(a['guard_ids']) == 2


def test_allocation_rejects_duplicate_source_or_audio():
    for field in ['source_filename', 'pcm_sha256']:
        r = rows()
        for x in r: x[field] = 'same'
        with pytest.raises(ValueError, match='distinct'): allocate(r, ['A', 'B'], 7)


def test_full_fake_run_resume_label_boundary_and_corruption(tmp_path):
    from test_rebuttal_runner import setup, RebuttalFake
    from akr_final.rebuttal_plan import build_study
    root, path = setup(tmp_path)
    original = build_study(root, path, ['core'])['datasets']['marmaudio']
    rr = original['plan']['rows']
    for c in ['A', 'B']:
        group = [r for r in rr if r['label'] == c]
        for j, r in enumerate(group):
            r.update(split='train' if j < 4 else 'valid',
                     source_filename=r['event_id'], pcm_sha256=r['event_id'])
    entry = make_entry(original['plan']['config'], original['options']['settings'], rr,
                       ['A', 'B'], {}, seed=7)
    runner = PilotRunner(root, root/'pilot', 'dogs', entry, '24gb', RebuttalFake)
    runner.preflight()
    fit_arrays = []
    original_fit = runner.router_fit
    def observed_fit(x, g, **kw):
        fit_arrays.append((x.copy(), g.copy()))
        return original_fit(x, g, **kw)
    runner.router_fit = observed_fit
    result = runner.run_pilot()
    e = runner.plan['episodes'][0]
    assert len(fit_arrays) == 4  # Full pool, split, two independent CV refits.
    pool_x, pool_g = fit_arrays[0]
    for arrays, ids in zip(fit_arrays[1:], [e['fit_ids'], e['folds'][1], e['folds'][0]]):
        index = [e['support_ids'].index(i) for i in ids]
        assert np.array_equal(arrays[0], pool_x[index])
        assert np.array_equal(arrays[1], pool_g[index])
    with pytest.raises(ValueError, match='restricted'):
        runner.loss(e, e['query_ids'], None, 0.)
    assert result['complete'] and len(result['rows']) == 7
    assert not set(RebuttalFake.loss_calls) & set(e['query_ids'])
    assert not set(RebuttalFake.grad_calls) & set(e['query_ids'])
    assert result['no_op_exact']
    before = (RebuttalFake.loads, RebuttalFake.predict_calls, len(RebuttalFake.loss_calls))
    runner.close()
    again = PilotRunner(root, root/'pilot', 'dogs', entry, '24gb', RebuttalFake)
    again.preflight(); assert again.run_pilot()['complete']
    assert before == (RebuttalFake.loads, RebuttalFake.predict_calls, len(RebuttalFake.loss_calls))
    # Query labels affect scoring, never selection or predictions.
    altered = copy.deepcopy(entry)
    for r in altered['plan']['rows']:
        if r['event_id'] in e['query_ids']: r['label'] = 'B' if r['label'] == 'A' else 'A'
    other = PilotRunner(root, root/'other', 'dogs', altered, '24gb', RebuttalFake)
    other.preflight(); changed = other.run_pilot()
    assert changed['policies'] == result['policies']
    for method in result['predictions']:
        assert changed['predictions'][method] == result['predictions'][method]
    item = next((again.out/'loss_cache').glob('*/items/*.json'))
    import json
    saved = item.read_text(); damaged = json.loads(saved)
    damaged.pop('result_sha256'); item.write_text(json.dumps(damaged))
    with pytest.raises((RuntimeError, ValueError, KeyError)): again.run_pilot()
    assert not json.loads((again.out/'STATUS.json').read_text())['complete']
    item.write_text(saved)
    audio = next((again.out/'audio').glob('**/*.wav'))
    audio.write_bytes(b'partial waveform')
    with pytest.raises(RuntimeError, match='processed audio checksum'): again.run_pilot()
    assert not json.loads((again.out/'STATUS.json').read_text())['complete']
