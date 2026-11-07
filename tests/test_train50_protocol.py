from copy import deepcopy
import pytest

from akr_final.train50_protocol import allocate_train_support, validate_population


def rows():
    result=[]
    for c in ['a','b','c']:
        for i in range(30):
            event=f'{c}{i}'
            result.append(dict(event_id=event,label=c,split='train',source_filename=event,pcm_sha256=event))
        for i in range(3):
            event=f'test{c}{i}'
            result.append(dict(event_id=event,label=c,split='test',source_filename=event,pcm_sha256=event))
    return result


def test_train_only_exact_budget_full_test_and_nested_prefix():
    r=rows()
    a=allocate_train_support(r,['a','b','c'],50,17,'source_filename')
    b=allocate_train_support(list(reversed(r)),['a','b','c'],60,17,'source_filename')
    assert len(a['support_ids'])==50
    assert sorted(a['support_class_counts'].values())==[16,17,17]
    assert a['query_ids']==['testa0','testa1','testa2','testb0','testb1','testb2','testc0','testc1','testc2']
    assert b['support_ids'][:50]==a['support_ids']
    assert all(not i.startswith('test') for i in a['support_ids'])


def test_transitive_source_pcm_overlap_excludes_train_never_test():
    r=rows()
    r[0]['source_filename']='testa0'
    r[1]['pcm_sha256']=r[0]['pcm_sha256']
    split=allocate_train_support(r,['a','b','c'],50,17,'source_filename')
    assert not {'a0','a1'} & set(split['support_ids'])
    assert {'a0','a1'} <= set(split['excluded_overlap_train_ids'])
    assert 'testa0' in split['query_ids']


def test_class_coverage_is_not_silently_relaxed():
    with pytest.raises(ValueError,match='class'):
        allocate_train_support(rows(),['a','b','c'],2,0,'source_filename')
    r=rows()
    for row in r:
        if row['label']=='c' and row['split']=='train':row['split']='valid'
    with pytest.raises(ValueError,match='class'):
        allocate_train_support(r,['a','b','c'],50,0,'source_filename')


def test_missing_group_unknown_split_and_duplicate_identity_rejected():
    for mutate in [lambda r:r[0].pop('source_filename'),lambda r:r[0].update(split='mystery'),lambda r:r[0].update(event_id=r[1]['event_id'])]:
        r=deepcopy(rows());mutate(r)
        with pytest.raises(ValueError):allocate_train_support(r,['a','b','c'],50,0,'source_filename')


def test_exact_prediction_population_does_not_trust_self_report():
    expected=['q1','q2']
    valid=[dict(event_id='q1',prediction='a'),dict(event_id='q2',prediction='')]
    validate_population(expected,valid)
    for bad in [valid[:1],valid[::-1],valid+[valid[0]],[valid[0],dict(event_id='foreign',prediction='a')]]:
        with pytest.raises(ValueError):validate_population(expected,bad)
