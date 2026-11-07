from copy import deepcopy
import pytest
from scripts.cal40_full_eval import allocate, recording_groups, check_complete, model_tag


def rows(nclass=10, perclass=8, split='train'):
    return [dict(event_id=f'{split}_{c}_{j}', label=str(c), split=split,
                 source=f'{split}_{c}_{j}', pcm_sha256=f'p_{split}_{c}_{j}')
            for c in range(nclass) for j in range(perclass)]


def test_exact40_full_official_test_and_coverage():
    data=rows(31,3)+rows(31,2,'test')+rows(31,1,'valid')
    a=allocate(data,[str(c) for c in range(31)],'source',True)
    b=allocate(list(reversed(data)),[str(c) for c in range(31)],'source',True)
    assert a==b
    assert len(a['support_ids'])==40 and len(a['query_ids'])==62
    assert set(a['support_class_counts'])=={str(c) for c in range(31)}
    assert all(i.startswith('test_') for i in a['query_ids'])


def test_transitive_pcm_bridge_excludes_train_not_test():
    data=rows()+rows(perclass=2,split='test')
    data[0]['source']=data[-1]['source']
    data[1]['pcm_sha256']=data[0]['pcm_sha256']
    groups=recording_groups(data,'source')
    assert groups[data[1]['event_id']]==groups[data[-1]['event_id']]
    a=allocate(data,[str(c) for c in range(10)],'source',True)
    assert data[0]['event_id'] not in a['support_ids']
    assert data[1]['event_id'] not in a['support_ids']
    assert len(a['query_ids'])==20


def test_derived_evaluates_all_rows_from_other_groups():
    data=rows(6,20,'pool')
    for c in range(6):
        for j in range(0,20,2): data[c*20+j+1]['source']=data[c*20+j]['source']
    a=allocate(data,[str(c) for c in range(6)],'source',False)
    assert len(a['support_ids'])==40
    assert len(a['query_ids'])==40
    assert len(a['excluded'])==40


def test_40_labels_cannot_cover41_classes():
    with pytest.raises(ValueError,match='40'):
        allocate(rows(41,3),[str(c) for c in range(41)],'source',False)


def test_complete_requires_every_exact_registered_id():
    with pytest.raises(ValueError): check_complete(['a','b'],[{'event_id':'a'}])
    with pytest.raises(ValueError): check_complete(['a','b'],[{'event_id':'a'},{'event_id':'a'}])
    check_complete(['a','b'],[{'event_id':'a'},{'event_id':'b'}])


def test_legacy_training_tag_routes_to_registered_model():
    assert model_tag('qwen3b','qwen7b')=='qwen7b'
    assert model_tag('qwen7b','qwen7b')=='qwen7b'
    with pytest.raises(ValueError): model_tag('glm4voice9b','qwen7b')


def test_gpu_admission_counts_not_yet_loaded_reservations():
    from scripts.cal40_queue import can_admit
    assert can_admit(34,34,[18],36,95.6)
    assert not can_admit(34,34,[18,36],18,95.6)
    assert not can_admit(80,34,[],18,95.6)
