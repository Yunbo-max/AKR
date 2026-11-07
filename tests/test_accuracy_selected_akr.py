import numpy as np
import pytest

from scripts.accuracy_selected_akr import CenteredRouter, select_recognition


@pytest.mark.parametrize('kind',['centered_regression','centered_class'])
def test_centering_removes_common_gradient_bias(kind):
    rng=np.random.default_rng(7)
    x=rng.normal(size=(12,5)).astype('float32'); g=rng.normal(size=(12,8)).astype('float32')
    y=np.repeat(np.arange(3),4); q=rng.normal(size=(4,5)).astype('float32')
    offset=np.arange(8,dtype='float32')*3
    first=CenteredRouter(x,g,y,3,kind).predict(q)
    second=CenteredRouter(x,g+offset,y,3,kind).predict(q)
    assert np.allclose(first,second,atol=1e-5)
    assert first.shape==(4,8)


def test_selection_uses_recognition_and_prefers_native_on_ties():
    truth=['A','A','B','B']; labels=['A','B']
    predictions={'native':['A']*4,'a':['A','A','B','A'],'b':['A','B','A','B']}
    choice=select_recognition(predictions,truth,labels,['native','b','a'])
    assert choice['candidate']=='a' and choice['scores']['a']['accuracy']==.75
    assert select_recognition({'native':['A']*4,'a':['A']*4},truth,labels,['native','a'])['candidate']=='native'
    with pytest.raises(ValueError): select_recognition({'native':['A']},truth,labels,['native'])


def test_truth_changes_scores_not_centered_query_directions():
    rng=np.random.default_rng(19)
    x=rng.normal(size=(8,4));g=rng.normal(size=(8,6));y=np.repeat([0,1],4)
    router=CenteredRouter(x,g,y,2,'centered_class')
    fields=router.predict(x[:2])
    select_recognition({'native':['A','B'],'a':['B','A']},['A','B'],['A','B'],['native','a'])
    select_recognition({'native':['A','B'],'a':['B','A']},['B','A'],['A','B'],['native','a'])
    assert np.array_equal(fields,router.predict(x[:2]))
