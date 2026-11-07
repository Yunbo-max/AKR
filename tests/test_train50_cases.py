import pytest
from akr_final.train50_cases import CASES, methods_for, validate_job


def test_main_preserves_original_and_new_names_and_no_external_selection():
    specs=methods_for('c01','full')
    assert {s['name'] for s in specs}=={'native','akr_regression_original','akr_hard_centered','akr_soft_cap'}
    new=next(s for s in specs if s['name']=='akr_soft_cap')
    assert new['kind']=='soft' and new['scale_mode']=='relative_cap'
    assert all(s.get('support_budget',50)==50 for s in specs)


def test_route_scale_factorial_and_icl_budget_are_real_methods():
    specs=methods_for('c05','full')
    assert {(s['kind'],s['scale_mode']) for s in specs if s['kind'] in ['hard','soft']}=={
        ('hard','relative_cap'),('hard','relative_norm'),('soft','relative_cap'),('soft','relative_norm')}
    icl=[s for s in methods_for('c04','full') if s['kind']=='icl']
    assert len(icl)==3 and all(s['support_budget']==50 for s in icl)


def test_cases_never_silently_accept_unsupported_models():
    validate_job('c12','dogs','glm4voice9b','full')
    with pytest.raises(ValueError):validate_job('c13','dogs','glm4voice9b','full')
    with pytest.raises(ValueError):validate_job('c01','marmaudio','qwen7b','full')
    for case in CASES:
        specs=methods_for(case,'full')
        assert len({s['name'] for s in specs})==len(specs)
        assert all(s['name'] and '/' not in s['name'] for s in specs)


def test_location_controls_use_actual_backend_mask_contract():
    import torch
    from akr_closing.backend import OmniBackend
    backend=object.__new__(OmniBackend);backend.audio_token_id=9;backend.torch=torch
    inputs={'input_ids':torch.tensor([[9,1,2]]),'attention_mask':torch.ones(1,3)}
    for spec in methods_for('c08'):
        assert backend._mask(inputs,spec['scope']).any()
