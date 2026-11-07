import json
import numpy as np
import pytest
import torch

from scripts.fair_speech_lm import append_answer, language_targets, attach_lora, select_epoch, fresh_queries, verify_stage


def test_answer_mask_matches_support_objective_and_context():
    x = {'input_ids':torch.tensor([[4,5,6]]), 'attention_mask':torch.ones(1,3,dtype=torch.long),
         'input_features':torch.ones(1,2)}
    out = append_answer(x, torch.tensor([[7,8]]), 6)
    assert out['labels'].tolist() == [[-100,-100,-100,7,8]]
    assert out['input_ids'].tolist() == [[4,5,6,7,8]]
    assert x['input_ids'].shape[1] == 3
    assert out['input_features'] is x['input_features']
    with pytest.raises(ValueError): append_answer(x, torch.tensor([[7,8]]), 4)


def test_epoch_selection_pairs_folds_and_prefers_native():
    folds = [{0:[3,3], 1:[2,2], 3:[1,1], 5:[2,2]},
             {0:[3,3], 1:[2,2], 3:[1,1], 5:[2,2]}]
    assert select_epoch(folds)['epoch'] == 3
    assert select_epoch([{0:[1],1:[1],3:[2],5:[2]}]*2)['epoch'] == 0
    with pytest.raises(ValueError): select_epoch([{0:[1],1:[float('nan')],3:[2],5:[2]}]*2)
    with pytest.raises(ValueError): select_epoch([{0:[1],1:[1,2],3:[2],5:[2]}]*2)


def test_fresh_queries_exclude_previous_groups_audio_and_official_test():
    rows = [dict(event_id=f'{c}-{i}', label=c, split='valid', source_filename=f'{c}-{i}',
                 pcm_sha256=f'{c}-{i}') for c in ['A','B'] for i in range(6)]
    excluded = [rows[0], rows[6]]
    rows[1]['pcm_sha256'] = rows[0]['pcm_sha256']
    selected = fresh_queries(rows, ['A','B'], excluded)
    assert len(selected)==8 and selected==fresh_queries(list(reversed(rows)),['A','B'],excluded)
    assert not {r['pcm_sha256'] for r in selected} & {r['pcm_sha256'] for r in excluded}
    for r in rows: r['split']='test'
    with pytest.raises(ValueError): fresh_queries(rows,['A','B'],excluded)


def test_real_tiny_peft_training_freezes_base_and_saves_reloads(tmp_path):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from peft import PeftModel
    torch.manual_seed(5)
    config=Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1)
    base=Qwen2ForCausalLM(config).eval()
    original={n:p.detach().clone() for n,p in base.named_parameters()}
    tokens=torch.tensor([[2,3,4,5]])
    with torch.no_grad(): expected=base(tokens).logits.clone()
    targets=language_targets(base)
    assert len(targets)==4 and all('self_attn' in n for n in targets)
    model, audit=attach_lora(base, seed=11)
    model.eval()
    with torch.no_grad(): assert torch.equal(model(tokens).logits, expected)
    assert audit['trainable_parameters'] > 0 and audit['audio_trainable_parameters']==0
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    model.train(); model(tokens,labels=tokens).loss.backward(); optimizer.step()
    assert any(torch.count_nonzero(p) for n,p in model.named_parameters() if 'lora_B' in n)
    for n,p in model.get_base_model().named_parameters():
        if 'lora_' not in n:
            assert not p.requires_grad
            assert torch.equal(p,original[n.replace('.base_layer','')])
    model.eval()
    with model.disable_adapter(),torch.no_grad(): assert torch.equal(model(tokens).logits,expected)
    with torch.no_grad(): adapted=model(tokens).logits.clone()
    model.save_pretrained(tmp_path/'adapter')
    restored=Qwen2ForCausalLM(config); restored.load_state_dict(original)
    restored=PeftModel.from_pretrained(restored,tmp_path/'adapter').eval()
    with torch.no_grad(): assert torch.equal(restored(tokens).logits,adapted)


def test_incomplete_or_corrupt_stage_rejected(tmp_path):
    from akr_closing.core import atomic_json, digest
    assert verify_stage(tmp_path,{'run':'a'}) is None
    result={'complete':True,'identity':{'run':'a'},'files':{},'losses':{}}
    atomic_json(tmp_path/'DONE.json',{'result':result,'sha256':'wrong'})
    with pytest.raises(RuntimeError): verify_stage(tmp_path,{'run':'a'})
    atomic_json(tmp_path/'DONE.json',{'result':result,'sha256':digest(result)})
    with pytest.raises(RuntimeError): verify_stage(tmp_path,{'run':'b'})


def test_selection_rejects_valid_but_foreign_fold_receipt(tmp_path,monkeypatch):
    import scripts.fair_speech_lm as module
    from akr_closing.core import atomic_json,digest
    class Audit:
        fingerprint='registered'
        def __init__(self,*args): pass
        def close(self): pass
    monkeypatch.setattr(module,'FairRunner',Audit)
    registration={'entry':{'plan':{'episodes':[{'support_ids':['a','b'],'folds':[['a'],['b']]}]}}}
    for lr in module.LRS:
        for fold,held in enumerate([['a'],['b']]):
            train=['b'] if fold==0 else ['a']
            result=dict(complete=True,identity=dict(run='registered',train=train,held=held,lr=lr,
                max_epoch=5,seed=20261108+fold,accumulation=4),files={},train_ids=train,heldout_ids=held,
                losses={'0':[3.],'1':[2.],'3':[1.],'5':[2.]},
                train_exposures=5,optimizer_steps=5,forward_calls=9,wall_seconds=1.)
            module.finish_stage(tmp_path/'training'/f'fold_{fold}'/f'lr_{lr:g}',result)
    assert module.choose_lora(tmp_path,registration)['epoch']==3
    path=tmp_path/'training/fold_0/lr_5e-05/DONE.json'
    envelope=json.loads(path.read_text()); envelope['result']['identity']['train']=['a']
    envelope['sha256']=digest(envelope['result']); atomic_json(path,envelope)
    with pytest.raises(RuntimeError,match='stage receipt'): module.choose_lora(tmp_path,registration)


def test_parallel_policy_verification_is_serialized(tmp_path,monkeypatch):
    import scripts.fair_speech_lm as module
    import time
    from concurrent.futures import ThreadPoolExecutor
    counters={'active':0,'peak':0}
    def verify(*_):
        counters['active']+=1; counters['peak']=max(counters['peak'],counters['active'])
        time.sleep(.03); counters['active']-=1
        return {'locked':True}
    monkeypatch.setattr(module,'_lock_policies',verify)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:module.lock_policies(tmp_path,{}),range(2)))
    assert counters['peak']==1 and all(v['locked'] for v in results)


def test_peft_counter_covers_wrapped_calls_without_double_counting_generation():
    from types import SimpleNamespace
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from scripts.fair_speech_lm import register_adapter_counter
    config=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
        eos_token_id=None,pad_token_id=0)
    backend=SimpleNamespace(thinker=Qwen2ForCausalLM(config).eval(),forward_count=0)
    def count(*_): backend.forward_count+=1
    backend.thinker.register_forward_hook(count)
    tokens=torch.tensor([[2,3,4,5]])
    backend.thinker(tokens); assert backend.forward_count==1
    backend.thinker,_=attach_lora(backend.thinker,11)
    register_adapter_counter(backend)
    backend.thinker.eval(); backend.thinker(tokens)
    backend.thinker.train(); backend.thinker(tokens,labels=tokens).loss.backward()
    backend.thinker.eval()
    with backend.thinker.disable_adapter(): backend.thinker(tokens)
    assert backend.forward_count==4
    output=backend.thinker.generate(tokens,attention_mask=torch.ones_like(tokens),
        max_new_tokens=3,do_sample=False,pad_token_id=0,eos_token_id=None)
    assert output.shape[1]-tokens.shape[1]==3
    assert backend.forward_count==7


def test_gpu_budget_accounts_for_nvidia_total_not_only_wsl_cuda_view(monkeypatch):
    import scripts.fair_speech_lm as module
    gib=1024**3; fractions=[]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(95*gib,96*gib))
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda f,d:fractions.append(f))
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**k:'81920, 98304\n')
    budget=module.gpu_budget(90.)
    assert budget['worker_cap_bytes']==10*gib
    assert budget['nvidia_occupied_bytes']==80*gib
    assert fractions==[10/96]


def test_gpu_budget_rejects_insufficient_global_headroom(monkeypatch):
    import scripts.fair_speech_lm as module
    gib=1024**3
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(95*gib,96*gib))
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**k:'84992, 98304\n')
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda *a:None)
    with pytest.raises(RuntimeError,match='available'): module.gpu_budget(90.)
