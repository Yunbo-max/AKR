"""Optional E6 uses the existing class-routed factorized implementation, not AKR."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
from .core import atomic_json,file_hash


def run_factorized(root:Path,config:dict,*,run_id,execute=False):
    folder=root/'results'/'akr_closing'/run_id/'E6';folder.mkdir(parents=True,exist_ok=True)
    split=root/'results/beans_dogs_AJ_factorized_validation_split.json'
    if not split.exists():raise FileNotFoundError('E6 requires the registered original selection/confirmation split')
    data=json.loads(split.read_text())
    if len(data['selection'])!=30 or len(data['confirmation'])!=109:raise ValueError('E6 expected registered 30/109 partition')
    if set(data['selection'])&set(data['confirmation']):raise ValueError('E6 partition overlap')
    meta={'split_sha256':file_hash(split),'batch_size':1,'old_batch5_partial':'not imported; unchanged',
          'scope':'class-routed factorized control, not primary continuous AKR',
          'test_execution':'never automatic; old failed gates remain unchanged'}
    lock=folder/'IDENTITY.json'
    if lock.exists() and json.loads(lock.read_text())!=meta:raise RuntimeError('E6 experiment identity changed')
    atomic_json(lock,meta)
    common=[sys.executable,str(root/'scripts/evaluate_probe_routed_class_kv.py'),
        '--config',str(root/'configs/beans_dogs.yaml'),
        '--manifest',str(root/'data/manifests/beans_dogs_all_full_lp1.csv'),
        '--condition','lp_0-1000','--query-split','valid',
        '--equal-support-split',str(root/'results/beans_dogs_lp1_tokenwise_equal_support_fullvalid_split.json'),
        '--support-k-per-class','2','--gradient-dir',str(root/'results/gradients_beans_dogs_train_lp1_tokenwise_7b_AJ'),
        '--representation-dir',str(root/'results/token_reps_beans_dogs_lp1_7b_layer28'),
        '--feature-layer','28','--ridge-alpha','1','--label-map',str(root/'configs/beans_dogs_arbitrary_AJ_labels.json'),
        '--model-id','Qwen/Qwen2.5-Omni-7B','--max-new-tokens','1',
        '--query-event-list',str(split),'--method-batch-size','1','--resume']
    selection=common+['--query-event-key','selection','--factorized-ranks','2','4','8',
        '--methods','probe_class_pooled','probe_class_factorized_r2','probe_class_factorized_r4','probe_class_factorized_r8','probe_class_tokenwise',
        '--relative-alphas','.01','--output',str(folder/'selection.csv')]
    select=[sys.executable,str(root/'scripts/select_factorized_rank.py'),'--predictions',str(folder/'selection.csv'),
            '--expected-n','30','--output',str(folder/'rank.json')]
    atomic_json(folder/'COMMANDS.json',{'selection':selection,'select':select,'execute_requested':execute})
    if not execute:return meta
    def call(cmd):
        from .process import run_checked
        run_checked(cmd,cwd=root,log_path=folder/'run.log')
    call(selection);call(select)
    chosen=json.loads((folder/'rank.json').read_text())
    locked=folder/'LOCK.json'
    if locked.exists() and json.loads(locked.read_text())!=chosen:raise RuntimeError('E6 selected rank changed')
    atomic_json(locked,chosen)
    confirmation=common+['--query-event-key','confirmation','--factorized-ranks',str(chosen['selected_rank']),
        '--methods','probe_class_pooled',chosen['selected_method'],'probe_class_tokenwise',
        '--relative-alphas','.003','.01','.03','--output',str(folder/'confirmation.csv')]
    summarize=[sys.executable,str(root/'scripts/summarize_factorized_kv.py'),'--predictions',str(folder/'confirmation.csv'),
       '--expected-n','109','--output',str(folder/'confirmation_summary.json')]
    call(confirmation);call(summarize)
    return json.loads((folder/'confirmation_summary.json').read_text())
