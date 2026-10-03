#!/usr/bin/env python3
"""Offline release audit: files, references, registered settings; never loads a model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
import yaml


def audit(root: Path) -> dict:
    root=root.resolve(); errors=[]; checks=0
    def require(ok: bool, message: str):
        nonlocal checks
        checks+=1
        if not ok: errors.append(message)
    required=['paper/main.tex','paper/references.bib','paper/iclr2027_conference.sty',
              'paper/iclr2027_conference.bst','configs/akr_registry.json',
              'configs/akr_standalone.yaml','akr_final/simple_baselines.py',
              'akr_final/standalone.py','scripts/run_akr_release.py',
              'scripts/check_akr_backend.py','requirements-cpu.txt','requirements-qwen.txt',
              'docs/research/RESEARCH_AUTOPILOT_AUDIT_20261003.md',
              'docs/research/SKILL_USE_20261003.json','paper/figure_prompts/figure2_prompt.md']
    for p in required: require((root/p).is_file(),'Missing file: '+p)
    registry={};cfg={}
    try:
        registry=json.loads((root/'configs/akr_registry.json').read_text())
        cfg=yaml.safe_load((root/'configs/akr_standalone.yaml').read_text())
    except (OSError,ValueError,yaml.YAMLError) as exc:
        errors.append('Invalid registry/config: '+str(exc))
    if registry and cfg:
        for tag,key in [('qwen7b','model_7b'),('qwen3b','model_3b')]:
            m=registry.get('models',{}).get(tag,{})
            c=cfg.get(key,{})
            require(bool(re.fullmatch(r'[0-9a-f]{40}',str(c.get('revision','')))),key+' revision is not immutable')
            require(c.get('id')==m.get('id') and c.get('revision')==m.get('revision'),key+' id/revision differs from registry')
        s=registry.get('experiments',{}).get('S1',{})
        require(s.get('status')=='implemented_not_run_on_real_model','S1 has no published real-model result in this release; do not mark it complete')
        require(registry.get('default_tasks')==['S1'],'Default queue must remain bounded to S1')
        require(s.get('support_per_class')==cfg.get('k_per_class'),'Support count differs from registry')
        require(cfg.get('seeds')==[s.get('seed')],'Seed differs from registry')
        for k,r in [('rank','fixed_rank'),('alpha','relative_alpha'),('ridge_alpha','ridge_alpha')]:
            require(cfg.get('fixed_settings',{}).get(k)==s.get(r),'Setting differs: '+k)
        require(cfg.get('labels')==registry.get('datasets',{}).get('MA-CT',{}).get('labels'),'MarmAudio label order differs')
        for name,e in registry.get('experiments',{}).items():
            if e.get('evidence'):require((root/e['evidence']).is_file(),'Missing '+name+' evidence: '+e['evidence'])
    main=root/'paper/main.tex'; bib=root/'paper/references.bib'
    if main.is_file() and bib.is_file():
        text=main.read_text(); bt=bib.read_text()
        keys=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',bt))
        used={k.strip() for group in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}',text) for k in group.split(',')}
        for k in sorted(used):require(k in keys,'Unresolved citation: '+k)
        labels=re.findall(r'\\label\{([^}]+)\}',text)
        require(len(labels)==len(set(labels)),'Duplicate LaTeX label')
        for ref in re.findall(r'\\(?:eqref|ref|pageref)\{([^}]+)\}',text):require(ref in labels,'Unresolved label: '+ref)
        for p in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text):
            require((main.parent/p).is_file(),'Missing figure: '+p)
        require('Acoustic-to-KV Regression for' in text,'Latest approved title missing')
    return {'complete':not errors,'checks':checks,'errors':errors,'gpu_started':False,
            'raw_audio_required_for_this_check':False,'scope':'release integrity, not scientific validation'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    a=p.parse_args();r=audit(a.root);print(json.dumps(r,indent=2));return 0 if r['complete'] else 1

if __name__=='__main__':sys.exit(main())
