"""Fixed, independently executable comparisons. Grids never select on test."""
from __future__ import annotations

PRIMARY_DATASETS=('dogs','watkins','bats','humbugdb')
PRIMARY_MODELS=('qwen7b','qwen3b')
CASES={
 'c00':dict(slug='protocol',title='Official Train-50 / full-test audit',reviewers=['tbdR','7PoR']),
 'c01':dict(slug='main',title='Native, original continuous AKR, hard and soft prototypes',reviewers=['tbdR','h8E8']),
 'c02':dict(slug='ridge_decoder',title='Matched ridge, text rendering and decoder utility',reviewers=['tbdR']),
 'c03':dict(slug='acoustic_features',title='MFCC, eGeMAPS and BEATs with the same ridge readout',reviewers=['h8E8']),
 'c04':dict(slug='audio_icl',title='All-50 audio ICL with three predetermined support orders',reviewers=['7PoR']),
 'c05':dict(slug='routing_scale',title='Hard / soft routing x fixed norm / cap',reviewers=['tbdR','h8E8']),
 'c06':dict(slug='centering',title='Centered versus uncentered class prototypes',reviewers=['tbdR','h8E8']),
 'c07':dict(slug='direction_controls',title='Fixed, random, shuffled pairing and shuffled prototypes',reviewers=['tbdR','h8E8']),
 'c08':dict(slug='kv_layers',title='K / V, equal-count single layers and injection positions',reviewers=['tbdR']),
 'c09':dict(slug='support_size',title='Nested 50 / 100 / 200 support scaling',reviewers=['tbdR','h8E8']),
 'c10':dict(slug='sensitivity',title='Prespecified rank, ridge, temperature, gain and cap diagnostics',reviewers=['tbdR']),
 'c11':dict(slug='frequency_transfer',title='Full / LP1 matched and crossed support conditions',reviewers=['h8E8','7PoR']),
 'c12':dict(slug='glm_architecture',title='GLM-4-Voice architecture comparison',reviewers=['h8E8'],models=('glm4voice9b',)),
 'c13':dict(slug='matched_lora',title='Same-50-label language-decoder LoRA',reviewers=['tbdR','h8E8']),
 'c14':dict(slug='cost_noop',title='Real intervention/no-op integrity and end-to-end component costs',reviewers=['tbdR','7PoR']),
 'c15':dict(slug='naturelm',title='Frozen NatureLM-audio specialist applicability',reviewers=['7PoR']),
}


def validate_job(case,dataset,model,condition):
    if case not in CASES:raise ValueError('unknown case')
    if dataset not in PRIMARY_DATASETS:raise ValueError('primary suite supports Dogs/Watkins/Bats/HumBugDB only')
    if model not in CASES[case].get('models',PRIMARY_MODELS):raise ValueError('model not supported by this case')
    if condition not in ('full','lp:1000'):raise ValueError('condition must be full or lp:1000')


def methods_for(case,condition='full'):
    if case not in CASES:raise ValueError('unknown case')
    def m(name,kind,**kw):
        return dict(name=name,kind=kind,ridge=10.,rank=4,temperature=1.,centered=True,
                    alpha=kw.pop('alpha',.3),raw_eta=kw.pop('raw_eta',300.),
                    scale_mode=kw.pop('scale_mode','relative_cap'),support_budget=kw.pop('support_budget',50),
                    feature='omni',scope='audio',kv_kind='kv',layer_group='all',**kw)
    native=m('native','native')
    original=m('akr_regression_original','original',alpha=.01,scale_mode='relative_norm')
    soft=m('akr_soft_cap','soft')
    hard=m('akr_hard_centered','hard',scale_mode='relative_norm')
    if case=='c00':return []
    if case in ('c01','c12'):return [native,original,hard,soft]
    if case=='c02':return [native,original,soft,m('ridge_direct','ridge'),m('probe_to_text','ridge'),
                          m('probe_to_lm','probe_lm'),m('probe_distribution_to_lm','probability_lm')]
    if case=='c03':
        specs=[native,m('omni_ridge','ridge')]
        for feature in ('mfcc','egemaps','beats'):
            s=m(feature+'_ridge','ridge');s['feature']=feature;specs.append(s)
        return specs
    if case=='c04':return [native,original,soft]+[m(f'audio_icl_50_order{o}','icl',order_seed=o) for o in (0,1,2)]
    if case=='c05':return [native,original]+[m(f'{kind}_{scale}',kind,scale_mode=scale) for kind in ('hard','soft') for scale in ('relative_norm','relative_cap')]
    if case=='c06':
        unc=m('soft_uncentered_cap','soft');unc['centered']=False
        return [native,soft,unc]
    if case=='c07':return [native,original,soft,m('continuous_cap','original')]+[m(kind+'_cap',kind) for kind in ('fixed','random','shuffled_pair','shuffled_prototypes')]
    if case=='c08':
        specs=[native,soft]
        for key,values in [('kv_kind',('k','v')),('layer_group',('early','middle','late')),('scope',('text','full_prefill'))]:
            for value in values:
                s=m(f'soft_{key}_{value}','soft');s[key]=value;specs.append(s)
        return specs
    if case=='c09':
        return [native]+[{**template,'name':f'{template["kind"]}_n{n}','support_budget':n}
                         for n in (50,100,200) for template in (original,soft,m('ridge','ridge'))]
    if case=='c10':
        specs=[native,original,soft]
        for key,values,kind in [('rank',(1,16),'original'),('ridge',(1.,100.,3000.),'original'),
                                ('temperature',(.25,4.),'soft'),('raw_eta',(30.,3000.),'soft'),
                                ('alpha',(.03,.1),'soft')]:
            for value in values:
                s=dict(original if kind=='original' else soft)
                s.update(name=f'{kind}_{key}_{value:g}',**{key:value});specs.append(s)
        return specs
    if case=='c11':
        specs=[native]
        for fit in ('full','lp:1000'):
            for template in (original,soft):
                s={**template,'name':template['name']+'_fit_'+fit.replace(':','_'),'fit_condition':fit};specs.append(s)
        return specs
    if case=='c13':return [native,original,soft,m('lora_same50','lora',lr=5e-4,epochs=5,lora_rank=8)]
    if case=='c14':return [native,original,soft,m('zero_alpha','soft',alpha=0.),m('zero_field','zero')]
    if case=='c15':return [native,m('naturelm_native','naturelm')]
    raise AssertionError('unimplemented case')


def case_plan(case,dataset='dogs',model=None,condition='full',seed=20261107):
    model=model or CASES[case].get('models',PRIMARY_MODELS)[0]
    validate_job(case,dataset,model,condition)
    specs=methods_for(case,condition)
    return dict(case=case,**CASES[case],dataset=dataset,model=model,condition=condition,seed=seed,
                methods=specs,support_budgets=sorted({50,*[s['support_budget'] for s in specs]}),
                query_population='complete official test; no max-query limit',
                fixed_policy=True,selection_uses_query_outcomes=False,
                status='planned; no scientific result',
                selection_note='tau=1, eta=300, cap=.3 fixed before test; no all-class two-fold selection with singleton supports; sensitivity never changes main policy')
