"""Real Qwen2.5-Omni Thinker backend and tested pre-RoPE interventions.

The backbone is frozen. Training-time support gradients are taken with respect
only to temporary K/V projection deltas. Query inference uses inference_mode.
This is a new direct-Thinker backend; historical wrapper parity must be checked
on the user's GPU before comparing new and historical numbers.
"""
from __future__ import annotations
from contextlib import AbstractContextManager
from pathlib import Path
import inspect
import numpy as np
from .core import Query


class ContextLimit(RuntimeError): pass


def candidate_logprob(logits, answer_ids, prompt_length):
    import torch
    n=answer_ids.shape[1]
    if n<1 or prompt_length<1: raise ValueError('nonempty prefix and answer required')
    selected=logits[0,prompt_length-1:prompt_length+n-1].float()
    lp=torch.log_softmax(selected,dim=-1).gather(1,answer_ids[0,:,None]).squeeze(1)
    return {'sequence_logprob':float(lp.sum().detach().cpu()),
            'mean_token_logprob':float(lp.mean().detach().cpu()),'token_count':n}


class ProjectionIntervention(AbstractContextManager):
    """One prefill-only additive intervention; alpha is relative active-state norm."""
    def __init__(self,modules,directions,mask,alpha):
        if not np.isfinite(alpha) or alpha<0: raise ValueError('alpha must be finite and nonnegative')
        self.modules=modules; self.directions=directions; self.mask=mask
        self.alpha=alpha; self.handles=[]; self.ratios={}; self.norms={}; self.called=set()
    def __enter__(self):
        import torch
        for key,module in self.modules.items():
            if key not in self.directions: continue
            def hook(_m,_x,out,key=key):
                if key in self.called or out.shape[:2]!=self.mask.shape: return out
                self.called.add(key)
                if self.alpha==0: return out
                mask=self.mask.to(out.device).bool()
                if not mask.any(): return out
                direction=torch.as_tensor(self.directions[key],device=out.device,dtype=torch.float32)
                if not torch.isfinite(direction).all(): raise ValueError('non-finite KV correction')
                delta=torch.zeros_like(out,dtype=torch.float32)
                if direction.ndim==1: delta[mask]=direction
                elif direction.shape==out.shape: delta=direction*mask[:,:,None]
                elif direction.ndim==2 and direction.shape[0]==int(mask.sum()): delta[mask]=direction
                else: raise ValueError(f'field shape incompatible at {key}: {direction.shape}')
                dn=torch.linalg.vector_norm(delta[mask]); bn=torch.linalg.vector_norm(out.float()[mask])
                scale=self.alpha*bn/dn.clamp_min(1e-30) if dn>0 else dn*0
                applied=(delta*scale).to(out.dtype)
                actual=torch.linalg.vector_norm(applied.float()[mask])
                self.ratios[str(key)]=float((actual/bn.clamp_min(1e-30)).detach().cpu())
                self.norms[str(key)]={'base_frobenius':float(bn.detach().cpu()),'delta_frobenius':float(actual.detach().cpu()),'active_tokens':int(mask.sum())}
                return out+applied
            self.handles.append(module.register_forward_hook(hook))
        return self
    def __exit__(self,*args):
        for h in self.handles: h.remove()
        self.handles=[]
        return False


class OmniBackend:
    def __init__(self,model_id,revision,*,profile='24gb',layer_fractions=(.25,.5,.75,.875),
                 feature_fraction=.875,max_context_tokens=32768):
        import torch
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration,Qwen2_5OmniProcessor
        self.torch=torch; self.max_context_tokens=max_context_tokens
        if not torch.cuda.is_available(): raise RuntimeError('Real experiments require CUDA; CPU tests do not run Omni.')
        budget={'16gb':'13GiB','24gb':'20GiB','48gb':'43GiB'}
        if profile not in budget: raise ValueError('profile must be 16gb/24gb/48gb')
        self.processor=Qwen2_5OmniProcessor.from_pretrained(model_id,revision=revision)
        self.thinker=Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_id,revision=revision,torch_dtype=torch.bfloat16,device_map='auto',
            max_memory={0:budget[profile],'cpu':'64GiB'},attn_implementation='sdpa')
        self.thinker.eval()
        for p in self.thinker.parameters(): p.requires_grad_(False)
        self.device=self.thinker.get_input_embeddings().weight.device
        layers=self.thinker.model.layers; n=len(layers)
        self.layer_ids=sorted({min(n-1,max(0,int(f*n))) for f in layer_fractions})
        self.feature_layer=min(n,max(0,round(feature_fraction*n)))
        self.forward_count=0
        self._count_handle=self.thinker.register_forward_hook(lambda *_: setattr(self,'forward_count',self.forward_count+1))
        self.modules={(i,k):getattr(layers[i].self_attn,k+'_proj') for i in self.layer_ids for k in ['k','v']}
        self.keys=sorted(self.modules)
        self.audio_token_id=self.thinker.config.audio_token_id
        self.layout=[(key,int(self.modules[key].out_features)) for key in self.keys]
        self._forward_parameters=inspect.signature(self.thinker.forward).parameters
        self.provenance={'model_id':model_id,'revision':revision,'dtype':'bfloat16','batch_size':1,
            'backend':'direct Thinker, SDPA attention, new locked re-evaluation protocol',
            'layers':self.layer_ids,'feature_layer':self.feature_layer,'profile':profile,
            'intervention_site':'pre-RoPE k_proj/v_proj output; not a post-RoPE cache edit'}

    def _prepare(self,q:Query,prompt:str,support=()):
        from qwen_omni_utils import process_mm_info
        conversations=[]
        for sq,label in support:
            conversations.extend([{'role':'user','content':[{'type':'audio','audio':sq.audio_path},
                                 {'type':'text','text':prompt}]},
                                  {'role':'assistant','content':[{'type':'text','text':label}]}])
        conversations.append({'role':'user','content':[{'type':'audio','audio':q.audio_path},
                                                        {'type':'text','text':prompt}]})
        text=self.processor.apply_chat_template(conversations,add_generation_prompt=True,tokenize=False)
        audio,images,videos=process_mm_info(conversations,use_audio_in_video=False)
        inputs=self.processor(text=text,audio=audio,images=images,videos=videos,
                              return_tensors='pt',padding=True,use_audio_in_video=False)
        if inputs['input_ids'].shape[1]>self.max_context_tokens:
            raise ContextLimit(f"{inputs['input_ids'].shape[1]} tokens exceeds budget {self.max_context_tokens}; no truncation")
        inputs=inputs.to(self.device)
        # Never cast token IDs or attention masks to floating point.
        for key,value in inputs.items():
            if self.torch.is_tensor(value) and value.is_floating_point(): inputs[key]=value.to(self.torch.bfloat16)
        return inputs

    def feature(self,q,prompt):
        x=self._prepare(q,prompt); mask=x['input_ids'].eq(self.audio_token_id)
        if not mask.any(): raise RuntimeError('no audio tokens')
        extra={}
        if 'logits_to_keep' in self._forward_parameters: extra['logits_to_keep']=1
        elif 'num_logits_to_keep' in self._forward_parameters: extra['num_logits_to_keep']=1
        with self.torch.inference_mode():
            out=self.thinker(**x,use_cache=False,output_hidden_states=True,return_dict=True,**extra)
            h=out.hidden_states[self.feature_layer]
            feature=h[mask.to(h.device)].float().mean(0).cpu().numpy()
        return feature

    def _mask(self,x,scope,prefix_length=None):
        ids=x['input_ids']; n=prefix_length or ids.shape[1]
        prefix=self.torch.arange(ids.shape[1],device=ids.device)[None,:]<n
        audio=ids.eq(self.audio_token_id)&prefix
        if scope=='audio': return audio
        if scope=='text': return prefix&~audio
        if scope=='full_prefill': return prefix.expand_as(ids)
        raise ValueError('scope must be audio/text/full_prefill')

    def target_token_ids(self,target):
        return self.processor.tokenizer(target,add_special_tokens=False)['input_ids']

    def support_gradients(self,q:Query,target:str,prompt:str):
        """Only invoke on registered labeled SUPPORT, never on held-out queries."""
        t=self.torch; x=self._prepare(q,prompt); prefix=x['input_ids'].shape[1]
        answer=self.processor.tokenizer(target,add_special_tokens=False,return_tensors='pt')['input_ids'].to(self.device)
        x['input_ids']=t.cat([x['input_ids'],answer],1)
        x['attention_mask']=t.cat([x['attention_mask'],t.ones_like(answer)],1)
        labels=t.full_like(x['input_ids'],-100); labels[:,prefix:]=answer; x['labels']=labels
        leaves={}; handles=[]
        try:
            for key,module in self.modules.items():
                def hook(_m,_i,out,key=key):
                    delta=t.zeros_like(out,requires_grad=True); leaves[key]=delta; return out+delta
                handles.append(module.register_forward_hook(hook))
            with t.enable_grad():
                output=self.thinker(**x,use_cache=False,return_dict=True)
                grads=t.autograd.grad(output.loss,[leaves[k] for k in self.keys],allow_unused=False)
            answer_free={}
            for scope in ['audio','text','full_prefill']:
                mask=self._mask(x,scope,prefix)
                if not mask.any(): raise RuntimeError(f'empty support scope {scope}')
                answer_free[scope]=np.concatenate([
                    (-g[mask.to(g.device)]).float().mean(0).detach().cpu().numpy() for g in grads])
            return answer_free
        finally:
            for handle in handles: handle.remove()

    def unflatten(self,vector,kind='kv',layer_group='all'):
        out={}; offset=0
        if layer_group not in {'all','early','middle','late'}: raise ValueError('unknown layer group')
        selected=self.layer_ids if layer_group=='all' else ([self.layer_ids[0]] if layer_group=='early' else [self.layer_ids[len(self.layer_ids)//2]] if layer_group=='middle' else [self.layer_ids[-1]])
        for key,width in self.layout:
            value=vector[offset:offset+width]; offset+=width
            if key[0] in selected and (kind=='kv' or key[1]==kind): out[key]=value
        if offset!=len(vector): raise ValueError('gradient/layout mismatch')
        return out

    def predict(self,q,prompt,labels,*,support=(),direction=None,alpha=0.,scope='audio',
                kind='kv',layer_group='all',max_new_tokens=12):
        x=self._prepare(q,prompt,support); n=x['input_ids'].shape[1]
        if direction is not None and support: raise ValueError('repair uses no audio ICL prefix in this protocol')
        directions={} if direction is None else self.unflatten(direction,kind,layer_group)
        with self.torch.inference_mode(),ProjectionIntervention(self.modules,directions,self._mask(x,scope),alpha) as hooks:
            output=self.thinker.generate(**x,do_sample=False,max_new_tokens=max_new_tokens,use_cache=True)
        raw=self.processor.batch_decode(output[:,n:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
        # Use the repository's authoritative normalizer; ambiguous strings remain invalid.
        from animal_omni.metrics import normalize_label
        prediction=normalize_label(raw,labels) or ''
        return {'prediction':prediction,'raw_prediction':raw,'applied_relative_norms':hooks.ratios,
                'applied_norms':hooks.norms,'generated_tokens':int(output.shape[1]-n),'input_tokens':int(n)}

    def candidates(self,q,prompt,labels,*,support=()):
        t=self.torch; base=self._prepare(q,prompt,support); n=base['input_ids'].shape[1]; scores=[]
        for label in labels:
            ids=self.processor.tokenizer(label,add_special_tokens=False,return_tensors='pt')['input_ids'].to(self.device)
            x={k:v.clone() if t.is_tensor(v) else v for k,v in base.items()}
            x['input_ids']=t.cat([x['input_ids'],ids],1)
            x['attention_mask']=t.cat([x['attention_mask'],t.ones_like(ids)],1)
            with t.inference_mode(): out=self.thinker(**x,use_cache=False,return_dict=True)
            result=candidate_logprob(out.logits,ids,n); result['candidate']=label; scores.append(result)
        return {'prediction':max(scores,key=lambda z:z['sequence_logprob'])['candidate'],
                'mean_prediction':max(scores,key=lambda z:z['mean_token_logprob'])['candidate'],
                'scores':scores}
