"""Independent runnable QP-R/QP-P wrapper. No changes to frozen parent files.

The parent object stays alive and its already-returned action is held verbatim.
All observation history remains genuine and visible. Review check side effects
on inherited prior-pass shortcuts are real experimental costs, not hidden.
"""
import copy
import json
import os
from pathlib import Path
from step4_agent import identity
from mp3_sql.sqlite_tool import validate_sql
from step107_qwen_proactive_protocol import ParentFrame, GenerationLedger, opportunity, prompt_context
from step107_qwen_integration1_evidence import analyze,compare,record

POLICIES=('keep_unknown_public','margin1','margin2','margin0_tie_candidate')
CURRENT=None


class CompletionCap(RuntimeError):pass


def install_meter():
    """Intercept shared active bundle, then its generate method (all call sites)."""
    import step63_agent as api
    if getattr(api.active_model,'integration1_meter',False):return
    original=api.active_model
    import step5_model
    original_base_loader=step5_model.load_model
    def active():
        bundle=original();_,model,info=bundle
        if not getattr(model,'_integration1_meter',False):
            generate=model.generate
            def metered(*args,**kwargs):
                if CURRENT is None:raise RuntimeError('Generation outside an episode owner')
                limit=kwargs.get('max_new_tokens')
                if type(limit) is not int or not 1<=limit<=2048:raise ValueError('Explicit max_new_tokens in [1,2048] required')
                if 'max_length' in kwargs or 'generation_config' in kwargs:raise ValueError('No alternate length/config override')
                config=getattr(model,'generation_config',None)
                for name,default in (('num_return_sequences',1),('num_beams',1)):
                    if kwargs.get(name,getattr(config,name,default))!=1:raise ValueError('Exactly one greedy sequence required')
                if kwargs.get('do_sample',getattr(config,'do_sample',False)):
                    raise ValueError('Greedy decoding required')
                if kwargs.get('return_dict_in_generate',getattr(config,'return_dict_in_generate',False)):
                    raise ValueError('Tensor-only generation output required')
                inputs=args[0] if args else kwargs.get('input_ids',kwargs.get('inputs'))
                shape=getattr(inputs,'shape',None)
                if shape is None or len(shape)!=2 or shape[0]!=1:raise ValueError('Batch size must be exactly one')
                if CURRENT.ledger.total>=6 or (CURRENT.extra and CURRENT.ledger.proactive>=1):
                    raise CompletionCap('completion_cap')
                CURRENT.ledger.register(model,info,CURRENT.extra)
                CURRENT.audit.append({'stage':'generation','extra':CURRENT.extra,'total':CURRENT.ledger.total,'model':info})
                result=generate(*args,**kwargs)
                outshape=getattr(result,'shape',None)
                if outshape is None or len(outshape)!=2 or outshape[0]!=1:raise ValueError('Unexpected generation output batch')
                actual=int(outshape[1]-shape[1])
                if not 0<=actual<=limit:raise ValueError('Actual generated tokens exceed explicit cap')
                CURRENT.audit.append({'stage':'generation_completed','extra':CURRENT.extra,'total':CURRENT.ledger.total,
                                      'input_tokens':int(shape[1]),'actual_output_tokens':actual,'max_new_tokens':limit})
                return result
            model.generate=metered;model._integration1_meter=True
        return bundle
    active.integration1_meter=True
    api.active_model=active
    # Older optional proposers must also receive this adapter bundle. Keep
    # api.load_model untouched: original active_model uses that bound base
    # loader to construct the bundle once, avoiding recursive construction.
    step5_model.load_model=active
    # Imported aliases also use the same patched bundle. No second model loads.
    import sys
    for module in list(sys.modules.values()):
        if module and getattr(module,'__name__','').startswith('step') and getattr(module,'active_model',None) is original:
            module.active_model=active
        if module is not api and module and getattr(module,'__name__','').startswith('step') and getattr(module,'load_model',None) is original_base_loader:
            module.load_model=active


def native_propose(obs,schema,mode):
    import time
    import torch
    import step63_agent as api
    system,context=prompt_context(obs['task']['question'],schema,obs['example']['columns'],'',mode,
                                  original_issue_sql=obs['issue_sql'])
    tokenizer,model,info=api.active_model();tokens=api.prompt_tokens(tokenizer,context,system)
    tensor=torch.tensor([tokens]);start=time.perf_counter()
    with torch.inference_mode():
        output=model.generate(tensor,attention_mask=torch.ones_like(tensor),do_sample=False,max_new_tokens=512,
            temperature=None,top_p=None,top_k=None,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    raw=tokenizer.decode(output[0,len(tokens):],skip_special_tokens=True).strip()
    entry={'id':obs['task']['id'],'context':context,'system':system,'output':raw,'model':info,
           'purpose':'QP-R' if mode=='repair_original' else 'QP-P','new_tokens':len(output[0])-len(tokens),
           'wall_seconds':time.perf_counter()-start,'max_new_tokens':512}
    with Path(os.environ['MP3_STEP63_MODEL_LOG']).open('a',encoding='utf-8') as file:file.write(json.dumps(entry,ensure_ascii=False)+'\n')
    return raw


def public(obs,sql):
    return any(e['tool']=='check' and e.get('sql')==sql and e['observation'].get('ok') and
               e['observation'].get('matches_example') for e in obs['history'])


class Agent:
    def __init__(self,parent,adapter_sha,policy='margin0_tie_candidate',mode='repair_original',proposer=None):
        assert policy in POLICIES and mode in ('repair_original','plan')
        self.parent=parent;self.policy=policy;self.mode=mode;self.proposer=proposer or native_propose
        self.ledger=GenerationLedger(expected_adapter_sha256=adapter_sha)
        self.extra=False;self.used=False;self.frame=None;self.phase=None;self.candidate=None
        self.rejected=set();self.audit=[];self.parent_sql=None;self.owned_sql=set();self.calls=0
        self.pre_review_owned=set()

    def fallback(self,obs):
        review=identity(self.candidate) if self.candidate else None
        eligible=[e['sql'] for e in obs['history'] if e['tool']=='check' and identity(e['sql']) not in self.rejected
                  and identity(e['sql'])!=review and identity(e['sql']) in self.owned_sql and public(obs,e['sql'])]
        if eligible:return eligible[-1]
        if self.frame:
            held=self.frame.restore_action().get('sql')
            if held and identity(held)!=review and identity(held) not in self.rejected:return held
        if self.parent_sql and identity(self.parent_sql)!=review and identity(self.parent_sql) not in self.rejected:return self.parent_sql
        return obs['issue_sql']

    def finish(self,obs,action,schema):
        parent_sql=action['sql']
        # A public-pass review may leak into an inherited frontier's output.
        # Compare it to a genuine nonreview parent alternative, not to itself.
        if self.candidate and identity(parent_sql)==identity(self.candidate) and identity(parent_sql) not in self.pre_review_owned:
            parent_sql=self.fallback(obs)
            self.audit.append({'stage':'review_origin_parent_redirect','inherited_submit':action['sql'],'comparison_parent':parent_sql})
            if identity(action['sql']) in self.rejected:
                self.audit.append({'stage':'firewall','blocked':action['sql'],'fallback':parent_sql})
        if identity(parent_sql) in self.rejected:
            parent_sql=self.fallback(obs);self.audit.append({'stage':'firewall','blocked':action['sql'],'fallback':parent_sql})
        if self.candidate and identity(self.candidate) not in self.rejected:
            pe,pf=analyze(obs,parent_sql,schema);ce,cf=analyze(obs,self.candidate,schema)
            take,reason=compare(pe,ce,public(obs,parent_sql),public(obs,self.candidate),self.policy)
            self.audit.append({'stage':'selection','parent':parent_sql,'candidate':self.candidate,'selected':take,'reason':reason,
                               'parent_evidence':record(pe),'candidate_evidence':record(ce),'facts':{'parent':pf,'candidate':cf}})
            if not take:self.rejected.add(identity(self.candidate))
            else:parent_sql=self.candidate
        return {'tool':'submit','sql':parent_sql}

    def act(self,obs):
        global CURRENT
        previous=CURRENT;CURRENT=self
        try:return self._act(obs)
        finally:CURRENT=previous

    def _act(self,obs):
        self.calls+=1
        if self.calls>9:raise RuntimeError('Official episode step cap')
        schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        if self.phase=='schema':
            assert schema is not None
            return self.generate_review(obs,schema)
        if self.phase=='check':
            assert obs['history'][-1].get('sql')==self.candidate and obs['history'][-1]['tool']=='check'
            ce,facts=analyze(obs,self.candidate,schema or [])
            if ce.status=='HARD_CONFLICT' or not public(obs,self.candidate):self.rejected.add(identity(self.candidate))
            self.audit.append({'stage':'checked','candidate':self.candidate,'public_pass':public(obs,self.candidate),
                               'evidence':record(ce),'facts':facts})
            self.phase='restore'
        if self.phase=='restore':
            action=self.frame.restore_action();self.phase=None
            self.audit.append({'stage':'restore','action':action,'same_parent_object':self.frame.parent is self.parent,
                               'actual_remaining':obs['remaining'],'actual_history_length':len(obs['history'])})
            if action['tool']=='submit':return self.finish(obs,action,schema or [])
            existing=self.frame.existing_pending(obs)
            if existing is None:return action
            # A same-action candidate check may satisfy the held callback only
            # when its true result is already the last global event.
            if existing!=len(obs['history'])-1:raise RuntimeError('Cannot focus an older result without forging history')
            self.audit.append({'stage':'reuse_exact_actual_pending_result','history_index':existing})
        try:
            action=self.parent.act(copy.deepcopy(obs))
        except CompletionCap:
            self.audit.append({'stage':'parent_completion_cap_interruption','remaining':obs['remaining']})
            return self.finish(obs,{'tool':'submit','sql':self.fallback(obs)},schema or [])
        if 'sql' in action:
            self.owned_sql.add(identity(action['sql']));self.parent_sql=action['sql']
        if not self.used and 'sql' in action:
            pe,facts=analyze(obs,action['sql'],schema or [])
            allowed,reason=opportunity(obs,action,pe,self.ledger)
            self.audit.append({'stage':'opportunity','parent_action':action,'eligible':allowed,'reason':reason,
                               'evidence':record(pe),'facts':facts})
            if allowed:
                self.used=True;self.frame=ParentFrame.capture(self.parent,action,obs)
                self.pre_review_owned=set(self.owned_sql)
                if schema is None:self.phase='schema';return {'tool':'schema'}
                return self.generate_review(obs,schema)
        if action['tool']=='submit':return self.finish(obs,action,schema or [])
        return action

    def generate_review(self,obs,schema):
        try:
            self.extra=True
            self.candidate=self.proposer(obs,schema,self.mode)
        except CompletionCap:
            self.audit.append({'stage':'extra_cap_rejected'});self.candidate=None
        finally:self.extra=False
        if self.candidate:
            try:
                validate_sql(self.candidate)
                evidence,facts=analyze(obs,self.candidate,schema)
                self.audit.append({'stage':'generated','candidate':self.candidate,'evidence':record(evidence),'facts':facts})
                if evidence.status=='HARD_CONFLICT':raise ValueError('hard_conflict')
                if any(identity(e.get('sql',''))==identity(self.candidate) for e in obs['history'] if 'sql' in e):
                    self.audit.append({'stage':'duplicate_generation_no_new_check'})
                    self.candidate=None
                else:
                    self.phase='check';return {'tool':'check','sql':self.candidate}
            except ValueError as error:
                self.rejected.add(identity(self.candidate));self.audit.append({'stage':'generation_rejected','reason':str(error)})
        self.phase='restore'
        # No extra official observation exists: resume within this call without
        # increasing the number of external act invocations.
        self.calls-=1
        return self._act(obs)


def create_agent(checkpoint=None):
    import hashlib
    directory=Path(os.environ['MP3_ADAPTER_DIR'])
    sha=hashlib.file_digest((directory/'adapter_model.safetensors').open('rb'),'sha256').hexdigest()
    # Execution config is independently frozen by the integration payload.
    config=json.loads(Path(os.environ.get('MP3_QPI_CONFIG','outputs/step107-qwen-integration1-preregister.json')).read_text(encoding='utf-8'))
    expected=config['adapters'][os.environ.get('MP3_QPI_ADAPTER','historical_D')]['sha256']
    assert sha==expected['adapter_model.safetensors']
    assert hashlib.sha256((directory/'adapter_config.json').read_bytes()).hexdigest()==expected['adapter_config.json']
    install_meter()
    from step107_polarity_selected import create_agent as parent_factory
    return Agent(parent_factory(checkpoint),sha,os.environ.get('MP3_QPI_POLICY','margin0_tie_candidate'),
                 os.environ.get('MP3_QPI_MODE','repair_original'))
