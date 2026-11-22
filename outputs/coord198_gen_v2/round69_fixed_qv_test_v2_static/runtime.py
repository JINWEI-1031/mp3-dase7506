"""Observational native+late generation counter. Native source/prompt/parse unchanged."""
import contextvars,copy,time
from common import append,digest
ACTIVE=contextvars.ContextVar('late_rescue_actual_episode',default=None)
class RuntimeIntegrityError(RuntimeError):pass
class InputUnavailable(ValueError):pass
class State:
 def __init__(self,tid):self.id=tid;self.completions=0;self.phase='native';self.observation=None;self.records=[];self.attempts=[];self.failure=None
class Runtime:
 def __init__(self,config,log):self.config=config;self.log=log;self.bundle=None;self.allowed=set();self.loads=0
 def observe_bundle(self,bundle):
  tok,model,info=bundle
  if info.get('adapter',{}).get('sha256')!=self.config['adapter_SHA256'] or info.get('weights_sha256')!=self.config['base_weights_SHA256'] or info.get('parameters')!=self.config['expected_total_parameters'] or info.get('adapter',{}).get('rank')!=self.config['adapter_rank'] or set(info.get('adapter',{}).get('target_modules',[]))!={'q_proj','v_proj'}:raise RuntimeIntegrityError('Not the one declared arm bundle')
  if self.bundle is not None and self.bundle[1] is not model:raise RuntimeIntegrityError('A second model bundle is forbidden')
  if self.bundle is None:
   self.bundle=bundle;self.loads+=1;self.allowed={id(model),id(model.base_model),id(model.base_model.model)}
   if model.training or any(p.device.type!='cpu' for p in model.parameters()):raise RuntimeIntegrityError('Expected declared CPU eval model')
  return bundle
 def install(self,step63,generation_mixin=None):
  original_loader=step63.active_model;runtime=self
  def observed_loader():
   try:return runtime.observe_bundle(original_loader())
   except BaseException as e:
    if ACTIVE.get() is not None:ACTIVE.get().failure=repr(e)
    raise
  step63.active_model=observed_loader;self.loader=observed_loader
  if generation_mixin is None:
   from transformers.generation.utils import GenerationMixin
   generation_mixin=GenerationMixin
  original_generate=generation_mixin.generate
  def tracked(model,*args,**kwargs):
   state=ACTIVE.get()
   if state is None:raise RuntimeIntegrityError('Generation outside actual episode')
   if self.bundle is None or id(model) not in self.allowed:raise RuntimeIntegrityError('Undeclared native model path; never silently replace weights')
   if state.completions>=6:raise RuntimeIntegrityError('native+late completion cap6')
   limit=kwargs.get('max_new_tokens')
   if type(limit) is not int or not 0<limit<=512:raise RuntimeIntegrityError('Unexpected native generation cap; preserve source, stop UNKNOWN')
   tensor=kwargs.get('input_ids',kwargs.get('inputs',args[0] if args else None))
   if tensor is None or tensor.shape[0]!=1:raise RuntimeIntegrityError('Expected one real input')
   ids=tensor[0].tolist()
   if not ids or len(ids)>3072:raise RuntimeIntegrityError('native or late input outside cap3072; no truncation')
   state.completions+=1;start=time.monotonic()
   attempt={'id':state.id,'phase':state.phase,'completion_ordinal':state.completions,'input_ids':ids,'max_new_tokens':limit,'status':'STARTED'};state.attempts.append(attempt);append(self.log,dict(attempt))
   try:output=original_generate(model,*args,**kwargs)
   except BaseException as error:
    attempt.update(status='FAILED',error=repr(error),seconds=time.monotonic()-start);append(self.log,dict(attempt));raise
   attempt.update(status='RETURNED',seconds=time.monotonic()-start);append(self.log,dict(attempt));tail=output[0,len(ids):]
   if len(tail)>limit:raise RuntimeIntegrityError('Actual output exceeds frozen native/late cap')
   tok=self.bundle[0];raw=tok.decode(tail,skip_special_tokens=True);record={'id':state.id,'phase':state.phase,'completion_ordinal':state.completions,'input_token_count':len(ids),'input_ids':ids,'input_ids_SHA256':__import__('hashlib').sha256(__import__('json').dumps(ids).encode()).hexdigest(),'actual_decoded_prompt':tok.decode(tensor[0],skip_special_tokens=False),'raw_output':raw,'returned_text':raw.strip(),'actual_new_tokens':len(tail),'max_new_tokens':limit,'actual_history':copy.deepcopy(state.observation['history']),'remaining_paid_tools':state.observation['remaining'],'model_adapter_SHA256':self.config['adapter_SHA256'],'base_weights_SHA256':self.config['base_weights_SHA256'],'seconds':time.monotonic()-start};state.records.append(record);append(self.log,record);return output
  def guarded_generate(model,*args,**kwargs):
   try:return tracked(model,*args,**kwargs)
   except BaseException as e:
    if ACTIVE.get() is not None:ACTIVE.get().failure=repr(e)
    raise
  generation_mixin.generate=guarded_generate
 def preflight_late(self,messages):
  if self.bundle is None:
   from transformers import AutoTokenizer
   from common import ROOT
   tok=AutoTokenizer.from_pretrained(str(ROOT/self.config['base_path']),local_files_only=True)
  else:tok=self.bundle[0]
  ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
  if not ids or len(ids)>3072:raise InputUnavailable('late input empty or >3072; no truncation')
  return ids
 def late(self,ids,**generation):
  state=ACTIVE.get()
  if state is None:raise RuntimeIntegrityError('No actual late episode')
  tok,model,info=self.loader()
  import torch
  tensor=torch.tensor([ids]);old=state.phase;state.phase='late'
  try:
   with torch.inference_mode():out=model.generate(tensor,attention_mask=torch.ones_like(tensor),do_sample=False,max_new_tokens=512,temperature=None,top_p=None,top_k=None,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
  finally:state.phase=old
  return tok.decode(out[0,len(ids):],skip_special_tokens=True).strip()
class ActiveAgent:
 def __init__(self,agent,state):self.agent=agent;self.state=state
 def act(self,obs):
  self.state.observation=obs;token=ACTIVE.set(self.state)
  try:return self.agent.act(obs)
  finally:
   ACTIVE.reset(token)
   if self.state.failure is not None:raise RuntimeIntegrityError('sticky actual model failure: '+self.state.failure)
class ParentTap:
 def __init__(self,parent,state):self.parent=parent;self.state=state;self.events=[];self.final=None
 def act(self,obs):
  before=self.state.completions;a=self.parent.act(obs);self.events.append({'observation':copy.deepcopy(obs),'action':copy.deepcopy(a),'native_completions_before':before,'native_completions_after':self.state.completions})
  if a['tool']=='submit':self.final={'SQL':a['sql'],'actual_history':copy.deepcopy(obs['history']),'native_model_records':[{'input_ids_SHA256':r['input_ids_SHA256'],'returned_text':r['returned_text'],'actual_new_tokens':r['actual_new_tokens'],'max_new_tokens':r['max_new_tokens']} for r in self.state.records if r['phase']=='native'],'native_completions':self.state.completions,'parent_action_events':copy.deepcopy(self.events)}
  return a
