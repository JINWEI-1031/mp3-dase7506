"""Refuse a second base/adapter load before it happens; identical in both arms."""
from pathlib import Path
from common import ROOT
def install(cfg):
 from transformers import AutoModelForCausalLM
 from peft import PeftModel
 original_base=AutoModelForCausalLM.from_pretrained
 original_adapter=PeftModel.from_pretrained
 state={'base_loads':0,'adapter_loads':0,'base_object':None}
 def deny(message):
  from runtime import ACTIVE
  active=ACTIVE.get()
  if active is not None:active.failure=message
  raise RuntimeError(message)
 def base(path,*args,**kwargs):
  if Path(path).resolve()!=(ROOT/cfg['base_path']).resolve() or state['base_loads']:
   deny('UNKNOWN: undeclared or repeated base loader')
  if kwargs.get('local_files_only') is not True or kwargs.get('trust_remote_code') is not False:
   deny('UNKNOWN: model must be pinned local only')
  state['base_loads']+=1
  try:state['base_object']=original_base(path,*args,**kwargs)
  except BaseException as e:deny('UNKNOWN: base loader failure '+repr(e))
  return state['base_object']
 def adapter(model,path,*args,**kwargs):
  if model is not state['base_object'] or Path(path).resolve()!=(ROOT/cfg['adapter_path']).resolve() or state['adapter_loads']:
   deny('UNKNOWN: undeclared/repeated adapter or wrong base object')
  if kwargs.get('local_files_only') is not True or kwargs.get('is_trainable') is not False:
   deny('UNKNOWN: adapter must be local eval only')
  state['adapter_loads']+=1
  try:return original_adapter(model,path,*args,**kwargs)
  except BaseException as e:deny('UNKNOWN: adapter loader failure '+repr(e))
 AutoModelForCausalLM.from_pretrained=base
 PeftModel.from_pretrained=adapter
 return state
