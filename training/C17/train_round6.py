"""Approved train coverage336/477 controls, explicit schedules only."""
import os,json,hashlib,time,resource,sys
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import Dataset,SequentialSampler
from transformers import AutoTokenizer,AutoModelForCausalLM,Trainer,TrainerCallback,TrainingArguments,set_seed
from peft import LoraConfig,get_peft_model
R=Path.cwd();S=R.parents[1];O=R/'data';WIN=S/'jobs/matrix-batch1/winner.json'
sys.path.insert(0,str(R/'experiments'))
from round4_encoder import encode
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
class Data(Dataset):
 def __init__(self,t,schedule,corpus):
  lookup={x['id']:x for x in corpus};self.examples=[]
  for i,s in enumerate(schedule):
   p,a=encode(t,lookup[s['row_id']]);assert len(p)+len(a)<=3072
   self.examples.append({'input_ids':p+a,'labels':[-100]*len(p)+a,'exposure':i})
 def __len__(self):return len(self.examples)
 def __getitem__(self,i):return self.examples[i]
class Collate:
 def __call__(self,rr):
  assert len(rr)==1
  r=rr[0];return {k:torch.tensor([r[k]]) for k in ['input_ids','labels']}|{'attention_mask':torch.ones((1,len(r['input_ids'])),dtype=torch.long),'exposure':torch.tensor([r['exposure']])}
class Audit(Trainer):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.indices=[];self.tokens=Counter()
 def _get_train_sampler(self,*a,**kw):return SequentialSampler(self.train_dataset)
 def compute_loss(self,model,inputs,*a,**kw):
  indices=inputs.pop('exposure').tolist()
  if model.training:
   self.indices.extend(indices);self.tokens['input']+=int(inputs['attention_mask'].sum());self.tokens['target']+=int((inputs['labels']!=-100).sum())
   assert self.indices==list(range(len(self.indices))),'Actual schedule changed'
  value=super().compute_loss(model,inputs,*a,**kw);loss=value[0] if isinstance(value,tuple) else value
  assert torch.isfinite(loss).all().item(),'Nonfinite loss';return value
class Callback(TrainerCallback):
 stopped=False;first_gradient=None;gradient_steps=0
 def on_pre_optimizer_step(self,args,state,control,model=None,**kw):
  grads=[]
  for n,p in model.named_parameters():
   if p.requires_grad:
    assert '.lora_A.' in n or '.lora_B.' in n
    assert p.grad is not None,n
    assert torch.isfinite(p.grad).all().item(),n
    grads.append((n,float(p.grad.norm())))
  self.gradient_steps+=1
  if self.first_gradient is None:
   assert any(v>0 for n,v in grads);self.first_gradient={'finite_tensors':len(grads),'nonzero_tensors':sum(v>0 for n,v in grads)}
 def on_step_end(self,args,state,control,**kw):
  pass # Global winner gates new dispatch; an already started training finishes.
  return control
def main():
 group=sys.argv[1];arm=group;plan=read(O/'training-plan.json');spec=plan['runs'][arm];smoke='--smoke' in sys.argv;steps=3 if smoke else spec['optimizer_steps'];exposures=steps*8
 assert read(R/'tokenizer-verification.json')['status']=='read_only_round6_token_schedule_verification_passed'
 for p,h in read(O/'index.json')['files'].items():assert sha(R/p)==h,p
 freeze=read(R/'trainer-freeze.json');assert sha(__file__)==freeze['trainer_sha256'];assert not WIN.exists()
 out=S/'outputs/gpu_lab_v1/runs'/(group+'-round6-smoke' if smoke else group);out.mkdir(parents=True);assert not any(out.iterdir())
 modeldir='/root/autodl-tmp/mp3-lora/model';assert sha(Path(modeldir)/'model.safetensors')=='f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
 t=AutoTokenizer.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False)
 schedule=rows(R/spec['schedule']);assert len(schedule)==spec['exposures'];schedule=schedule[:exposures]
 data=Data(t,schedule,rows(R/spec['corpus']));expected={'input':sum(len(x['input_ids']) for x in data.examples),'target':sum(sum(y!=-100 for y in x['labels']) for x in data.examples)}
 verified=read(R/'tokenizer-verification.json')['totals'][group];assert smoke or expected=={k:verified[k] for k in ['input','target']}
 set_seed(61);base=AutoModelForCausalLM.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,use_safetensors=True,attn_implementation='sdpa');base.config.use_cache=False
 model=get_peft_model(base,LoraConfig(task_type='CAUSAL_LM',r=spec['rank'],lora_alpha=spec['alpha'],lora_dropout=.05,target_modules=['q_proj','v_proj'],bias='none'))
 h=hashlib.sha256();params=0;trainable=0
 for n,p in model.named_parameters():
  params+=p.numel()
  if p.requires_grad:
   assert ('.lora_A.' in n or '.lora_B.' in n) and ('q_proj' in n or 'v_proj' in n);trainable+=p.numel();h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 assert trainable==1146880*spec['rank']//8 and params==596049920+trainable
 initial=h.hexdigest();shared=R/('initial-AB-'+group+'-sha.json')
 if shared.exists():assert read(shared)['sha256']==initial
 else:write(shared,{'sha256':initial,'seed':61,'fresh_base':True})
 write(out/'run_config.json',{'group':group,'arm':arm,'source_freeze':freeze,'plan_sha256':sha(O/'training-plan.json'),'schedule_sha256':sha(R/spec['schedule']),'corpus_sha256':sha(R/spec['corpus']),'actual_passes':1,'steps':steps,'exposures':exposures,'smoke_only':smoke,'old141_role':read(R/'exposure-roles.json')['models'][group]['old141_role'],'old141_gradient_rows':spec['old141_gradient_rows'],'initial_AB_sha256':initial,'parameters':params,'trainable_parameters':trainable,'expected_tokens':expected,'coverage_and_tokens_not_matched':True})
 args=TrainingArguments(output_dir=str(out),seed=61,data_seed=61,learning_rate=2e-4,lr_scheduler_type='cosine',warmup_ratio=.05,weight_decay=0.,optim='adamw_torch',num_train_epochs=1,max_steps=steps,per_device_train_batch_size=1,gradient_accumulation_steps=8,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},bf16=True,logging_steps=1,logging_nan_inf_filter=False,logging_first_step=True,report_to=[],eval_strategy='no',save_strategy='no',remove_unused_columns=False,label_names=['labels'])
 cb=Callback();trainer=Audit(model=model,args=args,train_dataset=data,data_collator=Collate(),callbacks=[cb]);loader=trainer.get_train_dataloader()
 def has_seq(obj,seen=None):
  seen=set() if seen is None else seen
  if id(obj) in seen:return False
  seen.add(id(obj));return isinstance(obj,SequentialSampler) or any(has_seq(getattr(obj,k),seen) for k in ['sampler','batch_sampler'] if hasattr(obj,k))
 assert has_seq(loader),'Sequential sampler not preserved'
 assert not WIN.exists();torch.cuda.reset_peak_memory_stats();start=time.time();result=trainer.train();n=trainer.state.global_step
 if not cb.stopped:assert n==steps and cb.gradient_steps==steps and trainer.indices==list(range(exposures)) and dict(trainer.tokens)==expected
 adapter=out/('interrupted_adapter' if cb.stopped else 'adapter');model.save_pretrained(adapter,safe_serialization=True,save_embedding_layers=False);(adapter/'README.md').unlink(missing_ok=True)
 assert {x.name for x in adapter.iterdir()}=={'adapter_config.json','adapter_model.safetensors'}
 usage=resource.getrusage(resource.RUSAGE_SELF)
 write(out/'metrics.json',dict(result.metrics)|{'optimizer_steps':n,'actual_input_tokens':trainer.tokens['input'],'actual_target_tokens':trainer.tokens['target'],'actual_exposures':len(trainer.indices),'actual_order_SHA256':hashlib.sha256(json.dumps(trainer.indices).encode()).hexdigest(),'actual_source_frequency':dict(Counter(schedule[i]['source_id'] for i in trainer.indices)),'actual_row_frequency':dict(Counter(schedule[i]['row_id'] for i in trainer.indices)),'first_gradient':cb.first_gradient,'all_gradient_steps_finite':cb.gradient_steps,'stopped_by_195_threshold':cb.stopped,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'CPU_peak_RSS_kib':usage.ru_maxrss,'CPU_seconds':usage.ru_utime+usage.ru_stime,'wall_training_seconds':time.time()-start,'history':trainer.state.log_history,'adapter_sha256':{p.name:sha(p) for p in adapter.iterdir()}})
if __name__=='__main__':main()
