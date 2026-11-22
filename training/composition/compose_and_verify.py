"""Original C4/C17 AB -> two fixed q/v module mixtures. Tiny CPU fixtures only."""
import os,sys,time,json,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def kind(key):
 tail=key.split('.self_attn.',1)[1].split('.',1)[0]
 assert tail in ['q_proj','v_proj'];return tail
def main():
 p=argparse.ArgumentParser();p.add_argument('--guard-receipt',required=True);args=p.parse_args();start=time.monotonic()
 while not Path(args.guard_receipt).exists():
  if time.monotonic()-start>30:raise RuntimeError('No guard receipt')
  time.sleep(.02)
 g=read(args.guard_receipt);assert g['pid']==os.getpid() and g['CPU_affinity_count']==1
 r=read(OUT/'RECIPE-PRECOMMITTED.json');assert sha(__file__)==r['compose_source_SHA256']
 assert [s['module_C4_coefficients'] for s in r['outputs']]==[{'q_proj':.25,'v_proj':.75},{'q_proj':.75,'v_proj':.25}]
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.environ['HF_HUB_OFFLINE']=='1'
 assert all(os.environ[n]=='1' for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
 sys.path[:0]=[str(ROOT/'outputs/coord198_learn_v2/round4/deps'),str(ROOT/'.venv/Lib/site-packages')]
 import torch
 from safetensors.torch import load_file,save_file
 from peft.tuners.lora.layer import Linear as PeftLinear
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_grad_enabled(False);torch.manual_seed(61)
 source={}
 for name,m in r['sources'].items():
  d=ROOT/m['directory'];assert sha(d/'adapter_model.safetensors')==m['weights_SHA256'] and sha(d/'adapter_config.json')==m['config_SHA256']
  source[name]=load_file(str(d/'adapter_model.safetensors'),device='cpu')
 a=source['C4'];b=source['C17'];assert set(a)==set(b) and len(a)==112
 modules=sorted(k[:-len('.lora_A.weight')] for k in a if k.endswith('.lora_A.weight'))
 assert len(modules)==56 and sum(kind(m)=='q_proj' for m in modules)==28 and sum(kind(m)=='v_proj' for m in modules)==28
 outputs=[];checks=[]
 for spec in r['outputs']:
  target={};module_map=spec['module_C4_coefficients']
  for k in sorted(a):
   assert a[k].dtype==b[k].dtype==torch.float32 and a[k].shape==b[k].shape
   assert torch.isfinite(a[k]).all() and torch.isfinite(b[k]).all()
   lam=module_map[kind(k)]
   if k.endswith('.lora_A.weight'):target[k]=torch.cat([a[k],b[k]],dim=0).contiguous()
   else:target[k]=torch.cat([a[k]*(2*lam),b[k]*(2*(1-lam))],dim=1).contiguous()
  rng=torch.Generator(device='cpu').manual_seed(61);probes=[]
  for m in modules:
   ak=m+'.lora_A.weight';bk=m+'.lora_B.weight';lam=module_map[kind(m)]
   x=torch.randn(5,a[ak].shape[1],generator=rng,dtype=torch.float64)
   expected=2*lam*(x@a[ak].double().T)@a[bk].double().T+2*(1-lam)*(x@b[ak].double().T)@b[bk].double().T
   actual=(x@target[ak].double().T)@target[bk].double().T
   stored=(x@a[ak].double().T)@target[bk][:,:16].double().T+(x@b[ak].double().T)@target[bk][:,16:].double().T
   torch.testing.assert_close(actual,expected,atol=1e-8,rtol=1e-6);torch.testing.assert_close(actual,stored,atol=1e-10,rtol=1e-10)
   assert torch.equal(target[ak][:16],a[ak]) and torch.equal(target[ak][16:],b[ak])
   probes.append({'module':m,'module_type':kind(m),'lambda_C4':lam,'formula_storage_max_abs_error':(actual-expected).abs().max().item(),
                  'stored_block_max_abs_error':(actual-stored).abs().max().item()})
  tiny=[]
  for typ in ['q_proj','v_proj']:
   m=next(m for m in modules if kind(m)==typ);ak=m+'.lora_A.weight';bk=m+'.lora_B.weight';lam=module_map[typ]
   base=torch.nn.Linear(17,13,bias=False,dtype=torch.float64)
   layer=PeftLinear(base,'default',r=32,lora_alpha=32,lora_dropout=.05,use_rslora=False,use_dora=False,lora_bias=False).double().eval();layer.requires_grad_(False)
   layer.lora_A['default'].weight.copy_(torch.cat([a[ak][:,:17],b[ak][:,:17]],dim=0).double())
   layer.lora_B['default'].weight.copy_(target[bk][:13].double());x=torch.randn(7,17,generator=rng,dtype=torch.float64)
   expected=torch.nn.functional.linear(x,base.weight)+2*lam*(x@a[ak][:,:17].double().T)@a[bk][:13].double().T+2*(1-lam)*(x@b[ak][:,:17].double().T)@b[bk][:13].double().T
   actual=layer(x);torch.testing.assert_close(actual,expected,atol=1e-8,rtol=1e-6)
   assert layer.scaling['default']==1 and layer.active_adapters==['default'] and not layer.training
   tiny.append({'module_type':typ,'lambda_C4':lam,'dimensions':[17,13],'max_abs_error':(actual-expected).abs().max().item()})
  dest=OUT/spec['name']/'adapter';dest.mkdir(parents=True,exist_ok=False)
  cfg=read(ROOT/r['sources']['C4']['directory']/'adapter_config.json')
  cfg.update(r=32,lora_alpha=32,base_model_name_or_path=r['base_repository'],revision=r['base_revision'],target_modules=['q_proj','v_proj'])
  write(dest/'adapter_config.json',cfg)
  save_file(target,str(dest/'adapter_model.safetensors'),metadata={'format':'pt','fixed_recipe_SHA256':sha(OUT/'RECIPE-PRECOMMITTED.json'),'module_C4_coefficients':json.dumps(module_map,sort_keys=True)})
  saved=load_file(str(dest/'adapter_model.safetensors'),device='cpu');assert all(torch.equal(saved[k],target[k]) for k in target)
  assert sum(v.numel() for v in target.values())==4587520 and (dest/'adapter_model.safetensors').stat().st_size<512*1024**2
  outputs.append({'name':spec['name'],'module_C4_coefficients':module_map,'directory':dest.relative_to(ROOT).as_posix(),
   'weights_SHA256':sha(dest/'adapter_model.safetensors'),'config_SHA256':sha(dest/'adapter_config.json'),
   'adapter_parameters':4587520,'total_deployed_parameters':600637440,'rank':32,'alpha':32,
   'weight_file_bytes':(dest/'adapter_model.safetensors').stat().st_size,'tensor_count':112})
  checks.append({'name':spec['name'],'all_passed':True,'probes':probes,'PEFT_fixtures':tiny,
   'A_source_bits_exact':True,'B_float32_rounding_disclosed':True,'save_reload_bit_exact':True,
   'grad_enabled':torch.is_grad_enabled(),'coefficients_constant_all_layers':True})
 for m in r['sources'].values():
  d=ROOT/m['directory'];assert sha(d/'adapter_model.safetensors')==m['weights_SHA256'] and sha(d/'adapter_config.json')==m['config_SHA256']
 write(OUT/'NUMERICAL-VERIFICATION.json',{'all_passed':True,'outputs':checks,'actual_modules_each':56,'q_modules_each':28,'v_modules_each':28,
  'vectors_each_module':5,'tiny_PEFT_fixtures_each':2,'tolerance':r['numeric_tolerance'],
  'full_base_LLM_predictions_gradients_GPU_SQL_score_or_test_rows':0})
 write(OUT/'COMPOSITION-RESULT.json',{'status':'TWO_FIXED_QV_MODULE_CROSS_SINGLE_R32_ADAPTERS_CREATED','outputs':outputs,
  'recipe_SHA256':sha(OUT/'RECIPE-PRECOMMITTED.json'),'original_source_assets_unchanged':True,
  'full_base_weights_or_LLM_loaded':False,'new_predictions_training_GPU_score_or_test_rows':0,'seconds':time.monotonic()-start})
 print(read(OUT/'COMPOSITION-RESULT.json'))
if __name__=='__main__':main()
