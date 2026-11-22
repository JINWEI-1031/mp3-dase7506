"""One local unchanged public Coder model; never load Qwen3 or an adapter."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import time

REPO='Qwen/Qwen2.5-Coder-0.5B-Instruct'
REVISION='ea3f2471cf1b1f0db85067f1ef93848e38e88c25'
SYSTEM='Repair the SQLite query to answer the question. Output only the corrected SQL.'
MAX_NEW=512

def memory():
    import psutil
    info=psutil.Process().memory_info()
    return {'rss_bytes':info.rss,'peak_rss_bytes':getattr(info,'peak_wset',info.rss)}

@lru_cache(maxsize=1)
def active_model():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    start=time.perf_counter();cpu=time.process_time()
    path=Path(os.environ.get('MP3_CODER_MODEL',str(Path(__file__).resolve().parents[1]/'models/step107-coderprobe'/REVISION)))
    assert path.name==REVISION
    provision=json.loads((Path(__file__).resolve().parents[1]/'outputs/step107-coderprobe-provision.json').read_text(encoding='utf-8'))
    with (path/'model.safetensors').open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    assert sha==provision['files']['model.safetensors']['sha256']
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    options={'local_files_only':True,'trust_remote_code':False}
    tok=AutoTokenizer.from_pretrained(path,**options)
    model=AutoModelForCausalLM.from_pretrained(path,use_safetensors=True,torch_dtype=torch.float32,**options).eval()
    n=sum(p.numel() for p in model.parameters());assert n==494032768 and n<=1_000_000_000
    assert not hasattr(model,'peft_config') and model.__class__.__name__=='Qwen2ForCausalLM'
    assert all(p.device.type=='cpu' for p in model.parameters())
    model.requires_grad_(False)
    info={'repository':REPO,'revision':REVISION,'parameters_including_embeddings':n,'weights_sha256':sha,'adapter':None,
          'device':'cpu','dtype':'float32','threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
          'torch':torch.__version__,'class':model.__class__.__name__,'load_wall_seconds':time.perf_counter()-start,
          'load_cpu_seconds':time.process_time()-cpu,'live_LLM_instances':1,'D_loaded':False,**memory()}
    return tok,model,info

def extract(raw):
    text=raw.strip();m=re.fullmatch(r'```(?:sql|sqlite)?\s*\n?(.*?)\n?```',text,re.S|re.I)
    return m[1].strip() if m else text

def generate(payload,source,phase):
    import torch
    tok,model,info=active_model()
    messages=[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(payload,ensure_ascii=False,separators=(',',':'))}]
    ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
    x=torch.tensor([ids]);wall=time.perf_counter();cpu=time.process_time()
    with torch.inference_mode():
        y=model.generate(x,attention_mask=torch.ones_like(x),do_sample=False,max_new_tokens=MAX_NEW,temperature=None,top_p=None,top_k=None,
                        pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    new=y[0,len(ids):];raw=tok.decode(new,skip_special_tokens=True).strip();sql=extract(raw)
    log={'id':source,'phase':phase,'model':info,'messages':messages,'payload':payload,'prompt':tok.decode(ids,skip_special_tokens=False),
         'payload_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),'output':raw,'extracted_sql':sql,
         'input_tokens':len(ids),'new_tokens':len(new),'max_new_tokens':MAX_NEW,'decode':'greedy','wall_seconds':time.perf_counter()-wall,
         'cpu_seconds':time.process_time()-cpu,**memory()}
    with Path(os.environ.get('MP3_CODER_LOG','outputs/step107-coderprobe-model.jsonl')).open('a',encoding='utf-8') as f:f.write(json.dumps(log,ensure_ascii=False)+'\n')
    return sql
