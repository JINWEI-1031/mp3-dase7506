"""Unchanged public native text-to-SQL model, question + observed schema only."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

REPO='gaussalgo/T5-LM-Large-text2sql-spider'
REVISION='f0a1f3ec0a40a26a7e94619d62c2992a82bf83bc'
MAX_NEW=512
ROOT=Path(__file__).resolve().parents[1]

def serialize(question,schema):
    """Card: quoted non-FK columns, FK typed relation declarations, ordered PKs."""
    quote=lambda s:'"'+s.replace('"','""')+'"'
    parts=[]
    for table in schema:
        columns={c['name']:c for c in table['columns']};fk={k['from'] for k in table['foreign_keys']}
        ordinary=[f"{quote(c['name'])} {c['type'].lower()}" for c in table['columns'] if c['name'] not in fk]
        foreign=[f"{quote(k['from'])} {columns[k['from']]['type'].lower()} from {quote(k['table'])} {quote(k['to'])}" for k in table['foreign_keys']]
        primary=[c for c in table['columns'] if c.get('pk')]
        primary.sort(key=lambda c:int(c['pk']))
        parts.append(quote(table['name'])+' '+ ' , '.join(ordinary)+' , foreign_key: '+' , '.join(foreign)+' primary key: '+' '.join(quote(c['name']) for c in primary))
    return ' '.join(['Question: ',question,'Schema:',' [SEP] '.join(parts)])

def memory():
    import psutil
    i=psutil.Process().memory_info();return {'rss_bytes':i.rss,'peak_rss_bytes':getattr(i,'peak_wset',i.rss)}

@lru_cache(maxsize=1)
def active_model():
    import torch
    from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
    start=time.perf_counter();cpu=time.process_time();path=Path(os.environ.get('MP3_T5_MODEL',str(ROOT/'models/step107-t5probe'/REVISION)))
    assert path.name==REVISION
    provision=json.loads((ROOT/'outputs/step107-t5probe-provision.json').read_text())
    with (path/'model.safetensors').open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    assert sha==provision['files']['model.safetensors']['sha256']
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    options={'local_files_only':True,'trust_remote_code':False}
    tokenizer=AutoTokenizer.from_pretrained(path,**options)
    model=AutoModelForSeq2SeqLM.from_pretrained(path,use_safetensors=True,torch_dtype=torch.float32,**options).eval()
    parameters=sum(p.numel() for p in model.parameters());assert parameters==783150080 and parameters<=1_000_000_000
    assert not hasattr(model,'peft_config') and model.__class__.__name__=='T5ForConditionalGeneration'
    assert all(p.device.type=='cpu' for p in model.parameters());model.requires_grad_(False)
    import gc
    live=[type(x).__name__ for x in gc.get_objects() if type(x).__name__ in ('T5ForConditionalGeneration','Qwen2ForCausalLM','Qwen3ForCausalLM','PeftModelForCausalLM')]
    assert live==['T5ForConditionalGeneration'],live
    info={'repository':REPO,'revision':REVISION,'parameters_including_embeddings':parameters,'weights_sha256':sha,'adapter':None,
          'device':'cpu','dtype':'float32','threads':4,'interop_threads':1,'class':model.__class__.__name__,'torch':torch.__version__,
          'load_wall_seconds':time.perf_counter()-start,'load_cpu_seconds':time.process_time()-cpu,'live_model_object_audit':live,
          'D_or_Coder_loaded':False,**memory()}
    return tokenizer,model,info

def generate(question,schema,source,phase='native_once'):
    import torch
    tok,model,info=active_model();prompt=serialize(question,schema)
    # T5 relative positions support the full observed schema. Do not silently
    # truncate to the tokenizer's nominal512 or append unsupported feedback.
    encoded=tok(prompt,return_tensors='pt',truncation=False);wall=time.perf_counter();cpu=time.process_time()
    with torch.inference_mode():
        output=model.generate(**encoded,do_sample=False,num_beams=1,max_new_tokens=MAX_NEW)
    ids=output[0];new=ids[1:];raw=tok.decode(ids,skip_special_tokens=True).strip()
    record={'id':source,'phase':phase,'model':info,'prompt':prompt,'schema':schema,'question':question,
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'output':raw,'input_tokens':len(encoded['input_ids'][0]),
            'new_tokens':len(new),'decoder_total_tokens':len(ids),'decoder_start_token_excluded':True,'max_new_tokens':MAX_NEW,
            'decode':'greedy;num_beams1','truncated_input':False,'chat_template_used':False,'feedback_in_model_input':False,
            'wall_seconds':time.perf_counter()-wall,'cpu_seconds':time.process_time()-cpu,**memory()}
    with Path(os.environ.get('MP3_T5_LOG','outputs/step107-t5probe-model.jsonl')).open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    return raw
