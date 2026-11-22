"""Exact original C4 optional-feedback payload, complete SQL+EOS, no truncation."""
import json
from round4_encoder import SYSTEM

def encode(t,r):
    payload={k:r[k] for k in ['question','draft','columns','schema']}
    if 'feedback' in r:payload['feedback']=r['feedback']
    prompt=t.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(payload,ensure_ascii=False,separators=(',',':'))}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    target=t.encode(r['target_sql'].strip(),add_special_tokens=False)+[t.eos_token_id]
    return prompt,target
