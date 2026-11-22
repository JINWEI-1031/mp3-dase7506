"""E3 selective semantic review and E4 reserved, feedback-conditioned repair."""
import hashlib
import json
import os
import re
import time
from pathlib import Path
from sqlglot import exp
from mp3_sql.sqlite_tool import validate_sql
from repair_rules import parse
from step4_agent import identity,passed,continuation_conflicts
import step63_agent as model_api
from step124_agent import create_agent as baseline

MAX_NEW=256
SYSTEM='Repair the SQLite query to answer the question. Output only the corrected SQL.'
_CACHE={}


def risks(sql,question):
    t=parse(sql)
    # UNION/INTERSECT/EXCEPT are legitimate query forms, not semantic contradictions.
    if t is None:return {'invalid_sql'}
    if not isinstance(t,exp.Select):return set()
    q=question.lower();out=set(continuation_conflicts(sql,question) or ())
    if re.search(r'\b(?:without|do not have|does not have|have no|has no|not any)\b',q) and not any(t.find_all(exp.Not,exp.Except)):
        out.add('negation_missing')
    direction=re.search(r'\b(?:ordered|sorted)\s+(ascending|descending)\b',q)
    if direction:
        order=t.args.get('order')
        if not order or len(order.expressions)==1 and bool(order.expressions[0].args.get('desc'))!=(direction[1]=='descending'):
            out.add('explicit_direction')
    return out


def payload(obs,schema,current,feedback=False):
    context=model_api.payload(obs,schema)
    if feedback:
        checks=[e for e in obs['history'] if e['tool']=='check']
        context={**context,'current_sql':current,'public_example':obs['example'],
            'feedback':[{'sql':e['sql'],'ok':e['observation']['ok'],'error':e['observation']['error'],
                         'columns':e['observation']['columns'],'rows':e['observation']['rows'],
                         'matches_example':e['observation']['matches_example']} for e in checks[-4:]],
            'instruction':'Use the execution feedback to fix remaining errors. Do not repeat a failed SQL. Public matching alone does not prove the question is answered.'}
    return context


def generate(obs,schema,current,feedback=False,purpose='repair'):
    import torch
    context=payload(obs,schema,current,feedback);key=json.dumps(context,sort_keys=True)
    if key in _CACHE:return _CACHE[key]
    tokenizer,model,info=model_api.active_model()
    tokens=model_api.prompt_tokens(tokenizer,context,SYSTEM);tensor=torch.tensor([tokens]);start=time.perf_counter()
    with torch.inference_mode():
        output=model.generate(tensor,attention_mask=torch.ones_like(tensor),do_sample=False,max_new_tokens=MAX_NEW,
            temperature=None,top_p=None,top_k=None,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    tail=output[0,len(tokens):];raw=tokenizer.decode(tail,skip_special_tokens=True).strip();_CACHE[key]=raw
    path=Path(os.environ.get('MP3_STEP125_MODEL_LOG',os.environ['MP3_STEP63_MODEL_LOG']))
    record={'id':obs['task']['id'],'model':info,'system_prompt':SYSTEM,'context':context,'output':raw,
            'max_new_tokens':MAX_NEW,'new_tokens':len(tail),'wall_seconds':time.perf_counter()-start,'purpose':purpose,
            'input_sha256':hashlib.sha256(key.encode()).hexdigest()}
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    return raw


def valid_candidate(sql,obs):
    try:t=validate_sql(sql)
    except ValueError:return False
    return len(t.selects)==len(obs['example']['columns']) and identity(sql) not in {identity(e['sql']) for e in obs['history'] if e['tool']=='check'}


class ConditionedModelAgent(model_api.Agent):
    def __init__(self,original,feedback):
        super().__init__(original.baseline);self.feedback=feedback

    def act(self,obs):
        if self.suffix is None:
            action=self.baseline.act(obs)
            if action['tool']!='submit':return action
            self.suffix=model_api.ModelSuffix(action,proposer=lambda state,schema:generate(state,schema,action['sql'],self.feedback,'internal_feedback' if self.feedback else 'internal_raw256'))
        return self.suffix.act(obs)


class Agent:
    def __init__(self,checkpoint=None):
        self.base=baseline(checkpoint);self.arms=set(os.environ.get('MP3_STEP125_ARMS','baseline').split(','))
        feedback=bool(self.arms & {'feedback','e4','all','e4_type','all_type'})
        if feedback or 'raw_control' in self.arms:
            node=self.base
            while hasattr(node,'base'):
                if isinstance(node.base,model_api.Agent):node.base=ConditionedModelAgent(node.base,feedback);break
                node=node.base
        self.feedback=feedback;self.fallback=None;self.pending=None;self.review=False;self.done=False;self.reserved=False

    def act(self,obs):
        history=obs['history'];remaining=obs['remaining'];schema=next((e['observation'] for e in history if e['tool']=='schema'),None)
        if self.pending is not None:
            good=passed(history[-1]['observation'])
            if self.review:good=good and risks(self.pending,obs['task']['question'])<risks(self.fallback['sql'],obs['task']['question'])
            return {'tool':'submit','sql':self.pending} if good else self.fallback
        if self.done:return self.fallback
        if self.reserved and schema is not None:
            sql=generate(obs,schema,self.fallback['sql'],self.feedback,'reserved_feedback' if self.feedback else 'reserved_raw')
            self.done=True
            if remaining and valid_candidate(sql,obs):self.pending=sql;return {'tool':'check','sql':sql}
            return self.fallback
        # Type-level variants reserve before asking the baseline for its next action:
        # otherwise the baseline may already generate or give up at the budget boundary.
        if self.fallback is None and self.arms & {'reserve_type','e4_type','all_type'}:
            failed=[e for e in history if e['tool']=='check' and not passed(e['observation'])]
            matched=any(e['tool']=='check' and passed(e['observation']) for e in history)
            errors=[e['observation'].get('error') for e in failed[-2:]]
            repeated=len(errors)==2 and all(isinstance(e,dict) for e in errors) and errors[0].get('type')==errors[1].get('type')
            if not matched and repeated and remaining==(1 if schema is not None else 2):
                good_execution=[e['sql'] for e in failed if e['observation']['ok']]
                self.fallback={'tool':'submit','sql':good_execution[-1] if good_execution else obs['issue_sql']};self.reserved=True
                if schema is None:return {'tool':'schema'}
                return self.act(obs)
        if self.fallback is None:
            action=self.base.act(obs)
            failed=[e for e in history if e['tool']=='check' and not passed(e['observation'])]
            matched=any(e['tool']=='check' and passed(e['observation']) for e in history)
            errors=[e['observation'].get('error') for e in failed[-2:]]
            type_mode=bool(self.arms & {'reserve_type','e4_type','all_type'})
            repeated_error=len(errors)==2 and all(isinstance(e,dict) for e in errors) and (errors[0].get('type')==errors[1].get('type') if type_mode else errors[0]==errors[1])
            reserve=bool(self.arms & {'reserve','e4','all','reserve_type','e4_type','all_type'}) and not matched and repeated_error and action['tool'] not in ('submit','schema') and remaining==(1 if schema is not None else 2)
            if reserve:
                good_execution=[e['sql'] for e in failed if e['observation']['ok']]
                self.fallback={'tool':'submit','sql':good_execution[-1] if good_execution else obs['issue_sql']};self.reserved=True
                if schema is None:return {'tool':'schema'}
                return self.act(obs)
            if action['tool']!='submit':return action
            self.fallback=action
        initial=risks(self.fallback['sql'],obs['task']['question'])
        matched=any(e['tool']=='check' and passed(e['observation']) for e in history)
        # A small example is not sufficient: require a concrete semantic contradiction as well.
        eligible=bool(self.arms & {'e3','all','all_type'}) and matched and bool(initial) and initial!={'distinct'}
        if not eligible or not remaining:self.done=True;return self.fallback
        if schema is None:
            if remaining<2:self.done=True;return self.fallback
            return {'tool':'schema'}
        self.done=True;self.review=True
        sql=generate(obs,schema,self.fallback['sql'],self.feedback,'semantic_review')
        if valid_candidate(sql,obs) and risks(sql,obs['task']['question'])<initial:
            self.pending=sql;return {'tool':'check','sql':sql}
        return self.fallback


def create_agent(checkpoint=None):return Agent(checkpoint)
