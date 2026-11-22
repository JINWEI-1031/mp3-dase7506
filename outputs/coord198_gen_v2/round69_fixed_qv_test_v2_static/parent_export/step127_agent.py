"""Schema-evidenced review and compact additive feedback repair over Step124."""
import os
import re
import json
import time
from pathlib import Path
from sqlglot import exp
from repair_rules import parse
from step4_agent import identity,passed
from step123_agent import singular,words
from step124_agent import create_agent as baseline
import step125_agent as old
import step63_agent as model_api


def scope_sources(select,schema):
    known={t['name'].lower():t for t in schema};tables=[]
    root=select.args.get('from')
    if root is not None and isinstance(root.this,exp.Table):tables.append(root.this)
    tables += [j.this for j in select.args.get('joins',[]) if isinstance(j.this,exp.Table)]
    return {t.alias_or_name.lower():known[t.name.lower()] for t in tables if t.name.lower() in known}


def flags(sql,question,schema=()):
    tree=parse(sql);q=question.lower();out=set()
    if tree is None:return {'invalid_sql'}
    if not isinstance(tree,exp.Select):return out
    order=tree.args.get('order')
    if re.search(r'\b(?:alphabetical|alphabetically|lexicographic)\b',q) and order:
        for item in order.expressions:
            col=item.this
            if isinstance(col,exp.Column) and re.search(r'(?:_id|id)$',col.name,re.I):out.add('alphabetical_order_uses_identifier')
    if not schema:return out
    for select in tree.find_all(exp.Select):
        sources=scope_sources(select,schema)
        def bind(col):
            if not isinstance(col,exp.Column):return None
            candidates=[(a,t,c) for a,t in sources.items() for c in t['columns'] if c['name'].lower()==col.name.lower() and (not col.table or a==col.table.lower())]
            return candidates[0] if len(candidates)==1 else None
        for join in select.args.get('joins',[]):
            on=join.args.get('on')
            if on is None:continue
            for eq in on.find_all(exp.EQ):
                a,b=bind(eq.this),bind(eq.expression)
                if not a or not b or a[0]==b[0]:continue
                pairs=[(a,b),(b,a)]
                for key,text in pairs:
                    numeric=bool(re.search(r'INT|REAL|NUM|DEC|FLOAT|DOUBLE',key[2].get('type',''),re.I))
                    is_key=bool(key[2].get('pk') or any(f['from'].lower()==key[2]['name'].lower() for f in key[1]['foreign_keys']))
                    if numeric and is_key and re.search(r'name|detail|description',text[2]['name'],re.I) and re.search(r'CHAR|TEXT|CLOB',text[2].get('type',''),re.I):out.add('identifier_joined_to_descriptive_text')
                links=[]
                for left,right in pairs:
                    links += [(left[0],f['from'].lower(),right[0],f['to'].lower()) for f in left[1]['foreign_keys'] if f['table'].lower()==right[1]['name'].lower()]
                if len(links)==1:
                    expected=links[0];actual={(a[0],a[2]['name'].lower(),b[0],b[2]['name'].lower()),(b[0],b[2]['name'].lower(),a[0],a[2]['name'].lower())}
                    if expected not in actual:out.add('join_conflicts_with_unique_declared_fk')
    return out


def preflight(sql,question):
    t=parse(sql)
    if not isinstance(t,exp.Select):return False
    if flags(sql,question):return True
    # A cheap syntactic warning is required before buying missing schema metadata.
    for join in t.args.get('joins',[]):
        on=join.args.get('on')
        if on is None:continue
        for eq in on.find_all(exp.EQ):
            if isinstance(eq.this,exp.Column) and isinstance(eq.expression,exp.Column):
                names=(eq.this.name,eq.expression.name)
                if any(re.search(r'(?:_id|id)$',a,re.I) and re.search(r'name|detail|description',b,re.I) for a,b in (names,names[::-1])):return True
    return False


def linked_schema(schema,sql,question):
    t=parse(sql);names={x.name.lower() for x in t.find_all(exp.Table)} if t else set()
    qwords={singular(w) for w in words(question)}
    names|={table['name'].lower() for table in schema if words(table['name']) and all(singular(w) in qwords for w in words(table['name']))}
    neighbors={fk['table'].lower() for table in schema if table['name'].lower() in names for fk in table['foreign_keys']}
    # Incoming relationships often contain the missing count/association entity.
    neighbors|={table['name'].lower() for table in schema if any(fk['table'].lower() in names for fk in table['foreign_keys'])}
    return [t for t in schema if t['name'].lower() in names|neighbors]


def focused_payload(obs,schema,current,mode,diagnosis):
    checked=[e for e in obs['history'] if e['tool']=='check']
    relevant=[e for e in checked if identity(e['sql'])==identity(current)]
    last=relevant[-1] if relevant else checked[-1] if checked else None
    feedback=None if last is None else {'sql':last['sql'],'error':last['observation']['error'],'matches_example':last['observation']['matches_example'],
                                       'returned_columns':last['observation']['columns'],'returned_row_count':last['observation']['row_count']}
    return {'question':obs['task']['question'],'draft':current,'columns':obs['example']['columns'],
            'schema':schema if mode=='full' else linked_schema(schema,current,obs['task']['question']),
            'public_example_shape':{'row_count':obs['example']['row_count'],'ordered':obs['ordered'],'truncated':obs['example'].get('truncated',False)},
            'repair_focus':sorted(diagnosis),'last_execution':feedback,
            'instruction':'Correct the draft for the question. Fix the listed structural problem or execution error. Preserve unrelated filters and named values. Return SQL only. Do not copy example answer rows.'}


def generate(obs,schema,current,mode,diagnosis):
    if mode=='raw':return old.generate(obs,schema,current,False,'new_risk_raw')
    import torch
    context=focused_payload(obs,schema,current,mode,diagnosis)
    key=json.dumps(context,sort_keys=True)
    if key in old._CACHE:return old._CACHE[key]
    tokenizer,model,info=model_api.active_model();tokens=model_api.prompt_tokens(tokenizer,context,old.SYSTEM)
    tensor=torch.tensor([tokens]);start=time.perf_counter()
    with torch.inference_mode():
        output=model.generate(tensor,attention_mask=torch.ones_like(tensor),do_sample=False,max_new_tokens=old.MAX_NEW,
            temperature=None,top_p=None,top_k=None,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    tail=output[0,len(tokens):];raw=tokenizer.decode(tail,skip_special_tokens=True).strip();old._CACHE[key]=raw
    record={'id':obs['task']['id'],'model':info,'context':context,'system_prompt':old.SYSTEM,'output':raw,
            'purpose':'focused_'+mode,'max_new_tokens':old.MAX_NEW,'new_tokens':len(tail),'wall_seconds':time.perf_counter()-start}
    with Path(os.environ['MP3_STEP125_MODEL_LOG']).open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    return raw


def preserves_named_values(oldsql,newsql,question):
    a,b=parse(oldsql),parse(newsql)
    if a is None or b is None:return False
    newvalues=' '.join(v.this.lower() for v in b.find_all(exp.Literal) if v.is_string)
    for value in a.find_all(exp.Literal):
        if not value.is_string:continue
        text=value.this.strip('%').lower()
        if len(text)>1 and text in question.lower() and not set(words(text))<=set(words(newvalues)):return False
    return True


class FocusedReservation(old.Agent):
    def act(self,obs):
        schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        if self.reserved and schema is not None and self.pending is None and not self.done:
            checked=[e['sql'] for e in obs['history'] if e['tool']=='check']
            current=checked[-1] if checked else self.fallback['sql']
            sql=generate(obs,schema,current,'linked',{'reserved_execution_error'})
            self.done=True
            if obs['remaining'] and old.valid_candidate(sql,obs) and preserves_named_values(current,sql,obs['task']['question']):
                self.pending=sql;return {'tool':'check','sql':sql}
            return self.fallback
        return super().act(obs)


class Agent:
    def __init__(self,checkpoint=None):
        self.arm=os.environ.get('MP3_STEP127_ARM','baseline');self.base=baseline(checkpoint)
        if self.arm in ('raw_reserve','combined_reserved','focused_reserve','combined_focused_reserve','combined_strong','combined_structured'):
            prior=os.environ.get('MP3_STEP125_ARMS');os.environ['MP3_STEP125_ARMS']='reserve_type'
            try:self.base=FocusedReservation(checkpoint) if 'focused_reserve' in self.arm else old.Agent(checkpoint)
            finally:
                if prior is None:os.environ.pop('MP3_STEP125_ARMS',None)
                else:os.environ['MP3_STEP125_ARMS']=prior
        self.fallback=None;self.pending=None;self.done=False;self.review=False;self.reason=None;self.initial=set()

    def act(self,obs):
        if self.pending:
            good=passed(obs['history'][-1]['observation'])
            return {'tool':'submit','sql':self.pending} if good else self.fallback
        if self.done:return self.fallback
        if self.fallback is None:
            action=self.base.act(obs)
            if action['tool']!='submit':return action
            self.fallback=action
        if self.arm in ('baseline','raw_reserve','focused_reserve'):self.done=True;return self.fallback
        remaining=obs['remaining'];schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        matched=any(e['tool']=='check' and passed(e['observation']) for e in obs['history'])
        review=self.arm in ('risk_raw','risk_full','risk_focus','combined','combined_reserved','combined_focused_reserve','risk_strong','combined_strong','risk_structured','structured_strong','combined_structured') and matched
        retry=self.arm in ('retry_focus','combined','combined_reserved','combined_focused_reserve','combined_strong','combined_structured') and not matched
        if not remaining or not (review or retry):self.done=True;self.reason='gate';return self.fallback
        if review and self.arm in ('risk_structured','structured_strong','combined_structured'):
            shape=parse(self.fallback['sql'])
            if not isinstance(shape,exp.Select) or len(list(shape.find_all(exp.Select)))!=1 or any(j.args.get('side') for j in shape.args.get('joins',[])):
                self.done=True;self.reason='structured_shape_preflight';return self.fallback
        if schema is None:
            if remaining<2 or review and not preflight(self.fallback['sql'],obs['task']['question']):self.done=True;self.reason='schema_or_budget_gate';return self.fallback
            return {'tool':'schema'}
        self.initial=flags(self.fallback['sql'],obs['task']['question'],schema)
        if review and not self.initial:self.done=True;self.reason='no_schema_evidenced_risk';return self.fallback
        if review and self.arm in ('risk_strong','combined_strong','structured_strong','combined_structured') and not self.initial & {'identifier_joined_to_descriptive_text','alphabetical_order_uses_identifier'}:
            self.done=True;self.reason='declared_fk_alone_insufficient';return self.fallback
        current=self.fallback['sql']
        if retry:
            checked=[e['sql'] for e in obs['history'] if e['tool']=='check']
            current=checked[-1] if checked else current
        diagnosis=self.initial if review else {'public_mismatch_or_execution_error'}
        mode='raw' if self.arm=='risk_raw' else 'full' if self.arm=='risk_full' else 'linked'
        self.review=review;self.done=True
        if review and self.arm in ('risk_structured','structured_strong','combined_structured'):
            from step127_structured import propose
            sql=propose(current,obs['task']['question'],schema,self.initial)
            if sql is None:self.reason='no_unique_structured_plan';return self.fallback
        else:sql=generate(obs,schema,current,mode,diagnosis)
        if not old.valid_candidate(sql,obs):self.reason='invalid_width_or_duplicate';return self.fallback
        if review and not flags(sql,obs['task']['question'],schema)<self.initial:self.reason='risk_not_reduced';return self.fallback
        if not preserves_named_values(current,sql,obs['task']['question']):self.reason='named_filter_dropped';return self.fallback
        self.pending=sql;self.reason='checked';return {'tool':'check','sql':sql}


def create_agent(checkpoint=None):return Agent(checkpoint)
