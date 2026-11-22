"""A small evidence ranker over the parent's already checked finite pool."""
import json
import re
from pathlib import Path
from functools import lru_cache
from sqlglot import exp
from repair_rules import parse
from step107_polarity_selected import create_agent as baseline
from step107_rank_features import features,NAMES,key
from step107_frontier_agent import eligible
from step107_frontier_search import sources
from step107_slots_fields import family

@lru_cache(maxsize=1)
def load_model():
    model=json.loads(Path(__file__).with_name('step107_rank_model.json').read_text(encoding='utf-8'))
    assert model['feature_names']==list(NAMES)
    return model

def certificate(parent,candidate,obs):
    if not eligible(candidate,obs):return False,'legacy_parse_width_value_guard'
    a,b=parse(parent),parse(candidate)
    if not isinstance(a,exp.Select) or not isinstance(b,exp.Select):return False,'unsupported_parent_shape'
    # This first ranker does not reopen population, joins, quantifiers or grouping.
    # These are guards, not numeric learned preferences.
    for clause in ('from','joins','where','group','having','distinct','limit','offset','with'):
        if a.args.get(clause)!=b.args.get(clause):return False,'protected_'+clause
    if len(list(a.find_all(exp.Select)))!=len(list(b.find_all(exp.Select))):return False,'scope_changed'
    if a.args.get('order')!=b.args.get('order'):
        q=obs['task']['question'].lower()
        if not re.search(r'\b(?:ascending|descending|increasing|decreasing)\b',q):return False,'order_direction_not_explicit'
        ao=a.args.get('order');bo=b.args.get('order')
        if not ao or not bo or len(ao.expressions)!=len(bo.expressions):return False,'order_shape_changed'
        if any(x.this!=y.this for x,y in zip(ao.expressions,bo.expressions)):return False,'order_object_changed'
    if a.selects==b.selects:return True,'same_projection_protected_context'
    schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
    if schema is None:return None,'need_observed_schema_for_type_guard'
    sa,sb=sources(a,schema),sources(b,schema)
    def typ(e,src):
        c=e.unalias()
        if not isinstance(c,exp.Column):return None
        fs=[f for alias,t in src.items() if not c.table or c.table.lower()==alias.lower() for f in t['columns'] if f['name'].lower()==c.name.lower()]
        return family(fs[0]['type']) if len(fs)==1 else None
    for x,y in zip(a.selects,b.selects):
        if x==y:continue
        tx,ty=typ(x,sa),typ(y,sb)
        if tx is None or ty is None or tx=='unknown' or tx!=ty:return False,'unknown_or_changed_projection_type'
    return True,'same_type_local_projection_only'

def choose(parent,obs,model,pool=None):
    checked=[];seen={key(parent['sql'])}
    for e in obs['history']:
        if e['tool']!='check' or not e['observation'].get('matches_example'):continue
        sql=e['sql'];k=key(sql)
        if pool is not None and k not in {key(s) for s in pool}:continue
        if k in seen:continue
        seen.add(k);checked.append(sql)
    threshold=float(model['threshold']);weights=model['weights'];scale=model['scale']
    def value(sql):return sum(x*w/s for x,w,s in zip(features(sql,obs),weights,scale))
    old=value(parent['sql']);best=old;action=parent;needs_schema=False;diagnostic=[]
    for sql in checked:
        ok,reason=certificate(parent['sql'],sql,obs);margin=value(sql)-old
        diagnostic.append({'guard':reason,'margin':margin})
        if margin<=threshold:continue
        if ok is None:needs_schema=True;continue
        if ok and value(sql)>best:best=value(sql);action={'tool':'submit','sql':sql}
    if needs_schema and action==parent and obs['remaining']>0:return {'tool':'schema'},diagnostic
    return action,diagnostic

class Agent:
    def __init__(self,checkpoint=None):
        self.base=baseline(checkpoint);self.held=None;self.done=False;self.audit=[]
    def act(self,obs):
        action=self.held if self.held is not None else self.base.act(obs)
        if self.done or action['tool']!='submit':return action
        selected,diagnostic=choose(action,obs,load_model());self.audit.append({'remaining':obs['remaining'],'candidates':diagnostic,'selected_tool':selected['tool']})
        if selected['tool']=='schema':self.held=action;return selected
        self.done=True;return selected

def create_agent(checkpoint=None):return Agent(checkpoint)
