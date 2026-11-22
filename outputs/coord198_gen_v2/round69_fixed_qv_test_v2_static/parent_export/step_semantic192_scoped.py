"""Head/modifier and relation-role checks for semantic proposal selection."""
import re
from sqlglot import exp
from repair_rules import parse,sql_text
from step_pool192_rules import schema_errors
from step_semantic192_rules import tokens,conjuncts
from step_semantic192_fresh import generate as original_generate,fresh_events
from step_semantic192_ranked import Agent as Ranked,score as ranked_score

def overlap(a,b):
    return any(x==y or min(len(x),len(y))>=3 and x[:min(4,len(x),len(y))]==y[:min(4,len(x),len(y))] for x in a for y in b)
def clean(sql,q):
    t=parse(sql)
    if not isinstance(t,exp.Select) or not t.args.get('where') or re.search(r'\b(?:description|details|type)\b',q,re.I):return sql
    ps=conjuncts(t.args['where'].this);named={v.this.lower() for p in ps for v in p.find_all(exp.Literal) if v.is_string and any(re.search(r'name',c.name,re.I) for c in p.find_all(exp.Column))}
    keep=[p for p in ps if not (any(re.search(r'description|details',c.name,re.I) for c in p.find_all(exp.Column)) and any(v.is_string and v.this.lower() in named for v in p.find_all(exp.Literal)))]
    if len(keep)==len(ps):return sql
    t.set('where',exp.Where(this=exp.and_(*(p.copy() for p in keep))) if keep else None);return sql_text(t)
def generate(seeds,obs,schema,families):
    q=obs['task']['question'];current=parse(seeds[0]) if seeds else None
    alternatives=strict_events(seeds,obs,schema) if 'event' in families else []
    alternatives+=original_generate(seeds,obs,schema,families-{'relation'})
    if 'relation' in families:alternatives+=original_generate([clean(s,q) for s in seeds],obs,schema,{'relation'})
    out=[];head=re.search(r'\b(?:average|total|sum)\s+(?:of\s+)?(.+?)(?:\s+(?:of|for|in|across|with|where|that|who)\b|[?.]|$)',q,re.I)
    head_tokens=tokens(head[1])-{'the','all','a','an'} if head else set()
    for item in alternatives:
        kind=item['path'][0];tree=parse(item['sql'])
        if kind=='relation_minimal' and schema_errors(seeds[0],obs,schema)==0:continue
        if kind=='measure_field' and head_tokens and not overlap(head_tokens,{w for e in tree.expressions for c in e.find_all(exp.Column) for w in tokens(c.name)}):continue
        if kind.startswith('event_') and re.search(r'\b(?:before|after|earlier than|later than)\b',q,re.I) and current is not None and any(s.find(exp.Max) or s.find(exp.Min) for s in current.find_all(exp.Subquery)):continue
        out.append(item)
    return out
def strict_events(seeds,obs,schema):
    normalized=[]
    for sql in seeds:
        t=parse(sql)
        if isinstance(t,exp.Select):
            for node in list(t.find_all(exp.Alias)):node.replace(node.this.copy())
            normalized.append(sql_text(t))
    return fresh_events(normalized,obs,schema)
def score(item,current,obs,schema):
    value=ranked_score(item,current,obs,schema)
    if item['family']=='negative':
        tree=parse(item['sql']);q=tokens(obs['task']['question'])
        if tree:
            for eq in tree.find_all(exp.EQ):
                if isinstance(eq.this,exp.Column) and isinstance(eq.expression,exp.Column):
                    value+=sum(.5 for col in (eq.this,eq.expression) for word in tokens(col.name)-{'id'} if overlap({word},q))
    return value
class Agent(Ranked):
    def __init__(self,checkpoint=None):
        super().__init__(checkpoint)
        import step_semantic192_agent as api
        api.generate=generate;api.semantic_score=score
        previous=api.ready
        if not getattr(previous,'scoped_output',False):
            def ready(sql,obs,schema,families):
                result=previous(sql,obs,schema,families)
                if 'event' in families and strict_events([sql],obs,schema):result.add('event')
                return result
            ready.scoped_output=True;ready.fresh_output=True;api.ready=ready
        self.base=Frontier(self.base,self)
class Frontier:
    def __init__(self,base,owner):self.base=base;self.owner=owner;self.held=None;self.tried=False
    def act(self,obs):
        schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        if self.held is not None:
            if not self.tried and schema is not None and obs['remaining']>=3:
                self.tried=True;action=self.owner.choose(obs,schema,self.held['sql'],[self.held['sql']],True)
                if action:return action
            action=self.held;self.held=None;return action
        action=self.base.act(obs)
        if self.tried or self.owner.sem_used or action['tool']!='check' or obs['remaining']<3 or not any(e['tool']=='check' for e in obs['history']):return action
        q=obs['task']['question'];t=parse(action['sql']);headers={h.lower() for h in obs['example']['columns']}
        actual={c.name.lower() for e in t.expressions for c in e.find_all(exp.Column)} if isinstance(t,exp.Select) else set()
        if not re.search(r'earliest|latest|recent|oldest|newest',q,re.I) or headers<=actual:return action
        self.held=action
        if schema is None:return {'tool':'schema'}
        return self.act(obs)
def create_agent(checkpoint=None):
    a=Agent(checkpoint);a.semantic_families={'event','relation','count','negative'};return a
