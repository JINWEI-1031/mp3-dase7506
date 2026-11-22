"""Rebuild an event plan when the existing plan projects the wrong entity field."""
import re
from sqlglot import exp
from repair_rules import parse,sql_text
from step_semantic192_rules import generate as original_generate,paths,join_path,tokens
from step_semantic192_agent import Agent as Parent

def fresh_events(seeds,obs,schema):
    q=obs['task']['question'];headers=obs['example']['columns']
    if not re.search(r'earliest|latest|recent|oldest|newest',q,re.I) or not all(re.fullmatch(r'\w+',h) for h in headers):return []
    current=parse(seeds[0]) if seeds else None
    if not isinstance(current,exp.Select) or {h.lower() for h in headers}<={e.alias_or_name.lower() for e in current.expressions}:return []
    if any(v.is_string and len(v.this)>1 and v.this.lower() in q.lower() for s in seeds for v in (parse(s).find_all(exp.Literal) if parse(s) is not None else [])):return []
    roots=[t for t in schema if tokens(t['name'])&tokens(q) and all(any(c['name'].lower()==h.lower() for c in t['columns']) for h in headers)]
    roots.sort(key=lambda t:-len(tokens(t['name'])&tokens(q))/max(1,len(tokens(t['name']))));out=[]
    for root in roots[:1]:
        base=exp.select(*(exp.column(h,table='event_entity') for h in headers)).from_(exp.to_table(root['name']).as_('event_entity'))
        for table in schema:
            for col in table['columns']:
                if not re.search(r'DATE|TIME',col['type'],re.I):continue
                for path in paths(schema,root['name'],table['name'])[:2]:
                    if len(path)>2:continue
                    new,aliases=join_path(base,{'event_entity':root},path)
                    new.set('order',exp.Order(expressions=[exp.Ordered(this=exp.column(col['name'],table=aliases[table['name'].lower()]),desc=bool(re.search(r'latest|recent|newest',q,re.I)))]));new.set('limit',exp.Limit(expression=exp.Literal.number(1)))
                    out.append({'sql':sql_text(new),'family':'event','path':['event_output_entity']})
    return out

def generate(seeds,obs,schema,families):
    fresh=fresh_events(seeds,obs,schema) if 'event' in families else []
    return [*fresh,*original_generate(seeds,obs,schema,families)][:64]

class Agent(Parent):
    def __init__(self,checkpoint=None):
        super().__init__(checkpoint)
        import step_semantic192_agent as api
        api.generate=generate
        old_ready=api.ready
        if not getattr(old_ready,'fresh_output',False):
            def ready(sql,obs,schema,families):
                result=old_ready(sql,obs,schema,families)
                if 'event' in families and fresh_events([sql],obs,schema):result.add('event')
                return result
            ready.fresh_output=True;api.ready=ready
    def act(self,obs):
        schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        if self.queue is not None and self.pending is None and not self.sem_used and schema is not None and obs['remaining']>=3 and fresh_events([self.queue['sql']],obs,schema):
            action=self.choose(obs,schema,self.queue['sql'],[self.queue['sql']],True)
            if action:self.pending=action['sql'];return action
        return super().act(obs)
def create_agent(checkpoint=None):
    agent=Agent(checkpoint);agent.semantic_families={'event','relation','count','negative'};return agent
