"""Terminal event-role review over the immutable complete frontier chain."""
from sqlglot import exp
from step107_frontier_selected import create_agent as baseline
from step107_countrole_portable import parse,event_counts,typed_literals,sources,role_words
from step107_frontier_agent import eligible
from step4_agent import identity,passed
import re

ARMS=('baseline','existing')

def all_entities(question):
    return bool(re.search(r'\b(?:each|every|all)\b',question,re.I))

def safe(tree,obs,schema):
    if len(list(tree.find_all(exp.Select)))!=1 or tree.args.get('distinct'):return False
    src=sources(tree,schema)
    if not src:return False
    projected=[c for e in tree.expressions if not e.find(exp.Count) for c in e.find_all(exp.Column)]
    def owner(c):
        choices=[a for a,t in src.items() if (not c.table or a.lower()==c.table.lower()) and any(f['name'].lower()==c.name.lower() for f in t['columns'])]
        return choices[0] if len(choices)==1 else None
    owners={owner(c) for c in projected}
    if None in owners or len(owners)!=1:return False
    root=next(iter(owners))
    if any(isinstance(c.this,exp.Distinct) for c in tree.find_all(exp.Count)):return False
    if any(not isinstance(j.args.get('on'),exp.EQ) for j in tree.args.get('joins',[])):return False
    for part in ('where','having','order','group'):
        clause=tree.args.get(part)
        if clause and any(owner(c)!=root for c in clause.find_all(exp.Column)):return False
    q=role_words(obs['task']['question'])
    if any(a!=root and role_words(t['name']) and role_words(t['name'])<=q for a,t in src.items()):return False
    return True

def proposals(sql,obs,schema,arm):
    tree=parse(sql)
    if not isinstance(tree,exp.Select):return []
    if arm=='existing':
        from step107_countrole_existing import generate
        return generate(sql,obs,schema) if all_entities(obs['task']['question']) else []
    if arm in ('guard','combined') and not safe(tree,obs,schema):return []
    pool=[p for p in event_counts(tree,obs,schema) if typed_literals(p['sql'],schema) and eligible(p['sql'],obs)]
    if arm in ('zero','combined') and all_entities(obs['task']['question']):
        # A missing child row is one NULL-extended row. COUNT(child FK), not
        # COUNT(*), contributes zero. Reject INNER rather than merely giving
        # LEFT a small preference that another score could overwhelm.
        pool=[p for p in pool if any(j.args.get('side')=='LEFT' for j in parse(p['sql']).args.get('joins',[]))]
    return sorted(pool,key=lambda p:-p['confidence'])

class Agent:
    def __init__(self,checkpoint=None,*,arm='existing'):
        self.base=baseline(checkpoint);self.arm=arm
        self.fallback=None;self.pending=None;self.done=False;self.audit=[]
    def act(self,obs):
        if self.arm=='baseline':return self.base.act(obs)
        if self.pending:
            self.done=True
            return {'tool':'submit','sql':self.pending} if passed(obs['history'][-1]['observation']) else self.fallback
        if self.fallback is None:
            action=self.base.act(obs)
            if action['tool']!='submit':return action
            self.fallback=action
        if self.done or not obs['remaining']:return self.fallback
        t=parse(self.fallback['sql'])
        if not isinstance(t,exp.Select) or not t.find(exp.Count) or not t.args.get('group'):return self.fallback
        schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        if schema is None:return {'tool':'schema'} if obs['remaining']>=2 else self.fallback
        seen={identity(e['sql']) for e in obs['history'] if e['tool']=='check'}
        for item in proposals(self.fallback['sql'],obs,schema,self.arm):
            if identity(item['sql']) in seen:continue
            self.pending=item['sql'];self.audit.append(item)
            return {'tool':'check','sql':self.pending}
        self.done=True;return self.fallback

def create_agent(checkpoint=None):return Agent(checkpoint)
