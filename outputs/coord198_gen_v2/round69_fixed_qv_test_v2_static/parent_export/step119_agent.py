"""Step117 plus a conservative unique two-edge foreign-key bridge review."""
from collections import deque
import os
from sqlglot import exp
from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step117_agent import create_agent as prior_agent
from step61_agent import create_agent as rules


def bridge(sql, schema):
    tree = parse(sql)
    if not isinstance(tree, exp.Select) or len(list(tree.find_all(exp.Select))) != 1:
        return None
    tables = list(tree.find_all(exp.Table))
    joins = tree.args.get('joins', [])
    if len(tables) != 2 or len(joins) != 1 or any(t.alias for t in tables):
        return None
    join = joins[0]
    if join.args.get('side') or join.args.get('kind') not in (None, '', 'INNER') or not isinstance(join.args.get('on'), exp.EQ):
        return None
    graph = {}
    # Composite relationships need a different reconstruction; skip them here.
    for table in schema:
        for fk in table['foreign_keys']:
            a,b=table['name'],fk['table']
            graph.setdefault(a.lower(), []).append((b,a,fk['from'],b,fk['to']))
            graph.setdefault(b.lower(), []).append((a,a,fk['from'],b,fk['to']))
    start,end=tables[0].name,tables[1].name
    queue=deque([(start,[],{start.lower()})]);paths=[]
    while queue:
        name,path,seen=queue.popleft()
        if len(path)>2:
            continue
        if name.lower()==end.lower():
            paths.append(path)
            continue
        for nxt,a,af,b,bf in graph.get(name.lower(),[]):
            if nxt.lower() not in seen:
                queue.append((nxt,[*path,(nxt,a,af,b,bf)],seen|{nxt.lower()}))
    if not paths:
        return None
    shortest=min(map(len,paths))
    choices=[p for p in paths if len(p)==shortest]
    if shortest!=2 or len(choices)!=1:
        return None
    new=tree.copy();new.set('joins',[])
    for nxt,a,af,b,bf in choices[0]:
        new.append('joins',exp.Join(this=exp.to_table(nxt),on=exp.EQ(this=exp.column(af,table=a),expression=exp.column(bf,table=b))))
    return sql_text(new)


class Agent:
    def __init__(self, checkpoint=None):
        self.base=prior_agent(checkpoint)
        if os.environ.get('MP3_STEP119_MODEL')=='off':
            self.base.base=rules(checkpoint)
        self.fallback=None
        self.candidate=None

    def act(self, obs):
        if self.candidate is not None:
            return {'tool':'submit','sql':self.candidate} if passed(obs['history'][-1]['observation']) else self.fallback
        if self.fallback is None:
            action=self.base.act(obs)
            if action['tool']!='submit':
                return action
            self.fallback=action
        if os.environ.get('MP3_STEP119_BRIDGE','on')=='off' or not obs['remaining']:
            return self.fallback
        tree=parse(self.fallback['sql'])
        if not isinstance(tree,exp.Select) or len(list(tree.find_all(exp.Table)))!=2 or len(list(tree.find_all(exp.Select)))!=1:
            return self.fallback
        schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        if schema is None:
            return {'tool':'schema'} if obs['remaining']>=2 else self.fallback
        sql=bridge(self.fallback['sql'],schema)
        seen={identity(e['sql']) for e in obs['history'] if e['tool']=='check'}
        if sql is None or identity(sql) in seen:
            return self.fallback
        self.candidate=sql
        return {'tool':'check','sql':sql}


def create_agent(checkpoint=None):
    return Agent(checkpoint)
