"""Scope-correct semantic ranking; no database/task IDs or grader labels."""
import re
from sqlglot import exp
from repair_rules import parse
from step123_agent import words,singular
from step124_pool_search import sources
from step127_agent import flags

def score(sql,obs,schema):
    t=parse(sql)
    if t is None:return -1000
    q=obs['task']['question'].lower();qw={singular(w) for w in words(q)};value=0.0
    for s in t.find_all(exp.Select):
        src=sources(s,schema);value-=.15*len(s.args.get('joins',[]))
        if s.args.get('distinct') and not re.search(r'\b(?:distinct|different|unique)\b',q):value-=1.2
        for table in src.values():value+=.25*sum(singular(w) in qw for w in words(table['name']))
        group=s.args.get('group')
        if group:
            for alias,table in src.items():
                keys={c['name'].lower() for c in table['columns'] if c.get('pk')}
                cols={c.name.lower() for c in group.expressions if isinstance(c,exp.Column) and (not c.table or c.table.lower()==alias.lower())}
                if keys and keys<=cols:value-=len(cols-keys)
        where=s.args.get('where')
        if where:
            for eq in where.find_all(exp.EQ):
                if eq.find_ancestor(exp.Select) is not s:continue
                for col,lit in ((eq.this,eq.expression),(eq.expression,eq.this)):
                    if not isinstance(col,exp.Column) or not isinstance(lit,exp.Literal) or lit.is_string:continue
                    owners=[table for alias,table in src.items() if not col.table or alias.lower()==col.table.lower()]
                    key=any(any(c['name'].lower()==col.name.lower() and c.get('pk') for c in table['columns']) or any(fk['from'].lower()==col.name.lower() for fk in table['foreign_keys']) for table in owners)
                    if key and not re.search(r'\b'+re.escape(lit.this)+r'\b',q):value-=2.5
        for col in s.find_all(exp.Column):
            if col.find_ancestor(exp.Select) is not s or not src:continue
            # Correlated columns resolve in an enclosing SELECT, not this scope.
            if col.table and col.table.lower() not in {a.lower() for a in src}:continue
            owners=[a for a,table in src.items() if any(c['name'].lower()==col.name.lower() for c in table['columns']) and (not col.table or a.lower()==col.table.lower())]
            if not owners and col.name.lower() not in {e.alias.lower() for e in s.expressions if e.alias}:value-=2
    if isinstance(t,exp.Select):
        order=t.args.get('order');limit=t.args.get('limit');top=re.search(r'\btop\s+(\d+)\b',q)
        first=bool(re.search(r'\bfirst\s+(?!name\b|and\b)[a-z]',q));expected=top[1] if top else '1' if first else None
        if expected:value+=4 if order and limit and limit.expression.name==expected else -4
        if re.search(r'\b(?:total|sum)\b',q):value+=2 if t.find(exp.Sum) else -2
        if re.search(r'\baverage\b',q):value+=2 if t.find(exp.Avg) else -2
        value-=2*len(flags(sql,obs['task']['question'],schema))
    return value

def create_agent(checkpoint=None):
    import step124_pool_agent as controller
    import step124_pool_search as search
    controller.score=score;search.score=score
    return controller.Agent(checkpoint)
