"""Independent semantic repair switches over Step119; no task identifiers or gold answers."""
import os
import re
from sqlglot import exp
from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step119_agent import create_agent as baseline


def quoted(s):
    return '"'+s.replace('"','""')+'"'


def plans(obs, sql, schema, arms):
    q=obs['task']['question'].lower();tree=parse(sql);out=[]
    if not isinstance(tree,exp.Select):
        return out
    if 'negation' in arms and re.search(r'\bhow many\b',q) and 'do not have' in q:
        threshold=re.search(r'longer than\s+(\d+(?:\.\d+)?)',q)
        for child in schema:
            lengths=[c['name'] for c in child['columns'] if c['name'].lower()=='length']
            for fk in child['foreign_keys']:
                parent=next((t for t in schema if t['name'].lower()==fk['table'].lower()),None)
                if parent is None:continue
                noun=parent['name'].lower();forms={noun,noun+'s',noun[:-1]+'ies' if noun.endswith('y') else noun}
                if threshold and len(lengths)==1 and len(obs['example']['columns'])==1 and any(re.search(r'\b'+re.escape(n)+r'\b',q) for n in forms):
                    out.append(('negation',f'SELECT COUNT(*) FROM {quoted(parent["name"])} AS p WHERE NOT EXISTS (SELECT 1 FROM {quoted(child["name"])} AS c WHERE c.{quoted(fk["from"])} = p.{quoted(fk["to"])} AND c.{quoted(lengths[0])} > {threshold[1]})'))
    if 'entity' in arms and not any(tree.args.get(k) for k in ('group','having','order','limit','distinct')):
        projections=[e.unalias() for e in tree.expressions]
        if projections and all(isinstance(e,exp.Column) and e.table for e in projections) and len({e.table.lower() for e in projections})==1:
            table=next((t for t in schema if t['name'].lower()==projections[0].table.lower()),None)
            literals=[v.this for v in tree.find_all(exp.Literal) if v.is_string]
            if table and len(literals)==1 and re.search(r'\b'+re.escape(table['name'].lower())+r'\s+'+re.escape(literals[0].lower())+r'\b',q):
                names=[c['name'] for c in table['columns'] if c['name'].lower().endswith('name')]
                if len(names)==1:
                    new=exp.select(*[e.copy() for e in tree.expressions]).from_(exp.to_table(table['name'])).where(exp.EQ(this=exp.column(names[0],table=table['name']),expression=exp.Literal.string(literals[0])))
                    out.append(('entity',sql_text(new)))
    seen={identity(e['sql']) for e in obs['history'] if e['tool']=='check'}
    out=list(dict.fromkeys(out))
    # Ambiguous schema bindings must not be selected by iteration order.
    return [(kind,sql) for kind,sql in out if sum(k==kind for k,s in out)==1 and identity(sql) not in seen]


def category_plan(obs,schema):
    q=obs['task']['question'].lower()
    if not re.search(r'\bteam\s+\w',q):return None
    heads=obs['example']['columns'];choices=[]
    for table in schema:
        if not all(any(c['name'].lower()==h.lower() for c in table['columns']) for h in heads):continue
        for col in table['columns']:
            if col['name'].lower()=='team':
                choices.append((table['name'],col['name'],f'SELECT DISTINCT {quoted(col["name"])} FROM {quoted(table["name"])} LIMIT 32'))
    return choices[0] if len(choices)==1 else None


class Agent:
    def __init__(self,checkpoint=None):
        self.base=baseline(checkpoint)
        self.arms=set(os.environ.get('MP3_STEP120_ARMS','negation,entity,category').split(','))
        self.category_done=False;self.probe=None;self.pending=None;self.fallback=None;self.tried=set()

    def act(self,obs):
        history=obs['history'];remaining=obs['remaining']
        if self.pending:
            if passed(history[-1]['observation']):return {'tool':'submit','sql':self.pending}
            self.pending=None
        if self.probe:
            table,col,probe=self.probe;self.probe=None
            sample=history[-1]['observation'];q=obs['task']['question'].lower()
            values=[r[0] for r in sample.get('rows',[]) if len(r)==1 and isinstance(r[0],str) and re.search(r'\b'+re.escape(r[0].lower())+r'\b',q)]
            if sample.get('ok') and not sample.get('truncated') and len(values)==1 and remaining:
                tree=exp.select(*[exp.column(h,quoted=True) for h in obs['example']['columns']]).from_(exp.to_table(table)).where(exp.EQ(this=exp.column(col,quoted=True),expression=exp.Literal.string(values[0])))
                sql=sql_text(tree)
                if not any(e['tool']=='check' and identity(e['sql'])==identity(sql) for e in history):
                    self.pending=sql;return {'tool':'check','sql':sql}
        schema=next((e['observation'] for e in history if e['tool']=='schema'),None)
        # Category grounding needs an execute and a check; reserve them before baseline exhaustion.
        if 'category' in self.arms and not self.category_done and any(e['tool']=='repair_hint' for e in history) and re.search(r'\bteam\s+\w',obs['task']['question'],re.I):
            if schema is None and remaining>=3:return {'tool':'schema'}
            self.category_done=True
            plan=category_plan(obs,schema or [])
            if plan and remaining>=2:
                self.probe=plan;return {'tool':'execute','sql':plan[2]}
        if self.fallback is None:
            action=self.base.act(obs)
            if action['tool']!='submit':return action
            self.fallback=action
        q=obs['task']['question'].lower()
        relevant=('negation' in self.arms and 'do not have' in q and 'longer than' in q or 'entity' in self.arms and isinstance(parse(self.fallback['sql']),exp.Select) and bool(list(parse(self.fallback['sql']).find_all(exp.Literal))))
        if schema is None and relevant and remaining>=2:return {'tool':'schema'}
        if schema and remaining:
            for kind,sql in plans(obs,self.fallback['sql'],schema,self.arms):
                if kind not in self.tried:
                    self.tried.add(kind);self.pending=sql;return {'tool':'check','sql':sql}
        return self.fallback


def create_agent(checkpoint=None):return Agent(checkpoint)
