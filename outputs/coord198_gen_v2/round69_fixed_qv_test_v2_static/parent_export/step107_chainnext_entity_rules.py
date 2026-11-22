import re
from sqlglot import exp
from repair_rules import parse,sql_text
from step123_agent import words

def shape(sql,obs):
    t=parse(sql)
    if not isinstance(t,exp.Select) or len(list(t.find_all(exp.Select)))!=1:return None
    if any(t.args.get(k) for k in ('group','having','limit','order','distinct')):return None
    root=t.args.get('from');where=t.args.get('where')
    if not root or not isinstance(root.this,exp.Table) or not where or not isinstance(where.this,exp.EQ):return None
    eq=where.this
    if not isinstance(eq.this,exp.Column) or not isinstance(eq.expression,exp.Literal) or not eq.expression.is_string:return None
    alias=root.this.alias_or_name
    if not all(isinstance(e.unalias(),exp.Column) and e.unalias().table.lower()==alias.lower() for e in t.expressions):return None
    if eq.this.table.lower()==alias.lower():return None
    noun=' '.join(words(root.this.name));literal=eq.expression.this
    if not re.search(r'\b'+re.escape(noun)+r'\s+(?:(?:named|called)\s+)?'+re.escape(literal.lower())+r'\b',obs['task']['question'].lower()):return None
    return t,root.this,literal

def entity_risk(sql,obs):return {'explicit_entity_role'} if shape(sql,obs) else set()

def entity_satisfied(sql,obs):
    t=parse(sql)
    if not isinstance(t,exp.Select) or t.args.get('joins') or len(list(t.find_all(exp.Select)))!=1:return False
    root=t.args.get('from');w=t.args.get('where')
    if not root or not isinstance(root.this,exp.Table) or not w or not isinstance(w.this,exp.EQ):return False
    eq=w.this
    if not isinstance(eq.this,exp.Column) or not eq.this.name.lower().endswith('name') or not isinstance(eq.expression,exp.Literal) or not eq.expression.is_string:return False
    noun=' '.join(words(root.this.name))
    return bool(re.search(r'\b'+re.escape(noun)+r'\s+(?:(?:named|called)\s+)?'+re.escape(eq.expression.this.lower())+r'\b',obs['task']['question'].lower()))

def entity_plan(sql,obs,schema):
    item=shape(sql,obs)
    if item is None:return None
    t,root,literal=item
    table=next((x for x in schema if x['name'].lower()==root.name.lower()),None)
    if table is None:return None
    names=[c['name'] for c in table['columns'] if c['name'].lower().endswith('name')]
    if len(names)!=1:return None
    if any(e.unalias().name.lower() not in {c['name'].lower() for c in table['columns']} for e in t.expressions):return None
    q=exp.select(*[e.copy() for e in t.expressions]).from_(root.copy()).where(exp.EQ(this=exp.column(names[0],table=root.alias_or_name),expression=exp.Literal.string(literal)))
    return sql_text(q)
