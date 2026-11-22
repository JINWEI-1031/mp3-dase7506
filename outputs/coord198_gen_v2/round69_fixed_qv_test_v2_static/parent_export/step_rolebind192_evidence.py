"""Reject role-unsupported anti-relations and temporal projection/top-1 confusion."""
import re
from sqlglot import exp
from repair_rules import parse
from step124_pool_search import sources
from step_rolebind192_rules import generate as proposed,role_words,ROLE_GROUPS

def supported_negative(tree,q,schema):
    src=sources(tree,schema);root=tree.args.get('from')
    if not root or not isinstance(root.this,exp.Table):return False
    alias=root.this.alias_or_name;parent=src.get(alias)
    if parent is None:return False
    # The counted entity must be the entity requested before its modifier.
    request=re.search(r'\bhow many\s+(.+?)(?:\s+(?:do|does|have|has|are|is|with|without|that)\b|[?.]|$)',q,re.I)
    if tree.find(exp.Count) and request and not role_words(parent['name'])<=role_words(request[1]):return False
    for exists in tree.find_all(exp.Exists):
        sub=exists.this
        if not isinstance(sub,exp.Select):return False
        children=sources(sub,schema);links=[]
        for eq in sub.find_all(exp.EQ):
            if eq.find_ancestor(exp.Select) is not sub:continue
            for a,b in ((eq.this,eq.expression),(eq.expression,eq.this)):
                if not isinstance(a,exp.Column) or not isinstance(b,exp.Column) or b.table.lower()!=alias.lower():continue
                child=next((t for ca,t in children.items() if ca.lower()==a.table.lower()),None)
                if child is None:continue
                declared=any(f['from'].lower()==a.name.lower() and f['table'].lower()==parent['name'].lower() and f['to'].lower()==b.name.lower() for f in child['foreign_keys'])
                explicit_role=any(role_words(a.name)&g and role_words(q)&g for g in ROLE_GROUPS)
                target_pk=any(c['name'].lower()==b.name.lower() and c.get('pk') for c in parent['columns'])
                links.append(declared or explicit_role and target_pk)
        if not links or not all(links):return False
    return True

def generate(seeds,obs,schema,*,new=True,cap=24):
    out=[];q=obs['task']['question']
    for item in proposed(seeds,obs,schema,new=new,cap=cap):
        tree=parse(item['sql'])
        if item['family']=='negative' and not supported_negative(tree,q,schema):continue
        if item['family']=='event' and tree.args.get('limit'):
            projected_time=all(any(re.search(r'date|time|year|logon',c.name,re.I) for c in e.find_all(exp.Column)) for e in tree.expressions)
            if projected_time and not re.search(r'\b(?:who|which person|which student|which employee)\b',q,re.I):continue
        out.append(item)
    return out
