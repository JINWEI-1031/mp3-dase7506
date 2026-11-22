"""Runtime-only provenance, structural diversity and explicit semantic guards."""
import re,json
from sqlglot import exp
from repair_rules import parse,sql_text
from step124_pool_search import sources
from step_semantic192_rules import typed_literals
from step_pool192_rules import schema_errors
from step4_agent import identity

def structural_key(sql,schema):
    tree=parse(sql)
    if tree is None:return identity(sql)
    scopes=list(tree.find_all(exp.Select));bindings={}
    for i,s in enumerate(scopes):
        src=sources(s,schema);bindings[id(s)]={a.lower():f's{i}_t{j}' for j,a in enumerate(src)}
    for s in scopes:
        local=bindings[id(s)]
        for c in s.find_all(exp.Column):
            if c.find_ancestor(exp.Select) is not s or not c.table:continue
            owner=s
            while owner is not None:
                bound=bindings.get(id(owner),{})
                if c.table.lower() in bound:c.set('table',exp.to_identifier(bound[c.table.lower()]));break
                owner=owner.find_ancestor(exp.Select)
        for t in s.find_all(exp.Table):
            if t.find_ancestor(exp.Select) is s and t.alias_or_name.lower() in local:t.set('alias',exp.TableAlias(this=exp.to_identifier(local[t.alias_or_name.lower()])))
    for n in tree.find_all(exp.Identifier):n.set('this',n.this.lower());n.set('quoted',False)
    return sql_text(tree)

def structural_bucket(sql,schema):
    t=parse(structural_key(sql,schema))
    if t is None:return 'invalid'
    return json.dumps({'paths':[sql_text(j) for j in t.find_all(exp.Join)],'aggregates':[sql_text(a) for a in t.find_all(exp.AggFunc)],'scopes':[sql_text(s.args['where']) if s.args.get('where') else '' for s in t.find_all(exp.Select)],'setops':[type(s).__name__ for s in t.find_all(exp.SetOperation)]},sort_keys=True)

def question_values(obs):
    q=obs['task']['question'].lower();t=parse(obs['issue_sql']);strings=[]
    if t:
        for v in t.find_all(exp.Literal):
            if v.is_string and len(v.this.strip('%'))>1 and v.this.strip('%').lower() in q:
                value=v.this.strip('%').lower();eq=v.find_ancestor(exp.EQ)
                entity_names={n.name.lower().rstrip('s') for n in t.find_all(exp.Table)}
                implicit_entity_code=eq is not None and any(re.search(r'type.*code|category.*code',c.name,re.I) for c in eq.find_all(exp.Column)) and value in entity_names and not re.search(r'\b(?:type|types|category|categories)\b',q) and not re.search(r"['\"]"+re.escape(value)+r"['\"]",q)
                if not implicit_entity_code:strings.extend(re.findall(r'[a-z0-9]+',value))
    return set(strings)

def guard(sql,obs,schema):
    t=parse(sql)
    if t is None:return 'invalid_sql'
    if schema_errors(sql,obs,schema):return 'schema_conflict'
    if not typed_literals(sql,schema):return 'type_conflict'
    if len(t.selects)!=len(obs['example']['columns']):return 'projection_contract'
    actual=set(re.findall(r'[a-z0-9]+',' '.join(v.this for v in t.find_all(exp.Literal) if v.is_string).lower()))
    if not question_values(obs)<=actual:return 'named_value_lost'
    q=obs['task']['question']
    required=set(re.findall(r'\b\d+(?:\.\d+)?\b',q));present={n for v in t.find_all(exp.Literal) for n in re.findall(r'\b\d+(?:\.\d+)?\b',v.this)}
    if not required<=present:return 'numeric_constraint_lost'
    negative_boolean=False
    for e in t.find_all(exp.EQ):
        column=e.this if isinstance(e.this,exp.Column) else e.this.this if isinstance(e.this,exp.Lower) and isinstance(e.this.this,exp.Column) else None
        if column is not None and isinstance(e.expression,exp.Literal) and e.expression.this.lower() in ('no','false','0','n') and re.sub(r'\W','',column.name).lower() in re.sub(r'\W','',q).lower():negative_boolean=True
    if re.search(r'\b(?:without|not|no|never|doesn.t|don.t)\b',q,re.I) and not (negative_boolean or t.find(exp.Not) or t.find(exp.Except) or any(j.args.get('side')=='LEFT' for j in t.find_all(exp.Join))):return 'negative_scope_lost'
    if re.search(r'\b(?:ascending|descending|alphabetical|alphabetically)\b',q,re.I) and not t.args.get('order'):return 'ordering_lost'
    # SQLite accepts arbitrary ungrouped values; such a query is not a justified entity plan.
    for select in t.find_all(exp.Select):
        group=select.args.get('group')
        if not group:continue
        grouped={sql_text(c).lower() for c in group.find_all(exp.Column)};src=sources(select,schema)
        for e in select.expressions:
            if e.find(exp.AggFunc):continue
            for c in e.find_all(exp.Column):
                if sql_text(c).lower() in grouped:continue
                owners=[(a,table) for a,table in src.items() if (not c.table or c.table.lower()==a.lower()) and any(x['name'].lower()==c.name.lower() for x in table['columns'])]
                dependent=False
                for a,table in owners:
                    keys=[x['name'].lower() for x in table['columns'] if x.get('pk')]
                    names={x.name.lower() for x in group.find_all(exp.Column) if not x.table or x.table.lower()==a.lower()}
                    if keys and set(keys)<=names:dependent=True
                if not dependent:return 'ungrouped_output_entity'
    return None

def seed_set(current,obs,*,multi=False):
    values=[{'sql':current,'source':'current'}]
    if multi:
        values.append({'sql':obs['issue_sql'],'source':'issue'})
        checks=[e for e in obs['history'] if e['tool']=='check']
        values.extend({'sql':e['sql'],'source':'checked_parent'} for e in reversed(checks))
        for e in obs['history']:
            if e['tool']=='repair_hint':values.extend({'sql':s,'source':'reached_hint'} for s in e['observation'])
    selected=[];seen=set();sources_seen=set()
    # Keep one seed per source first, then remaining structurally different seeds.
    for pass_ in (0,1):
        for value in values:
            key=identity(value['sql'])
            if key in seen or pass_==0 and value['source'] in sources_seen:continue
            selected.append(value);seen.add(key);sources_seen.add(value['source'])
            if len(selected)==4:return selected
    return selected

def diverse(items,obs,schema,cap=24):
    unique={};rejected=[]
    for c in items:
        reason=guard(c['sql'],obs,schema)
        if reason:rejected.append({'source':c.get('source'),'path':c.get('path'),'reason':reason});continue
        key=structural_key(c['sql'],schema)
        if key not in unique or unique[key]['confidence']<c['confidence']:unique[key]=c
    buckets={}
    for c in sorted(unique.values(),key=lambda c:-c['confidence']):buckets.setdefault(structural_bucket(c['sql'],schema),[]).append(c)
    out=[]
    while buckets and len(out)<cap:
        for key in list(buckets):
            out.append(buckets[key].pop(0))
            if not buckets[key]:del buckets[key]
            if len(out)==cap:break
    return sorted(out,key=lambda c:-c['confidence']),rejected
