"""Semantic alternatives derived only from question, schema and reached SQL."""
import re
from collections import deque
from sqlglot import exp
from repair_rules import parse,sql_text
from step123_agent import words
from step124_pool_search import sources
from step_pool192_rules import shape
from step4_agent import identity

def lemma(word):
    if word.endswith('sses'):return word[:-2]
    if word.endswith('ies'):return word[:-3]+'y'
    return word[:-1] if word.endswith('s') and not word.endswith('ss') else word
def tokens(s):return {lemma(x) for x in words(s)}
def paths(schema,start,end):
    graph={}
    for t in schema:
        for fk in t['foreign_keys']:
            edge=(t['name'],fk['from'],fk['table'],fk['to'])
            graph.setdefault(t['name'].lower(),[]).append((fk['table'],edge));graph.setdefault(fk['table'].lower(),[]).append((t['name'],edge))
    queue=deque([(start,[],{start.lower()})]);out=[]
    while queue:
        name,path,seen=queue.popleft()
        if name.lower()==end.lower():out.append(path);continue
        if len(path)>=3:continue
        for nxt,edge in graph.get(name.lower(),[]):
            if nxt.lower() not in seen:queue.append((nxt,[*path,(nxt,edge)],seen|{nxt.lower()}))
    return sorted(out,key=len)[:3]
def join_path(tree,src,path):
    aliases={t['name'].lower():a for a,t in src.items()};new=tree.copy()
    for nxt,(left,lc,right,rc) in path:
        if nxt.lower() in aliases:continue
        alias='sem_'+str(len(aliases));aliases[nxt.lower()]=alias
        new.append('joins',exp.Join(this=exp.to_table(nxt).as_(alias),on=exp.EQ(this=exp.column(lc,table=aliases[left.lower()]),expression=exp.column(rc,table=aliases[right.lower()]))))
    return new,aliases
def grounded(predicate,q):
    literals=list(predicate.find_all(exp.Literal))
    return bool(literals) and all(v.this.strip('%').lower() in q.lower() for v in literals)
def conjuncts(node):
    if isinstance(node,exp.And):return conjuncts(node.this)+conjuncts(node.expression)
    return [node]

def events(tree,obs,schema):
    q=obs['task']['question'].lower()
    src=sources(tree,schema)
    aggregate=[]
    agg=next(tree.find_all(exp.Sum),None) if re.search(r'\b(?:total|sum)\b',q) else next(tree.find_all(exp.Avg),None) if re.search(r'\baverage\b',q) else None
    if agg is not None and not tree.args.get('group') and not re.search(r'\b(?:who|which|that|where|having|top)\b',q):
        mentioned=[t for t in schema if tokens(t['name']) and tokens(t['name'])<=tokens(q)]
        for table in mentioned:
            for col in table['columns']:
                if not re.search(r'INT|REAL|NUM|FLOAT|DOUBLE|DECIMAL',col['type'],re.I) or col.get('pk') or re.search(r'(?:_id|^id)$',col['name'],re.I):continue
                if not tokens(col['name'])&tokens(q):continue
                new=exp.select(type(agg)(this=exp.column(col['name'],table='measure_entity'))).from_(exp.to_table(table['name']).as_('measure_entity'))
                if tree.args.get('where'):
                    predicate=tree.args['where'].this.copy();supported=True
                    for c in predicate.find_all(exp.Column):
                        owners=[t for a,t in src.items() if (not c.table or a.lower()==c.table.lower()) and any(k['name'].lower()==c.name.lower() for k in t['columns'])]
                        if len(owners)!=1 or owners[0]['name'].lower()!=table['name'].lower():supported=False;break
                        c.set('table',exp.to_identifier('measure_entity'))
                    if not supported:continue
                    new=new.where(predicate)
                aggregate.append(('measure_field',sql_text(new)))
    if not re.search(r'\b(?:earliest|latest|first|last|oldest|newest|recent(?:ly)?)\b',q):return aggregate
    if re.search(r'\b(?:first|last) names?\b',q) and not re.search(r'\b(?:earliest|latest|oldest|newest|recent)\b',q):return []
    if tree.args.get('group') or any(e.find(exp.AggFunc) for e in tree.expressions):return []
    cols=[c for e in tree.expressions for c in e.find_all(exp.Column)];owners=set()
    for col in cols:
        matches=[a for a,t in src.items() if (not col.table or a.lower()==col.table.lower()) and any(c['name'].lower()==col.name.lower() for c in t['columns'])]
        if len(matches)!=1:return []
        owners.add(matches[0])
    if len(owners)!=1:return []
    alias=next(iter(owners));table=src[alias];out=[]
    candidates=[(a,t,c) for a,t in src.items() for c in t['columns'] if re.search(r'DATE|TIME',c['type'],re.I) or re.search(r'(?:date|time|year|born|opened|founded|release|launch)',c['name'],re.I)]
    candidates.sort(key=lambda item:-len(tokens(item[2]['name'])&tokens(q)))
    for a,t,c in candidates:
        new=tree.copy();order=exp.Order(expressions=[exp.Ordered(this=exp.column(c['name'],table=a),desc=bool(re.search(r'latest|last|newest|recent',re.sub(r'last names?','',q))))]);new.set('order',order);new.set('limit',exp.Limit(expression=exp.Literal.number(1)))
        out.append(('event_field',sql_text(new)))
        where=tree.args.get('where');grounded_predicates=[p for p in conjuncts(where.this) if grounded(p,q)] if where else []
        root_scope_preserved=all(all(not col.table or col.table.lower()==alias.lower() for col in p.find_all(exp.Column)) for p in grounded_predicates)
        if a==alias and root_scope_preserved:
            minimal=exp.select(*(e.copy() for e in tree.expressions)).from_(exp.to_table(table['name']).as_(alias))
            where=tree.args.get('where');keep=[]
            if where:
                for p in conjuncts(where.this):
                    if grounded(p,q) and all(not col.table or col.table.lower()==alias.lower() for col in p.find_all(exp.Column)):keep.append(p.copy())
            keep.append(exp.Not(this=exp.Is(this=exp.column(c['name'],table=alias),expression=exp.Null())))
            minimal=minimal.where(exp.and_(*keep));minimal.set('order',order.copy());minimal.set('limit',exp.Limit(expression=exp.Literal.number(1)));out.append(('event_entity',sql_text(minimal)))
        if a!=alias:
            event_col=c
            base=exp.select(*(e.copy() for e in tree.expressions)).from_(exp.to_table(table['name']).as_(alias))
            for path in paths(schema,table['name'],t['name'])[:2]:
                planned,bindings=join_path(base,{alias:table},path);event_alias=bindings[t['name'].lower()]
                preserved=[];supported=True
                for predicate in grounded_predicates:
                    p=predicate.copy()
                    for c in p.find_all(exp.Column):
                        owners=[s for sa,s in src.items() if not c.table or sa.lower()==c.table.lower()]
                        if len(owners)!=1 or owners[0]['name'].lower() not in bindings:supported=False;break
                        c.set('table',exp.to_identifier(bindings[owners[0]['name'].lower()]))
                    preserved.append(p)
                if not supported:continue
                if preserved:planned=planned.where(exp.and_(*preserved))
                planned.set('order',exp.Order(expressions=[exp.Ordered(this=exp.column(event_col['name'],table=event_alias),desc=bool(re.search(r'latest|last|newest|recent',re.sub(r'last names?','',q))))]));planned.set('limit',exp.Limit(expression=exp.Literal.number(1)))
                out.append(('event_path',sql_text(planned)))
    return out

def counts(tree,obs,schema):
    q=obs['task']['question'];request=re.search(r'\b(?:how many|number of)\s+(.+?)(?:\s+(?:do|does|have|has|are|is|exist|with|without|for|per|in|that|who|were)\b|[?.]|$)',q,re.I)
    if not request or not tree.find(exp.Count):return []
    targets=[t for t in schema if tokens(t['name']) and tokens(t['name'])<=tokens(request[1])]
    if len(targets)!=1:return []
    target=targets[0];keys=[c['name'] for c in target['columns'] if c.get('pk')]
    if len(keys)!=1:return []
    src=sources(tree,schema);out=[]
    present=[a for a,t in src.items() if t['name'].lower()==target['name'].lower()]
    alternatives=[]
    if present:alternatives=[(tree.copy(),present[0])]
    else:
        for a,t in src.items():
            for path in paths(schema,t['name'],target['name']):
                new,aliases=join_path(tree,src,path);alternatives.append((new,aliases[target['name'].lower()]))
    for new,alias in alternatives[:6]:
        for distinct in (True,False):
            variant=new.copy()
            for count in variant.find_all(exp.Count):
                col=exp.column(keys[0],table=alias);count.set('this',exp.Distinct(expressions=[col]) if distinct else col)
            out.append(('count_entity_distinct' if distinct else 'count_entity',sql_text(variant)))
    return out

def relations(tree,obs,schema):
    if not tree.args.get('group') and any(e.find(exp.AggFunc) for e in tree.expressions):
        src=sources(tree,schema);root=tree.args.get('from');out=[]
        if not root or not isinstance(root.this,exp.Table) or root.this.alias_or_name not in src:return []
        alias=root.this.alias_or_name;parent=src[alias]
        for child_alias,child in src.items():
            if child_alias==alias or not tokens(child['name'])<=tokens(obs['task']['question']):continue
            links=[fk for fk in child['foreign_keys'] if fk['table'].lower()==parent['name'].lower()]
            if len(links)!=1:continue
            fk=links[0];sub=exp.select(exp.Literal.number(1)).from_(exp.to_table(child['name']).as_(child_alias)).where(exp.EQ(this=exp.column(fk['from'],table=child_alias),expression=exp.column(fk['to'],table=alias)))
            root_predicates=[];supported=True
            if tree.args.get('where'):
                for p in conjuncts(tree.args['where'].this):
                    names={c.table.lower() for c in p.find_all(exp.Column) if c.table}
                    if names<={alias.lower()}:root_predicates.append(p.copy())
                    elif names<={child_alias.lower()}:sub=sub.where(p.copy())
                    else:supported=False
            if supported:
                new=exp.select(*(e.copy() for e in tree.expressions)).from_(root.this.copy()).where(exp.and_(exp.Exists(this=sub),*root_predicates));out.append(('relation_semijoin',sql_text(new)))
        return out
    if tree.args.get('group'):return []
    src=sources(tree,schema);root=tree.args.get('from');where=tree.args.get('where')
    if not root or not isinstance(root.this,exp.Table) or not where or where.find(exp.Or):return []
    predicates=[p for p in conjuncts(where.this) if grounded(p,obs['task']['question'])]
    needed={c.table.lower() for e in tree.expressions for c in e.find_all(exp.Column) if c.table}
    needed|={c.table.lower() for p in predicates for c in p.find_all(exp.Column) if c.table}
    if not needed or not needed<={a.lower() for a in src}:return []
    first=root.this.alias_or_name;start=src.get(first)
    if start is None:return []
    base=exp.select(*(e.copy() for e in tree.expressions)).from_(root.this.copy());aliases={start['name'].lower():first};known={start['name'].lower():first};out=[]
    for alias,t in src.items():
        if alias.lower() not in needed or alias==first:continue
        options=paths(schema,start['name'],t['name'])
        if not options:return []
        current_src={v:next(x for x in schema if x['name'].lower()==k) for k,v in aliases.items()}
        base,aliases=join_path(base,current_src,options[0]);known[t['name'].lower()]=alias
    mapping={a.lower():aliases[t['name'].lower()] for a,t in src.items() if t['name'].lower() in aliases}
    base.set('expressions',[e.copy() for e in tree.expressions])
    if predicates:base=base.where(exp.and_(*(p.copy() for p in predicates)))
    if tree.args.get('distinct'):base.set('distinct',tree.args['distinct'].copy())
    for col in base.find_all(exp.Column):
        if col.table.lower() in mapping:col.set('table',exp.to_identifier(mapping[col.table.lower()]))
    return [('relation_minimal',sql_text(base))]

def negations(tree,obs,schema):
    q=obs['task']['question'];src=sources(tree,schema);root=tree.args.get('from')
    if not re.search(r"\b(?:no|without|not|never|don't|doesn't)\b",q,re.I) or not root or not isinstance(root.this,exp.Table):return []
    if tree.find(exp.Exists) or tree.args.get('group'):return []
    parent=src.get(root.this.alias_or_name)
    if parent is None:return []
    targets=[t for t in schema if t['name'].lower()!=parent['name'].lower() and tokens(t['name'])&(tokens(q)-tokens(parent['name'])-{'in','to','be','for','of'})]
    where=tree.args.get('where');out=[]
    for child in targets:
        links=[fk for fk in child['foreign_keys'] if fk['table'].lower()==parent['name'].lower()]
        keys=[c for c in parent['columns'] if c.get('pk')]
        if not links and len(keys)==1:
            links=[{'from':c['name'],'to':keys[0]['name']} for c in child['columns'] if re.search(r'INT|NUM',c['type'],re.I) and any(len(w)>=4 and any(v.startswith(w[:4]) for v in tokens(q)) for w in tokens(c['name']))]
        if not links:continue
        links.sort(key=lambda fk:-sum(any(v.startswith(w[:4]) for v in tokens(q)) for w in tokens(fk['from']) if len(w)>=4))
        fk=links[0];inner=exp.select(exp.Literal.number(1)).from_(exp.to_table(child['name']).as_('neg_event'))
        conditions=[exp.EQ(this=exp.column(fk['from'],table='neg_event'),expression=exp.column(fk['to'],table=root.this.alias_or_name))];outer_conditions=[];supported=True
        if where:
            for predicate in conjuncts(where.this):
                columns=list(predicate.find_all(exp.Column))
                if not columns or not grounded(predicate,q):continue
                bound=[]
                for col in columns:
                    owners=[t['name'].lower() for a,t in src.items() if (not col.table or a.lower()==col.table.lower()) and any(c['name'].lower()==col.name.lower() for c in t['columns'])]
                    bound.append(owners[0] if len(owners)==1 else None)
                if all(owner==parent['name'].lower() for owner in bound):outer_conditions.append(predicate.copy());continue
                if all(owner==child['name'].lower() for owner in bound):
                    p=predicate.copy()
                    for col in p.find_all(exp.Column):col.set('table',exp.to_identifier('neg_event'))
                    for comp in list(p.find_all(exp.LT)):
                        if isinstance(comp.expression,exp.Literal) and re.search(r'(?:longer|higher|greater|more) than\s+'+re.escape(comp.expression.this),q,re.I):comp.replace(exp.GT(this=comp.this.copy(),expression=comp.expression.copy()))
                    conditions.append(p)
                else:supported=False
        if not supported:continue
        inner=inner.where(exp.and_(*conditions));new=exp.select(*(e.copy() for e in tree.expressions)).from_(root.this.copy()).where(exp.and_(exp.Not(this=exp.Exists(this=inner)),*outer_conditions))
        if tree.args.get('distinct'):new.set('distinct',tree.args['distinct'].copy())
        out.append(('negative_relation',sql_text(new)))
    if where:
        for eq in where.find_all(exp.EQ):
            if not isinstance(eq.this,exp.Column) or not isinstance(eq.expression,exp.Literal) or eq.expression.this.lower() not in {'yes','no','y','n','true','false','0','1'}:continue
            if not tokens(eq.this.name)<=tokens(q):continue
            new=tree.copy()
            for e in list(new.find_all(exp.EQ)):
                if sql_text(e)==sql_text(eq):
                    value=exp.Lower(this=exp.Cast(this=e.this.copy(),to=exp.DataType.build('TEXT')))
                    e.replace(exp.In(this=value,expressions=[exp.Literal.string(v) for v in ('no','n','false','0')]))
            out.append(('negative_boolean',sql_text(new)))
    return out

def typed_literals(sql,schema):
    tree=parse(sql)
    if tree is None:return False
    for select in tree.find_all(exp.Select):
        src=sources(select,schema)
        for eq in select.find_all(exp.EQ):
            if eq.find_ancestor(exp.Select) is not select:continue
            for col,lit in ((eq.this,eq.expression),(eq.expression,eq.this)):
                if not isinstance(col,exp.Column) or not isinstance(lit,exp.Literal) or not lit.is_string:continue
                owners=[c for a,t in src.items() if not col.table or a.lower()==col.table.lower() for c in t['columns'] if c['name'].lower()==col.name.lower()]
                if owners and all(re.search(r'INT|REAL|NUM|FLOAT|DOUBLE|DECIMAL',c['type'],re.I) for c in owners):
                    try:float(lit.this)
                    except ValueError:return False
    return True

def generate(seeds,obs,schema,families):
    outputs={};operators={'event':events,'relation':relations,'count':counts,'negative':negations}
    for sql in list(dict.fromkeys(seeds))[:16]:
        variants=[sql,*shape(sql,obs,schema)]
        for variant in variants[:3]:
            tree=parse(variant)
            if not isinstance(tree,exp.Select) or len(list(tree.find_all(exp.Select)))>2:continue
            for family in sorted(families):
                for kind,proposal in operators[family](tree,obs,schema):
                    key=identity(proposal)
                    if key==identity(sql) or len(proposal.encode())>16000 or not typed_literals(proposal,schema):continue
                    top=re.search(r'\btop\s+(\d+)\b',obs['task']['question'],re.I)
                    if top and not any(limit.expression.name==top[1] for limit in parse(proposal).find_all(exp.Limit)):continue
                    outputs.setdefault(key,{'sql':proposal,'path':[kind],'family':family})
    return list(outputs.values())[:64]
