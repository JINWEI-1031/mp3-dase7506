"""Schema-grounded candidate families; no task IDs, database IDs, or offline labels."""
import re
from sqlglot import exp
from repair_rules import parse,sql_text
from step124_pool_search import sources
from step124_pool_neutral import score as prior_score
from step4_agent import identity

FAMILIES={'shape','entity','value','ordinal','categories'}

def owned_columns(select):
    return [c for c in select.find_all(exp.Column) if c.find_ancestor(exp.Select) is select]

def shape(sql,obs,schema):
    tree=parse(sql)
    if tree is None:return []
    out=[]
    for preference in (0,-1):
        t=tree.copy()
        for select in t.find_all(exp.Select):
            src=sources(select,schema)
            if not src:continue
            # Public output headers describe only the outer result, never subqueries.
            if select is t and len(select.expressions)!=len(obs['example']['columns']):
                chosen=[]
                for header in obs['example']['columns']:
                    matches=[e for e in select.expressions if e.alias_or_name.lower()==header.lower()]
                    if len(matches)==1:chosen.append(matches[0].copy());continue
                    owners=[(a,c['name']) for a,table in src.items() for c in table['columns'] if c['name'].lower()==header.lower()]
                    if not owners:chosen=[];break
                    a,name=owners[preference];chosen.append(exp.column(name,table=a,quoted=True))
                if chosen:select.set('expressions',chosen)
            for col in owned_columns(select):
                if col.table:continue
                if any(e.alias and e.alias.lower()==col.name.lower() for e in select.expressions):continue
                owners=[(a,c['name']) for a,table in src.items() for c in table['columns'] if c['name'].lower()==col.name.lower()]
                if owners:
                    a,name=owners[preference];col.set('table',exp.to_identifier(a,quoted=True));col.set('this',exp.to_identifier(name,quoted=True))
        out.append(sql_text(t))
    return list(dict.fromkeys(out))

def ordinal(sql,obs,schema):
    t=parse(sql);q=obs['task']['question'].lower()
    if not isinstance(t,exp.Select) or t.args.get('limit') or t.args.get('group'):return []
    first=bool(re.search(r'\bfirst\s+(?!name\b|and\b|,)[a-z]',q));last=bool(re.search(r'\b(?:last|earliest|latest)\s+(?!name\b)[a-z]',q))
    if not (first or last):return []
    src=sources(t,schema);owners=set()
    for expression in t.expressions:
        for col in expression.find_all(exp.Column):
            matches=[a for a,table in src.items() if (not col.table or a.lower()==col.table.lower()) and any(c['name'].lower()==col.name.lower() for c in table['columns'])]
            if len(matches)==1:owners.add(matches[0])
    if len(owners)!=1:return []
    alias=next(iter(owners));table=src[alias];keys=[c for c in table['columns'] if c.get('pk')]
    if len(keys)!=1:return []
    new=t.copy();new.set('order',exp.Order(expressions=[exp.Ordered(this=exp.column(keys[0]['name'],table=alias),desc=not first and bool(re.search(r'\b(?:last|latest)\b',q)))]));new.set('limit',exp.Limit(expression=exp.Literal.number(1)))
    return [sql_text(new)]

def entity(sql,obs,schema):
    tree=parse(sql)
    if not isinstance(tree,exp.Select) or len(list(tree.find_all(exp.Select)))!=1:return []
    out=[];src=sources(tree,schema);new=tree.copy();changed=False
    for eq in list(new.find_all(exp.EQ)):
        col,lit=eq.this,eq.expression
        if not isinstance(col,exp.Column) or not isinstance(lit,exp.Literal) or not lit.is_string or len(lit.this.split())<2:continue
        tables=[(a,t) for a,t in src.items() if (not col.table or a.lower()==col.table.lower()) and any(c['name'].lower()==col.name.lower() for c in t['columns'])]
        if len(tables)!=1 or not re.search(r'last.?name|lname|surname',col.name,re.I):continue
        alias,table=tables[0];first=[c for c in table['columns'] if re.search(r'(?:^fname$|first.?name|given.?name)',c['name'],re.I)]
        if len(first)!=1 or lit.this.lower() not in obs['task']['question'].lower():continue
        given,last=lit.this.rsplit(' ',1)
        eq.replace(exp.and_(exp.EQ(this=exp.column(first[0]['name'],table=alias),expression=exp.Literal.string(given)),exp.EQ(this=exp.column(col.name,table=alias),expression=exp.Literal.string(last))));changed=True
    if changed:out.append(sql_text(new))
    # Remove a schema-invalid join only when no output depends on it and every
    # question-grounded value it filters is already bound by a retained predicate.
    where=tree.args.get('where')
    if where and where.find(exp.Or) is None:
        predicates=[]
        def conjuncts(e):
            if isinstance(e,exp.And):conjuncts(e.this);conjuncts(e.expression)
            else:predicates.append(e)
        conjuncts(where.this)
        for join in tree.args.get('joins',[]):
            if not isinstance(join.this,exp.Table) or join.args.get('side'):continue
            alias=join.this.alias_or_name
            on=join.args.get('on')
            source_map={a.lower():table for a,table in src.items()}
            invalid=on is not None and any(c.table.lower() in source_map and not any(k['name'].lower()==c.name.lower() for k in source_map[c.table.lower()]['columns']) for c in on.find_all(exp.Column))
            if not invalid:continue
            if any(c.table.lower()==alias.lower() for e in tree.expressions for c in e.find_all(exp.Column)):continue
            removed=[p for p in predicates if any(c.table.lower()==alias.lower() for c in p.find_all(exp.Column))];kept=[p for p in predicates if p not in removed]
            supported={v.this.lower() for p in kept for v in p.find_all(exp.Literal) if v.is_string}
            grounded={v.this.lower() for p in removed for v in p.find_all(exp.Literal) if v.is_string and v.this.lower() in obs['task']['question'].lower()}
            if not grounded<=supported:continue
            if any(c.table.lower()==alias.lower() for j in tree.args.get('joins',[]) if j is not join for c in j.find_all(exp.Column)):continue
            pruned=tree.copy();pruned.set('joins',[j.copy() for j in tree.args.get('joins',[]) if j is not join]);pruned.set('where',exp.Where(this=exp.and_(*(p.copy() for p in kept))) if kept else None)
            out.append(sql_text(pruned))
    return out

def casefold_needed(obs):
    executable=[e for e in obs['history'] if e['tool']=='check' and e['observation']['ok']]
    matched=any(e['observation'].get('matches_example',False) for e in executable)
    return obs['example']['row_count']>0 and executable and executable[-1]['observation']['row_count']==0 and not matched

def value(sql,obs,schema):
    tree=parse(sql)
    mismatch=casefold_needed(obs)
    if tree is None or not mismatch:return []
    new=tree.copy();changed=False
    for eq in list(new.find_all(exp.EQ)):
        if isinstance(eq.this,exp.Column) and isinstance(eq.expression,exp.Literal) and eq.expression.is_string and eq.expression.this.lower() in obs['task']['question'].lower():
            eq.set('this',exp.Lower(this=eq.this.copy()));eq.set('expression',exp.Lower(this=eq.expression.copy()));changed=True
    return [sql_text(new)] if changed else []

def pair_terms(question):
    m=re.search(r'\bboth\s+([a-z]+)\s+and\s+([a-z]+)\s*[?.]?\s*$',question.lower())
    return tuple(m.groups()) if m else None

def categories(sql,obs,schema):
    t=parse(sql);terms=pair_terms(obs['task']['question'])
    if not terms or not isinstance(t,exp.Select):return []
    src=sources(t,schema);projected=[c for e in t.expressions for c in e.find_all(exp.Column)]
    owners={a for col in projected for a,table in src.items() if (not col.table or a.lower()==col.table.lower()) and any(c['name'].lower()==col.name.lower() for c in table['columns'])}
    if len(owners)!=1:return []
    alias=next(iter(owners));parent=src[alias];known={s['name'].lower():s for s in schema};global_alias={x.alias_or_name.lower():x.name.lower() for x in t.find_all(exp.Table)};choices=[]
    for node in t.find_all(exp.In):
        if not isinstance(node.this,exp.Column) or len(node.expressions)!=2 or not all(isinstance(v,exp.Literal) and v.is_string for v in node.expressions):continue
        table=known.get(global_alias.get(node.this.table.lower(),''))
        if table is None:continue
        column=next((c for c in table['columns'] if c['name'].lower()==node.this.name.lower()),None)
        links=[fk for fk in table['foreign_keys'] if fk['table'].lower()==parent['name'].lower()]
        if column and re.search(r'CHAR|TEXT|CLOB',column['type'],re.I) and len(links)==1:choices.append((table,column,links[0]))
    unique={(t['name'],c['name'],fk['from'],fk['to']):(t,c,fk) for t,c,fk in choices}
    if len(unique)!=1:return []
    table,column,fk=next(iter(unique.values()));child='category_event'
    sub=exp.select(exp.column(fk['from'],table=child)).from_(exp.to_table(table['name']).as_(child))
    category=exp.Lower(this=exp.column(column['name'],table=child));sub=sub.where(exp.In(this=category.copy(),expressions=[exp.Literal.string(v) for v in terms])).group_by(exp.column(fk['from'],table=child)).having(exp.EQ(this=exp.Count(this=exp.Distinct(expressions=[category.copy()])),expression=exp.Literal.number(2)))
    new=t.copy();new.set('joins',[]);new.set('where',exp.Where(this=exp.In(this=exp.column(fk['to'],table=alias),query=exp.Subquery(this=sub))));new.set('group',None);new.set('having',None)
    return [sql_text(new)]

def schema_errors(sql,obs,schema):
    t=parse(sql)
    if t is None:return 20
    known={x['name'].lower():x for x in schema};errors=sum(x.name.lower() not in known for x in t.find_all(exp.Table))
    for select in t.find_all(exp.Select):
        src=sources(select,schema)
        for col in owned_columns(select):
            if col.table and col.table.lower() not in {a.lower() for a in src}:continue
            owners=[a for a,s in src.items() if (not col.table or col.table.lower()==a.lower()) and any(c['name'].lower()==col.name.lower() for c in s['columns'])]
            if not owners and col.name.lower() not in {e.alias.lower() for e in select.expressions if e.alias}:errors+=1
            if not col.table and len(owners)>1:errors+=1
    if isinstance(t,exp.Select) and len(t.expressions)!=len(obs['example']['columns']):errors+=2
    return errors

def score(sql,obs,schema):
    value=prior_score(sql,obs,schema)-4*schema_errors(sql,obs,schema);t=parse(sql);q=obs['task']['question'].lower()
    terms=pair_terms(q)
    if terms and t:
        literals={v.this.lower() for v in t.find_all(exp.Literal) if v.is_string}
        value+=3*sum(v in literals for v in terms)
        if all(v in literals for v in terms) and (t.find(exp.Having) or isinstance(t,exp.Intersect)):value+=2
    if t is not None and casefold_needed(obs) and any(lower.find(exp.Literal) for lower in t.find_all(exp.Lower)):value+=1.5
    return value

def generate(seeds,obs,schema,families,depth=2):
    outputs={};frontier=[]
    def add(sql,path):
        if not isinstance(sql,str) or len(sql.encode())>16000:return
        key=identity(sql)
        if key in outputs:return
        item={'sql':sql,'path':path,'score':score(sql,obs,schema)};outputs[key]=item;return item
    for seed in seeds:
        item=add(seed,[])
        if item:frontier.append(item)
    operators={'shape':shape,'entity':entity,'value':value,'ordinal':ordinal,'categories':categories}
    for _ in range(depth):
        newer=[]
        # Breadth before pruning: low-scoring broken hints may contain a correct
        # predicate absent from the highest-scoring existing candidates.
        for item in frontier[:64]:
            for family in sorted(families):
                for sql in operators[family](item['sql'],obs,schema):
                    result=add(sql,[*item['path'],family])
                    if result:newer.append(result)
        frontier=sorted(newer,key=lambda r:r['score'],reverse=True)[:24]
    return sorted([x for x in outputs.values() if x['path']],key=lambda r:r['score'],reverse=True)[:64]
