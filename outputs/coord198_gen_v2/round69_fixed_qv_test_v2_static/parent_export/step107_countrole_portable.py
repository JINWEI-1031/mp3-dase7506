"""Portable role-evidenced generators. Dependency: sqlglot only."""
import re
import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
def parse(sql):
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except (SqlglotError, RecursionError):
        return None
    if len(statements) == 1 and isinstance(statements[0], (exp.Select, exp.SetOperation)):
        return statements[0]
    return None

def sql_text(tree):
    return tree.sql(dialect="sqlite")
def words(value):return re.findall(r'[a-z]+',re.sub(r'([a-z])([A-Z])',r'\1 \2',value).lower())
def lemma(word):
    if word.endswith('sses'):return word[:-2]
    if word.endswith('ies'):return word[:-3]+'y'
    return word[:-1] if word.endswith('s') and not word.endswith('ss') else word

def tokens(s):return {lemma(x) for x in words(s)}
def sources(tree,schema):
    known={t['name'].lower():t for t in schema};result={}
    if not isinstance(tree,exp.Select):return result
    root=tree.args.get('from');ts=[root.this] if root and isinstance(root.this,exp.Table) else []
    ts += [j.this for j in tree.args.get('joins',[]) if isinstance(j.this,exp.Table)]
    for t in ts:
        if t.name.lower() in known:result[t.alias_or_name]=known[t.name.lower()]
    return result
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
ROLE_GROUPS=[{'graduate','graduation','graduated','left','leaving'}, {'register','registered','registration','enroll','enrolled','enrolment'}, {'access','accessed','attendance','attend','attended'}, {'win','won','winning'}, {'authorize','authorise','authorised','authorized','authorization','authorisation'}, {'report','reported','reporting'}, {'start','started','starting','source','src'}, {'end','ended','destination','dst'}, {'advise','advises','advisor','adviser'}, {'settlement','settled','settle'}]

STOP={'id','code','name','the','of','in','a','an','all','for','and','to','by','any'}

def role_words(s):
    normalized=re.sub(r'\b(\w*(?:ch|sh|ss|x|z))es\b',r'\1',s.lower())
    out=tokens(normalized)-STOP
    for group in ROLE_GROUPS:
        if out&group:out|=group
    return out

def affinity(field,q):return len(role_words(field)&role_words(q))

def signals(sql,obs,schema=()):
    q=obs['task']['question'];t=parse(sql);out=[]
    if not isinstance(t,exp.Select):return out
    if re.search(r'earliest|latest|recent|oldest|newest|\bfirst\b|\blast\b',re.sub(r'(first|last) names?','',q,flags=re.I),re.I):out.append('event_role')
    if t.find(exp.Count) and re.search(r'number|times|how many|count',q,re.I):out.append('counted_role')
    if re.search(r"\b(no|without|not|never|don't|doesn't)\b",q,re.I):out.append('negative_role')
    if t.find(exp.Join):out.append('relation_role')
    if t.find(exp.Avg) or t.find(exp.Sum):out.append('measure_role')
    return out

def event_counts(tree,obs,schema):
    """Replace an unrelated counted association by a uniquely evidenced event FK."""
    q=obs['task']['question'];src=sources(tree,schema)
    if not tree.find(exp.Count) or not tree.args.get('group') or len(list(tree.find_all(exp.Select)))!=1:return []
    projections=[e for e in tree.expressions if not e.find(exp.Count)]
    columns=[c for e in projections for c in e.find_all(exp.Column)]
    owners=[]
    for c in columns:
        candidates=[a for a,t in src.items() if (not c.table or c.table.lower()==a.lower()) and any(x['name'].lower()==c.name.lower() for x in t['columns'])]
        if len(candidates)!=1:return []
        owners.append(candidates[0])
    if not owners or len(set(owners))!=1:return []
    root_alias=owners[0];root=src[root_alias];links=[]
    for child in schema:
        if child['name'].lower()==root['name'].lower():continue
        # Both the event noun and the declared role must have question evidence.
        if not role_words(child['name'])<=role_words(q):continue
        for fk in child['foreign_keys']:
            if fk['table'].lower()==root['name'].lower():links.append((affinity(fk['from'],q),child,fk))
    if not links:return []
    best=max(v for v,_,_ in links);links=[x for x in links if x[0]==best]
    if len(links)!=1:return []
    _,child,fk=links[0]
    if any(t['name'].lower()==child['name'].lower() for t in src.values()):return []
    # No predicate or ordering may be silently dropped with the old relation.
    for part in ('where','having','order'):
        if tree.args.get(part) and any(c.table and c.table.lower()!=root_alias.lower() for c in tree.args[part].find_all(exp.Column)):return []
    group=tree.args['group']
    if any(c.table and c.table.lower()!=root_alias.lower() for c in group.find_all(exp.Column)):return []
    out=[]
    for side in ('','LEFT'):
        t=tree.copy();t.set('joins',[exp.Join(this=exp.to_table(child['name']).as_('role_event'),side=side,on=exp.EQ(this=exp.column(fk['to'],table=root_alias),expression=exp.column(fk['from'],table='role_event')))])
        for c in t.find_all(exp.Count):c.set('this',exp.column(fk['from'],table='role_event'))
        out.append({'sql':sql_text(t),'family':'count','path':['count_event_role'],'evidence':{'event_table':child['name'],'declared_fk':fk,'question_role_overlap':best},'confidence':8 if not side else 7})
    return out

def direct_fk(tree,obs,schema):
    """Unique direct FK correction per local scope, preserving all predicates."""
    out=[]
    for select in tree.find_all(exp.Select):
        src=sources(select,schema);local={a.lower():(a,t) for a,t in src.items()}
        for join in select.args.get('joins',[]):
            on=join.args.get('on')
            # Composite and disjunctive join predicates need a separate proof.
            if not isinstance(on,exp.EQ) or not isinstance(on.this,exp.Column) or not isinstance(on.expression,exp.Column):continue
            l,r=on.this,on.expression
            if l.table.lower() not in local or r.table.lower() not in local or l.table.lower()==r.table.lower():continue
            la,lt=local[l.table.lower()];ra,rt=local[r.table.lower()];edges=[]
            for a,t,b,u in ((la,lt,ra,rt),(ra,rt,la,lt)):
                edges.extend((a,f['from'],b,f['to']) for f in t['foreign_keys'] if f['table'].lower()==u['name'].lower())
            if len(edges)!=1:continue
            a,af,b,bf=edges[0]
            if {(l.table.lower(),l.name.lower()),(r.table.lower(),r.name.lower())}=={(a.lower(),af.lower()),(b.lower(),bf.lower())}:continue
            new=tree.copy();scope=list(new.find_all(exp.Select))[list(tree.find_all(exp.Select)).index(select)];j=scope.args['joins'][select.args['joins'].index(join)]
            j.set('on',exp.EQ(this=exp.column(af,table=a),expression=exp.column(bf,table=b)))
            out.append({'sql':sql_text(new),'family':'relation','path':['unique_local_fk'],'evidence':{'declared_edge':[a,af,b,bf],'scope_local':True},'confidence':9})
    return out
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

def generate(seeds,obs,schema,cap=24):
    """New primitives only. Input schema from runtime Tools.schema, no labels."""
    out={}
    for sql in seeds[:4]:
        tree=parse(sql)
        if not isinstance(tree,exp.Select):continue
        for item in event_counts(tree,obs,schema)+direct_fk(tree,obs,schema):
            if not typed_literals(item['sql'],schema):continue
            item['source']='rolebind192_portable'
            out.setdefault(item['sql'],item)
    return sorted(out.values(),key=lambda c:-c['confidence'])[:cap]
