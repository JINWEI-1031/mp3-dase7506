"""Independent, bounded role-evidence provider. No labels or task identifiers."""
import re
from sqlglot import exp
from repair_rules import parse,sql_text
from step124_pool_search import sources
from step_semantic192_rules import tokens,conjuncts
from step_semantic192_selective import generate as previous_generate
from step4_agent import identity

FAMILIES={'event','relation','count','negative'}
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

def generate(seeds,obs,schema,*,new=True,cap=24):
    """Return <=cap SQL proposals with source, evidence and confidence. Pure API."""
    if not seeds:return []
    output=[]
    for sql in seeds[:4]:
        t=parse(sql)
        if not isinstance(t,exp.Select):continue
        if new:output+=event_counts(t,obs,schema)+direct_fk(t,obs,schema)
    for item in previous_generate(seeds[:4],obs,schema,FAMILIES):
        item={**item,'evidence':{},'confidence':3}
        t=parse(item['sql']);q=obs['task']['question']
        if item['family']=='event' and item['path'][0].startswith('event_'):
            order=t.args.get('order')
            strength=max((affinity(c.name,q) for c in order.find_all(exp.Column)),default=0) if order else 0
            # A temporal type alone is insufficient event-role evidence.
            if new and not strength:continue
            item['evidence']={'time_role_overlap':strength};item['confidence']=5+strength
        elif item['path'][0]=='measure_field':
            strength=max((affinity(c.name,q) for e in t.expressions for c in e.find_all(exp.Column)),default=0)
            item['confidence']=5+strength;item['evidence']={'measure_role_overlap':strength}
        elif item['path'][0]=='relation_semijoin':item['confidence']=7
        elif item['family']=='negative':item['confidence']=6
        output.append(item)
    result={}
    for item in output:
        key=identity(item['sql'])
        if key in {identity(s) for s in seeds[:1]}:continue
        item['source']='rolebind192' if new else 'semantic192_previous'
        if key not in result or result[key]['confidence']<item['confidence']:result[key]=item
    return sorted(result.values(),key=lambda x:-x['confidence'])[:cap]
