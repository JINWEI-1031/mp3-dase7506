"""Bounded typed sketch completion. Unknown roles remain hypotheses, never labels."""
import re,collections
from sqlglot import exp
from repair_rules import parse,sql_text
from step124_pool_search import sources
from step_semantic192_rules import tokens
from step_scope441_ir import table,column,key,declared,qi,col,linked
CAP=24
STOP={'name','id','key','number','num','hardware','code','the','of','in'}
def role_score(tb,q):
    qt=tokens(q.lower());ts=tokens(tb['name'].lower())-STOP
    ks=tokens(key(tb) or '')-STOP
    return 2*len(ts&qt)+len(ks&qt)
def rebind(node,src,aliases):
    n=node.copy()
    for c in n.find_all(exp.Column):
        owners=[tb for a,tb in src.items() if (not c.table or a.lower()==c.table.lower()) and column(tb,c.name)]
        if len(owners)!=1:raise ValueError('unresolved_predicate_owner')
        c.set('table',exp.to_identifier(aliases[owners[0]['name']]))
    return sql_text(n)
def emit(sql,family,slots,uncertain,positive,parent):
    return {'sql':sql,'source':'synth441_'+family,'family':family,'slots':slots,'uncertainty':uncertain,'positive':positive,'parent':parent}
def count_sketch(current,obs,schema):
    t=parse(current);q=obs['task']['question'];out=[]
    if not isinstance(t,exp.Select) or len(list(t.find_all(exp.Select)))!=1:return [],'count_requires_single_parent_scope'
    if not re.search(r'\bhow many\b|\bnumber of\b|\bcount\b',q,re.I):return [],'no_quantity_evidence'
    if any(t.args.get(k) for k in ['joins','group','having','limit','order','distinct']):return [],'count_group_or_relation_sketch_not_implemented'
    if len(t.expressions)!=1 or not isinstance(t.expressions[0].unalias(),exp.Count) or not isinstance(t.expressions[0].unalias().this,exp.Star):return [],'not_scalar_row_count'
    src=sources(t,schema)
    if len(src)!=1:return [],'count_seed_root_unresolved'
    alias,base=next(iter(src.items()));where=t.args.get('where')
    if not where or where.find(exp.Subquery):return [],'no_flat_population_predicate'
    owners=[];needed=[]
    for c in where.find_all(exp.Column):
        if column(base,c.name) and (not c.table or c.table.lower()==alias.lower()):tb=base
        else:
            xs=[z for z in schema if column(z,c.name) and (not c.table or z['name'].lower()==c.table.lower())]
            if len(xs)!=1:return [],'attribute_owner_not_resolvable'
            tb=xs[0]
        owners.append(tb)
        if tb not in needed:needed.append(tb)
    if len(needed)>2:return [],'too_many_predicate_owners'
    related=[z for z in needed if z is not base]
    if len(related)!=1:return [],'no_role_diversity_in_count_population'
    other=related[0];edges=declared(schema,base,other)
    if len(edges)!=1:return [],'count_path_not_uniquely_declared_typed'
    edge=edges[0]
    for root in sorted([base,other],key=lambda z:-role_score(z,q))[:2]:
        if not key(root) or role_score(root,q)<=0:continue
        # The entity slot can change; every explicit current predicate is retained.
        amap={root['name']:'sk_root'};opposite=other if root is base else base;amap[opposite['name']]='sk_related'
        pred=where.this.copy()
        for c,tb in zip(pred.find_all(exp.Column),owners):c.set('table',exp.to_identifier(amap[tb['name']]))
        used_other=any(tb is opposite for tb in owners)
        aliases={base['name']:amap[base['name']],other['name']:amap[other['name']]}
        eq=f'{col(aliases[edge["child"]],edge["child_field"])}={col(aliases[edge["parent"]],edge["parent_field"])}'
        if used_other:
            p=f'EXISTS(SELECT 1 FROM {qi(opposite["name"])} AS sk_related WHERE {eq} AND ({sql_text(pred)}))'
        else:p=sql_text(pred)
        new=t.copy();new.set('from',exp.From(this=exp.to_table(root['name']).as_('sk_root')));new.set('where',exp.Where(this=parse('SELECT 1 WHERE '+p).args['where'].this))
        out.append(emit(sql_text(new),'count_membership',{'root':root['name'],'root_key':key(root),'attribute_owners':[z['name'] for z in needed],'edge':edge,'grain':'one keyed root row','count':'COUNT(*)','root_role_score':role_score(root,q)},['counted_entity_not_unique']+(['root_changes_from_parent'] if root is not base else []),['quantity_question','typed_declared_path','all_predicates_preserved'],current))
    return out,None
def extreme_sketch(current,obs,schema):
    t=parse(current);q=obs['task']['question'];out=[]
    if not isinstance(t,exp.Select):return [],'invalid_extreme_seed'
    cues=[v for v in ['fewest','least','most','largest'] if v in tokens(q)]
    if not cues:return [],'no_extreme_evidence'
    direction='ASC' if any(v in cues for v in ['fewest','least']) else 'DESC'
    group=t.args.get('group');having=t.args.get('having');src=sources(t,schema)
    if len(src)!=3 or not group or len(group.expressions)!=1 or not isinstance(group.expressions[0],exp.Column):return [],'no_three_role_group_skeleton'
    g=group.expressions[0]
    if g.table not in src:return [],'group_role_not_resolved'
    population=src[g.table]
    if key(population)!=g.name:return [],'group_not_entity_key'
    projected=[e.unalias() for e in t.expressions]
    if not projected or any(not isinstance(e,exp.Column) or e.table not in src for e in projected):return [],'output_role_not_resolved'
    output=src[projected[0].table]
    if any(src[e.table] is not output for e in projected) or output is population or not key(output):return [],'not_separate_output_entity'
    events=[z for z in src.values() if z is not population and z is not output]
    if len(events)!=1:return [],'event_role_ambiguous'
    event=events[0];pe=declared(schema,event,population);oe=declared(schema,event,output)
    if len(pe)!=1 or len(oe)!=1 or pe[0]['child']!=event['name'] or oe[0]['child']!=event['name']:return [],'no_unique_typed_role_star'
    pe,oe=pe[0],oe[0]
    if not all(role_score(tb,q)>0 for tb in [population,output,event]):return [],'no_positive_role_evidence'
    if not having or not isinstance(having.this,exp.EQ) or not isinstance(having.this.this,exp.Count) or not isinstance(having.this.this.this,exp.Star) or not isinstance(having.this.expression,exp.Subquery):return [],'count_extreme_skeleton_absent'
    dist=having.this.expression.this;dg=dist.args.get('group');order=dist.args.get('order');lim=dist.args.get('limit')
    if not isinstance(dist,exp.Select) or not dg or len(dg.expressions)!=1 or not isinstance(dg.expressions[0],exp.Column) or dg.expressions[0].name.lower()!=pe['child_field'].lower() or not order or len(order.expressions)!=1 or not isinstance(order.expressions[0].this,exp.Count) or not lim or lim.expression.this!='1':return [],'extreme_distribution_not_resolved'
    if bool(order.expressions[0].args.get('desc'))!=(direction=='DESC'):return [],'explicit_extreme_direction_conflict'
    if any(t.args.get(k) for k in ['limit','distinct']):return [],'outer_limit_or_dedup_scope_unimplemented'
    expected={frozenset([(event['name'].lower(),e['child_field'].lower()),(e['parent'].lower(),e['parent_field'].lower())]) for e in [pe,oe]};actual=set()
    for j in t.args.get('joins') or []:
        on=j.args.get('on')
        if j.args.get('side') or j.args.get('kind') not in [None,'INNER'] or not isinstance(on,exp.EQ) or any(not isinstance(c,exp.Column) or c.table not in src for c in [on.this,on.expression]):return [],'unpreserved_implicit_join_eligibility'
        actual.add(frozenset((src[c.table]['name'].lower(),c.name.lower()) for c in [on.this,on.expression]))
    if actual!=expected:return [],'unpreserved_join_roles'
    aliases={event['name']:'sk_event',population['name']:'sk_population',output['name']:'sk_output_filter'}
    joins=f'{qi(event["name"])} AS sk_event JOIN {qi(population["name"])} AS sk_population ON {col("sk_event",pe["child_field"])}={col("sk_population",pe["parent_field"])} JOIN {qi(output["name"])} AS sk_output_filter ON {col("sk_event",oe["child_field"])}={col("sk_output_filter",oe["parent_field"])}'
    try:pred=rebind(t.args['where'].this,src,aliases) if t.args.get('where') else '1=1'
    except ValueError as e:return [],str(e)
    if t.args.get('where') and t.args['where'].find(exp.Select):return [],'nested_population_filter_scope_unimplemented'
    for ties in ['single_winner','all_ties']:
        if ties=='single_winner' and dist.args.get('where'):continue # cannot drop a separate explicit inner filter
        winners=f'SELECT {col("sk_population",pe["parent_field"])} FROM {joins} WHERE ({pred}) GROUP BY {col("sk_population",pe["parent_field"])}'
        if ties=='single_winner':winners+=f' ORDER BY COUNT(*) {direction} LIMIT 1'
        else:winners+=' HAVING COUNT(*)=('+sql_text(dist)+')'
        relevant=f'SELECT 1 FROM {joins} WHERE ({pred}) AND {col("sk_event",oe["child_field"])}={col("sk_root",oe["parent_field"])} AND {col("sk_event",pe["child_field"])} IN ({winners})'
        new=exp.select(*[exp.column(e.name,table='sk_root') for e in projected]).from_(exp.to_table(output['name']).as_('sk_root')).where(exp.Exists(this=parse(relevant)))
        for i,e in enumerate(t.expressions):
            if isinstance(e,exp.Alias):new.expressions[i].replace(exp.alias_(new.expressions[i].copy(),e.alias))
        if t.args.get('order'):
            o=t.args['order'].copy()
            if any(c.table not in src or src[c.table] is not output for c in o.find_all(exp.Column)):continue
            for c in o.find_all(exp.Column):c.set('table',exp.to_identifier('sk_root'))
            new.set('order',o)
        out.append(emit(sql_text(new),'extreme_projection',{'root':output['name'],'population':population['name'],'event':event['name'],'edges':[pe,oe],'ties':ties,'direction':direction,'grain':'output entity root','implicit_joins_preserved':True},['tie_policy_not_unique','output_entity_grain_assumed'],['three_typed_declared_roles','all_outer_filters_and_join_eligibility_preserved','count_direction_preserved'],current))
    return out,None
def predicate_contract(tree,schema):
    def parts(n):
        while isinstance(n,exp.Paren):n=n.this
        return parts(n.this)+parts(n.expression) if isinstance(n,exp.And) else [n]
    found=[]
    for w in tree.find_all(exp.Where):
        sel=w.find_ancestor(exp.Select);visible={};p=sel
        while p:
            for a,tb in sources(p,schema).items():visible.setdefault(a.lower(),tb)
            p=p.find_ancestor(exp.Select)
        for atom in parts(w.this):
            if atom.find(exp.Select):continue
            a=atom.copy()
            for c in a.find_all(exp.Column):
                owners=[tb for alias,tb in visible.items() if (not c.table or alias==c.table.lower()) and column(tb,c.name)]
                if len(owners)!=1:owners=[tb for tb in schema if column(tb,c.name) and (not c.table or tb['name'].lower()==c.table.lower())]
                if len(owners)!=1:return None
                c.set('table',exp.to_identifier(owners[0]['name'].lower()));c.set('this',exp.to_identifier(c.name.lower()))
            found.append(sql_text(a))
    return collections.Counter(found)
def static_check(c,obs,schema):
    t=parse(c['sql']);parent=parse(c['parent']);q=obs['task']['question']
    if not isinstance(t,exp.Select):return False,'invalid_select'
    known={tb['name'].lower() for tb in schema}
    if any(x.name.lower() not in known for x in t.find_all(exp.Table)):return False,'unknown_table'
    width=len(obs.get('example',{}).get('columns',[]))
    if width and len(t.expressions)!=width:return False,'output_width_conflict'
    needed=collections.Counter((x.is_string,x.this) for w in parent.find_all(exp.Where) for x in w.find_all(exp.Literal))
    actual=collections.Counter((x.is_string,x.this) for x in t.find_all(exp.Literal))
    if any(actual[k]<v for k,v in needed.items()):return False,'explicit_filter_constant_lost'
    old_filters=predicate_contract(parent,schema);new_filters=predicate_contract(t,schema)
    if old_filters is None or new_filters is None:return False,'filter_scope_not_resolved'
    if any(new_filters[k]<v for k,v in old_filters.items()):return False,'explicit_filter_operator_or_owner_lost'
    if c['family']=='count_membership' and re.search(r'\b(?:each|every|per|respectively)\b',q,re.I):return False,'explicit_separate_populations_conflict'
    if c['family']=='extreme_projection' and c['slots']['ties']=='single_winner' and re.search(r'\ball\s+(?:the\s+)?tie(?:d|s)\b',q,re.I):return False,'explicit_all_ties_conflict'
    named=re.findall(r'\b(?:named|called)\s+[\'"]([^\'"]+)[\'"]',q,re.I)
    for eq in t.find_all(exp.EQ):
        if isinstance(eq.this,exp.Column) and isinstance(eq.expression,exp.Literal) and eq.expression.this in named and re.search(r'topic|theme|category|status|address|price|date|time',eq.this.name,re.I) and not re.search(r'name|title',eq.this.name,re.I):return False,'explicit_name_bound_to_non_name_role'
    return True,None
def strict_check(c,obs,schema):
    if c['family']=='count_membership':
        root=table(schema,c['slots']['root']);src=sources(parse(c['parent']),schema)
        return bool(root in src.values() and linked(root,obs['task']['question']))
    return c['slots']['ties']=='all_ties' and bool(re.search(r'\ball\s+(?:the\s+)?tie(?:d|s)\b',obs['task']['question'],re.I))
def generate(current,obs,schema,family='combined'):
    raw=[];reasons={}
    for f,fn in [('count',count_sketch),('extreme',extreme_sketch)]:
        if family not in ['combined',f]:continue
        cs,r=fn(current,obs,schema);raw+=cs;reasons[f]=r
    seen=set();out=[]
    for c in raw:
        if c['sql'] in seen:continue
        seen.add(c['sql']);ok,r=static_check(c,obs,schema);c['static_admit']=ok;c['static_reason']=r;c['old_strict_admit']=strict_check(c,obs,schema)
        c['priority']=c['slots'].get('root_role_score',0)+(1 if c['slots'].get('ties')=='single_winner' else 0)
        out.append(c)
    return sorted(out,key=lambda c:-c['priority'])[:CAP],reasons
