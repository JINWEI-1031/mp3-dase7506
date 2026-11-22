"""Two bounded typed relational plans; runtime question/schema/current SQL only."""
import re
from sqlglot import exp
from repair_rules import parse,sql_text
from step124_pool_search import sources
from step_semantic192_rules import tokens
from step_graph439_edges import family

def qi(s):return '"'+s.replace('"','""')+'"'
def col(a,c):return qi(a)+'.'+qi(c)
def table(schema,name):return next((t for t in schema if t['name'].lower()==name.lower()),None)
def column(tb,name):return next((c for c in tb['columns'] if c['name'].lower()==name.lower()),None)
def key(tb):
    ks=[c['name'] for c in tb['columns'] if c.get('pk')]
    return ks[0] if len(ks)==1 else None
def linked(tb,q):return bool(tokens(tb['name'])) and tokens(tb['name'])<=tokens(q)
def declared(schema,left,right):
    edges=[]
    for t in [left,right]:
        other=right if t is left else left
        for fk in t.get('foreign_keys',[]):
            if fk['table'].lower()!=other['name'].lower():continue
            a,b=column(t,fk['from']),column(other,fk['to'])
            if not a or not b or family(a['type'])=='other' or family(a['type'])!=family(b['type']):continue
            # A scalar endpoint, never silently interpret components of a composite FK.
            if key(other)!=b['name']:continue
            edges.append({'child':t['name'],'child_field':a['name'],'parent':other['name'],'parent_field':b['name'],'declared':True})
    return list({tuple(e.values()):e for e in edges}.values())
def candidate(ir,sql):return {'sql':sql,'source':'scope441_'+ir['operator'],'path':[ir['operator']],'ir':ir,'evidence':ir['evidence']}

def membership_count(current,obs,schema):
    t=parse(current);q=obs['task']['question']
    if not isinstance(t,exp.Select) or len(list(t.find_all(exp.Select)))!=1:return None,'nested_or_invalid_count_scope'
    if not re.search(r'\bhow many\b|\bnumber of\b',q,re.I):return None,'no_entity_quantity_request'
    if re.search(r'\b(?:each|every|per|respectively|both|neither|either|any|all|never)\b',q,re.I):return None,'population_or_quantifier_ambiguous'
    if any(t.args.get(k) for k in ['joins','group','having','order','limit','distinct']):return None,'not_single_global_entity_count'
    if len(t.expressions)!=1 or not isinstance(t.expressions[0].unalias(),exp.Count) or not isinstance(t.expressions[0].unalias().this,exp.Star):return None,'count_object_not_root_rows'
    src=sources(t,schema)
    if len(src)!=1:return None,'unknown_count_root'
    alias,root=next(iter(src.items()));pk=key(root)
    if not pk:return None,'root_key_not_unique'
    # Preserve the parent's counted root, but require an explicit entity-role anchor.
    anchored=linked(root,q) or bool(tokens(pk)-{'name','id','key','number','num','hardware'}<=tokens(q) and tokens(pk)-{'name','id','key','number','num','hardware'})
    if not anchored:return None,'count_root_not_question_grounded'
    w=t.args.get('where')
    if not w or w.find(exp.Or) or w.find(exp.Not) or w.find(exp.Subquery):return None,'ambiguous_count_predicate_scope'
    missing=[c for c in w.find_all(exp.Column) if column(root,c.name) is None]
    if not missing or len({c.name.lower() for c in missing})!=1:return None,'no_unique_missing_count_filter'
    field=missing[0].name
    if not tokens(field.lower())<=tokens(q.lower()):return None,'filter_field_not_question_grounded'
    owners=[tb for tb in schema if tb is not root and column(tb,field)]
    if len(owners)!=1:return None,'count_filter_owner_ambiguous'
    owner=owners[0];edges=declared(schema,root,owner)
    if len(edges)!=1:return None,'no_unique_declared_count_path'
    edge=edges[0]
    if edge['child']!=root['name']:return None,'filter_relation_is_to_many_requires_other_plan'
    pred=w.this.copy()
    for c in pred.find_all(exp.Column):
        if c.table and c.table.lower() not in [alias.lower(),owner['name'].lower()]:return None,'unknown_filter_qualifier'
        c.set('table',exp.to_identifier('population_root' if column(root,c.name) else 'population_filter'))
    condition=f'{col("population_root",edge["child_field"])} = {col("population_filter",edge["parent_field"])}'
    sub=f'SELECT 1 FROM {qi(owner["name"])} AS population_filter WHERE {condition} AND ({sql_text(pred)})'
    new=t.copy();new.set('from',exp.From(this=exp.to_table(root['name']).as_('population_root')));new.set('where',exp.Where(this=exp.Exists(this=parse(sub))))
    ir={'operator':'entity_membership_count','root_entity':root['name'],'root_key':pk,'population_predicate':sql_text(w.this),'event_relation':edge,'count_object':'root rows (one per declared root key)','distinct':'root key defines population; no added DISTINCT','zero_events':'not relevant: positive many-to-one attribute membership','output_grouping':'single scalar','evidence':{'contradiction':'count predicate field is absent from counted root; unique declared typed path owns explicitly named field','preserved':'COUNT object, root population, predicate operators/literals and scalar output alias','assumption':'parent root denotes requested entity; no to-many edge or root substitution allowed'}}
    return candidate(ir,sql_text(new)),None

def extreme_projection(current,obs,schema):
    t=parse(current);q=obs['task']['question'];qt=tokens(q)
    if not isinstance(t,exp.Select):return None,'invalid_extreme_scope'
    dirs=[v for v in ['fewest','most'] if v in qt]
    if len(dirs)!=1:return None,'no_unique_count_extreme'
    if re.search(r'\b(?:not|without|neither|both|either|respectively|before|after)\b',q,re.I):return None,'extreme_scope_ambiguous'
    if not re.search(r'\bwho\b',q,re.I):return None,'extreme_population_role_not_explicit'
    if t.args.get('where') or t.args.get('limit') or t.args.get('distinct'):return None,'outer_filter_limit_or_dedup_requires_unimplemented_scope'
    group=t.args.get('group');having=t.args.get('having')
    if not group or len(group.expressions)!=1 or not isinstance(group.expressions[0],exp.Column) or not having:return None,'no_single_population_group'
    hv=having.this
    if not isinstance(hv,exp.EQ) or not isinstance(hv.this,exp.Count) or not isinstance(hv.this.this,exp.Star) or not isinstance(hv.expression,exp.Subquery):return None,'not_tie_preserving_count_extreme_contract'
    dist=hv.expression.this
    if not isinstance(dist,exp.Select) or len(dist.expressions)!=1 or not isinstance(dist.expressions[0].unalias(),exp.Count) or not isinstance(dist.expressions[0].unalias().this,exp.Star):return None,'count_measure_not_row_count'
    if dist.args.get('where') or dist.args.get('having') or dist.args.get('joins'):return None,'inner_population_filter_not_supported'
    dg=dist.args.get('group');order=dist.args.get('order');limit=dist.args.get('limit')
    if not dg or len(dg.expressions)!=1 or not isinstance(dg.expressions[0],exp.Column) or not order or len(order.expressions)!=1 or not isinstance(order.expressions[0].this,exp.Count) or not limit or limit.expression.this!='1':return None,'no_scalar_extreme_distribution'
    descending=bool(order.expressions[0].args.get('desc'))
    if descending!=(dirs[0]=='most'):return None,'question_extreme_direction_conflicts'
    src=sources(t,schema);ds=sources(dist,schema)
    if len(src)!=3 or len(ds)!=1:return None,'requires_one_event_two_entity_star'
    event=next(iter(ds.values()));groupfield=dg.expressions[0].name
    groupcol=group.expressions[0]
    if groupcol.table not in src:return None,'population_group_owner_not_resolved'
    population=src[groupcol.table]
    if population is event or not key(population) or key(population).lower()!=groupcol.name.lower():return None,'group_not_population_entity_key'
    projected=[e.unalias() for e in t.expressions]
    if not projected or any(not isinstance(e,exp.Column) or e.table not in src for e in projected):return None,'output_scope_not_entity_columns'
    output=src[projected[0].table]
    if output is population or output is event or any(src[e.table] is not output for e in projected) or not key(output):return None,'output_not_separate_keyed_entity'
    if not all(linked(tb,q) for tb in [event,population,output]):return None,'event_population_output_not_question_grounded'
    pe=declared(schema,event,population);oe=declared(schema,event,output)
    if len(pe)!=1 or len(oe)!=1 or pe[0]['child']!=event['name'] or oe[0]['child']!=event['name'] or pe[0]['child_field'].lower()!=groupfield.lower():return None,'ambiguous_or_unmatched_event_role_edges'
    pe,oe=pe[0],oe[0]
    # Every current join must be exactly one of these declared roles, no discarded ON filter.
    expected={frozenset([(event['name'].lower(),pe['child_field'].lower()),(population['name'].lower(),pe['parent_field'].lower())]),frozenset([(event['name'].lower(),oe['child_field'].lower()),(output['name'].lower(),oe['parent_field'].lower())])}
    actual=set()
    for j in t.args.get('joins') or []:
        on=j.args.get('on')
        if j.args.get('side') or j.args.get('kind') not in [None,'INNER'] or not isinstance(on,exp.EQ) or not all(isinstance(c,exp.Column) and c.table in src for c in [on.this,on.expression]):return None,'unpreserved_join_scope'
        actual.add(frozenset((src[c.table]['name'].lower(),c.name.lower()) for c in [on.this,on.expression]))
    if actual!=expected:return None,'unpreserved_relation_path'
    direction='DESC' if descending else 'ASC';ek=qi(event['name']);pk=qi(population['name']);ok=qi(output['name']);pc=pe['parent_field'];ef=pe['child_field']
    def joined(a,b):return f'{pk} AS {a} JOIN {ek} AS {b} ON {col(a,pc)}={col(b,ef)}'
    measure=f'SELECT COUNT(*) FROM {joined("extreme_population","extreme_event")} GROUP BY {col("extreme_population",pc)} ORDER BY COUNT(*) {direction} LIMIT 1'
    winners=f'SELECT {col("winning_population",pc)} FROM {joined("winning_population","winning_event")} GROUP BY {col("winning_population",pc)} HAVING COUNT(*)=({measure})'
    relevant=f'SELECT 1 FROM {ek} AS output_event WHERE {col("output_event",oe["child_field"])}={col("output_entity",oe["parent_field"])} AND {col("output_event",ef)} IN ({winners})'
    new=exp.select(*[exp.column(e.name,table='output_entity') for e in projected]).from_(exp.to_table(output['name']).as_('output_entity')).where(exp.Exists(this=parse(relevant)))
    # Preserve original projection aliases and sorting contract, with exact owner rebinding.
    for i,e in enumerate(t.expressions):
        if isinstance(e,exp.Alias):new.expressions[i].replace(exp.alias_(new.expressions[i].copy(),e.alias))
    if t.args.get('order'):
        o=t.args['order'].copy()
        for c in o.find_all(exp.Column):
            if c.table not in src or src[c.table] is not output:return None,'sorting_not_output_owned'
            c.set('table',exp.to_identifier('output_entity'))
        new.set('order',o)
    ir={'operator':'extreme_population_projection','root_entity':output['name'],'root_key':key(output),'population_entity':population['name'],'population_key':pc,'event_entity':event['name'],'event_edges':[pe,oe],'count_object':'event rows COUNT(*) from current parent','zero_events':'exclude: question has filed and current INNER event population','ties':'all tied population keys, preserved scalar minimum/maximum count equality','direction':direction,'filters':{'outer':None,'inner':None},'output_grouping':'output entity root via EXISTS, not one arbitrary row per winning population','evidence':{'contradiction':'current group key belongs to counted person/entity while projection belongs to separate output entity; aggregate collapses related output rows','preserved':'all declared relation roles, row count object, extrema direction and all-ties equality, original output columns/aliases/order','new_assumption':'question requests output entity population, so repeated events do not duplicate that entity; reject any unimplemented local filter or scope'}}
    return candidate(ir,sql_text(new)),None

def generate(current,obs,schema,arm='combined'):
    cs=[];reasons={}
    for name,fn in [('count',membership_count),('extreme',extreme_projection)]:
        if arm not in ['combined',name]:continue
        c,r=fn(current,obs,schema);reasons[name]=r
        if c:cs.append(c)
    return cs,reasons
