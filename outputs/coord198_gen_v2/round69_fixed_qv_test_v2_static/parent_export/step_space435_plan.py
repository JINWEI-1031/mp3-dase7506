"""Declarative relation plans: entity -> paths -> filters -> aggregate -> output."""
import re
from sqlglot import exp
from repair_rules import parse,sql_text
from step_semantic192_rules import tokens,paths,join_path,conjuncts
from step_rolebind192_rules import role_words,affinity
from step124_pool_search import sources
from step_space435_common import guard

def proposal(tree,kind,plan,confidence=10):
    return {'sql':sql_text(tree),'family':'relation_plan','path':[kind],'source':'space435_plan','confidence':confidence,'evidence':plan,'plan':plan}

def roots_for_headers(headers,schema):
    if not all(re.fullmatch(r'\w+',h) for h in headers):return []
    return [t for t in schema if all(any(c['name'].lower()==h.lower() for c in t['columns']) for h in headers)]

def anchored_filters(current,obs,schema):
    tree=parse(current);out=[]
    if not isinstance(tree,exp.Select) or not tree.args.get('where'):return out
    src=sources(tree,schema)
    for p in conjuncts(tree.args['where'].this):
        if p.find(exp.Subquery) or p.find(exp.Exists):continue
        literals=list(p.find_all(exp.Literal))
        if not literals or any(v.is_string and v.this.strip('%').lower() not in obs['task']['question'].lower() for v in literals):continue
        bindings=[];supported=True
        for c in p.find_all(exp.Column):
            owners=[t for a,t in src.items() if (not c.table or c.table.lower()==a.lower()) and any(f['name'].lower()==c.name.lower() for f in t['columns'])]
            if len(owners)!=1:supported=False;break
            bindings.append(owners[0]['name'])
        if supported and len(set(bindings))==1:out.append((bindings[0],p.copy()))
    return out

def attach_filters(root,filters,schema):
    base=exp.select().from_(exp.to_table(root['name']).as_('p0'));bindings={root['name'].lower():'p0'};path_evidence=[]
    for name,predicate in filters:
        if name.lower() not in bindings:
            options=paths(schema,root['name'],name)
            if not options or len(options)>1 and len(options[0])==len(options[1]):return None,None,{'reason':'ambiguous_or_missing_filter_path','filter_entity':name}
            src={a:next(t for t in schema if t['name'].lower()==n) for n,a in bindings.items()};base,bindings=join_path(base,src,options[0]);path_evidence+=options[0]
        p=predicate.copy()
        for c in p.find_all(exp.Column):c.set('table',exp.to_identifier(bindings[name.lower()]))
        base=base.where(p)
    return base,bindings,{'paths':path_evidence}

def count_plan(current,obs,schema,log):
    q=obs['task']['question'];request=re.search(r'\b(?:how many|number of)\s+(?:distinct\s+)?(?:types of\s+)?([a-z]+)',q,re.I)
    if not request:return []
    original=parse(current)
    if isinstance(original,exp.Select) and (len(list(original.find_all(exp.Select)))>1 or original.args.get('having') or original.args.get('where') and original.args['where'].find(exp.Or)):
        log.append({'provider':'count_plan','reason':'count_population_complex_scope_unsupported'});return []
    targets=[t for t in schema if tokens(t['name'])==tokens(request[1])]
    if not targets:targets=[t for t in schema if tokens(request[1])<=tokens(t['name'])]
    if len(targets)!=1:log.append({'provider':'count_plan','reason':'count_object_ambiguous'});return []
    target=targets[0];keys=[c['name'] for c in target['columns'] if c.get('pk')]
    fields=keys if len(keys)==1 else []
    # Public output expression can resolve a count object without reference SQL.
    for h in obs['example']['columns']:
        parsed=parse('SELECT '+h)
        if parsed and parsed.find(exp.Count):
            names=[c.name for c in parsed.find_all(exp.Column)]
            matching=[c['name'] for c in target['columns'] if any(c['name'].lower()==n.lower() for n in names)]
            if len(matching)==1:fields=matching
    if not fields:log.append({'provider':'count_plan','reason':'count_key_missing'});return []
    filters=anchored_filters(current,obs,schema)
    if not filters:
        negative=re.search(r'\b(?:do not|does not|don.t|doesn.t)\s+have\s+(?:the\s+)?([a-z]+)\s+(?:function|feature)',q,re.I)
        matches=[c['name'] for c in target['columns'] if negative and re.sub(r'\W','',c['name']).lower()==negative[1].lower()]
        if len(matches)==1:
            base=exp.select(exp.Count(this=exp.Distinct(expressions=[exp.column(fields[0],table='p0')]))).from_(exp.to_table(target['name']).as_('p0')).where(exp.EQ(this=exp.Lower(this=exp.column(matches[0],table='p0')),expression=exp.Literal.string('no')))
            return [proposal(base,'count_boolean_feature_plan',{'target_entity':target['name'],'count_field':fields[0],'feature_from_question':negative[1],'field':matches[0],'negative_boolean':'no'},11)]
        log.append({'provider':'count_plan','reason':'population_filter_evidence_missing'});return []
    base,bindings,evidence=attach_filters(target,filters,schema)
    if base is None:log.append({'provider':'count_plan',**evidence});return []
    base.set('expressions',[exp.Count(this=exp.Distinct(expressions=[exp.column(fields[0],table='p0')]))])
    return [proposal(base,'count_entity_population_plan',{'target_entity':target['name'],'count_field':fields[0],'filters':[n for n,p in filters],**evidence,'constraints_preserved':'Question-grounded values retained; unrelated address/count-object plan replaced.'},11)]

def extremal_plan(current,obs,schema,log):
    q=obs['task']['question'];match=re.search(r'\b(most|fewest|least)\s+([a-z]+)',q,re.I)
    if not match:return []
    noun=tokens(match[2]);roots=roots_for_headers(obs['example']['columns'],schema)
    if len(roots)!=1:log.append({'provider':'extremal_plan','reason':'output_entity_ambiguous'});return []
    root=roots[0];desc=match[1].lower()=='most';candidates=[]
    for event in schema:
        if not noun<=tokens(event['name']) and not any(noun<=tokens(c['name']) for c in event['columns']):continue
        if not tokens(event['name'])&tokens(q):continue
        for fk in event['foreign_keys']:
            parent=next((t for t in schema if t['name'].lower()==fk['table'].lower()),None)
            if parent is None or not tokens(parent['name'])<=tokens(q):continue
            # The ranked parent is explicitly introduced before a relative clause.
            if not any(re.search(r'\b'+re.escape(w)+r's?\s+(?:who|which|that)\b',q,re.I) for w in tokens(parent['name'])):continue
            candidates.append((event,parent,fk))
    if len(candidates)>1:
        scored=[]
        for candidate in candidates:
            parent=candidate[1];anchors=[m.end() for w in tokens(parent['name']) for m in re.finditer(r'\b'+re.escape(w)+r's?\s+(?:who|which|that)\b',q,re.I) if m.end()<=match.start()]
            scored.append((max(anchors,default=-1),candidate))
        best=max(s for s,c in scored);candidates=[c for s,c in scored if s==best]
    if len(candidates)!=1:log.append({'provider':'extremal_plan','reason':'ranked_entity_or_event_role_ambiguous','choices':len(candidates)});return []
    event,parent,fk=candidates[0];rank=exp.select(exp.column(fk['from'],table='rank_event')).from_(exp.to_table(event['name']).as_('rank_event')).where(exp.Not(this=exp.Is(this=exp.column(fk['from'],table='rank_event'),expression=exp.Null()))).group_by(exp.column(fk['from'],table='rank_event')).order_by(exp.Ordered(this=exp.Count(this=exp.Star()),desc=desc)).limit(1)
    filters=anchored_filters(current,obs,schema)
    if filters:log.append({'provider':'extremal_plan','reason':'extremal_population_filter_scope_unresolved'});return []
    base,bindings,evidence=attach_filters(root,filters,schema)
    if base is None:log.append({'provider':'extremal_plan',**evidence});return []
    options=paths(schema,root['name'],event['name']) if root['name']!=event['name'] else [[]]
    if len(options)!=1:log.append({'provider':'extremal_plan','reason':'rank_output_path_ambiguous'});return []
    if options[0]:
        src={a:next(t for t in schema if t['name'].lower()==n) for n,a in bindings.items()};base,bindings=join_path(base,src,options[0])
    event_alias=bindings[event['name'].lower()];base=base.where(exp.In(this=exp.column(fk['from'],table=event_alias),query=exp.Subquery(this=rank)))
    base.set('expressions',[exp.column(h,table='p0') for h in obs['example']['columns']])
    p={'output_entity':root['name'],'ranked_entity':parent['name'],'event_entity':event['name'],'rank_fk':fk,'rank_population':'entities with recorded events','direction':'DESC' if desc else 'ASC','limit':1,'output_path':options[0],'filters_preserved':[n for n,_ in filters]}
    # Distinguish all mathematical ties from a single ranked entity without silently
    # assuming a unique extremum. Both variants remain explicit in provenance.
    output=[proposal(base,'extremal_population_output_plan',{**p,'tie_policy':'single'},10)]
    tie_rank=rank.copy();tie_rank.set('order',None);tie_rank.set('limit',None);best_count=exp.select(exp.Count(this=exp.Star())).from_(exp.to_table(event['name']).as_('tie_event')).where(exp.Not(this=exp.Is(this=exp.column(fk['from'],table='tie_event'),expression=exp.Null()))).group_by(exp.column(fk['from'],table='tie_event')).order_by(exp.Ordered(this=exp.Count(this=exp.Star()),desc=desc)).limit(1)
    tie_rank=tie_rank.having(exp.EQ(this=exp.Count(this=exp.Star()),expression=exp.Subquery(this=best_count)))
    tied=base.copy()
    for sub in tied.find_all(exp.Subquery):
        if sql_text(sub.this)==sql_text(rank):sub.set('this',tie_rank.copy())
    output.append(proposal(tied,'extremal_population_output_plan',{**p,'tie_policy':'all'},11))
    # If the output entity is merely qualified by child events, preserve entity
    # multiplicity with EXISTS rather than multiplying it by event rows.
    if len(options[0])==1 and root['name']!=event['name']:
        links=[f for f in event['foreign_keys'] if f['table'].lower()==root['name'].lower()]
        if len(links)==1 and not filters:
            rf=links[0]
            for rank_sql,tie,confidence in [(rank,'single',11),(tie_rank,'all',12)]:
                sub=exp.select(exp.Literal.number(1)).from_(exp.to_table(event['name']).as_('qualified_event')).where(exp.and_(exp.EQ(this=exp.column(rf['from'],table='qualified_event'),expression=exp.column(rf['to'],table='p0')),exp.In(this=exp.column(fk['from'],table='qualified_event'),query=exp.Subquery(this=rank_sql.copy()))))
                fresh=exp.select(*(exp.column(h,table='p0') for h in obs['example']['columns'])).from_(exp.to_table(root['name']).as_('p0')).where(exp.Exists(this=sub))
                output.append(proposal(fresh,'extremal_population_output_plan',{**p,'tie_policy':tie,'output_grain':'one row per target entity'},confidence))
    return output

def union_population(current,obs,schema,log):
    q=obs['task']['question'];headers=obs['example']['columns']
    if len(headers)!=1 or not re.fullmatch(r'\w+',headers[0]) or not re.search(r'\ball\b.+\band\b',q,re.I):return []
    targets=[t for t in roots_for_headers(headers,schema) if tokens(t['name']) and tokens(t['name'])<=tokens(q)]
    if len(targets)!=2:return []
    if anchored_filters(current,obs,schema):log.append({'provider':'union_population','reason':'filtered_population_union_requires_scope'});return []
    a,b=[exp.select(headers[0]).from_(t['name']) for t in targets]
    return [proposal(exp.Union(this=a,expression=b,distinct=True),'independent_population_union',{'populations':[t['name'] for t in targets],'output':headers,'relation':'independent populations, not identifier equality','duplicate_policy':'set'},11)]

def temporal_reference(current,obs,schema,log):
    q=obs['task']['question'];request=re.search(r'\b(before|after)\s+(?:the\s+date\s+of\s+)?(any|every|all)\b',q,re.I)
    if not request:return []
    if request[2].lower()!='any':log.append({'provider':'temporal_reference','reason':'universal_empty_null_scope_requires_separate_plan'});return []
    roots=roots_for_headers(obs['example']['columns'],schema)
    roots=[t for t in roots if tokens(t['name']) and tokens(t['name'])<=tokens(q)]
    if len(roots)!=1:log.append({'provider':'temporal_reference','reason':'event_output_entity_ambiguous'});return []
    root=roots[0];links=[]
    for f in root['foreign_keys']:
        actor=next((t for t in schema if t['name'].lower()==f['table'].lower()),None)
        if actor is None or not tokens(actor['name'])&tokens(q):continue
        first=[c['name'] for c in actor['columns'] if re.search(r'first_?name|fname',c['name'],re.I)];last=[c['name'] for c in actor['columns'] if re.search(r'last_?name|lname',c['name'],re.I)]
        if len(first)==len(last)==1:links.append((affinity(f['from'],q),f,actor,first[0],last[0]))
    if not links:return []
    best=max(x[0] for x in links);links=[x for x in links if x[0]==best]
    if len(links)!=1:log.append({'provider':'temporal_reference','reason':'actor_relation_role_ambiguous'});return []
    _,fk,actor,first,last=links[0];t=parse(current);full=[]
    if t:
        full=[v.this for v in t.find_all(exp.Literal) if v.is_string and len(v.this.split())==2 and v.this.lower() in q.lower()]
    if len(set(full))!=1:log.append({'provider':'temporal_reference','reason':'actor_name_evidence_missing'});return []
    name=full[0].split();dates=[c for c in root['columns'] if re.search(r'DATE|TIME',c['type'],re.I)]
    if not dates:return []
    score=max(affinity(c['name'],q) for c in dates);dates=[c for c in dates if affinity(c['name'],q)==score]
    if len(dates)!=1:return []
    field=dates[0]['name'];before=request[1].lower()=='before';any_=request[2].lower()=='any';aggregate=exp.Max if before==any_ else exp.Min
    actor_ids=exp.select(exp.column(fk['to'],table='actor')).from_(exp.to_table(actor['name']).as_('actor')).where(exp.and_(exp.EQ(this=exp.column(first,table='actor'),expression=exp.Literal.string(name[0])),exp.EQ(this=exp.column(last,table='actor'),expression=exp.Literal.string(name[1]))))
    reference=exp.select(aggregate(this=exp.column(field,table='reference_event'))).from_(exp.to_table(root['name']).as_('reference_event')).where(exp.In(this=exp.column(fk['from'],table='reference_event'),query=exp.Subquery(this=actor_ids)))
    compare=exp.LT if before else exp.GT;tree=exp.select(*(exp.column(h,table='outer_event') for h in obs['example']['columns'])).from_(exp.to_table(root['name']).as_('outer_event')).where(compare(this=exp.column(field,table='outer_event'),expression=exp.Subquery(this=reference)))
    return [proposal(tree,'temporal_actor_quantifier_plan',{'target_entity':root['name'],'event_time':field,'actor_entity':actor['name'],'actor_fk':fk,'name_from_question':full[0],'quantifier':request[2].lower(),'reference_aggregate':aggregate.__name__,'population':'all outer events; actor filter inside reference only'},12)]

def generate_plans(current,obs,schema):
    log=[];out=[]
    for provider in (count_plan,extremal_plan,union_population,temporal_reference):out+=provider(current,obs,schema,log)
    tree=parse(current)
    if tree is None:log.append({'provider':'local_edit','reason':'seed_structure_unsupported'})
    for c in out:
        reason=guard(c['sql'],obs,schema)
        if reason:log.append({'provider':c['path'][0],'reason':reason})
    return [c for c in out if guard(c['sql'],obs,schema) is None],log
