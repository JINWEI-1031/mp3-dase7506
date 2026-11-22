"""Independent A seeds, B plans and C selective evidence/guard diagnostics."""
import re
from sqlglot import exp
from repair_rules import parse,sql_text
from step_evidence28_192_scoped import generate as local_generate
from step_space435_common import seed_set,diverse,guard,structural_key
from step_space435_plan import generate_plans
from step124_pool_search import sources

def zero_conflict(sql,current,obs,schema):
    tree=parse(sql);old=parse(current)
    if not isinstance(tree,exp.Select) or not any(j.args.get('side')=='LEFT' for j in tree.args.get('joins',[])):return None
    explicit=bool(re.search(r'(?:including|include|even).*\b(?:zero|no|without)\b|whether or not',obs['task']['question'],re.I))
    if not explicit:return 'weak_zero_population_domain'
    root=tree.args.get('from');src=sources(tree,schema)
    if not root:return 'zero_root_unbound'
    ra=root.this.alias_or_name;children={j.this.alias_or_name for j in tree.args.get('joins',[]) if isinstance(j.this,exp.Table)}
    where=tree.args.get('where')
    if where:
        for col in where.find_all(exp.Column):
            owners=[a for a,t in src.items() if (not col.table or a.lower()==col.table.lower()) and any(c['name'].lower()==col.name.lower() for c in t['columns'])]
            if len(owners)!=1:return 'zero_filter_scope_ambiguous'
            if owners[0] in children:return 'zero_child_filter_in_where'
    for count in tree.find_all(exp.Count):
        if isinstance(count.this,exp.Star):return 'zero_null_extended_count_star'
        cols=list(count.find_all(exp.Column))
        if not cols or any(not c.table or c.table==ra for c in cols):return 'zero_count_object_not_child'
    if old and any(isinstance(c.this,exp.Distinct) for c in old.find_all(exp.Count)):
        original={c.name.lower() for count in old.find_all(exp.Count) for c in count.find_all(exp.Column)}
        revised={c.name.lower() for count in tree.find_all(exp.Count) for c in count.find_all(exp.Column)}
        if not all(isinstance(c.this,exp.Distinct) for c in tree.find_all(exp.Count)) or original!=revised:return 'zero_distinct_object_changed'
    return None

def probe(current,obs,schema):
    t=parse(current)
    if not isinstance(t,exp.Select) or len(list(t.find_all(exp.Select)))!=1 or t.args.get('joins'):return None
    src=sources(t,schema)
    if len(src)!=1 or not t.args.get('where'):return None
    alias,table=next(iter(src.items()));pairs=[]
    for eq in t.args['where'].find_all(exp.EQ):
        if isinstance(eq.this,exp.Column) and isinstance(eq.expression,exp.Literal) and eq.expression.is_string and eq.expression.this.lower() in obs['task']['question'].lower() and len(eq.expression.this)<=60:pairs.append(eq)
    if len(pairs)!=1:return None
    pair=pairs[0];fields=[c['name'] for c in table['columns'] if re.search(r'CHAR|TEXT|CLOB',c['type'],re.I)]
    if pair.this.name.lower() not in {f.lower() for f in fields} or len(fields)>8:return None
    expressions=[exp.alias_(exp.Sum(this=exp.Case(ifs=[exp.If(this=exp.EQ(this=exp.Lower(this=exp.column(f)),expression=exp.Lower(this=pair.expression.copy())),true=exp.Literal.number(1))],default=exp.Literal.number(0))),f'evidence_{i}') for i,f in enumerate(fields)]
    sql=sql_text(exp.select(*expressions).from_(table['name']))
    return {'sql':sql,'fields':fields,'current_field':pair.this.name,'literal':pair.expression.this,'table':table['name'],'alias':alias,'original':current}

def apply_probe(plan,feedback,obs,schema):
    if not feedback.get('ok') or len(feedback.get('rows',[]))!=1:return [],'value_probe_failed'
    row=feedback['rows'][0];hits=[f for f,n in zip(plan['fields'],row) if isinstance(n,(int,float)) and n>0];old=next((n for f,n in zip(plan['fields'],row) if f.lower()==plan['current_field'].lower()),None)
    if not hits:return [],'public_value_absent'
    if len(hits)!=1:return [],'value_role_evidence_ambiguous'
    if old not in (0,None) or hits[0].lower()==plan['current_field'].lower():return [],'current_value_field_supported'
    tree=parse(plan['original'])
    for eq in tree.args['where'].find_all(exp.EQ):
        if isinstance(eq.this,exp.Column) and eq.this.name.lower()==plan['current_field'].lower() and isinstance(eq.expression,exp.Literal) and eq.expression.this==plan['literal']:eq.this.set('this',exp.to_identifier(hits[0]))
    c={'sql':sql_text(tree),'family':'supplemental_evidence','path':['unique_public_value_field'],'source':'space435_value_probe','confidence':12,'evidence':{'probe_sql':plan['sql'],'unique_value_field':hits[0],'prior_field_zero_hits':True,'literal':plan['literal']}}
    return ([c],None) if guard(c['sql'],obs,schema) is None else ([],guard(c['sql'],obs,schema))

def candidates(current,obs,schema,arm='abc',cap=24):
    multi=arm in ('a','a48','abc');plans=arm in ('b','abc');seeds=seed_set(current,obs,multi=multi);items=[];log=[]
    for seed in seeds:
        for c in local_generate([seed['sql']],obs,schema,cap=cap):items.append({**c,'seed_source':seed['source']})
        if multi and seed['source']!='current':
            reason=guard(seed['sql'],obs,schema)
            if reason:log.append({'source':seed['source'],'reason':reason})
            else:items.append({'sql':seed['sql'],'family':'seed','path':['visible_seed'],'source':'space435_'+seed['source'],'seed_source':seed['source'],'confidence':5,'evidence':{'runtime_source':seed['source']}})
    if plans:
        fresh,reasons=generate_plans(current,obs,schema);items+=fresh;log+=reasons
    if arm in ('c','abc'):
        accepted=[]
        for c in items:
            reason=zero_conflict(c['sql'],current,obs,schema)
            if reason=='weak_zero_population_domain':
                c={**c,'confidence':min(c['confidence'],4)};log.append({'source':c.get('source'),'reason':reason,'policy':'retained_low_priority'})
            elif reason:log.append({'source':c.get('source'),'reason':reason});continue
            accepted.append(c)
        items=accepted
    result,rejected=diverse(items,obs,schema,cap);log+=rejected
    if not result:
        t=parse(current)
        log.append({'reason':'seed_structure_unsupported' if not isinstance(t,exp.Select) else 'no_supported_plan_or_edit','schema_observed':bool(schema)})
    return result,{'seeds':[{'source':s['source'],'structure':structural_key(s['sql'],schema)} for s in seeds],'rejections':log,'cap':cap,'schema_source':'runtime_observation'}
