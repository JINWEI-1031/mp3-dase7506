"""Small explicit relational IR; compilers never read labels or database files."""
import re
from sqlglot import exp
from repair_rules import parse,sql_text
from step123_agent import words
from step_semantic192_rules import tokens
from step_space435_common import guard
from step_space435_plan import generate_plans as old_plans

def compile_plan(plan):
    op=plan['op']
    if op=='scan':return exp.select('*').from_(exp.to_table(plan['table']).as_(plan['alias']))
    if op in ('filter','project','aggregate'):
        tree=compile_plan(plan['input'])
        if op=='filter':tree=tree.where(parse('SELECT 1 WHERE '+plan['predicate']).args['where'].this)
        if op=='project':tree.set('expressions',[parse('SELECT '+s).expressions[0] for s in plan['columns']])
        if op=='aggregate':
            tree.set('expressions',[parse('SELECT '+s).expressions[0] for s in plan['measures']])
            if plan.get('keys'):tree=tree.group_by(*plan['keys'])
        return tree
    if op in ('semi','anti'):
        tree=compile_plan(plan['input']);sub=exp.select(exp.Literal.number(1)).from_(exp.to_table(plan['related']).as_(plan['alias'])).where(parse('SELECT 1 WHERE '+plan['correlation']).args['where'].this)
        if plan.get('predicate'):sub=sub.where(parse('SELECT 1 WHERE '+plan['predicate']).args['where'].this)
        condition=exp.Exists(this=sub)
        return tree.where(exp.Not(this=condition) if op=='anti' else condition)
    if op=='map_join':
        tree=compile_plan(plan['input']);return tree.join(exp.to_table(plan['table']).as_(plan['alias']),on=plan['condition'])
    if op=='union':return exp.Union(this=compile_plan(plan['left']),expression=compile_plan(plan['right']),distinct=plan.get('distinct',True))
    if op=='sql_plan':return parse(plan['sql'])
    raise ValueError('Unsupported plan operator '+op)

def candidate(plan,kind,evidence,confidence=11):
    return {'sql':sql_text(compile_plan(plan)),'family':'proof_ir','path':[kind],'source':'proof435_ir','confidence':confidence,'ir':plan,'evidence':evidence}

def simple_root(current,schema):
    tree=parse(current)
    if not isinstance(tree,exp.Select):return None,None
    root=tree.args.get('from')
    if not root or not isinstance(root.this,exp.Table):return None,None
    table=next((t for t in schema if t['name'].lower()==root.this.name.lower()),None)
    return tree,table

def branch_scope(current,obs,schema):
    q=obs['task']['question'];m=re.search(r'\beither\s+(\d+(?:\.\d+)?)\s+(\w+)\s+or\s+(\d+(?:\.\d+)?)\s+(\w+)\s+but\s+(\d+(?:\.\d+)?)\s+(\w+)',q,re.I)
    if not m or tokens(m[2])!=tokens(m[4]) or re.search(r'\bboth\s+(?:must|have|require)|\beach\s+(?:must|require)',q,re.I):return []
    tree,table=simple_root(current,schema)
    if table is None or tree.args.get('joins') or len(list(tree.find_all(exp.Select)))!=1:return []
    first=[c['name'] for c in table['columns'] if tokens(c['name'])==tokens(m[2])];second=[c['name'] for c in table['columns'] if tokens(c['name'])==tokens(m[6])]
    if len(first)!=1 or len(second)!=1:return []
    alias=tree.args['from'].this.alias_or_name
    # Only these two attribute conditions can be replaced; any extra population
    # filter must be represented by a separate supported operator first.
    where=tree.args.get('where')
    if not where or any(c.name.lower() not in {first[0].lower(),second[0].lower()} for c in where.find_all(exp.Column)):return []
    a,b=exp.column(first[0],table=alias),exp.column(second[0],table=alias)
    condition=exp.or_(exp.EQ(this=a.copy(),expression=exp.Literal.number(m[1])),exp.and_(exp.EQ(this=a.copy(),expression=exp.Literal.number(m[3])),exp.EQ(this=b,expression=exp.Literal.number(m[5]))))
    plan={'op':'project','columns':[sql_text(e) for e in tree.expressions],'input':{'op':'filter','predicate':sql_text(condition),'input':{'op':'scan','table':table['name'],'alias':alias}}}
    if tree.args.get('order'):return []
    return [candidate(plan,'branch_local_population',{'grammar':'either A or B but C; C belongs to second branch','attributes':[first[0],second[0]],'numbers':[m[1],m[3],m[5]],'assumptions':['nearest-branch interpretation of trailing but clause'],'preserved':['all numeric values','output fields','same root entity']})]

def extremum_or_filter(current,obs,schema):
    q=obs['task']['question'];tree,table=simple_root(current,schema)
    if table is None or not re.search(r'\bor\b',q,re.I) or not re.search(r'largest|highest|maximum|smallest|lowest|minimum',q,re.I):return []
    if tree.args.get('joins') or tree.args.get('group') or len(tree.expressions)!=1:return []
    agg=tree.expressions[0].unalias()
    if not isinstance(agg,(exp.Max,exp.Min)) or not isinstance(agg.this,exp.Column):return []
    where=tree.args.get('where');like=where.this if where else None
    if not isinstance(like,exp.Like) or not isinstance(like.this,exp.Column) or not isinstance(like.expression,exp.Literal) or like.this.name.lower()!=agg.this.name.lower():return []
    pattern=like.expression.this.strip('%')
    if not pattern or pattern.lower() not in q.lower():return []
    alias=tree.args['from'].this.alias_or_name;col=exp.column(agg.this.name,table=alias)
    max_query=exp.select(type(agg)(this=exp.column(agg.this.name))).from_(table['name'])
    predicate=exp.or_(exp.EQ(this=col.copy(),expression=exp.Subquery(this=max_query)),exp.Like(this=col.copy(),expression=like.expression.copy()))
    plan={'op':'project','columns':[sql_text(col)],'input':{'op':'filter','predicate':sql_text(predicate),'input':{'op':'scan','table':table['name'],'alias':alias}}}
    evidence={'grammar':'global extremum OR original literal predicate','aggregate_field':agg.this.name,'literal_pattern':like.expression.this,'assumptions':['existing SQL comparison semantics of the same field'],'preserved':['root entity','extremum field','literal branch']}
    value_union={'op':'union','distinct':True,'left':{'op':'aggregate','keys':[],'measures':[sql_text(type(agg)(this=col.copy()))],'input':{'op':'scan','table':table['name'],'alias':alias}},'right':{'op':'project','columns':[sql_text(col.copy())],'input':{'op':'filter','predicate':sql_text(exp.Like(this=col.copy(),expression=like.expression.copy())),'input':{'op':'scan','table':table['name'],'alias':alias}}}}
    return [candidate(value_union,'global_extremum_union_filter',{**evidence,'null_empty_policy':'preserve scalar aggregate NULL on empty input'},12),candidate(plan,'global_extremum_union_filter',{**evidence,'null_empty_policy':'entity rows only'},10)]

def participant_anti(current,obs,schema):
    q=obs['task']['question']
    if not re.search(r'\bsome\b.+?\b(?:playing|working|participating|taking)\b.+?\bbut not\b',q,re.I):return []
    tree=parse(current)
    if tree is None:return []
    # Require two question-backed category literals bound to the same event field.
    bindings=[]
    for eq in tree.find_all((exp.EQ,exp.NEQ)):
        if isinstance(eq.this,exp.Column) and isinstance(eq.expression,exp.Literal) and eq.expression.is_string and eq.expression.this.lower() in q.lower():bindings.append((eq.this.name,eq.expression.this))
    for membership in tree.find_all(exp.In):
        if isinstance(membership.this,exp.Column):
            bindings.extend((membership.this.name,v.this) for v in membership.expressions if isinstance(v,exp.Literal) and v.is_string and v.this.lower() in q.lower())
    fields={f.lower() for f,v in bindings}
    if len(fields)!=1 or len({v.lower() for f,v in bindings})!=2:return []
    field=bindings[0][0];values=list(dict.fromkeys(v for f,v in bindings));values.sort(key=lambda v:q.lower().find(v.lower()))
    output=re.search(r'\bhow many\s+(\w+)',q,re.I)
    if not output:return []
    owners=[(t,c['name']) for t in schema for c in t['columns'] if tokens(c['name'])==tokens(output[1])]
    if len(owners)!=1:return []
    root,attribute=owners[0];events=[t for t in schema if any(c['name'].lower()==field.lower() for c in t['columns']) and any(f['table'].lower()==root['name'].lower() for f in t['foreign_keys'])]
    if len(events)!=1:return []
    event=events[0];to_root=[f for f in event['foreign_keys'] if f['table'].lower()==root['name'].lower()];actors=[f for f in event['foreign_keys'] if f['table'].lower()!=root['name'].lower()]
    if len(to_root)!=1 or len(actors)!=1:return []
    mapping,actor=to_root[0],actors[0];positive=exp.EQ(this=exp.column(field,table='event'),expression=exp.Literal.string(values[0]));nonnull=exp.Not(this=exp.Is(this=exp.column(actor['from'],table='event'),expression=exp.Null()))
    plan={'op':'aggregate','keys':[],'measures':['COUNT(DISTINCT output.'+attribute+')'],'input':{'op':'map_join','table':root['name'],'alias':'output','condition':sql_text(exp.EQ(this=exp.column(mapping['from'],table='event'),expression=exp.column(mapping['to'],table='output'))),'input':{'op':'anti','related':event['name'],'alias':'negative_event','correlation':sql_text(exp.EQ(this=exp.column(actor['from'],table='negative_event'),expression=exp.column(actor['from'],table='event'))),'predicate':sql_text(exp.EQ(this=exp.column(field,table='negative_event'),expression=exp.Literal.string(values[1]))),'input':{'op':'filter','predicate':sql_text(exp.and_(positive,nonnull)),'input':{'op':'scan','table':event['name'],'alias':'event'}}}}}
    return [candidate(plan,'participant_anti_then_output_aggregate',{'negation_subject':'participant, not output attribute','category_field':field,'actor_fk':actor,'output_fk':mapping,'positive_category':values[0],'negative_category':values[1],'assumptions':['unique non-output FK is the relative-clause participant'],'preserved':['both category values','output state count','NULL participant exclusion']},12)]

def generate(current,obs,schema,extended=True,controls=True):
    items=[];reasons=[]
    if controls:
        # Reuse previously measured relation-plan controls, never old local edits.
        prior,diag=old_plans(current,obs,schema);reasons+=diag
        for c in prior:
            if c['path'][0] in ('count_entity_population_plan','count_boolean_feature_plan','independent_population_union'):
                items.append({**c,'ir':{'op':'sql_plan','sql':c['sql']},'source':'proof435_control_plan','evidence':{**c['evidence'],'historical_operator':True}})
    if extended:
        for provider in (branch_scope,extremum_or_filter,participant_anti):items+=provider(current,obs,schema)
    out=[]
    for c in items:
        reason=guard(c['sql'],obs,schema)
        if reason:reasons.append({'operator':c['path'][0],'reason':reason})
        else:out.append(c)
    if not out:reasons.append({'reason':'no_supported_ir_operator'})
    return sorted(out,key=lambda c:-c['confidence'])[:24],reasons
