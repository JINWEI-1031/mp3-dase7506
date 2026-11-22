"""Question/role/type-grounded missing-edge hypotheses; public witness is not FK proof."""
import re
from sqlglot import exp
from repair_rules import parse,sql_text
from step124_pool_search import sources
from step_semantic192_rules import tokens,paths

def qi(v):return '"'+v.replace('"','""')+'"'
def family(t):
    t=t.upper()
    return 'number' if any(x in t for x in ['INT','REAL','NUM','DEC','FLOAT','DOUBLE']) else 'text' if any(x in t for x in ['CHAR','TEXT','CLOB']) else 'other'
def role_evidence(field,q):
    w=tokens(field)
    if re.search(r'(^id$|_id$|^id_|name|date|time|year|age|price|value|amount)',field,re.I):return False
    if w&tokens(q):return True
    return 'classroom' in w and bool(re.search(r'\bteach(?:er|ers|es|ing)?\b',q,re.I)) and 'grade' in tokens(q)

def output_role(table,q):
    words=[w for w in tokens(table) if w not in {'record','list','data','detail'}]
    return any(re.search(r'\b(?:names?|ids?|identifiers?)\s+of\s+(?:the\s+)?'+re.escape(w)+r's?\b',q,re.I) or re.search(r'\b(?:which|what|list|show|find|give)\s+(?:the\s+)?'+re.escape(w)+r's?\b',q,re.I) or re.search(r'\b'+re.escape(w)+r's?[\u2019\']\s*(?:first\s+)?names?\b',q,re.I) for w in words)

def request(current,obs,schema):
    tree=parse(current);q=obs['task']['question']
    if re.search(r'\b(?:both|neither|either|not|no|none|without|never|cannot)\b|\b(?:do|does|is|are)n[\u2019\']t\b',q,re.I):return None,'question_polarity_or_quantifier_unsupported'
    if re.search(r'\b(?:per|each|every)\b',q,re.I):return None,'explicit_partition_grain_unsupported'
    if not isinstance(tree,exp.Select) or len(list(tree.find_all(exp.Select)))!=1:return None,'not_simple_missing_filter_owner'
    if tree.args.get('group') or tree.args.get('having') or tree.find(exp.AggFunc):return None,'aggregate_grain_unsupported'
    root=tree.args.get('from');src=sources(tree,schema)
    if not root or not isinstance(root.this,exp.Table):return None,'missing_output_root'
    alias=root.this.alias_or_name;table=src.get(alias)
    if table is None or not tokens(table['name'])<=tokens(q) or not output_role(table['name'],q):return None,'question_root_role_not_linked'
    where=tree.args.get('where')
    if not where:return None,'missing_predicate'
    if where.find(exp.Or) or where.find(exp.Not):return None,'predicate_scope_unsupported'
    missing=[c for c in where.find_all(exp.Column) if not any(f['name'].lower()==c.name.lower() for f in table['columns'])]
    if len({c.name.lower() for c in missing})!=1:return None,'not_one_missing_filter_field'
    field=missing[0].name
    if not tokens(field)<=tokens(q):return None,'missing_field_not_question_grounded'
    owners=[t for t in schema if t is not table and any(c['name'].lower()==field.lower() for c in t['columns'])]
    if len(owners)!=1:return None,'filter_owner_ambiguous'
    target=owners[0]
    joins=tree.args.get('joins') or []
    if joins:
        if len(joins)!=1 or len(src)!=2 or not isinstance(joins[0].this,exp.Table) or joins[0].this.name.lower()!=target['name'].lower():return None,'unsupported_existing_join_path'
        j=joins[0]
        if j.args.get('side') or j.args.get('kind') not in (None,'INNER'):return None,'outer_join_population_unsupported'
        child_alias=j.this.alias_or_name
        for c in tree.find_all(exp.Column):
            if c.find_ancestor(exp.Join) is j or c.find_ancestor(exp.Where) is where:continue
            if c.table and c.table.lower()==child_alias.lower():return None,'filter_owner_used_outside_predicate'
    column_roles=[]
    for c in where.find_all(exp.Column):
        if c.name.lower()==field.lower():
            if c.table and c.table.lower() not in {alias.lower(),child_alias.lower() if joins else alias.lower()}:return None,'unknown_predicate_qualifier'
            column_roles.append('filter');continue
        if c.table:
            if c.table.lower()==alias.lower():column_roles.append('root')
            elif joins and c.table.lower()==child_alias.lower():column_roles.append('filter')
            else:return None,'unknown_predicate_qualifier'
        else:
            bindings=[a for a,tb in src.items() if any(z['name'].lower()==c.name.lower() for z in tb['columns'])]
            if bindings!=[alias]:return None,'ambiguous_other_predicate_field'
            column_roles.append('root')
    if paths(schema,table['name'],target['name']):return None,'declared_graph_already_connected'
    edges=[]
    for a in table['columns']:
        for b in target['columns']:
            if a['name'].lower()!=b['name'].lower() or family(a['type'])!=family(b['type']) or family(a['type'])=='other':continue
            if role_evidence(a['name'],q):edges.append({'from_table':table['name'],'from_field':a['name'],'to_table':target['name'],'to_field':b['name'],'type_family':family(a['type']),'role':'question root -> shared semantic grouping attribute -> missing filter owner','declared':False})
    if len(edges)!=1:return None,'zero_or_multiple_role_supported_edges'
    shared_scope=[a['name'] for a in table['columns'] for b in target['columns'] if a['name'].lower()==b['name'].lower() and a['name'].lower()!=edges[0]['from_field'].lower() and re.search(r'school|campus|year|term|department|division|branch',a['name'],re.I)]
    if shared_scope:return None,'possible_composite_or_temporal_context'
    edge=edges[0];a=qi(edge['from_field']);b=qi(edge['to_field']);x=qi(table['name']);y=qi(target['name']);pf=qi(field)
    if joins:
        on=joins[0].args.get('on')
        expected={(alias.lower(),edge['from_field'].lower()),(child_alias.lower(),edge['to_field'].lower())}
        if not isinstance(on,exp.EQ) or not isinstance(on.this,exp.Column) or not isinstance(on.expression,exp.Column) or {(on.this.table.lower(),on.this.name.lower()),(on.expression.table.lower(),on.expression.name.lower())}!=expected:return None,'existing_join_has_unpreserved_constraints'
    probe=f'SELECT (SELECT COUNT(*) FROM {x}) AS roots, (SELECT COUNT(*) FROM {x} WHERE {a} IS NULL) AS null_roots, (SELECT COUNT(*) FROM {x} r WHERE r.{a} IS NOT NULL AND NOT EXISTS(SELECT 1 FROM {y} f WHERE f.{b}=r.{a})) AS missing_matches, (SELECT COUNT(*) FROM (SELECT {b} FROM {y} WHERE {b} IS NOT NULL GROUP BY {b} HAVING COUNT(*)>1)) AS duplicate_groups, (SELECT COUNT(*) FROM (SELECT {b} FROM {y} WHERE {b} IS NOT NULL GROUP BY {b} HAVING COUNT(DISTINCT {pf})>1)) AS conflicting_filter_groups'
    return {'sql':probe,'edge':edge,'parent':current,'root_alias':alias,'filter_field':field,'filter_owner':target['name'],'column_roles':column_roles,'source':'graph439_missing_edge'},None

def compile_candidate(plan,obs):
    tree=parse(plan['parent']);edge=plan['edge'];outer='edge_root';inner='edge_filter';old_alias=plan['root_alias'];pred=tree.args['where'].this.copy()
    for c,role in zip(pred.find_all(exp.Column),plan['column_roles']):c.set('table',exp.to_identifier(inner if role=='filter' else outer))
    correlation=exp.EQ(this=exp.column(edge['to_field'],table=inner),expression=exp.column(edge['from_field'],table=outer))
    sub=exp.select(exp.Literal.number(1)).from_(exp.to_table(plan['filter_owner']).as_(inner)).where(exp.and_(correlation,pred))
    t=tree.copy();t.set('from',exp.From(this=exp.to_table(edge['from_table']).as_(outer)));t.set('joins',None);t.set('where',exp.Where(this=exp.Exists(this=sub)))
    for c in t.find_all(exp.Column):
        if c.find_ancestor(exp.Select) is t and (not c.table or c.table.lower()==old_alias.lower()):c.set('table',exp.to_identifier(outer))
    return {'sql':sql_text(t),'source':'graph439_edge','path':['undeclared_role_semijoin'],'confidence':12,'evidence':{'edge':edge,'contradiction':'question-grounded predicate field absent from current output table, uniquely owned in disconnected schema','preserved':'original projection, DISTINCT, root entity grain, predicate values/operators, order and limit','cardinality_policy':'EXISTS; duplicate filter rows cannot multiply output roots; NULL never matches','limitation':'public profile validates only this instance, not a hidden FK'}}

def apply(plan,feedback,obs):
    if not feedback.get('ok') or feedback.get('truncated') or len(feedback.get('rows',[]))!=1:return [],'invalid_or_truncated_relation_profile'
    roots,nulls,missing,duplicates,conflicts=feedback['rows'][0]
    if roots<=0 or missing or conflicts:return [],'empty_uncovered_or_conflicting_relation'
    c=compile_candidate(plan,obs);c['evidence']['public_profile']={'roots':roots,'null_roots':nulls,'missing_matches':missing,'duplicate_groups':duplicates,'conflicting_filter_groups':conflicts}
    return [c],None
