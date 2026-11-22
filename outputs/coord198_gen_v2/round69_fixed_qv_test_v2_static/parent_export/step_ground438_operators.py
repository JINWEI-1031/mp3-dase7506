"""Question-backed entity multiplicity probe, distinct from value grounding."""
import re
from sqlglot import exp
from repair_rules import parse,sql_text
from step124_pool_search import sources
from step_semantic192_rules import tokens

def ascending(tree,original):
    t=tree.copy()
    explicit_null_policy=bool(re.search(r'\bNULLS\s+(?:LAST|FIRST)\b',original,re.I))
    for e in t.args['order'].find_all(exp.Ordered):
        e.set('desc',False)
        if not explicit_null_policy:e.set('nulls_first',True)
    return t

def generate(current,obs,schema):
    tree=parse(current);q=obs['task']['question'];out=[]
    if not isinstance(tree,exp.Select):return [],'seed_not_select'
    if re.search(r'\bnot\s+both\b|\bneither\b|\b(?:both|either)\b',q,re.I):return [],'quantifier_scope_requires_explicit_plan'
    src=sources(tree,schema)
    if tree.args.get('distinct') and not re.search(r'\bdistinct|\bunique|different names',q,re.I) and tree.find(exp.Exists) and not tree.args.get('group'):
        root=tree.args.get('from');table=src.get(root.this.alias_or_name) if root else None
        if table and any(c.get('pk') for c in table['columns']) and tokens(table['name'])<=tokens(q):
            t=tree.copy();t.set('distinct',None);out.append({'sql':sql_text(t),'source':'ground438_entity_grain','path':['entity_grain_projection'],'evidence':{'root':table['name'],'contradiction':'DISTINCT projected attributes merges different qualified root entities','preserved':'same entity EXISTS population, WHERE, ordering, limit, output fields','assumption':'question asks entity list rather than distinct attribute set'},'confidence':8})
    return out,None if out else 'no_supported_entity_grain_operator'

def population_join(current,obs,schema,support=1):
    tree=parse(current);q=obs['task']['question'].lower()
    if not support:return [],'no_out_of_fold_training_support'
    if not isinstance(tree,exp.Select) or not re.search(r'\b(?:every|all)\b',q):return [],'no_explicit_complete_population'
    if re.search(r'\b(?:who|which|that|with|having|have|both|neither|without)\b|\bnot\s+both\b',q):return [],'related_population_or_negation_scope'
    if tree.args.get('group') or tree.args.get('having') or tree.find(exp.AggFunc) or len(list(tree.find_all(exp.Select)))!=1:return [],'aggregate_or_nested_scope'
    src=sources(tree,schema);root=tree.args.get('from')
    if not root or not isinstance(root.this,exp.Table):return [],'unsupported_root'
    alias=root.this.alias_or_name;owner=src.get(alias)
    if owner is None:return [],'missing_root_schema'
    if any(not isinstance(e.unalias(),exp.Column) or e.unalias().table.lower()!=alias.lower() for e in tree.expressions):return [],'output_not_explicit_root'
    joins=tree.args.get('joins') or []
    # The same role plan can arrive after the parent has already eliminated the
    # provenance join. Complete its remaining order constraint, without needing
    # the original draft as another seed or reintroducing the removed relation.
    if not joins:
        if not tree.args.get('order') or not re.search(r'\border(?:ed)?\s+by\b',q) or re.search(r'\bdescending|decreas|highest|largest|most|top|reverse|desc\b',q):return [],'complete_population_and_order_already_resolved'
        if not any(e.args.get('desc') for e in tree.args['order'].find_all(exp.Ordered)):return [],'complete_population_and_order_already_resolved'
        t=ascending(tree,current)
        return [{'sql':sql_text(t),'source':'ground438_role_operator','path':['complete_population_default_order'],'confidence':13,'evidence':{'root':owner['name'],'contradiction':'parent introduces DESC for bare ordered-by request; complete root population already restored','preserved':'entire AST except default sort direction; no new relation or predicate','ordering_assumption':'bare ordered by uses ascending convention; explicit descending/extreme requests excluded','training_support_count':support,'plan_stage':'resume order after parent population repair'}}],None
    if len(joins)!=1:return [],'requires_single_declared_provenance_join'
    join=joins[0]
    if join.args.get('side') or join.args.get('kind') not in (None,'INNER'):return [],'not_simple_inner'
    if not isinstance(join.this,exp.Table):return [],'unsupported_child'
    ca=join.this.alias_or_name;child=src.get(ca)
    if child is None:return [],'missing_child_schema'
    specific=tokens(child['name'])-tokens(owner['name'])
    if not specific or specific&tokens(q):return [],'question_mentions_related_role'
    links=[f for f in child['foreign_keys'] if f['table'].lower()==owner['name'].lower()]
    if len(links)!=1:return [],'ambiguous_declared_role'
    fk=links[0];on=join.args.get('on')
    if not isinstance(on,exp.EQ) or not isinstance(on.this,exp.Column) or not isinstance(on.expression,exp.Column):return [],'unsupported_join_predicate'
    actual={(on.this.table.lower(),on.this.name.lower()),(on.expression.table.lower(),on.expression.name.lower())}
    if actual!={(ca.lower(),fk['from'].lower()),(alias.lower(),fk['to'].lower())}:return [],'join_not_declared_role'
    for col in tree.find_all(exp.Column):
        if col.find_ancestor(exp.Join) is join:continue
        if col.table and col.table.lower()==ca.lower():return [],'child_or_ambiguous_attribute_used'
        if not col.table:
            owners=[a for a,tb in src.items() if any(c['name'].lower()==col.name.lower() for c in tb['columns'])]
            if owners!=[alias]:return [],'child_or_ambiguous_attribute_used'
    t=tree.copy();t.set('joins',None)
    evidence={'root':owner['name'],'child_role':child['name'],'declared_fk':fk,'contradiction':'INNER provenance join excludes zero-related root entities and duplicates multiply-related entities despite every/all request','preserved':'all original projections, root filters, ordering, distinct and limit; only unused declared join removed','training_support_count':support}
    out=[{'sql':sql_text(t),'source':'ground438_role_operator','path':['complete_population_unrequested_join'],'confidence':12,'evidence':evidence}]
    if t.args.get('order') and re.search(r'\border(?:ed)?\s+by\b',q) and not re.search(r'\bdescending|decreas|highest|largest|most|top|reverse|desc\b',q):
        ordered=list(t.args['order'].find_all(exp.Ordered))
        if any(e.args.get('desc') for e in ordered):
            a=ascending(t,current)
            out.insert(0,{'sql':sql_text(a),'source':'ground438_role_operator','path':['complete_population_default_order'],'confidence':13,'evidence':{**evidence,'preserved':'all projection/filter/range/order keys/DISTINCT/LIMIT; unrequested DESC replaced by default ASC','ordering_assumption':'bare ordered by uses ascending convention; explicit descending/extreme requests are excluded'}})
    return out,None
