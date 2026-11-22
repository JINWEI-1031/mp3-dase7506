"""Replacement certificate built only from runtime-visible contradictions."""
import re
from sqlglot import exp
from repair_rules import parse,sql_text
from step_pool192_rules import schema_errors
from step_space435_common import guard,question_values
from step123_agent import words
from step124_pool_search import sources

def filter_atoms(sql,schema):
    tree=parse(sql);out=set()
    if tree is None:return out
    for select in tree.find_all(exp.Select):
        src=sources(select,schema);where=select.args.get('where')
        if where is None:continue
        for node in where.find_all((exp.EQ,exp.NEQ,exp.GT,exp.GTE,exp.LT,exp.LTE,exp.Like,exp.Is,exp.In)):
            col=node.this
            if not isinstance(col,exp.Column):continue
            owners=[t['name'].lower() for a,t in src.items() if (not col.table or col.table.lower()==a.lower()) and any(c['name'].lower()==col.name.lower() for c in t['columns'])]
            if len(owners)!=1:continue
            if isinstance(node,exp.In) and not node.args.get('query'):
                out.update((owners[0],col.name.lower(),'EQ',str(logical_shape(v))) for v in node.expressions)
            elif isinstance(node.expression,(exp.Literal,exp.Null)):
                out.add((owners[0],col.name.lower(),type(node).__name__,str(logical_shape(node.expression))))
    return out

def logical_shape(node):
    if node is None:return None
    if isinstance(node,(exp.Paren,exp.Where)):return logical_shape(node.this)
    if isinstance(node,exp.Column):return ('column',node.name.lower())
    if isinstance(node,exp.Literal):
        value=node.this
        try:return ('number',float(value))
        except ValueError:return ('text',value.lower())
    if isinstance(node,exp.In) and not node.args.get('query'):
        return ('Or',tuple(sorted((('EQ',logical_shape(node.this),logical_shape(v)) for v in node.expressions),key=str)))
    if isinstance(node,(exp.And,exp.Or)):
        children=[]
        def collect(n):
            if isinstance(n,type(node)):collect(n.this);collect(n.expression)
            else:children.append(logical_shape(n))
        collect(node);return (type(node).__name__,tuple(sorted(children,key=str)))
    if isinstance(node,exp.Binary):return (type(node).__name__,logical_shape(node.this),logical_shape(node.expression))
    return type(node).__name__

def name_conflict(sql,obs,schema):
    tree=parse(sql)
    if not isinstance(tree,exp.Select):return []
    q=obs['task']['question'];headers=obs['example']['columns'];roots=[t for t in schema if all(any(c['name'].lower()==h.lower() for c in t['columns']) for h in headers)]
    if len(roots)!=1:return []
    root=roots[0];namecols=[c['name'] for c in root['columns'] if re.search(r'name$',c['name'],re.I)]
    if len(namecols)!=1:return []
    terms=words(root['name']);actual=[]
    for literal in tree.find_all(exp.Literal):
        if not literal.is_string or len(literal.this)<2:continue
        direct=any(re.search(r'\b'+re.escape(t)+r's?\s+(?:named\s+|called\s+)?'+re.escape(literal.this)+r'\b',q,re.I) for t in terms)
        if not direct:continue
        parent=literal.find_ancestor(exp.EQ)
        if parent is not None and isinstance(parent.this,exp.Column) and parent.this.name.lower()!=namecols[0].lower():actual.append({'kind':'explicit_root_name_bound_elsewhere','root':root['name'],'field':namecols[0],'value':literal.this})
    return actual

def certificate(parent,candidate,obs,schema,tier='preserve'):
    sql=candidate['sql'];old=parse(parent);new=parse(sql);kind=candidate['path'][0];contradictions=[];new_errors=[]
    if schema_errors(parent,obs,schema):contradictions.append('schema_or_output_contract')
    evidence=candidate.get('evidence',{})
    if kind=='branch_local_population' and old is not None and new is not None and logical_shape(old.args.get('where'))!=logical_shape(new.args.get('where')):contradictions.append('question_branch_scope')
    if kind=='global_extremum_union_filter' and old is not None and isinstance(old,exp.Select) and any(e.find(exp.AggFunc) for e in old.expressions):contradictions.append('disjunctive_population_collapsed_into_aggregate')
    if kind=='participant_anti_then_output_aggregate' and old is not None and (old.find(exp.Except) or old.find(exp.NEQ)):contradictions.append('negation_on_output_instead_of_participant')
    if kind=='count_entity_population_plan' and old is not None:
        wanted=evidence.get('count_field','').lower();old_count=old.find(exp.Count)
        if wanted and old_count and wanted not in {c.name.lower() for c in old_count.find_all(exp.Column)}:contradictions.append('public_count_object_disagrees')
    if kind=='independent_name_fk_plan' and name_conflict(parent,obs,schema):contradictions.append('explicit_named_entity_role')
    if not (new is not None and schema_errors(sql,obs,schema)==0):new_errors.append('new_schema_or_projection_conflict')
    reason=guard(sql,obs,schema)
    if reason:new_errors.append(reason)
    checks={'type':True,'role':True,'filter':True,'order':True,'aggregate_scope':True}
    removed=filter_atoms(parent,schema)-filter_atoms(sql,schema)
    if kind=='independent_name_fk_plan':
        wrong={x['value'].lower() for x in name_conflict(parent,obs,schema)}
        removed={a for a in removed if not any(v in a[3].lower() for v in wrong)}
    if kind=='participant_anti_then_output_aggregate':removed={a for a in removed if a[1]!=evidence.get('category_field','').lower()}
    if removed:checks['filter']=False;new_errors.append('parent_filter_atom_lost')
    if old is not None and new is not None and old.args.get('order'):
        if not new.args.get('order'):checks['order']=False;new_errors.append('parent_order_removed_without_evidence')
    # A newly inferred tie population never displaces an already resolved ranking.
    if kind=='extremal_population_output_plan':new_errors.append('unsupported_rank_population_replacement');checks['aggregate_scope']=False
    if kind=='independent_name_fk_plan' and not name_conflict(parent,obs,schema):new_errors.append('name_role_not_proved');checks['role']=False
    approved=tier=='sample' or bool(contradictions) and (tier=='contradiction' or not new_errors)
    return {'approved':approved,'tier':tier,'parent_contradictions':contradictions,'explains':kind,'preservation_checks':checks,'removed_filter_atoms':sorted(removed),'new_conflicts':new_errors,'new_assumptions':evidence.get('assumptions',[]),'public_match_required':True,'uses_labels':False}
