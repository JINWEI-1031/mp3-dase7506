"""Explicit set-branch output contracts; no relaxation of predicate preservation."""
from sqlglot import exp
from repair_rules import parse,sql_text
from step_proof435_evidence import certificate as prior_certificate

def branch_widths(tree):
    if tree is None:return []
    if isinstance(tree,exp.SetOperation):return branch_widths(tree.this)+branch_widths(tree.expression)
    if isinstance(tree,exp.Subquery):return branch_widths(tree.this)
    return [len(tree.expressions)] if isinstance(tree,exp.Select) else []

def certificate(parent,candidate,obs,schema,tier='preserve'):
    result=prior_certificate(parent,candidate,obs,schema,tier);old=parse(parent);new=parse(candidate['sql']);width=len(obs['example']['columns'])
    widths=branch_widths(old);new_widths=branch_widths(new)
    if widths and any(n!=width for n in widths):result['parent_contradictions'].append('set_branch_output_arity_disagrees')
    if not new_widths or any(n!=width for n in new_widths):result['new_conflicts'].append('new_set_branch_output_arity_conflict')
    def representation(value):return sql_text(value) if isinstance(value,exp.Expression) else str(value)
    for key in ('distinct','limit','having'):
        if old is not None and old.args.get(key) and (new is None or not new.args.get(key) or representation(old.args[key])!=representation(new.args[key])):result['new_conflicts'].append('parent_'+key+'_lost_without_evidence')
    # These two exploratory readings explicitly introduce unresolved antecedent
    # assumptions. A public match does not establish the intended role/scope.
    if candidate['path'][0] in ('branch_local_population','participant_anti_then_output_aggregate') and candidate.get('evidence',{}).get('assumptions'):
        result['new_conflicts'].append('unresolved_role_or_branch_scope_assumption');result['preservation_checks']['role']=False
    result['branch_widths']={'parent':widths,'candidate':new_widths,'required':width}
    result['approved']=tier=='sample' or bool(result['parent_contradictions']) and (tier=='contradiction' or not result['new_conflicts'])
    return result
