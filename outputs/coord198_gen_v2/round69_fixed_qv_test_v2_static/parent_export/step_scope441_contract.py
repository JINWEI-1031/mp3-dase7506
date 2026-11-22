"""Admission prototype, deliberately not connected to the frozen parent."""
import re
from step_scope441_ir import generate,linked,table,column
from step_graph439_contract import plan_key
def approve(candidate,parent,obs,schema):
    ir=candidate['ir'];q=obs['task']['question']
    if ir['operator']=='entity_membership_count':
        if not linked(table(schema,ir['root_entity']),q):return False,'counted_entity_not_independently_named'
    elif ir['operator']=='extreme_population_projection':
        if not re.search(r'\ball\s+(?:the\s+)?tie(?:d|s)\b',q,re.I):return False,'question_tie_policy_not_established'
        output_edge=next(e for e in ir['event_edges'] if e['parent']==ir['root_entity'])
        f=column(table(schema,ir['event_entity']),output_edge['child_field'])
        if not f.get('not_null'):return False,'output_join_event_population_not_proven_preserved'
    else:return False,'unsupported_scope_operator'
    cs,_=generate(parent,obs,schema)
    target=plan_key(candidate['sql'],schema)
    if target is None or target==plan_key(parent,schema):return False,'same_or_ambiguous_scoped_plan'
    return any(plan_key(c['sql'],schema)==target and c['ir']==candidate['ir'] for c in cs),'regenerated_typed_scope_plan'
