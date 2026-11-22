from step_graph439_contract import approve as raw_approve
from step_graph439_qualified_items import binding_guard

def approve(candidate,parent,obs,schema,support):
    if candidate['source']=='graph439_edge':
        profile=candidate['evidence'].get('public_profile')
        if not profile or profile['roots']<=0 or profile['missing_matches'] or profile['conflicting_filter_groups']:return False,'missing_valid_public_relation_profile'
    elif candidate['source']=='graph439_item' and candidate['path'][0]=='binary_state_literal':
        why=binding_guard(candidate['evidence']['binding'],schema)
        if why:return False,why
    else:return False,'unqualified_repair_item'
    return raw_approve(candidate,parent,obs,schema,support)
