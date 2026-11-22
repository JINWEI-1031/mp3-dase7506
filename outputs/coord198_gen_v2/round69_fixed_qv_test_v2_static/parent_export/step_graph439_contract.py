"""Bounded plan identity: scope ownership plus unused output labels only."""
from sqlglot import exp
from repair_rules import parse,sql_text
from step_ground438_contract import semantic_key
import step_graph439_edges as E,step_graph439_items as I

def plan_key(sql,schema):
    tree=parse(sql)
    if tree is None:return None
    for s in tree.find_all(exp.Select):
        aliases={e.alias.lower() for e in s.expressions if isinstance(e,exp.Alias)}
        referenced=False
        for k in ['order','group','having','where']:
            part=s.args.get(k)
            if part and any(not c.table and c.name.lower() in aliases for c in part.find_all(exp.Column)):referenced=True
        if referenced:return None
        s.set('expressions',[e.unalias().copy() for e in s.expressions])
    return semantic_key(sql_text(tree),schema)

def approve(candidate,parent,obs,schema,support):
    target=plan_key(candidate['sql'],schema)
    if target is None or target==plan_key(parent,schema):return False,'already_same_or_ambiguous_scope'
    if candidate['source']=='graph439_edge':
        p,reason=E.request(parent,obs,schema)
        if not p:return False,reason
        if p['edge']!=candidate['evidence']['edge']:return False,'edge_role_changed'
        c=E.compile_candidate(p,obs)
        return plan_key(c['sql'],schema)==target,'regenerated_edge_plan'
    if candidate['path'][0]=='binary_state_literal':
        p,reason=I.request(parent,obs,schema,support['binary_state_literal'])
        if not p:return False,reason
        b=candidate['evidence']['binding']
        if p['binding']['table'].lower()!=b['table'].lower() or p['binding']['field'].lower()!=b['field'].lower():return False,'state_field_role_changed'
        feedback={'ok':True,'truncated':False,'rows':list(candidate['evidence']['observed_frequencies'].items())}
        cs,reason=I.apply(p,feedback)
    else:cs,reason=I.generate(parent,obs,schema,support)
    return any(plan_key(c['sql'],schema)==target for c in cs),'regenerated_independent_item'
